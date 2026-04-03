"""
optimize.py — Bayesian optimization for VQE circuit configuration.

Systematically searches for the optimal excitation subset and
hyperparameters using Gaussian Process-based Bayesian optimization.

Usage:
    uv run optimize.py --molecule h2 --n-trials 30
    uv run optimize.py --molecule lih --n-trials 50
    uv run optimize.py --molecule beh2 --n-trials 30 --time-budget 120
"""

import argparse
import time
import traceback
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

from prepare import (
    MOLECULE,
    MOLECULES,
    TIME_BUDGET_SECONDS,
    CHEMICAL_ACCURACY_HA,
    build_hamiltonian,
    compute_exact_energy,
    evaluate,
    TimeBudget,
)

PENALTY_VALUE = 100.0


def rank_excitations(
    hamiltonian: qml.operation.Operator,
    n_qubits: int,
    n_electrons: int,
    hf_state: np.ndarray,
    singles: list[list[int]],
    doubles: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Rank excitations by gradient magnitude at the Hartree-Fock state.

    Builds a circuit with all excitations at params=0, computes gradients,
    and returns excitations sorted by |gradient| descending.

    Args:
        hamiltonian: Qubit Hamiltonian operator.
        n_qubits: Number of qubits.
        n_electrons: Number of active electrons.
        hf_state: Hartree-Fock occupation state vector.
        singles: List of single excitation wire pairs.
        doubles: List of double excitation wire quads.

    Returns:
        Tuple of (ranked_singles, ranked_doubles) sorted by |gradient| descending.
    """
    n_singles = len(singles)
    n_doubles = len(doubles)
    n_params = n_singles + n_doubles

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params: pnp.tensor) -> float:
        qml.BasisState(hf_state, wires=range(n_qubits))
        for i, s in enumerate(singles):
            qml.SingleExcitation(params[i], wires=s)
        for j, d in enumerate(doubles):
            qml.DoubleExcitation(params[n_singles + j], wires=d)
        return qml.expval(hamiltonian)

    params = pnp.zeros(n_params, requires_grad=True)
    grads = qml.grad(circuit)(params)

    # Rank singles
    single_grads = [(abs(float(grads[i])), singles[i]) for i in range(n_singles)]
    single_grads.sort(key=lambda x: x[0], reverse=True)
    ranked_singles = [s for _, s in single_grads]

    # Rank doubles
    double_grads = [(abs(float(grads[n_singles + j])), doubles[j]) for j in range(n_doubles)]
    double_grads.sort(key=lambda x: x[0], reverse=True)
    ranked_doubles = [d for _, d in double_grads]

    print("Gradient ranking complete:")
    for i, (g, s) in enumerate(single_grads):
        print(f"  Single {i}: wires={s} |grad|={g:.6f}")
    for i, (g, d) in enumerate(double_grads):
        print(f"  Double {i}: wires={d} |grad|={g:.6f}")

    return ranked_singles, ranked_doubles


def run_vqe_trial(
    hamiltonian: qml.operation.Operator,
    n_qubits: int,
    hf_state: np.ndarray,
    exact_energy: float,
    selected_singles: list[list[int]],
    selected_doubles: list[list[int]],
    step_size: float,
    optimizer_name: str,
    init_scale: float,
    time_budget: float,
    conv_threshold: float = 1e-8,
) -> tuple[float, int, float]:
    """Run a single VQE optimization with the given configuration.

    Args:
        hamiltonian: Qubit Hamiltonian operator.
        n_qubits: Number of qubits.
        hf_state: Hartree-Fock occupation state vector.
        exact_energy: Exact ground-state energy for error calculation.
        selected_singles: Single excitation wire pairs to include.
        selected_doubles: Double excitation wire quads to include.
        step_size: Optimizer learning rate.
        optimizer_name: One of 'gradient_descent', 'adam', 'nesterov'.
        init_scale: Parameter initialization scale (0 = zero init).
        time_budget: Wall-clock time budget in seconds.
        conv_threshold: Convergence threshold for energy change.

    Returns:
        Tuple of (energy_error, n_params, wall_time).
    """
    n_s = len(selected_singles)
    n_d = len(selected_doubles)
    n_params = n_s + n_d

    if n_params == 0:
        return PENALTY_VALUE, 0, 0.0

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def cost_fn(params: pnp.tensor) -> float:
        qml.BasisState(hf_state, wires=range(n_qubits))
        for i, s in enumerate(selected_singles):
            qml.SingleExcitation(params[i], wires=s)
        for j, d in enumerate(selected_doubles):
            qml.DoubleExcitation(params[n_s + j], wires=d)
        return qml.expval(hamiltonian)

    # Initialize parameters
    if init_scale == 0.0:
        params = pnp.zeros(n_params, requires_grad=True)
    else:
        params = pnp.array(
            np.random.uniform(-init_scale, init_scale, n_params),
            requires_grad=True,
        )

    # Select optimizer
    if optimizer_name == "gradient_descent":
        opt = qml.GradientDescentOptimizer(stepsize=step_size)
    elif optimizer_name == "adam":
        opt = qml.AdamOptimizer(stepsize=step_size)
    elif optimizer_name == "nesterov":
        opt = qml.NesterovMomentumOptimizer(stepsize=step_size)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    best_energy = float("inf")
    prev_energy = float("inf")

    with TimeBudget(time_budget) as budget:
        for step in range(500):
            if budget.expired:
                break

            params, energy = opt.step_and_cost(cost_fn, params)
            energy = float(energy)

            if energy < best_energy:
                best_energy = energy

            if abs(energy - prev_energy) < conv_threshold:
                break

            prev_energy = energy

    results = evaluate(best_energy, exact_energy)
    return results["energy_error"], n_params, budget.wall_time


def main() -> None:
    """Run Bayesian optimization to find the best VQE circuit configuration."""
    parser = argparse.ArgumentParser(
        description="BO search for optimal VQE configuration"
    )
    parser.add_argument("--molecule", type=str, default=MOLECULE)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET_SECONDS)
    args = parser.parse_args()

    molecule_key = args.molecule
    n_trials = args.n_trials
    trial_time_budget = args.time_budget

    # Setup
    print(f"=== Bayesian Optimization for {MOLECULES[molecule_key]['name']} ===")
    hamiltonian, n_qubits, n_electrons, hf_state = build_hamiltonian(molecule_key)
    exact_energy = compute_exact_energy(hamiltonian, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)

    print(f"Qubits: {n_qubits}, Electrons: {n_electrons}")
    print(f"Singles: {len(singles)}, Doubles: {len(doubles)}")
    print(f"Exact energy: {exact_energy:.8f} Ha")
    print()

    # Gradient ranking (once)
    print("Ranking excitations by gradient magnitude...")
    t0 = time.time()
    ranked_singles, ranked_doubles = rank_excitations(
        hamiltonian, n_qubits, n_electrons, hf_state, singles, doubles
    )
    print(f"Ranking took {time.time() - t0:.1f}s\n")

    n_singles_total = len(singles)
    n_doubles_total = len(doubles)

    # Define search space (handle edge cases where ranges collapse)
    space = [
        Integer(0, max(n_singles_total, 1), name="n_singles"),
        Integer(1, max(n_doubles_total, 2), name="n_doubles"),
        Real(0.01, 1.0, prior="log-uniform", name="step_size"),
        Categorical(["gradient_descent", "adam", "nesterov"], name="optimizer"),
        Real(0.0, 0.5, name="init_scale"),
    ]

    # Track all trial results for reporting
    trial_results: list[dict[str, Any]] = []

    @use_named_args(space)
    def objective(
        n_singles: int,
        n_doubles: int,
        step_size: float,
        optimizer: str,
        init_scale: float,
    ) -> float:
        trial_num = len(trial_results) + 1
        # Clamp to actual available excitations
        n_singles = min(n_singles, n_singles_total)
        n_doubles = min(n_doubles, n_doubles_total)
        sel_singles = ranked_singles[:n_singles]
        sel_doubles = ranked_doubles[:n_doubles]

        try:
            energy_error, n_params, wall_time = run_vqe_trial(
                hamiltonian=hamiltonian,
                n_qubits=n_qubits,
                hf_state=hf_state,
                exact_energy=exact_energy,
                selected_singles=sel_singles,
                selected_doubles=sel_doubles,
                step_size=step_size,
                optimizer_name=optimizer,
                init_scale=init_scale,
                time_budget=trial_time_budget,
            )
        except Exception:
            traceback.print_exc()
            energy_error = PENALTY_VALUE
            n_params = n_singles + n_doubles
            wall_time = 0.0

        trial_results.append(
            {
                "trial": trial_num,
                "n_singles": n_singles,
                "n_doubles": n_doubles,
                "step_size": step_size,
                "optimizer": optimizer,
                "init_scale": init_scale,
                "energy_error": energy_error,
                "n_params": n_singles + n_doubles,
                "wall_time": wall_time,
            }
        )

        print(
            f"Trial {trial_num:3d}/{n_trials}: "
            f"n_s={n_singles} n_d={n_doubles} "
            f"step={step_size:.4f} opt={optimizer} "
            f"init={init_scale:.2f} "
            f"-> error={energy_error:.8f} Ha ({wall_time:.1f}s)"
        )

        return energy_error

    # Run BO
    print(f"Starting {n_trials} BO trials (time budget {trial_time_budget}s per trial)...\n")
    result = gp_minimize(
        objective,
        space,
        n_calls=n_trials,
        n_random_starts=min(5, n_trials // 3),
        acq_func="EI",
        random_state=42,
        verbose=False,
    )

    # Find best trial
    best_idx = min(range(len(trial_results)), key=lambda i: trial_results[i]["energy_error"])
    best = trial_results[best_idx]
    chem_acc = best["energy_error"] < CHEMICAL_ACCURACY_HA

    # Report
    print(f"\n{'=' * 50}")
    print(f"=== Bayesian Optimization Results ===")
    print(f"Molecule: {MOLECULES[molecule_key]['name']} ({n_qubits} qubits, {n_electrons} electrons)")
    print(f"Trials: {n_trials}")
    print(f"Best energy_error: {best['energy_error']:.8f} Ha ({best['energy_error'] * 1000:.4f} mHa)")
    print(f"Chemical accuracy: {chem_acc}")

    print(f"\nBest configuration:")
    print(f"  n_singles: {best['n_singles']}" + (f" (all)" if best["n_singles"] == n_singles_total else ""))
    print(f"  n_doubles: {best['n_doubles']}" + (f" (all)" if best["n_doubles"] == n_doubles_total else ""))
    print(f"  step_size: {best['step_size']:.4f}")
    print(f"  optimizer: {best['optimizer']}")
    print(f"  init_scale: {best['init_scale']:.2f}" + (" (zero init)" if best["init_scale"] == 0.0 else ""))
    print(f"  n_params: {best['n_params']}")

    # Top 5 trials
    sorted_trials = sorted(trial_results, key=lambda t: t["energy_error"])
    print(f"\nTop 5 trials:")
    print(f"  {'#':>3}  {'n_singles':>9}  {'n_doubles':>9}  {'step_size':>9}  {'optimizer':>16}  {'init_scale':>10}  {'energy_error':>14}  {'n_params':>8}")
    for t in sorted_trials[:5]:
        print(
            f"  {t['trial']:3d}  {t['n_singles']:9d}  {t['n_doubles']:9d}  "
            f"{t['step_size']:9.4f}  {t['optimizer']:>16}  {t['init_scale']:10.2f}  "
            f"{t['energy_error']:14.8f}  {t['n_params']:8d}"
        )

    # Recommended config
    print(f"\nRecommended circuit.py config:")
    print(f"  N_SINGLES = {best['n_singles']}")
    print(f"  N_DOUBLES = {best['n_doubles']}")
    print(f"  STEP_SIZE = {best['step_size']:.4f}")
    print(f"  OPTIMIZER = \"{best['optimizer']}\"")
    print(f"  INIT_SCALE = {best['init_scale']:.2f}")
    print(f"  CONVERGENCE_THRESHOLD = 1e-8")

    # Pareto front (accuracy vs params)
    print(f"\nPareto front (accuracy vs params):")
    seen_params: set[int] = set()
    pareto: list[dict[str, Any]] = []
    for t in sorted_trials:
        if t["n_params"] not in seen_params and t["energy_error"] < PENALTY_VALUE:
            seen_params.add(t["n_params"])
            pareto.append(t)
    pareto.sort(key=lambda t: t["n_params"])
    # Filter to actual Pareto-optimal points (non-dominated)
    filtered_pareto: list[dict[str, Any]] = []
    best_error_so_far = float("inf")
    for t in reversed(pareto):
        if t["energy_error"] <= best_error_so_far:
            best_error_so_far = t["energy_error"]
            filtered_pareto.append(t)
    filtered_pareto.reverse()
    for t in filtered_pareto:
        acc_str = "*" if t["energy_error"] < CHEMICAL_ACCURACY_HA else " "
        print(f"  {t['n_params']:3d} params:  error = {t['energy_error'] * 1000:.4f} mHa {acc_str}")

    # Save TSV
    tsv_path = f"optimize_results_{molecule_key}.tsv"
    with open(tsv_path, "w") as f:
        headers = ["trial", "n_singles", "n_doubles", "step_size", "optimizer", "init_scale", "energy_error", "energy_error_mha", "n_params", "wall_time"]
        f.write("\t".join(headers) + "\n")
        for t in trial_results:
            row = [
                str(t["trial"]),
                str(t["n_singles"]),
                str(t["n_doubles"]),
                f"{t['step_size']:.6f}",
                t["optimizer"],
                f"{t['init_scale']:.4f}",
                f"{t['energy_error']:.10f}",
                f"{t['energy_error'] * 1000:.6f}",
                str(t["n_params"]),
                f"{t['wall_time']:.2f}",
            ]
            f.write("\t".join(row) + "\n")
    print(f"\nResults saved to {tsv_path}")


if __name__ == "__main__":
    main()
