"""
optimize_noisy.py — Bayesian optimization for noise-optimal VQE circuits.

Finds the circuit configuration that minimizes energy error AFTER ZNE
mitigation under a given noise level. The optimal circuit under noise
may differ from the optimal noiseless circuit — this tool finds it.

Usage:
    uv run optimize_noisy.py --molecule h2 --noise 0.005 --n-trials 20
    uv run optimize_noisy.py --molecule lih --noise 0.005 --n-trials 25
    uv run optimize_noisy.py --molecule lih --noise 0.01 --n-trials 25 --time-budget 120
    uv run optimize_noisy.py --molecule lih --noise 0.005 --rank-only
"""

import argparse
import time
import traceback
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import noise as qml_noise
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
    build_device,
    get_zne_config,
)

PENALTY_VALUE = 100.0
ZNE_SCALE_FACTORS = (1.0, 2.0, 3.0)


def rank_excitations(
    hamiltonian: qml.operation.Operator,
    n_qubits: int,
    n_electrons: int,
    hf_state: np.ndarray,
    singles: list[list[int]],
    doubles: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Rank excitations by gradient magnitude at the HF state (noiseless).

    Returns (ranked_singles, ranked_doubles) sorted by |gradient| descending.
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

    single_grads = [(abs(float(grads[i])), singles[i]) for i in range(n_singles)]
    single_grads.sort(key=lambda x: x[0], reverse=True)
    ranked_singles = [s for _, s in single_grads]

    double_grads = [(abs(float(grads[n_singles + j])), doubles[j]) for j in range(n_doubles)]
    double_grads.sort(key=lambda x: x[0], reverse=True)
    ranked_doubles = [d for _, d in double_grads]

    print("Gradient ranking (noiseless):")
    for i, (g, s) in enumerate(single_grads):
        print(f"  Single {i}: wires={s} |grad|={g:.6f}")
    for i, (g, d) in enumerate(double_grads):
        print(f"  Double {i}: wires={d} |grad|={g:.6f}")

    return ranked_singles, ranked_doubles


def rank_excitations_noisy(
    hamiltonian: qml.operation.Operator,
    n_qubits: int,
    n_electrons: int,
    hf_state: np.ndarray,
    singles: list[list[int]],
    doubles: list[list[int]],
    noise_strength: float,
) -> tuple[list[list[int]], list[list[int]]]:
    """Rank excitations by gradient magnitude at the HF state under noise.

    Same as rank_excitations but uses a noisy device. Gradients reflect
    the noisy energy landscape, which may reorder excitation importance.
    """
    n_singles = len(singles)
    n_doubles = len(doubles)
    n_params = n_singles + n_doubles

    dev = build_device(n_qubits, noise_strength=noise_strength)

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

    single_grads = [(abs(float(grads[i])), singles[i]) for i in range(n_singles)]
    single_grads.sort(key=lambda x: x[0], reverse=True)
    ranked_singles = [s for _, s in single_grads]

    double_grads = [(abs(float(grads[n_singles + j])), doubles[j]) for j in range(n_doubles)]
    double_grads.sort(key=lambda x: x[0], reverse=True)
    ranked_doubles = [d for _, d in double_grads]

    print(f"Gradient ranking (noisy, p={noise_strength}):")
    for i, (g, s) in enumerate(single_grads):
        print(f"  Single {i}: wires={s} |grad|={g:.6f}")
    for i, (g, d) in enumerate(double_grads):
        print(f"  Double {i}: wires={d} |grad|={g:.6f}")

    return ranked_singles, ranked_doubles


def run_noisy_vqe_trial(
    hamiltonian: qml.operation.Operator,
    n_qubits: int,
    hf_state: np.ndarray,
    exact_energy: float,
    selected_singles: list[list[int]],
    selected_doubles: list[list[int]],
    step_size: float,
    optimizer_name: str,
    init_scale: float,
    noise_strength: float,
    time_budget: float,
    conv_threshold: float = 1e-8,
) -> tuple[float, float, float, int, float]:
    """Run a single noisy VQE trial with ZNE evaluation.

    Optimizes using the raw noisy cost function (no ZNE during optimization).
    After convergence, evaluates final params on ideal, noisy, and ZNE-mitigated
    cost functions.

    Returns:
        Tuple of (mitigated_error, noisy_error, ideal_error, n_params, wall_time).
    """
    n_s = len(selected_singles)
    n_d = len(selected_doubles)
    n_params = n_s + n_d

    if n_params == 0:
        return PENALTY_VALUE, PENALTY_VALUE, PENALTY_VALUE, 0, 0.0

    # Build devices
    dev_noisy = build_device(n_qubits, noise_strength=noise_strength)
    dev_ideal = qml.device("default.qubit", wires=n_qubits)

    # Noisy cost function — used for VQE optimization
    # parameter-shift required: default.mixed doesn't support autograd backprop
    @qml.qnode(dev_noisy, diff_method="parameter-shift")
    def cost_fn_noisy(params: pnp.tensor) -> float:
        qml.BasisState(hf_state, wires=range(n_qubits))
        for i, s in enumerate(selected_singles):
            qml.SingleExcitation(params[i], wires=s)
        for j, d in enumerate(selected_doubles):
            qml.DoubleExcitation(params[n_s + j], wires=d)
        return qml.expval(hamiltonian)

    # Ideal cost function — reference only
    @qml.qnode(dev_ideal)
    def cost_fn_ideal(params: pnp.tensor) -> float:
        qml.BasisState(hf_state, wires=range(n_qubits))
        for i, s in enumerate(selected_singles):
            qml.SingleExcitation(params[i], wires=s)
        for j, d in enumerate(selected_doubles):
            qml.DoubleExcitation(params[n_s + j], wires=d)
        return qml.expval(hamiltonian)

    # ZNE-mitigated cost function — for final evaluation
    zne_config = get_zne_config(scale_factors=ZNE_SCALE_FACTORS)
    cost_fn_mitigated = qml_noise.mitigate_with_zne(cost_fn_noisy, **zne_config)

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
    best_params = params.copy()
    prev_energy = float("inf")

    with TimeBudget(time_budget) as budget:
        for step in range(500):
            if budget.expired:
                break

            params, energy = opt.step_and_cost(cost_fn_noisy, params)
            energy = float(energy)

            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()

            if abs(energy - prev_energy) < conv_threshold:
                break

            prev_energy = energy

    # Final three-way evaluation
    energy_ideal = float(cost_fn_ideal(best_params))
    energy_noisy = float(cost_fn_noisy(best_params))
    energy_mitigated = float(cost_fn_mitigated(best_params))

    result_ideal = evaluate(energy_ideal, exact_energy)
    result_noisy = evaluate(energy_noisy, exact_energy)
    result_mitigated = evaluate(energy_mitigated, exact_energy)

    return (
        result_mitigated["energy_error"],
        result_noisy["energy_error"],
        result_ideal["energy_error"],
        n_params,
        budget.wall_time,
    )


def main() -> None:
    """Run noisy Bayesian optimization for noise-optimal circuit discovery."""
    parser = argparse.ArgumentParser(
        description="Noisy BO search for noise-optimal VQE configuration"
    )
    parser.add_argument("--molecule", type=str, default=MOLECULE,
                        choices=list(MOLECULES.keys()))
    parser.add_argument("--noise", type=float, default=0.005,
                        help="Depolarizing noise strength per gate (default: 0.005)")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET_SECONDS)
    parser.add_argument("--rank-only", action="store_true",
                        help="Print gradient rankings and exit (no BO)")
    args = parser.parse_args()

    molecule_key = args.molecule
    noise_strength = args.noise
    n_trials = args.n_trials
    trial_time_budget = args.time_budget

    # Setup
    print(f"=== Noisy Bayesian Optimization for {MOLECULES[molecule_key]['name']} ===")
    hamiltonian, n_qubits, n_electrons, hf_state = build_hamiltonian(molecule_key)
    exact_energy = compute_exact_energy(hamiltonian, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)

    print(f"Qubits: {n_qubits}, Electrons: {n_electrons}")
    print(f"Singles: {len(singles)}, Doubles: {len(doubles)}")
    print(f"Exact energy: {exact_energy:.8f} Ha")
    print(f"Noise strength: {noise_strength}")
    print(f"ZNE scale factors: {list(ZNE_SCALE_FACTORS)}")
    print()

    # Gradient ranking (noiseless — stable reference)
    print("Ranking excitations by gradient magnitude...")
    t0 = time.time()
    ranked_singles, ranked_doubles = rank_excitations(
        hamiltonian, n_qubits, n_electrons, hf_state, singles, doubles
    )
    print(f"Ranking took {time.time() - t0:.1f}s\n")

    # --rank-only mode: also compute noisy ranking and compare
    if args.rank_only:
        print("Computing noisy gradient ranking for comparison...")
        t0 = time.time()
        rank_excitations_noisy(
            hamiltonian, n_qubits, n_electrons, hf_state,
            singles, doubles, noise_strength
        )
        print(f"Noisy ranking took {time.time() - t0:.1f}s")
        return

    n_singles_total = len(singles)
    n_doubles_total = len(doubles)

    # Search space (same as optimize.py)
    space = [
        Integer(0, max(n_singles_total, 1), name="n_singles"),
        Integer(1, max(n_doubles_total, 2), name="n_doubles"),
        Real(0.01, 1.0, prior="log-uniform", name="step_size"),
        Categorical(["gradient_descent", "adam", "nesterov"], name="optimizer"),
        Real(0.0, 0.5, name="init_scale"),
    ]

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
        n_singles = min(n_singles, n_singles_total)
        n_doubles = min(n_doubles, n_doubles_total)
        sel_singles = ranked_singles[:n_singles]
        sel_doubles = ranked_doubles[:n_doubles]

        try:
            err_mitigated, err_noisy, err_ideal, n_params, wall_time = run_noisy_vqe_trial(
                hamiltonian=hamiltonian,
                n_qubits=n_qubits,
                hf_state=hf_state,
                exact_energy=exact_energy,
                selected_singles=sel_singles,
                selected_doubles=sel_doubles,
                step_size=step_size,
                optimizer_name=optimizer,
                init_scale=init_scale,
                noise_strength=noise_strength,
                time_budget=trial_time_budget,
            )
        except Exception:
            traceback.print_exc()
            err_mitigated = PENALTY_VALUE
            err_noisy = PENALTY_VALUE
            err_ideal = PENALTY_VALUE
            n_params = n_singles + n_doubles
            wall_time = 0.0

        improvement = err_noisy / err_mitigated if err_mitigated > 1e-12 else float("inf")

        trial_results.append({
            "trial": trial_num,
            "n_singles": n_singles,
            "n_doubles": n_doubles,
            "step_size": step_size,
            "optimizer": optimizer,
            "init_scale": init_scale,
            "energy_error_ideal": err_ideal,
            "energy_error_noisy": err_noisy,
            "energy_error_mitigated": err_mitigated,
            "improvement_factor": improvement,
            "n_params": n_singles + n_doubles,
            "wall_time": wall_time,
        })

        print(
            f"Trial {trial_num:3d}/{n_trials}: "
            f"n_s={n_singles} n_d={n_doubles} "
            f"step={step_size:.2f} opt={optimizer} "
            f"init={init_scale:.2f} noise={noise_strength}"
        )
        print(
            f"  -> ideal={err_ideal:.4f} Ha | noisy={err_noisy:.4f} Ha | "
            f"mitigated={err_mitigated:.4f} Ha | improvement={improvement:.1f}x "
            f"({wall_time:.1f}s)"
        )

        return err_mitigated

    # Run BO
    print(f"Starting {n_trials} noisy BO trials (time budget {trial_time_budget}s per trial)...\n")
    gp_minimize(
        objective,
        space,
        n_calls=n_trials,
        n_random_starts=min(5, n_trials // 3),
        acq_func="EI",
        random_state=42,
        verbose=False,
    )

    # --- Full UCCSD baseline ---
    print(f"\n{'=' * 50}")
    print("Running full UCCSD baseline (all excitations, same noise)...")
    try:
        uccsd_mit, uccsd_noisy, uccsd_ideal, uccsd_params, uccsd_time = run_noisy_vqe_trial(
            hamiltonian=hamiltonian,
            n_qubits=n_qubits,
            hf_state=hf_state,
            exact_energy=exact_energy,
            selected_singles=ranked_singles,
            selected_doubles=ranked_doubles,
            step_size=0.4,
            optimizer_name="nesterov",
            init_scale=0.0,
            noise_strength=noise_strength,
            time_budget=trial_time_budget * 2,
        )
    except Exception:
        traceback.print_exc()
        uccsd_mit = PENALTY_VALUE
        uccsd_noisy = PENALTY_VALUE
        uccsd_ideal = PENALTY_VALUE
        uccsd_params = n_singles_total + n_doubles_total
        uccsd_time = 0.0

    uccsd_improvement = uccsd_noisy / uccsd_mit if uccsd_mit > 1e-12 else float("inf")
    print(
        f"Full UCCSD ({uccsd_params} params): "
        f"ideal={uccsd_ideal:.4f} Ha | noisy={uccsd_noisy:.4f} Ha | "
        f"mitigated={uccsd_mit:.4f} Ha | improvement={uccsd_improvement:.1f}x "
        f"({uccsd_time:.1f}s)"
    )

    # --- Report ---
    best_idx = min(range(len(trial_results)),
                   key=lambda i: trial_results[i]["energy_error_mitigated"])
    best = trial_results[best_idx]
    chem_acc = best["energy_error_mitigated"] < CHEMICAL_ACCURACY_HA

    print(f"\n{'=' * 50}")
    print(f"=== Noise-Optimal Circuit Discovery Results ===")
    print(f"Molecule: {MOLECULES[molecule_key]['name']} ({n_qubits} qubits, {n_electrons} electrons)")
    print(f"Noise: {noise_strength}")
    print(f"Trials: {n_trials}")
    print(f"\nBest mitigated error: {best['energy_error_mitigated']:.8f} Ha "
          f"({best['energy_error_mitigated'] * 1000:.4f} mHa)")
    print(f"Chemical accuracy: {chem_acc}")

    print(f"\nBest configuration:")
    print(f"  n_singles: {best['n_singles']}"
          + (" (all)" if best["n_singles"] == n_singles_total else ""))
    print(f"  n_doubles: {best['n_doubles']}"
          + (" (all)" if best["n_doubles"] == n_doubles_total else ""))
    print(f"  step_size: {best['step_size']:.4f}")
    print(f"  optimizer: {best['optimizer']}")
    print(f"  init_scale: {best['init_scale']:.2f}"
          + (" (zero init)" if best["init_scale"] == 0.0 else ""))
    print(f"  n_params: {best['n_params']}")

    # Comparison with full UCCSD
    print(f"\nComparison with full UCCSD:")
    print(f"  Full UCCSD ({uccsd_params} params) mitigated: "
          f"{uccsd_mit:.8f} Ha ({uccsd_mit * 1000:.4f} mHa)")
    print(f"  Best BO    ({best['n_params']} params) mitigated: "
          f"{best['energy_error_mitigated']:.8f} Ha "
          f"({best['energy_error_mitigated'] * 1000:.4f} mHa)")
    if uccsd_mit > 1e-12 and best["energy_error_mitigated"] > 1e-12:
        ratio = uccsd_mit / best["energy_error_mitigated"]
        if ratio > 1:
            print(f"  BO advantage: {ratio:.1f}x better error with "
                  f"{100 * (1 - best['n_params'] / uccsd_params):.0f}% fewer params")
        else:
            print(f"  Full UCCSD wins: {1/ratio:.1f}x better than best BO subset")

    # Top 5 trials
    sorted_trials = sorted(trial_results, key=lambda t: t["energy_error_mitigated"])
    print(f"\nTop 5 trials:")
    print(f"  {'#':>3}  {'n_s':>3}  {'n_d':>3}  {'params':>6}  {'step':>6}  "
          f"{'optimizer':>10}  {'init':>5}  "
          f"{'ideal_mHa':>9}  {'noisy_mHa':>9}  {'mitigated_mHa':>13}  {'improv':>6}")
    for t in sorted_trials[:5]:
        print(
            f"  {t['trial']:3d}  {t['n_singles']:3d}  {t['n_doubles']:3d}  "
            f"{t['n_params']:6d}  {t['step_size']:6.2f}  "
            f"{t['optimizer']:>10}  {t['init_scale']:5.2f}  "
            f"{t['energy_error_ideal'] * 1000:9.4f}  "
            f"{t['energy_error_noisy'] * 1000:9.4f}  "
            f"{t['energy_error_mitigated'] * 1000:13.4f}  "
            f"{t['improvement_factor']:6.1f}x"
        )

    # Pareto front (mitigated error vs params)
    print(f"\nPareto front (mitigated error vs params):")
    seen_params: set[int] = set()
    pareto: list[dict[str, Any]] = []
    for t in sorted_trials:
        if t["n_params"] not in seen_params and t["energy_error_mitigated"] < PENALTY_VALUE:
            seen_params.add(t["n_params"])
            pareto.append(t)
    pareto.sort(key=lambda t: t["n_params"])
    filtered_pareto: list[dict[str, Any]] = []
    best_error_so_far = float("inf")
    for t in reversed(pareto):
        if t["energy_error_mitigated"] <= best_error_so_far:
            best_error_so_far = t["energy_error_mitigated"]
            filtered_pareto.append(t)
    filtered_pareto.reverse()
    for t in filtered_pareto:
        acc_str = "*" if t["energy_error_mitigated"] < CHEMICAL_ACCURACY_HA else " "
        print(f"  {t['n_params']:3d} params:  mitigated = "
              f"{t['energy_error_mitigated'] * 1000:.4f} mHa {acc_str}")

    # Save TSV
    tsv_path = f"optimize_noisy_results_{molecule_key}_{noise_strength}.tsv"
    with open(tsv_path, "w") as f:
        headers = [
            "trial", "n_singles", "n_doubles", "step_size", "optimizer",
            "init_scale", "energy_error_ideal", "energy_error_noisy",
            "energy_error_mitigated", "improvement_factor", "n_params", "wall_time",
        ]
        f.write("\t".join(headers) + "\n")
        for t in trial_results:
            row = [
                str(t["trial"]),
                str(t["n_singles"]),
                str(t["n_doubles"]),
                f"{t['step_size']:.6f}",
                t["optimizer"],
                f"{t['init_scale']:.4f}",
                f"{t['energy_error_ideal']:.10f}",
                f"{t['energy_error_noisy']:.10f}",
                f"{t['energy_error_mitigated']:.10f}",
                f"{t['improvement_factor']:.4f}",
                str(t["n_params"]),
                f"{t['wall_time']:.2f}",
            ]
            f.write("\t".join(row) + "\n")
    print(f"\nResults saved to {tsv_path}")


if __name__ == "__main__":
    main()
