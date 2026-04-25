"""
validate_sweep.py — Direct excitation count sweep for noise-optimal circuit validation.

Validates that shallower circuits outperform full UCCSD under depolarizing noise + ZNE,
independent of BO sampling artifacts or optimization difficulty confounds.

Two evaluation modes per (n_doubles, noise_level):
  Mode A — Noisy-optimized: params trained under noise, evaluated with ZNE
  Mode B — Fixed-params: params trained noiseless, evaluated with noisy+ZNE (no re-opt)

If both modes show the same pattern, the finding is about noise accumulation (real physics),
not optimization difficulty (confound).

Usage:
    uv run validate_sweep.py --molecule h2
    uv run validate_sweep.py --molecule lih
"""

import argparse
import sys
import traceback
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import noise as qml_noise
from pennylane import numpy as pnp

from prepare import (
    MOLECULES,
    CHEMICAL_ACCURACY_HA,
    build_hamiltonian,
    compute_exact_energy,
    evaluate,
    TimeBudget,
    build_device,
    get_zne_config,
    molecule_choice,
)
from optimize_noisy import rank_excitations, run_noisy_vqe_trial

# Fixed hyperparameters — NOT searched
STEP_SIZE = 0.4
OPTIMIZER = "nesterov"
INIT_SCALE = 0.0
CONV_THRESHOLD = 1e-8
TIME_BUDGET = 300
ZNE_SCALE_FACTORS = (1.0, 2.0, 3.0)

NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01]


def run_noiseless_optimization(
    hamiltonian: qml.operation.Operator,
    n_qubits: int,
    hf_state: np.ndarray,
    exact_energy: float,
    selected_singles: list[list[int]],
    selected_doubles: list[list[int]],
    time_budget: float,
) -> tuple[float, np.ndarray, float]:
    """Optimize VQE on noiseless device, return (ideal_error, best_params, wall_time)."""
    n_s = len(selected_singles)
    n_d = len(selected_doubles)
    n_params = n_s + n_d

    if n_params == 0:
        return 100.0, np.array([]), 0.0

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def cost_fn(params: pnp.tensor) -> float:
        qml.BasisState(hf_state, wires=range(n_qubits))
        for i, s in enumerate(selected_singles):
            qml.SingleExcitation(params[i], wires=s)
        for j, d in enumerate(selected_doubles):
            qml.DoubleExcitation(params[n_s + j], wires=d)
        return qml.expval(hamiltonian)

    params = pnp.zeros(n_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=STEP_SIZE)

    best_energy = float("inf")
    best_params = params.copy()
    prev_energy = float("inf")

    with TimeBudget(time_budget) as budget:
        for step in range(500):
            if budget.expired:
                break
            params, energy = opt.step_and_cost(cost_fn, params)
            energy = float(energy)
            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()
            if abs(energy - prev_energy) < CONV_THRESHOLD:
                break
            prev_energy = energy

    result = evaluate(best_energy, exact_energy)
    return result["energy_error"], np.array(best_params), budget.wall_time


def evaluate_fixed_params(
    hamiltonian: qml.operation.Operator,
    n_qubits: int,
    hf_state: np.ndarray,
    exact_energy: float,
    params: np.ndarray,
    selected_singles: list[list[int]],
    selected_doubles: list[list[int]],
    noise_strength: float,
) -> tuple[float, float]:
    """Evaluate pre-optimized params on noisy+ZNE without re-optimizing.

    Returns (mitigated_error, noisy_error).
    """
    n_s = len(selected_singles)

    dev_noisy = build_device(n_qubits, noise_strength=noise_strength)
    eval_params = pnp.array(params, requires_grad=False)

    @qml.qnode(dev_noisy, diff_method="parameter-shift")
    def cost_fn_noisy(p: pnp.tensor) -> float:
        qml.BasisState(hf_state, wires=range(n_qubits))
        for i, s in enumerate(selected_singles):
            qml.SingleExcitation(p[i], wires=s)
        for j, d in enumerate(selected_doubles):
            qml.DoubleExcitation(p[n_s + j], wires=d)
        return qml.expval(hamiltonian)

    zne_config = get_zne_config(scale_factors=ZNE_SCALE_FACTORS)
    cost_fn_mitigated = qml_noise.mitigate_with_zne(cost_fn_noisy, **zne_config)

    energy_noisy = float(cost_fn_noisy(eval_params))
    energy_mitigated = float(cost_fn_mitigated(eval_params))

    noisy_result = evaluate(energy_noisy, exact_energy)
    mitigated_result = evaluate(energy_mitigated, exact_energy)

    return mitigated_result["energy_error"], noisy_result["energy_error"]


def print_results_table(
    results: list[dict[str, Any]], molecule_key: str, n_doubles_total: int
) -> None:
    """Print clean comparison tables to stdout."""
    for variant in ["all_singles", "zero_singles"]:
        vr = [r for r in results if r["variant"] == variant]
        if not vr:
            continue

        n_s = vr[0]["n_singles"]
        print(f"\n{'=' * 80}")
        print(f"Variant: {variant} (n_singles={n_s})")
        print(f"{'=' * 80}")

        noise_levels = sorted(set(r["noise_level"] for r in vr))

        # Table 1: Mode A — noisy-optimized
        print(f"\nMode A — Noisy-Optimized (ZNE-mitigated error, mHa):")
        header = f"  {'n_d':>3}  {'params':>6}"
        for nl in noise_levels:
            label = "noiseless" if nl == 0.0 else f"p={nl}"
            header += f"  {label:>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for n_d in range(1, n_doubles_total + 1):
            row_str = f"  {n_d:3d}  {n_s + n_d:6d}"
            for nl in noise_levels:
                match = [r for r in vr if r["n_doubles"] == n_d and r["noise_level"] == nl]
                if match:
                    err = match[0]["noisy_opt_mitigated_error"]
                    acc = "*" if err < CHEMICAL_ACCURACY_HA else " "
                    row_str += f"  {err * 1000:9.4f}{acc}"
                else:
                    row_str += f"  {'N/A':>10}"
            print(row_str)

        # Table 2: Mode B — fixed-params
        print(f"\nMode B — Fixed-Params (ZNE-mitigated error, mHa):")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for n_d in range(1, n_doubles_total + 1):
            row_str = f"  {n_d:3d}  {n_s + n_d:6d}"
            for nl in noise_levels:
                match = [r for r in vr if r["n_doubles"] == n_d and r["noise_level"] == nl]
                if match:
                    err = match[0]["fixed_mitigated_error"]
                    acc = "*" if err < CHEMICAL_ACCURACY_HA else " "
                    row_str += f"  {err * 1000:9.4f}{acc}"
                else:
                    row_str += f"  {'N/A':>10}"
            print(row_str)

        # Confound check summary
        print(f"\nOptimal n_doubles by noise level:")
        for nl in noise_levels:
            subset_a = [r for r in vr if r["noise_level"] == nl]
            subset_b = [r for r in vr if r["noise_level"] == nl]
            if subset_a:
                best_a = min(subset_a, key=lambda r: r["noisy_opt_mitigated_error"])
                best_b = min(subset_b, key=lambda r: r["fixed_mitigated_error"])
                label = "noiseless" if nl == 0.0 else f"p={nl}"
                print(f"  {label:>10}: Mode A -> n_d={best_a['n_doubles']} "
                      f"({best_a['noisy_opt_mitigated_error'] * 1000:.4f} mHa), "
                      f"Mode B -> n_d={best_b['n_doubles']} "
                      f"({best_b['fixed_mitigated_error'] * 1000:.4f} mHa)")


def save_tsv(results: list[dict[str, Any]], molecule_key: str) -> None:
    """Save raw results to TSV file."""
    tsv_path = f"validate_sweep_{molecule_key}.tsv"
    headers = [
        "molecule", "variant", "n_singles", "n_doubles", "n_params",
        "noise_level", "ideal_error", "ideal_wall_time",
        "noisy_opt_mitigated_error", "noisy_opt_noisy_error",
        "noisy_opt_ideal_error", "noisy_opt_wall_time",
        "fixed_mitigated_error", "fixed_noisy_error",
    ]
    with open(tsv_path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for r in results:
            row = [
                r["molecule"], r["variant"],
                str(r["n_singles"]), str(r["n_doubles"]), str(r["n_params"]),
                f"{r['noise_level']:.4f}",
                f"{r['ideal_error']:.10f}", f"{r['ideal_wall_time']:.2f}",
                f"{r['noisy_opt_mitigated_error']:.10f}",
                f"{r['noisy_opt_noisy_error']:.10f}",
                f"{r['noisy_opt_ideal_error']:.10f}",
                f"{r['noisy_opt_wall_time']:.2f}",
                f"{r['fixed_mitigated_error']:.10f}",
                f"{r['fixed_noisy_error']:.10f}",
            ]
            f.write("\t".join(row) + "\n")
    print(f"\nResults saved to {tsv_path}")


def main() -> None:
    """Run the validation sweep."""
    parser = argparse.ArgumentParser(
        description="Direct excitation count sweep for noise-optimal circuit validation"
    )
    parser.add_argument("--molecule", type=molecule_choice, default="lih")
    parser.add_argument("--bond-length", type=float, default=None,
                        help="Override default bond length (Å) for diatomics/chains; "
                             "errors for fixed-geometry molecules.")
    args = parser.parse_args()

    molecule_key = args.molecule

    # Setup
    print(f"=== VALIDATION SWEEP: {MOLECULES[molecule_key]['name']} ===")
    hamiltonian, n_qubits, n_electrons, hf_state = build_hamiltonian(
        molecule_key, bond_length=args.bond_length
    )
    exact_energy = compute_exact_energy(hamiltonian, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)

    print(f"Qubits: {n_qubits}, Electrons: {n_electrons}")
    print(f"Singles: {len(singles)}, Doubles: {len(doubles)}")
    print(f"Exact energy: {exact_energy:.8f} Ha")
    geom = MOLECULES[molecule_key].get("geometry")
    if geom is not None:
        actual_bl = args.bond_length if args.bond_length is not None else geom["default_bond_length"]
        suffix = "" if args.bond_length is None else " (override)"
        print(f"Bond length: {actual_bl} A{suffix}")
    print(f"Hyperparameters: step={STEP_SIZE}, opt={OPTIMIZER}, init={INIT_SCALE}")
    print(f"ZNE: scale_factors={list(ZNE_SCALE_FACTORS)}, extrapolation=linear")
    print()

    # Gradient ranking
    print("Ranking excitations by gradient magnitude (noiseless)...")
    ranked_singles, ranked_doubles = rank_excitations(
        hamiltonian, n_qubits, n_electrons, hf_state, singles, doubles
    )
    print()

    n_singles_total = len(singles)
    n_doubles_total = len(doubles)

    variants = [
        ("all_singles", ranked_singles),
        ("zero_singles", []),
    ]

    all_results: list[dict[str, Any]] = []

    for variant_name, variant_singles in variants:
        print(f"\n{'=' * 60}")
        print(f"Variant: {variant_name} (n_singles={len(variant_singles)})")
        print(f"{'=' * 60}")

        for n_d in range(1, n_doubles_total + 1):
            sel_doubles = ranked_doubles[:n_d]
            sel_singles = list(variant_singles)
            n_params = len(sel_singles) + n_d

            print(f"\n--- n_doubles={n_d}, {variant_name} ({n_params} params) ---")

            # Noiseless optimization (once per n_doubles)
            sys.stdout.write("  Noiseless optimization... ")
            sys.stdout.flush()
            try:
                ideal_error, ideal_params, ideal_time = run_noiseless_optimization(
                    hamiltonian, n_qubits, hf_state, exact_energy,
                    sel_singles, sel_doubles, TIME_BUDGET,
                )
            except Exception:
                traceback.print_exc()
                ideal_error, ideal_params, ideal_time = 100.0, np.zeros(n_params), 0.0
            print(f"ideal_error={ideal_error * 1000:.4f} mHa ({ideal_time:.1f}s)")

            for noise in NOISE_LEVELS:
                row: dict[str, Any] = {
                    "molecule": molecule_key,
                    "variant": variant_name,
                    "n_singles": len(sel_singles),
                    "n_doubles": n_d,
                    "n_params": n_params,
                    "noise_level": noise,
                    "ideal_error": ideal_error,
                    "ideal_wall_time": ideal_time,
                }

                if noise == 0.0:
                    # Noiseless: all errors equal ideal
                    row["noisy_opt_mitigated_error"] = ideal_error
                    row["noisy_opt_noisy_error"] = ideal_error
                    row["noisy_opt_ideal_error"] = ideal_error
                    row["noisy_opt_wall_time"] = 0.0
                    row["fixed_mitigated_error"] = ideal_error
                    row["fixed_noisy_error"] = ideal_error
                else:
                    # Mode A: noisy-optimized
                    sys.stdout.write(f"  p={noise}: noisy VQE... ")
                    sys.stdout.flush()
                    try:
                        mit_err, noisy_err, ideal_err_a, _, wt = run_noisy_vqe_trial(
                            hamiltonian=hamiltonian,
                            n_qubits=n_qubits,
                            hf_state=hf_state,
                            exact_energy=exact_energy,
                            selected_singles=sel_singles,
                            selected_doubles=sel_doubles,
                            step_size=STEP_SIZE,
                            optimizer_name=OPTIMIZER,
                            init_scale=INIT_SCALE,
                            noise_strength=noise,
                            time_budget=TIME_BUDGET,
                            conv_threshold=CONV_THRESHOLD,
                        )
                    except Exception:
                        traceback.print_exc()
                        mit_err = noisy_err = ideal_err_a = 100.0
                        wt = 0.0

                    row["noisy_opt_mitigated_error"] = mit_err
                    row["noisy_opt_noisy_error"] = noisy_err
                    row["noisy_opt_ideal_error"] = ideal_err_a
                    row["noisy_opt_wall_time"] = wt
                    print(f"mitigated={mit_err * 1000:.4f} mHa ({wt:.1f}s)")

                    # Mode B: fixed-params
                    sys.stdout.write(f"  p={noise}: fixed-params... ")
                    sys.stdout.flush()
                    try:
                        fixed_mit, fixed_noisy = evaluate_fixed_params(
                            hamiltonian, n_qubits, hf_state, exact_energy,
                            ideal_params, sel_singles, sel_doubles, noise,
                        )
                    except Exception:
                        traceback.print_exc()
                        fixed_mit = fixed_noisy = 100.0

                    row["fixed_mitigated_error"] = fixed_mit
                    row["fixed_noisy_error"] = fixed_noisy
                    print(f"mitigated={fixed_mit * 1000:.4f} mHa")

                all_results.append(row)

    # Print tables
    print_results_table(all_results, molecule_key, n_doubles_total)

    # Save TSV
    save_tsv(all_results, molecule_key)


if __name__ == "__main__":
    main()
