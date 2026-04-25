"""
phase_scan.py — Sweep (bond_length, noise, n_doubles) for the noise-chemistry
phase diagram.

For each combination, runs a noisy VQE + ZNE evaluation (Mode A: parameters
optimized under noise) and records the mitigated energy error. Designed to be
resumable: if interrupted, can restart from where it left off using the data
file as state.

Hyperparameters are fixed across the grid to match validate_sweep.py and the
v3 noisy_circuit.py defaults: Nesterov, step=0.4, zero init, conv=1e-8, ZNE
scale_factors=[1, 2, 3], linear extrapolation. The "all singles" variant is
used (matches the validated all_singles configuration in validate_sweep.py).

Usage:
    uv run phase_scan.py --molecule lih
    uv run phase_scan.py --molecule lih --bond-lengths 1.0 1.5 2.0 2.5 3.0 3.5
    uv run phase_scan.py --molecule lih --noise-levels 0.0 0.001 0.005 0.01
    uv run phase_scan.py --molecule lih --bond-lengths 1.546 \\
        --noise-levels 0.005 --n-doubles 2          # pre-flight single point
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml

from prepare import (
    MOLECULES,
    CHEMICAL_ACCURACY_HA,
    build_hamiltonian,
    compute_exact_energy,
    molecule_choice,
)
from optimize_noisy import rank_excitations, run_noisy_vqe_trial
from validate_sweep import run_noiseless_optimization

# ============================================================
# DEFAULT GRID
# ============================================================
DEFAULT_BOND_LENGTHS = [1.0, 1.546, 2.0, 2.5, 3.0, 3.5]
DEFAULT_NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01]
DEFAULT_N_DOUBLES = [1, 2, 3, 4]

# Fixed hyperparameters — match validate_sweep.py / v3 noisy recipe
STEP_SIZE = 0.4
OPTIMIZER = "nesterov"
INIT_SCALE = 0.0
CONV_THRESHOLD = 1e-8
TIME_BUDGET = 300

# TSV column order
OUTPUT_FIELDS = [
    "molecule",
    "bond_length",
    "noise",
    "n_doubles",
    "n_singles",
    "n_params",
    "exact_energy",
    "ideal_error",
    "noisy_error",
    "mitigated_error",
    "mitigated_error_mha",
    "chemical_accuracy",
    "wall_time",
    "timestamp",
]

# Float-equality keying tolerance for the resume set
KEY_PRECISION = 6


def output_path(molecule_key: str) -> Path:
    """Path to the per-molecule TSV in the repo root."""
    return Path(f"phase_data_{molecule_key}.tsv")


def _key(bond_length: float, noise: float, n_doubles: int) -> tuple[float, float, int]:
    """Canonical (bond_length, noise, n_doubles) tuple for the resume set."""
    return (round(float(bond_length), KEY_PRECISION),
            round(float(noise), KEY_PRECISION),
            int(n_doubles))


def load_done(path: Path) -> set[tuple[float, float, int]]:
    """Load (bond_length, noise, n_doubles) tuples already present in the TSV.

    Returns an empty set when the file is absent or has only a header.
    Raises RuntimeError if the file is present but the header is unexpected,
    so a stale schema can't silently corrupt a resume.
    """
    if not path.exists():
        return set()
    done: set[tuple[float, float, int]] = set()
    with open(path) as f:
        header_line = f.readline()
        if not header_line.strip():
            return done
        cols = header_line.rstrip("\n").split("\t")
        try:
            bl_idx = cols.index("bond_length")
            p_idx = cols.index("noise")
            nd_idx = cols.index("n_doubles")
        except ValueError as e:
            raise RuntimeError(
                f"Unexpected TSV header in {path}: missing column ({e})"
            ) from e
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(cols):
                continue
            try:
                done.add(_key(float(parts[bl_idx]),
                              float(parts[p_idx]),
                              int(parts[nd_idx])))
            except ValueError:
                # Skip malformed rows rather than crashing a long resume
                continue
    return done


def write_header_if_new(path: Path) -> None:
    """Write the header row only when the file doesn't already exist."""
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w") as f:
        f.write("\t".join(OUTPUT_FIELDS) + "\n")


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one formatted row and fsync. fsync is intentional: a SIGKILL
    mid-grid must not lose the latest result, since the resume logic uses
    the file as the only source of state."""
    formatted = _format_row(row)
    with open(path, "a") as f:
        f.write("\t".join(formatted[k] for k in OUTPUT_FIELDS) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _format_row(row: dict[str, Any]) -> dict[str, str]:
    """Format a result row to TSV-safe strings with stable precision."""
    return {
        "molecule": str(row["molecule"]),
        "bond_length": f"{row['bond_length']:.4f}",
        "noise": f"{row['noise']:.6f}",
        "n_doubles": str(row["n_doubles"]),
        "n_singles": str(row["n_singles"]),
        "n_params": str(row["n_params"]),
        "exact_energy": f"{row['exact_energy']:.10f}",
        "ideal_error": f"{row['ideal_error']:.10f}",
        "noisy_error": f"{row['noisy_error']:.10f}",
        "mitigated_error": f"{row['mitigated_error']:.10f}",
        "mitigated_error_mha": f"{row['mitigated_error_mha']:.6f}",
        "chemical_accuracy": str(bool(row["chemical_accuracy"])),
        "wall_time": f"{row['wall_time']:.2f}",
        "timestamp": str(row["timestamp"]),
    }


def evaluate_grid_point(
    molecule_key: str,
    bond_length: float,
    noise: float,
    n_d: int,
    hamiltonian: qml.operation.Operator,
    n_qubits: int,
    hf_state: np.ndarray,
    exact_energy: float,
    ranked_singles: list[list[int]],
    ranked_doubles: list[list[int]],
) -> dict[str, Any]:
    """Run one (bond_length, noise, n_doubles) evaluation.

    At noise == 0.0 we use the noiseless optimizer (backprop, no ZNE folding —
    much faster than running run_noisy_vqe_trial with a default.qubit device).
    All three error fields collapse to the same ideal error in that case.

    At noise > 0 we use Mode A: optimize parameters under noise on
    default.mixed, then evaluate on ideal / raw-noisy / ZNE-mitigated cost
    functions. Returns the row dict to write to the TSV.
    """
    sel_singles = list(ranked_singles)  # all singles, by validated convention
    sel_doubles = ranked_doubles[:n_d]
    n_s = len(sel_singles)
    n_params = n_s + n_d

    if noise == 0.0:
        ideal_error, _, wall_time = run_noiseless_optimization(
            hamiltonian, n_qubits, hf_state, exact_energy,
            sel_singles, sel_doubles, TIME_BUDGET,
        )
        noisy_error = ideal_error
        mitigated_error = ideal_error
    else:
        mitigated_error, noisy_error, ideal_error, _, wall_time = run_noisy_vqe_trial(
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

    return {
        "molecule": molecule_key,
        "bond_length": bond_length,
        "noise": noise,
        "n_doubles": n_d,
        "n_singles": n_s,
        "n_params": n_params,
        "exact_energy": exact_energy,
        "ideal_error": ideal_error,
        "noisy_error": noisy_error,
        "mitigated_error": mitigated_error,
        "mitigated_error_mha": mitigated_error * 1000,
        "chemical_accuracy": mitigated_error < CHEMICAL_ACCURACY_HA,
        "wall_time": wall_time,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_bl_context(
    molecule_key: str, bond_length: float
) -> tuple[Any, int, np.ndarray, float, list[list[int]], list[list[int]]]:
    """Build the per-bond-length context used across all (n_d, noise) points.

    Hamiltonian construction and gradient ranking are expensive enough to
    cache, but they only depend on the geometry — not on the noise level or
    the number of doubles being kept. Returns
    (hamiltonian, n_qubits, hf_state, exact_energy, ranked_singles, ranked_doubles).
    """
    hamiltonian, n_qubits, n_electrons, hf_state = build_hamiltonian(
        molecule_key, bond_length=bond_length
    )
    exact_energy = compute_exact_energy(hamiltonian, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    ranked_singles, ranked_doubles = rank_excitations(
        hamiltonian, n_qubits, n_electrons, hf_state, singles, doubles
    )
    return (
        hamiltonian, n_qubits, hf_state, exact_energy,
        ranked_singles, ranked_doubles,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Each list-valued flag falls back to the module
    default when omitted, so passing nothing runs the full 6×4×4 grid."""
    ap = argparse.ArgumentParser(
        description="Phase-diagram sweep over (bond_length, noise, n_doubles)."
    )
    ap.add_argument("--molecule", type=molecule_choice, default="lih",
                    help="Molecule key (default: lih)")
    ap.add_argument("--bond-lengths", type=float, nargs="+", default=None,
                    metavar="BL",
                    help=f"Bond lengths (Å). Default: {DEFAULT_BOND_LENGTHS}")
    ap.add_argument("--noise-levels", type=float, nargs="+", default=None,
                    metavar="P",
                    help=f"Depolarizing noise per gate. Default: {DEFAULT_NOISE_LEVELS}")
    ap.add_argument("--n-doubles", type=int, nargs="+", default=None,
                    metavar="ND",
                    help=f"Doubles counts to sweep. Default: {DEFAULT_N_DOUBLES}")
    ap.add_argument("--output", type=str, default=None,
                    help="Override output TSV path (default: phase_data_<molecule>.tsv)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    molecule_key = args.molecule
    bond_lengths = args.bond_lengths if args.bond_lengths is not None else DEFAULT_BOND_LENGTHS
    noise_levels = args.noise_levels if args.noise_levels is not None else DEFAULT_NOISE_LEVELS
    n_doubles_list = args.n_doubles if args.n_doubles is not None else DEFAULT_N_DOUBLES

    path = Path(args.output) if args.output else output_path(molecule_key)

    config = MOLECULES[molecule_key]
    geom = config.get("geometry")
    if geom is None and bond_lengths != [None]:
        # Fixed-geometry molecule: a bond-length sweep makes no sense
        raise SystemExit(
            f"--molecule {molecule_key} has fixed coordinates; "
            f"a bond-length sweep is not supported."
        )

    grid: list[tuple[float, int, float]] = []
    for bl in bond_lengths:
        for n_d in n_doubles_list:
            for p in noise_levels:
                grid.append((bl, n_d, p))

    write_header_if_new(path)
    done = load_done(path)
    total = len(grid)

    print(f"=== Phase scan: {config['name']} ({molecule_key}) ===")
    print(f"Bond lengths : {bond_lengths}")
    print(f"Noise levels : {noise_levels}")
    print(f"n_doubles    : {n_doubles_list}")
    print(f"Hyperparams  : opt={OPTIMIZER}, step={STEP_SIZE}, "
          f"init={INIT_SCALE}, conv={CONV_THRESHOLD}, "
          f"time_budget={TIME_BUDGET}s")
    print(f"ZNE          : scale_factors=[1.0, 2.0, 3.0], linear extrapolation")
    print(f"Grid size    : {total} evaluations")
    print(f"Output       : {path}")
    print(f"Already done : {len(done)} / {total}")
    print()

    bl_cache: dict[float, tuple[Any, int, np.ndarray, float,
                                list[list[int]], list[list[int]]]] = {}
    overall_start = time.time()
    completed_this_run = 0

    for idx, (bl, n_d, p) in enumerate(grid, start=1):
        key = _key(bl, p, n_d)
        if key in done:
            continue

        bl_round = round(float(bl), KEY_PRECISION)
        if bl_round not in bl_cache:
            sys.stdout.write(f"  Building Hamiltonian at bl={bl}... ")
            sys.stdout.flush()
            try:
                bl_cache[bl_round] = build_bl_context(molecule_key, bl)
            except Exception:
                traceback.print_exc()
                print(f"  FAILED to build context at bl={bl}; skipping.")
                continue
            ham, n_qubits, _, exact_e, rs, rd = bl_cache[bl_round]
            print(f"exact={exact_e:.6f} Ha, "
                  f"singles={len(rs)}, doubles={len(rd)}")

        hamiltonian, n_qubits, hf_state, exact_energy, rs, rd = bl_cache[bl_round]

        if n_d > len(rd):
            print(f"[{idx}/{total}] {config['name']} bl={bl} p={p} n_d={n_d} "
                  f"-> SKIP (only {len(rd)} doubles available)")
            continue

        try:
            row = evaluate_grid_point(
                molecule_key, bl, p, n_d,
                hamiltonian, n_qubits, hf_state, exact_energy, rs, rd,
            )
        except Exception:
            traceback.print_exc()
            print(f"[{idx}/{total}] {config['name']} bl={bl} p={p} n_d={n_d} "
                  f"-> FAILED (see traceback above)")
            continue

        append_row(path, row)
        done.add(key)
        completed_this_run += 1

        acc = " (chem.acc.)" if row["chemical_accuracy"] else ""
        arrow = "->"
        print(
            f"[{idx}/{total}] {config['name']} bl={bl} p={p} n_d={n_d} "
            f"{arrow} mitigated={row['mitigated_error_mha']:.4f} mHa{acc} "
            f"{row['wall_time']:.1f}s"
        )

    elapsed = time.time() - overall_start
    print()
    print(f"Completed this run: {completed_this_run} / {total} cells")
    print(f"Wall time         : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Output            : {path}")


if __name__ == "__main__":
    main()
