"""
phase_agent.py — GP active-learning sampler for the chemical accuracy boundary.

Uses level-set estimation to localize the contour where mitigated_error =
1.6 mHa on the (bond_length, noise) plane, with n_doubles fixed at the grid's
optimum (n_d=1, all_singles).

Compares budget-for-budget against phase_scan.py (60 evaluations baseline).
The agent draws 5 quasi-random Sobol points to initialize a Gaussian process
over log10(mitigated_error_mha), then picks each subsequent point to maximize
the straddle acquisition

    score(x) = sigma(x) * exp(-|mu(x) - log10(1.6)|^2 / (2 * tau^2))

on a 50x50 candidate grid in standardized [0, 1]^2 input space.

Outputs:
    phase_agent_<molecule>.tsv     per-eval results, same columns as phase_scan
                                   plus iter, phase, gp_mean_log10, gp_std,
                                   acquisition, boundary_dist_log10.
    phase_agent_<molecule>.log     one rationale entry per pick.
    phase_agent_<molecule>.gp.pkl  fitted GP plus standardization metadata.

Usage:
    uv run --extra optimize phase_agent.py --molecule lih --budget 20
    uv run --extra optimize phase_agent.py --molecule lih --budget 5      # smoke
"""

import argparse
import os
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from scipy.stats import qmc
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        WhiteKernel,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "phase_agent requires scikit-learn (and scipy.stats.qmc). "
        "Run with `uv run --extra optimize phase_agent.py ...`."
    ) from exc

from prepare import MOLECULES, molecule_choice
from phase_scan import (
    OUTPUT_FIELDS,
    build_bl_context,
    evaluate_grid_point,
)

# ============================================================
# CONSTANTS
# ============================================================
DEFAULT_BUDGET = 20
HARD_CAP_BUDGET = 25
N_INIT_SOBOL = 5
DEFAULT_BL_RANGE: tuple[float, float] = (1.0, 3.0)   # skip 3.5 (ansatz wall)
DEFAULT_P_RANGE: tuple[float, float] = (0.0, 0.012)
DEFAULT_TAU = 0.3
DEFAULT_THRESHOLD_MHA = 1.6
PREFLIGHT_THRESHOLD_S = 60.0
PREFLIGHT_FALLBACK_BUDGET = 15
ERROR_FLOOR_MHA = 1e-3                              # log10 floor for GP target
N_DOUBLES_FIXED = 1                                 # all_singles grid winner
DEFAULT_SEED = 42
ACQ_GRID = 50

# Extra columns appended after phase_scan's OUTPUT_FIELDS in the TSV.
AGENT_FIELDS = [
    "iter",
    "phase",
    "gp_mean_log10",
    "gp_std",
    "acquisition",
    "boundary_dist_log10",
]
ALL_FIELDS = OUTPUT_FIELDS + AGENT_FIELDS


# ============================================================
# STANDARDIZATION
# ============================================================
def standardize(x_real: float | np.ndarray,
                lo: float, hi: float) -> float | np.ndarray:
    """Map [lo, hi] -> [0, 1]."""
    return (x_real - lo) / (hi - lo)


def unstandardize(x_unit: float | np.ndarray,
                  lo: float, hi: float) -> float | np.ndarray:
    """Map [0, 1] -> [lo, hi]."""
    return x_unit * (hi - lo) + lo


def floor_log10_mha(err_mha: float) -> float:
    """Clip error to ERROR_FLOOR_MHA before taking log10. Noiseless points
    can hit ~1e-7 mHa which would distort the GP scale; the same clip is
    used by plot_phase_diagram.py for its heatmap.
    """
    return float(np.log10(max(err_mha, ERROR_FLOOR_MHA)))


# ============================================================
# SOBOL INITIALIZATION
# ============================================================
def sobol_init(n_init: int, seed: int) -> np.ndarray:
    """Return n_init scrambled Sobol points in [0, 1]^2.

    Suppresses the "not a power of 2" UserWarning since the brief mandates
    5 init points and the discrepancy property we need still holds at
    non-power-of-2 counts.
    """
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="scipy.stats._qmc"
        )
        return sampler.random(n=n_init)


# ============================================================
# GP MODEL + ACQUISITION
# ============================================================
def make_gp(seed: int) -> GaussianProcessRegressor:
    """Anisotropic Matern-2.5 + WhiteKernel GP regressor.

    Anisotropic length scales let bl and p have different smoothness. The
    WhiteKernel absorbs eval variance (depolarizing channel + finite-step
    optimization stochasticity) so it doesn't inflate the model's epistemic
    std. normalize_y=True applies z-score normalization to the targets,
    appropriate because log10(error) spans several units across the search
    space.
    """
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=[0.3, 0.3],
                 length_scale_bounds=(1e-2, 1e1),
                 nu=2.5)
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-6, 1.0))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=seed,
    )


def acquisition_score(mu: np.ndarray, sigma: np.ndarray,
                      threshold_log10: float, tau: float) -> np.ndarray:
    """Straddle criterion (Bryan & Schneider 2005): maximized where the GP
    is both uncertain AND predicts a value near the target contour.
    """
    return sigma * np.exp(-((mu - threshold_log10) ** 2) / (2.0 * tau ** 2))


def candidate_grid(n: int) -> np.ndarray:
    """n*n equally spaced points in [0, 1]^2, returned as (n*n, 2)."""
    g = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


# ============================================================
# REASONING TEMPLATES
# ============================================================
def _region_label(bl: float, p: float) -> str:
    """Coarse human-readable region label for justification strings."""
    if bl < 1.4:
        bl_s = "compressed-bond"
    elif bl < 1.8:
        bl_s = "equilibrium"
    elif bl < 2.7:
        bl_s = "stretched"
    else:
        bl_s = "far-stretched"

    if p < 1e-4:
        p_s = "noiseless"
    elif p < 3e-3:
        p_s = "low-noise"
    elif p < 7e-3:
        p_s = "mid-noise"
    else:
        p_s = "high-noise"
    return f"{bl_s} {p_s}"


def justify_active(mu: float, sigma: float, threshold_log10: float,
                   tau: float, sigma_grid_median: float,
                   bl: float, p: float) -> str:
    """Templated justification for an active-phase pick. Branches off the
    relationship between mu, the threshold, and sigma relative to the grid
    median. Output is interpretable, not free-text.
    """
    region = _region_label(bl, p)
    dist_log = abs(mu - threshold_log10)
    high_sigma = sigma > sigma_grid_median

    if dist_log < tau / 3.0:
        return (f"on predicted boundary in {region} region; "
                f"refines contour position")
    if mu > threshold_log10 and high_sigma:
        return (f"above predicted boundary in uncertain {region} region; "
                f"tests where boundary curves")
    if mu < threshold_log10 and high_sigma:
        return (f"below predicted boundary in uncertain {region} region; "
                f"tests how far chem.acc. extends")
    if not high_sigma:
        return (f"low uncertainty but argmax of straddle in {region} region; "
                f"consolidates contour")
    return f"high uncertainty within tau of predicted boundary in {region} region"


# ============================================================
# TSV I/O
# ============================================================
def output_path(molecule_key: str) -> Path:
    return Path(f"phase_agent_{molecule_key}.tsv")


def log_path_default(molecule_key: str) -> Path:
    return Path(f"phase_agent_{molecule_key}.log")


def gp_pickle_path(molecule_key: str) -> Path:
    return Path(f"phase_agent_{molecule_key}.gp.pkl")


def write_header_if_new(path: Path) -> None:
    """Create the TSV with the agent's extended header if it doesn't exist."""
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w") as f:
        f.write("\t".join(ALL_FIELDS) + "\n")


def _opt_float(value: float | None, fmt: str) -> str:
    """Format a possibly-NaN/None float as a string, '' for missing."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return format(value, fmt)


def _format_full_row(row: dict[str, Any],
                     extras: dict[str, Any]) -> dict[str, str]:
    """Stringify a row including phase_scan grid columns + agent extras."""
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
        "iter": str(extras["iter"]),
        "phase": str(extras["phase"]),
        "gp_mean_log10": _opt_float(extras["gp_mean_log10"], ".6f"),
        "gp_std": _opt_float(extras["gp_std"], ".6f"),
        "acquisition": _opt_float(extras["acquisition"], ".6f"),
        "boundary_dist_log10": _opt_float(extras["boundary_dist_log10"], ".6f"),
    }


def append_row(path: Path, formatted: dict[str, str]) -> None:
    """Append one row and fsync. fsync is intentional: a SIGKILL mid-run
    must not lose the latest result, since resume reads the TSV as state.
    """
    with open(path, "a") as f:
        f.write("\t".join(formatted[k] for k in ALL_FIELDS) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    """Read prior rows for resume. Returns [] if missing or header-only."""
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(path) as f:
        header_line = f.readline()
        if not header_line.strip():
            return rows
        header = header_line.rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def _load_history(rows: list[dict[str, str]],
                  bl_range: tuple[float, float],
                  p_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (X_unit, y_log10) from prior TSV rows. Skips any row
    with a malformed numeric column rather than aborting the resume.
    """
    if not rows:
        return np.empty((0, 2)), np.empty(0)
    xs: list[list[float]] = []
    ys: list[float] = []
    for r in rows:
        try:
            bl = float(r["bond_length"])
            p = float(r["noise"])
            err_mha = float(r["mitigated_error_mha"])
        except (KeyError, ValueError):
            continue
        xs.append([float(standardize(bl, *bl_range)),
                   float(standardize(p, *p_range))])
        ys.append(floor_log10_mha(err_mha))
    return np.array(xs), np.array(ys)


# ============================================================
# EVALUATION (delegates to phase_scan helpers)
# ============================================================
def eval_point(molecule_key: str, bl: float, p: float) -> dict[str, Any]:
    """Build the bl context fresh and run one (bl, p, n_d=1) evaluation.

    Mirrors phase_scan exactly: same hyperparameters, same noiseless
    optimizer at p=0.0, same Mode A noisy VQE + ZNE at p>0. The Hamiltonian
    is built per call since each agent pick is at a new continuous bond
    length so caching cannot help.
    """
    ham, n_qubits, hf, exact_e, rs, rd = build_bl_context(molecule_key, bl)
    return evaluate_grid_point(
        molecule_key, bl, p, N_DOUBLES_FIXED,
        ham, n_qubits, hf, exact_e, rs, rd,
    )


# ============================================================
# REASONING LOG
# ============================================================
def log_pick(iteration: int, budget: int, bl: float, p: float,
             row: dict[str, Any], extras: dict[str, Any],
             is_init: bool, sobol_idx: int,
             threshold_log10: float, tau: float,
             sigma_grid_median: float,
             log_file: Any) -> None:
    """Print and persist the 3-line entry for one pick."""
    err_mha = row["mitigated_error_mha"]
    chem = "chem.acc." if row["chemical_accuracy"] else "above 1.6 mHa"

    if is_init:
        reason = (
            f"  reason: Sobol scrambled point {sobol_idx + 1}/{N_INIT_SOBOL}; "
            f"observed mitigated error = {err_mha:.4f} mHa ({chem})"
        )
        justification = (
            f"  justification: init point {sobol_idx + 1}/{N_INIT_SOBOL}; "
            f"low-discrepancy coverage of (bl, p) before GP takes over"
        )
    else:
        gp_mean = extras["gp_mean_log10"]
        gp_std = extras["gp_std"]
        boundary_dist = extras["boundary_dist_log10"]
        predicted_mha = 10.0 ** gp_mean
        side = "above" if gp_mean > threshold_log10 else "below"
        reason = (
            f"  reason: GP std={gp_std:.3f}, "
            f"predicted error={predicted_mha:.3f} mHa ({side} boundary), "
            f"boundary distance={boundary_dist:.3f}; "
            f"observed = {err_mha:.4f} mHa ({chem})"
        )
        justification = (
            f"  justification: "
            f"{justify_active(gp_mean, gp_std, threshold_log10, tau, sigma_grid_median, bl, p)}"
        )

    head = f"[iter {iteration}/{budget}] picked (bl={bl:.3f}, p={p:.4f})"
    text = f"{head}\n{reason}\n{justification}"
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


# ============================================================
# FINAL STATS
# ============================================================
def _crossing(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    """First sign change of ys evaluated at xs; returns the interpolated
    x at the crossing or None if there is no crossing."""
    sign = np.sign(ys)
    where = np.where(np.diff(sign) != 0)[0]
    if len(where) == 0:
        return None
    i = int(where[0])
    denom = ys[i + 1] - ys[i]
    if abs(denom) < 1e-12:
        return float(xs[i])
    t = -ys[i] / denom
    return float(xs[i] + t * (xs[i + 1] - xs[i]))


def print_final_stats(gp: GaussianProcessRegressor,
                      bl_range: tuple[float, float],
                      p_range: tuple[float, float],
                      X_unit: np.ndarray, y_log10: np.ndarray,
                      threshold_log10: float, threshold_mha: float,
                      log_file: Any) -> None:
    """Predict over a fine grid for sigma stats + classify reference cells +
    estimate the chem.acc. boundary p_chem(bl) with 95% CIs. Writes to
    stdout and to the reasoning log.
    """
    fine = candidate_grid(60)
    _mu_fine, sigma_fine = gp.predict(fine, return_std=True)
    sigma_mean = float(np.mean(sigma_fine))
    sigma_max = float(np.max(sigma_fine))

    ref_bls = [1.0, 1.546, 2.0, 2.5, 3.0]
    ref_ps = [0.0, 0.005, 0.01]
    pairs = [(b, p) for b in ref_bls for p in ref_ps]
    ref_unit = np.array([
        [float(standardize(b, *bl_range)), float(standardize(p, *p_range))]
        for (b, p) in pairs
    ])
    ref_mu, _ref_sigma = gp.predict(ref_unit, return_std=True)

    below: list[tuple[float, float, float]] = []
    above: list[tuple[float, float, float]] = []
    for k, (b, p) in enumerate(pairs):
        m_mha = float(10.0 ** ref_mu[k])
        if ref_mu[k] < threshold_log10:
            below.append((b, p, m_mha))
        else:
            above.append((b, p, m_mha))

    p_sweep_unit = np.linspace(0.0, 1.0, 200)
    p_sweep_real = unstandardize(p_sweep_unit, *p_range)
    boundary_lines: list[str] = []
    for b in ref_bls:
        b_unit = float(standardize(b, *bl_range))
        sweep = np.column_stack([np.full(200, b_unit), p_sweep_unit])
        mu_b, sigma_b = gp.predict(sweep, return_std=True)
        p_central = _crossing(p_sweep_real, mu_b - threshold_log10)
        # Optimistic upper bound on p_chem: where (mu - 1.96*sigma) crosses.
        # Pessimistic lower bound: where (mu + 1.96*sigma) crosses.
        p_upper = _crossing(p_sweep_real,
                            (mu_b - 1.96 * sigma_b) - threshold_log10)
        p_lower = _crossing(p_sweep_real,
                            (mu_b + 1.96 * sigma_b) - threshold_log10)
        if p_central is None:
            boundary_lines.append(
                f"    bl={b:5.3f}:  no crossing in p in "
                f"[{p_range[0]:g}, {p_range[1]:g}]"
            )
        else:
            lo_s = f"{p_lower:.5f}" if p_lower is not None else "unbounded"
            hi_s = f"{p_upper:.5f}" if p_upper is not None else "unbounded"
            boundary_lines.append(
                f"    bl={b:5.3f}:  p_chem ~= {p_central:.5f}   "
                f"(95% CI: [{lo_s}, {hi_s}])"
            )

    lines = [
        "",
        f"Final: {len(y_log10)} evaluations",
        f"  GP fit: sigma_mean = {sigma_mean:.3f}, sigma_max = {sigma_max:.3f}  "
        f"(in log10(mHa) space)",
        f"  Threshold: {threshold_mha} mHa  (log10 = {threshold_log10:.4f})",
        "",
        "  Cells claimed below 1.6 mHa (chem.acc.):",
    ]
    if below:
        lines.extend(
            f"    (bl={b:5.3f}, p={p:.4f})  predicted {m:7.3f} mHa"
            for (b, p, m) in below
        )
    else:
        lines.append("    (none)")
    lines.append("")
    lines.append("  Cells claimed above 1.6 mHa:")
    if above:
        lines.extend(
            f"    (bl={b:5.3f}, p={p:.4f})  predicted {m:7.3f} mHa"
            for (b, p, m) in above
        )
    else:
        lines.append("    (none)")
    lines.append("")
    lines.append("  Boundary uncertainty (95% CI on chem.acc. p threshold per bl):")
    lines.extend(boundary_lines)

    text = "\n".join(lines)
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="GP active-learning sampler for the chem.acc. boundary."
    )
    ap.add_argument("--molecule", type=molecule_choice, default="lih",
                    help="Molecule key (default: lih)")
    ap.add_argument("--budget", type=int, default=None, metavar="N",
                    help=(f"Total evaluations (default: {DEFAULT_BUDGET}; "
                          f"hard cap {HARD_CAP_BUDGET}; "
                          f"pre-flight may auto-cap to "
                          f"{PREFLIGHT_FALLBACK_BUDGET})."))
    ap.add_argument("--bl-range", type=float, nargs=2,
                    default=list(DEFAULT_BL_RANGE), metavar=("LO", "HI"),
                    help=f"Bond-length range in A (default: {DEFAULT_BL_RANGE})")
    ap.add_argument("--p-range", type=float, nargs=2,
                    default=list(DEFAULT_P_RANGE), metavar=("LO", "HI"),
                    help=f"Noise range (default: {DEFAULT_P_RANGE})")
    ap.add_argument("--threshold-mha", type=float,
                    default=DEFAULT_THRESHOLD_MHA,
                    help=("Chemical-accuracy threshold in mHa "
                          f"(default: {DEFAULT_THRESHOLD_MHA})"))
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU,
                    help=("Boundary tolerance in log10 space "
                          f"(default: {DEFAULT_TAU})"))
    ap.add_argument("--n-init", type=int, default=N_INIT_SOBOL,
                    help=f"Sobol init points (default: {N_INIT_SOBOL})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"RNG seed (default: {DEFAULT_SEED})")
    ap.add_argument("--output", type=str, default=None,
                    help="TSV output path (default: phase_agent_<molecule>.tsv)")
    ap.add_argument("--log", type=str, default=None,
                    help="Reasoning log path (default: phase_agent_<molecule>.log)")
    ap.add_argument("--gp-path", type=str, default=None,
                    help="GP pickle path (default: phase_agent_<molecule>.gp.pkl)")
    ap.add_argument("--no-preflight-cap", action="store_true",
                    help="Disable the >60s pre-flight budget auto-cap.")
    return ap.parse_args()


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    molecule_key = args.molecule
    bl_range: tuple[float, float] = (float(args.bl_range[0]), float(args.bl_range[1]))
    p_range: tuple[float, float] = (float(args.p_range[0]), float(args.p_range[1]))
    threshold_log10 = float(np.log10(args.threshold_mha))
    tau = float(args.tau)
    n_init = int(args.n_init)

    budget_user_override = args.budget is not None
    budget = int(args.budget) if budget_user_override else DEFAULT_BUDGET
    if budget > HARD_CAP_BUDGET:
        print(f"WARN: budget {budget} exceeds hard cap {HARD_CAP_BUDGET}, capping.")
        budget = HARD_CAP_BUDGET

    out_path = Path(args.output) if args.output else output_path(molecule_key)
    log_path = Path(args.log) if args.log else log_path_default(molecule_key)
    gp_path = Path(args.gp_path) if args.gp_path else gp_pickle_path(molecule_key)

    config = MOLECULES[molecule_key]
    if config.get("geometry") is None:
        raise SystemExit(
            f"--molecule {molecule_key} has fixed coordinates; the agent "
            f"requires a continuous bond-length axis."
        )

    print(f"=== Phase agent (GP active learning): {config['name']} ({molecule_key}) ===")
    print(f"Bond length range : {bl_range}")
    print(f"Noise range       : {p_range}")
    print(f"Threshold         : {args.threshold_mha} mHa  "
          f"(log10 = {threshold_log10:.4f}, tau = {tau})")
    print(f"n_doubles fixed   : {N_DOUBLES_FIXED}  (all_singles)")
    n_active = max(0, budget - n_init)
    print(f"Budget            : {budget}  ({n_init} Sobol init + {n_active} active)")
    print(f"Seed              : {args.seed}")
    print(f"Output TSV        : {out_path}")
    print(f"Reasoning log     : {log_path}")
    print(f"GP pickle         : {gp_path}")
    print()

    write_header_if_new(out_path)

    existing_rows = read_existing_rows(out_path)
    n_done = len(existing_rows)
    if n_done > 0:
        print(f"Resuming with {n_done} existing rows from {out_path}.")
        if n_done >= budget:
            print(f"Budget {budget} already reached; nothing to do.")
            return

    X_unit_done, y_log10_done = _load_history(existing_rows, bl_range, p_range)
    sobol_unit_all = sobol_init(n_init, args.seed)

    log_file = open(log_path, "a", encoding="utf-8")
    if n_done == 0:
        log_file.write(
            f"# phase_agent run start: molecule={molecule_key}, "
            f"budget={budget}, n_init={n_init}, seed={args.seed}, "
            f"bl_range={bl_range}, p_range={p_range}, "
            f"threshold_mha={args.threshold_mha}, tau={tau}\n"
        )
        log_file.flush()

    overall_start = time.time()
    completed_this_run = 0

    try:
        i = n_done
        while i < budget:
            iteration = i + 1
            is_init = i < n_init

            mu_pick = float("nan")
            sigma_pick = float("nan")
            score_pick = float("nan")
            sigma_grid_median = float("nan")

            if is_init:
                x_unit = sobol_unit_all[i]
                bl = float(unstandardize(x_unit[0], *bl_range))
                p = float(unstandardize(x_unit[1], *p_range))
            else:
                gp = make_gp(args.seed)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    gp.fit(X_unit_done, y_log10_done)
                grid_unit = candidate_grid(ACQ_GRID)
                mu_grid, sigma_grid = gp.predict(grid_unit, return_std=True)
                scores = acquisition_score(mu_grid, sigma_grid,
                                           threshold_log10, tau)
                sigma_grid_median = float(np.median(sigma_grid))
                argmax = int(np.argmax(scores))
                x_unit = grid_unit[argmax]
                bl = float(unstandardize(x_unit[0], *bl_range))
                p = float(unstandardize(x_unit[1], *p_range))
                mu_pick = float(mu_grid[argmax])
                sigma_pick = float(sigma_grid[argmax])
                score_pick = float(scores[argmax])

            bl = float(np.clip(bl, *bl_range))
            p = float(np.clip(p, *p_range))

            sys.stdout.write(
                f"  Evaluating (bl={bl:.3f}, p={p:.4f}, n_d={N_DOUBLES_FIXED})... "
            )
            sys.stdout.flush()
            t_eval_start = time.time()
            try:
                row = eval_point(molecule_key, bl, p)
            except Exception:
                traceback.print_exc()
                print(f"  FAILED at (bl={bl}, p={p}); aborting run.")
                break
            t_eval = time.time() - t_eval_start
            print(f"-> {row['mitigated_error_mha']:.4f} mHa  "
                  f"({row['wall_time']:.1f}s)")

            # Pre-flight cap on the first eval of this run (resume-safe).
            if (
                completed_this_run == 0
                and not args.no_preflight_cap
                and not budget_user_override
                and t_eval > PREFLIGHT_THRESHOLD_S
                and budget > PREFLIGHT_FALLBACK_BUDGET
            ):
                old_budget = budget
                budget = PREFLIGHT_FALLBACK_BUDGET
                msg = (
                    f"  Pre-flight: first eval took {t_eval:.1f}s > "
                    f"{PREFLIGHT_THRESHOLD_S:.0f}s threshold; "
                    f"budget capped {old_budget} -> {budget}."
                )
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()

            X_unit_done = (np.vstack([X_unit_done, x_unit])
                           if X_unit_done.size else np.array([x_unit]))
            y_log10_done = np.append(
                y_log10_done, floor_log10_mha(row["mitigated_error_mha"])
            )

            boundary_dist = (abs(mu_pick - threshold_log10)
                             if not np.isnan(mu_pick) else float("nan"))
            extras = {
                "iter": iteration,
                "phase": "init" if is_init else "active",
                "gp_mean_log10": mu_pick,
                "gp_std": sigma_pick,
                "acquisition": score_pick,
                "boundary_dist_log10": boundary_dist,
            }

            sobol_idx = i if is_init else -1
            log_pick(iteration, budget, bl, p, row, extras,
                     is_init, sobol_idx, threshold_log10, tau,
                     sigma_grid_median, log_file)
            append_row(out_path, _format_full_row(row, extras))
            completed_this_run += 1
            i += 1

    finally:
        log_file.flush()

    elapsed = time.time() - overall_start
    print()
    print(f"Completed this run: {completed_this_run} new evaluations")
    print(f"Wall time         : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Total in TSV      : {n_done + completed_this_run}")

    # Need at least 3 points for a meaningful GP fit.
    if X_unit_done.shape[0] >= 3:
        gp_final = make_gp(args.seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp_final.fit(X_unit_done, y_log10_done)
        print_final_stats(gp_final, bl_range, p_range,
                          X_unit_done, y_log10_done,
                          threshold_log10, args.threshold_mha, log_file)
        with open(gp_path, "wb") as f:
            pickle.dump({
                "gp": gp_final,
                "bl_range": bl_range,
                "p_range": p_range,
                "threshold_log10": threshold_log10,
                "threshold_mha": args.threshold_mha,
                "tau": tau,
                "X_unit": X_unit_done,
                "y_log10": y_log10_done,
                "molecule_key": molecule_key,
                "n_init": n_init,
                "n_doubles": N_DOUBLES_FIXED,
                "seed": args.seed,
                "error_floor_mha": ERROR_FLOOR_MHA,
            }, f)
        print(f"\nSaved GP model to {gp_path}")

    log_file.close()


if __name__ == "__main__":
    main()
