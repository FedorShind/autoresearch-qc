"""
plot_agent_comparison.py — Comparison plots: GP active-learning sampler
vs the phase-scan grid baseline.

Three plots:
  1. agent_trajectory_<molecule>.png   GP posterior heatmap with grid
                                       cells, numbered agent picks, and
                                       both 1.6 mHa boundary contours.
  2. agent_convergence_<molecule>.png  RMSE in p_chem(bl) versus eval
                                       count, agent vs grid head-to-head.
  3. agent_combined_<molecule>.png     Side-by-side hero figure.

Inputs:
    phase_agent_<molecule>.tsv     agent run TSV (with iter, phase columns)
    phase_agent_<molecule>.gp.pkl  fitted GP plus standardization metadata
    phase_data_<molecule>.tsv      grid baseline TSV (60 cells for LiH)

Usage:
    uv run --extra analysis plot_agent_comparison.py --molecule lih
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        WhiteKernel,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "plot_agent_comparison requires scikit-learn. "
        "Run with `uv run --extra analysis --extra optimize plot_agent_comparison.py ...`."
    ) from exc

# ============================================================
# CONSTANTS (mirrored from phase_agent.py to keep this script light)
# ============================================================
CHEMICAL_ACCURACY_MHA = 1.6
THRESHOLD_LOG10 = float(np.log10(CHEMICAL_ACCURACY_MHA))
ERROR_FLOOR_MHA = 1e-3
BG = "#ffffff"
MONO = "monospace"
DEFAULT_SEED = 42

# Reference bond lengths for boundary RMSE (intersect of grid + agent bl
# ranges). bl=3.5 is excluded since the agent skips it (ansatz wall).
REF_CELL_BLS = [1.546, 2.0, 2.5, 3.0]
REF_CELL_PS = [0.0, 0.005, 0.01]

# Inverse boundary search resolution.
P_SEARCH_N = 400


# ============================================================
# SHARED HELPERS
# ============================================================
def standardize(x: float | np.ndarray, lo: float, hi: float):
    return (x - lo) / (hi - lo)


def floor_log10_mha_vec(errs_mha: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(errs_mha, dtype=float),
                            ERROR_FLOOR_MHA, None))


def make_gp(seed: int) -> GaussianProcessRegressor:
    """Mirror of phase_agent.make_gp — same kernel and hyperparameters so
    the convergence comparison is apples-to-apples."""
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=[0.3, 0.3],
                 length_scale_bounds=(1e-2, 1e1),
                 nu=2.5)
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-6, 1.0))
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True,
        n_restarts_optimizer=8, random_state=seed,
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(MONO)


# ============================================================
# DATA LOADING
# ============================================================
def load_agent(molecule: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    tsv = Path(f"phase_agent_{molecule}.tsv")
    pkl = Path(f"phase_agent_{molecule}.gp.pkl")
    if not tsv.exists():
        sys.exit(f"Error: {tsv} not found. Run phase_agent.py first.")
    if not pkl.exists():
        sys.exit(f"Error: {pkl} not found. Run phase_agent.py first.")
    df = pd.read_csv(tsv, sep="\t")
    df = df.sort_values("iter").reset_index(drop=True)
    with open(pkl, "rb") as f:
        meta = pickle.load(f)
    return df, meta


def load_grid(molecule: str) -> pd.DataFrame:
    tsv = Path(f"phase_data_{molecule}.tsv")
    if not tsv.exists():
        sys.exit(f"Error: {tsv} not found. Run phase_scan.py first.")
    return pd.read_csv(tsv, sep="\t")


def aggregate_grid_per_cell(df_grid: pd.DataFrame) -> pd.DataFrame:
    """Per (bond_length, noise), pick the smallest n_doubles achieving the
    minimum mitigated error. Tie tolerance 1e-3 mHa, same as
    plot_phase_diagram.aggregate_best.
    """
    out = []
    for _, group in df_grid.groupby(["bond_length", "noise"]):
        min_err = group["mitigated_error_mha"].min()
        ties = group[group["mitigated_error_mha"] <= min_err + 1e-3]
        best = ties.sort_values("n_doubles").iloc[0]
        out.append(best)
    return pd.DataFrame(out).reset_index(drop=True)


# ============================================================
# BOUNDARY HELPERS
# ============================================================
def first_crossing(xs: np.ndarray, ys: np.ndarray) -> float | None:
    """First sign change of ys evaluated at xs; linear interpolation."""
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


def gp_p_chem(gp: GaussianProcessRegressor, bl: float,
              bl_range: tuple[float, float],
              p_range: tuple[float, float]) -> float:
    """For a fixed bl, find p where mu(bl, p) crosses THRESHOLD_LOG10.
    Clipped to the search range when no crossing is found inside.
    """
    p_grid = np.linspace(p_range[0], p_range[1], P_SEARCH_N)
    b_unit = float(standardize(bl, *bl_range))
    p_unit = standardize(p_grid, *p_range)
    pts_unit = np.column_stack([np.full(P_SEARCH_N, b_unit), p_unit])
    mu = gp.predict(pts_unit)
    cross = first_crossing(p_grid, mu - THRESHOLD_LOG10)
    if cross is not None:
        return cross
    # No crossing: clip to whichever side mu falls on.
    return p_range[1] if mu[-1] < THRESHOLD_LOG10 else p_range[0]


def boundary_rmse(gp: GaussianProcessRegressor,
                  truth_p_chem: dict[float, float],
                  bl_range: tuple[float, float],
                  p_range: tuple[float, float]) -> float:
    """RMSE in p_chem(bl) over REF_CELL_BLS, comparing GP to truth."""
    diffs = []
    for bl in REF_CELL_BLS:
        p_pred = gp_p_chem(gp, bl, bl_range, p_range)
        diffs.append(p_pred - truth_p_chem[bl])
    diffs = np.asarray(diffs)
    return float(np.sqrt(np.mean(diffs ** 2)))


def fit_gp_safe(X_unit: np.ndarray, y_log10: np.ndarray,
                seed: int) -> GaussianProcessRegressor | None:
    """Fit a GP guarded against the n<2 case."""
    if len(X_unit) < 2:
        return None
    gp = make_gp(seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gp.fit(X_unit, y_log10)
    return gp


# ============================================================
# PLOT 1 — TRAJECTORY
# ============================================================
def plot_trajectory(ax: plt.Axes, df_agent: pd.DataFrame,
                    meta: dict[str, Any],
                    df_grid_agg: pd.DataFrame) -> None:
    bl_range = tuple(meta["bl_range"])
    p_range = tuple(meta["p_range"])
    gp = meta["gp"]

    # Heatmap: GP posterior mu over a fine grid in real space.
    nx, ny = 220, 220
    bl_axis = np.linspace(bl_range[0], bl_range[1], nx)
    p_axis = np.linspace(p_range[0], p_range[1], ny)
    BL, PG = np.meshgrid(bl_axis, p_axis, indexing="xy")
    pts_unit = np.column_stack([
        standardize(BL.ravel(), *bl_range),
        standardize(PG.ravel(), *p_range),
    ])
    mu = gp.predict(pts_unit).reshape(PG.shape)
    extent = [bl_range[0], bl_range[1], p_range[0], p_range[1]]
    im = ax.imshow(mu, origin="lower", extent=extent, aspect="auto",
                   cmap="viridis_r", alpha=0.55, zorder=1)

    # Grid points within the agent's bl range (excludes bl=3.5).
    grid_in = df_grid_agg[
        (df_grid_agg["bond_length"] >= bl_range[0]) &
        (df_grid_agg["bond_length"] <= bl_range[1])
    ]
    ax.scatter(
        grid_in["bond_length"], grid_in["noise"],
        marker="s", s=80, c="#dddddd", edgecolors="#666666",
        linewidths=0.8, zorder=3,
    )

    # Agent picks: numbered circles, color by iteration.
    iters = df_agent["iter"].values.astype(int)
    norm_iter = plt.Normalize(vmin=int(iters.min()), vmax=int(iters.max()))
    cmap_iter = plt.cm.plasma
    colors = cmap_iter(norm_iter(iters))
    ax.scatter(
        df_agent["bond_length"], df_agent["noise"],
        c=colors, s=210, edgecolors="black", linewidths=1.2, zorder=5,
    )
    for _, r in df_agent.iterrows():
        ax.text(
            float(r["bond_length"]), float(r["noise"]),
            str(int(r["iter"])),
            ha="center", va="center",
            fontsize=8, fontfamily=MONO, fontweight="bold",
            color="black", zorder=6,
        )

    # Agent (GP) boundary contour — solid red.
    ax.contour(BL, PG, mu, levels=[THRESHOLD_LOG10],
               colors=["#cc0000"], linestyles=["-"], linewidths=[2.2])

    # Grid boundary contour — dashed red, drawn from per-cell best mit_err.
    if len(grid_in):
        gbls = sorted(grid_in["bond_length"].unique().tolist())
        gps = sorted(grid_in["noise"].unique().tolist())
        Z = np.full((len(gps), len(gbls)), np.nan)
        for i, p in enumerate(gps):
            for j, b in enumerate(gbls):
                m = grid_in[
                    (np.isclose(grid_in["bond_length"], b)) &
                    (np.isclose(grid_in["noise"], p))
                ]
                if not m.empty:
                    Z[i, j] = max(float(m["mitigated_error_mha"].iloc[0]),
                                  ERROR_FLOOR_MHA)
        log_Z = np.log10(Z)
        if (np.nanmin(log_Z) < THRESHOLD_LOG10 < np.nanmax(log_Z)):
            BLg, Pg = np.meshgrid(gbls, gps)
            ax.contour(BLg, Pg, log_Z, levels=[THRESHOLD_LOG10],
                       colors=["#cc0000"], linestyles=["--"],
                       linewidths=[2.2])

    # Heatmap colorbar on the right.
    cbar_h = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, location="right")
    cbar_h.set_label("GP posterior log10(mHa)", fontsize=9, fontfamily=MONO)
    cbar_h.ax.yaxis.label.set_fontfamily(MONO)
    for lab in cbar_h.ax.get_yticklabels():
        lab.set_fontfamily(MONO)

    # Iteration colorbar along the bottom (horizontal) so it doesn't crowd
    # the heatmap colorbar on the right.
    sm = plt.cm.ScalarMappable(cmap=cmap_iter, norm=norm_iter)
    sm.set_array([])
    cbar_i = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.13,
                          location="bottom", aspect=35)
    cbar_i.set_label("agent iteration (early -> late)",
                     fontsize=9, fontfamily=MONO)
    cbar_i.ax.xaxis.label.set_fontfamily(MONO)
    for lab in cbar_i.ax.get_xticklabels():
        lab.set_fontfamily(MONO)

    ax.set_xlim(bl_range)
    ax.set_ylim(p_range)
    ax.set_xlabel("bond length (A)", fontsize=11, fontfamily=MONO)
    ax.set_ylabel("noise (depolarizing p)", fontsize=11, fontfamily=MONO)
    ax.set_title(
        "agent sampling trajectory on (bl, p)\n"
        "gray squares: grid cells   solid red: GP 1.6 mHa   "
        "dashed red: grid 1.6 mHa",
        fontsize=11, fontweight="bold", fontfamily=MONO,
        pad=10,
    )
    style_axes(ax)


# ============================================================
# PLOT 2 — CONVERGENCE
# ============================================================
def grid_first_k_data(df_grid: pd.DataFrame, k: int,
                      bl_range: tuple[float, float],
                      p_range: tuple[float, float]
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Take the first k rows of the grid TSV (chronological), then
    deduplicate by (bond_length, noise) keeping the min mitigated_error
    seen so far for that cell. Returns (X_unit, y_log10).
    """
    df = df_grid.iloc[:k].copy()
    df["mitigated_error_mha"] = (
        df["mitigated_error_mha"].astype(float).clip(lower=ERROR_FLOOR_MHA)
    )
    agg = (
        df.groupby(["bond_length", "noise"])["mitigated_error_mha"]
        .min().reset_index()
    )
    bls = agg["bond_length"].values
    ps = agg["noise"].values
    ys = np.log10(agg["mitigated_error_mha"].values)
    X_unit = np.column_stack([
        standardize(bls, *bl_range),
        standardize(ps, *p_range),
    ])
    return X_unit, ys


def truth_p_chem_from_full_grid(df_grid_agg: pd.DataFrame,
                                bl_range: tuple[float, float],
                                p_range: tuple[float, float],
                                seed: int) -> dict[float, float]:
    """Fit a GP on the full grid (60 cells aggregated to 15 (bl,p) cells)
    and read off p_chem(bl) for each REF_CELL_BLS bond length. This is
    the boundary against which both agent-k and grid-k are compared.
    """
    bls = df_grid_agg["bond_length"].values
    ps = df_grid_agg["noise"].values
    errs_clipped = np.clip(df_grid_agg["mitigated_error_mha"].values.astype(float),
                           ERROR_FLOOR_MHA, None)
    ys = np.log10(errs_clipped)
    X_unit = np.column_stack([
        standardize(bls, *bl_range),
        standardize(ps, *p_range),
    ])
    gp = make_gp(seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gp.fit(X_unit, ys)
    truth = {}
    for bl in REF_CELL_BLS:
        truth[bl] = gp_p_chem(gp, bl, bl_range, p_range)
    return truth


def plot_convergence(ax: plt.Axes, df_agent: pd.DataFrame,
                     df_grid: pd.DataFrame, df_grid_agg: pd.DataFrame,
                     meta: dict[str, Any]) -> None:
    bl_range = tuple(meta["bl_range"])
    p_range = tuple(meta["p_range"])
    seed = meta.get("seed", DEFAULT_SEED)

    # Truth boundary from a GP fit on the full 60-cell grid.
    truth = truth_p_chem_from_full_grid(df_grid_agg, bl_range, p_range, seed)

    # Agent: refit GP on first k samples, compute boundary RMSE.
    X_agent = np.column_stack([
        standardize(df_agent["bond_length"].values, *bl_range),
        standardize(df_agent["noise"].values, *p_range),
    ])
    y_agent = floor_log10_mha_vec(df_agent["mitigated_error_mha"].values)
    n_agent = len(df_agent)
    ks_agent = list(range(2, n_agent + 1))
    rmse_agent = []
    for k in ks_agent:
        gp = fit_gp_safe(X_agent[:k], y_agent[:k], seed)
        rmse_agent.append(boundary_rmse(gp, truth, bl_range, p_range))

    # Grid baseline: refit GP on first k grid evals (dedup-by-cell).
    n_grid = len(df_grid)
    ks_grid = list(range(2, n_grid + 1))
    rmse_grid = []
    for k in ks_grid:
        Xg, yg = grid_first_k_data(df_grid, k, bl_range, p_range)
        if len(Xg) < 2:
            rmse_grid.append(np.nan)
            continue
        gp = fit_gp_safe(Xg, yg, seed)
        rmse_grid.append(boundary_rmse(gp, truth, bl_range, p_range))

    # Convert RMSE to mHa-equivalent units: keep raw p (Hz-style noise scalar).
    rmse_agent_arr = np.asarray(rmse_agent)
    rmse_grid_arr = np.asarray(rmse_grid)

    # Plot.
    ax.plot(ks_agent, rmse_agent_arr * 1000.0,
            color="#1565C0", lw=2.2, marker="o", markersize=4.5,
            label=f"agent ({n_agent} active picks)")
    ax.plot(ks_grid, rmse_grid_arr * 1000.0,
            color="#D32F2F", lw=1.5, marker=".", markersize=3,
            alpha=0.75,
            label=f"grid baseline (chronological order, {n_grid} cells)")

    # Find the eval count at which grid first matches agent's final RMSE.
    final_agent = rmse_agent_arr[-1]
    matched_k = None
    for k, val in zip(ks_grid, rmse_grid_arr):
        if not np.isnan(val) and val <= final_agent:
            matched_k = k
            break

    ax.set_xlim(1, max(n_grid, n_agent) + 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("evaluations spent", fontsize=11, fontfamily=MONO)
    ax.set_ylabel("p_chem RMSE  (x10$^{-3}$)", fontsize=11, fontfamily=MONO)
    title = (
        "boundary recovery: RMSE between predicted and true p_chem(bl)\n"
    )
    if matched_k is not None:
        title += (f"agent reaches {final_agent * 1000:.2f}e-3 at k={n_agent}; "
                  f"grid matches at k={matched_k} "
                  f"({matched_k / n_agent:.1f}x more evals)")
    else:
        title += (f"agent reaches {final_agent * 1000:.2f}e-3 at k={n_agent}; "
                  f"grid does not reach this within 60 evals")
    ax.set_title(title, fontsize=11, fontweight="bold", fontfamily=MONO,
                 pad=10)

    ax.axvline(n_agent, color="#1565C0", ls=":", lw=1.0, alpha=0.5)
    if matched_k is not None:
        ax.axvline(matched_k, color="#D32F2F", ls=":", lw=1.0, alpha=0.5)
    ax.legend(loc="upper right", prop={"family": MONO, "size": 9},
              frameon=False)
    style_axes(ax)


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comparison plots: agent vs grid baseline."
    )
    p.add_argument("--molecule", type=str, default="lih",
                   help="Molecule key (default: lih)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df_agent, meta = load_agent(args.molecule)
    df_grid = load_grid(args.molecule)
    df_grid_agg = aggregate_grid_per_cell(df_grid)

    Path("images").mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    n_agent = len(df_agent)
    n_grid = len(df_grid)

    # Plot 1 — trajectory standalone
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    fig.patch.set_facecolor(BG)
    plot_trajectory(ax, df_agent, meta, df_grid_agg)
    fig.tight_layout()
    out1 = f"images/agent_trajectory_{args.molecule}.png"
    fig.savefig(out1, dpi=180, facecolor=BG, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig)

    # Plot 2 — convergence standalone
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fig.patch.set_facecolor(BG)
    plot_convergence(ax, df_agent, df_grid, df_grid_agg, meta)
    fig.tight_layout()
    out2 = f"images/agent_convergence_{args.molecule}.png"
    fig.savefig(out2, dpi=180, facecolor=BG, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig)

    # Plot 3 — combined hero
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    plot_trajectory(axes[0], df_agent, meta, df_grid_agg)
    plot_convergence(axes[1], df_agent, df_grid, df_grid_agg, meta)
    fig.suptitle(
        f"Active learning recovers the {args.molecule.upper()} "
        f"chemical-accuracy boundary in {n_agent} evaluations vs grid's {n_grid}",
        fontsize=15, fontweight="bold", fontfamily=MONO, y=0.97,
    )
    fig.text(
        0.5, 0.92,
        f"GP active-learning sampler vs phase-scan grid baseline | {today}",
        ha="center", fontsize=10, fontfamily=MONO, color="#666666",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    out3 = f"images/agent_combined_{args.molecule}.png"
    fig.savefig(out3, dpi=180, facecolor=BG, bbox_inches="tight")
    print(f"Saved {out3}")
    plt.close(fig)


if __name__ == "__main__":
    main()
