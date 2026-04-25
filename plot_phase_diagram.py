"""
plot_phase_diagram.py — Generate phase diagram plots from phase_data_*.tsv.

For each (bond_length, noise) cell, picks the n_doubles that minimises the
ZNE-mitigated error (ties broken in favour of the smaller circuit). Produces:

  phase_diagram_optimal_nd_<molecule>.png  — discrete heatmap, optimal n_d
  phase_diagram_error_<molecule>.png       — continuous heatmap, log10 error
  phase_diagram_combined_<molecule>.png    — 2-panel hero figure (16:9)

Usage:
    uv run --extra analysis plot_phase_diagram.py --molecule lih
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

CHEMICAL_ACCURACY_MHA = 1.6
BG = "#ffffff"
MONO = "monospace"

# Discrete palette for optimal n_doubles. Picked so the extremes (1 = pure
# noise-optimised, 4 = full UCCSD) are dark and the middle is light, which
# lets the eye pick out gradient-of-doubles trends in the heatmap.
N_D_COLORS = {
    1: "#2E7D32",  # dark green
    2: "#9CCC65",  # lime
    3: "#FFB300",  # amber
    4: "#E65100",  # deep orange
}
# Text colour per cell — chosen for contrast against each fill above.
N_D_TEXT_COLORS = {1: "white", 2: "black", 3: "black", 4: "white"}


def load_data(molecule: str) -> pd.DataFrame:
    """Load phase_data_<molecule>.tsv into a DataFrame."""
    path = Path(f"phase_data_{molecule}.tsv")
    if not path.exists():
        print(f"Error: {path} not found. Run phase_scan.py first.")
        sys.exit(1)
    return pd.read_csv(path, sep="\t")


def aggregate_best(df: pd.DataFrame) -> pd.DataFrame:
    """Per (bond_length, noise), pick the smallest n_doubles achieving the
    minimum mitigated error (within a 1e-3 mHa tie tolerance).

    The tolerance matters in the noiseless column where n_d in {2, 3, 4}
    are numerically tied at machine precision; we report the smallest such
    circuit since fewer parameters is the tiebreaker spelled out in
    program.md.
    """
    out: list[pd.Series] = []
    for _, group in df.groupby(["bond_length", "noise"]):
        min_err = group["mitigated_error_mha"].min()
        ties = group[group["mitigated_error_mha"] <= min_err + 1e-3]
        best = ties.sort_values("n_doubles").iloc[0]
        out.append(best)
    return pd.DataFrame(out).reset_index(drop=True)


def style_axes(ax: plt.Axes) -> None:
    """Apply the project's chart aesthetic: white bg, monospace, clean spines."""
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(MONO)


def _format_y_ticklabels(noise_levels: list[float]) -> list[str]:
    return ["0 (noiseless)" if p == 0.0 else f"{p:g}" for p in noise_levels]


def plot_optimal_nd(
    ax: plt.Axes,
    best: pd.DataFrame,
    bond_lengths: list[float],
    noise_levels: list[float],
) -> None:
    """Discrete heatmap of optimal n_doubles. Hatched cells are those where
    no n_d configuration reaches chemical accuracy under that (bl, p)."""
    nb = len(bond_lengths)
    nn = len(noise_levels)
    grid_nd = np.zeros((nn, nb), dtype=int)
    grid_err = np.zeros((nn, nb))
    grid_chem = np.zeros((nn, nb), dtype=bool)

    for i, p in enumerate(noise_levels):
        for j, bl in enumerate(bond_lengths):
            row = best[(best["bond_length"] == bl) & (best["noise"] == p)]
            if row.empty:
                continue
            grid_nd[i, j] = int(row["n_doubles"].iloc[0])
            grid_err[i, j] = float(row["mitigated_error_mha"].iloc[0])
            grid_chem[i, j] = bool(row["chemical_accuracy"].iloc[0])

    cmap = ListedColormap([N_D_COLORS[k] for k in (1, 2, 3, 4)])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], ncolors=4)
    im = ax.imshow(grid_nd, cmap=cmap, norm=norm,
                   aspect="auto", origin="lower")

    # Per-cell labels: optimal n_d on top, mitigated error mHa below.
    for i in range(nn):
        for j in range(nb):
            n_d = grid_nd[i, j]
            err = grid_err[i, j]
            tcolor = N_D_TEXT_COLORS.get(n_d, "black")
            ax.text(j, i + 0.12, f"n_d={n_d}",
                    ha="center", va="center",
                    fontsize=11, fontfamily=MONO,
                    color=tcolor, fontweight="bold")
            ax.text(j, i - 0.20, f"{err:.2f} mHa",
                    ha="center", va="center",
                    fontsize=8.5, fontfamily=MONO,
                    color=tcolor)

    # Hatch cells where the best mitigated error fails chemical accuracy.
    for i in range(nn):
        for j in range(nb):
            if not grid_chem[i, j]:
                rect = Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="///",
                    edgecolor="#cc0000", linewidth=0.0, alpha=0.55,
                )
                ax.add_patch(rect)

    ax.set_xticks(range(nb))
    ax.set_xticklabels([f"{bl:.2f}" for bl in bond_lengths])
    ax.set_yticks(range(nn))
    ax.set_yticklabels(_format_y_ticklabels(noise_levels))
    ax.set_xlabel("bond length (A)", fontsize=11, fontfamily=MONO)
    ax.set_ylabel("noise (depolarizing p)", fontsize=11, fontfamily=MONO)
    ax.set_title(
        "optimal n_doubles per (bond_length, noise)\n"
        "red hatching: chemical accuracy unreachable at this cell",
        fontsize=12, fontweight="bold", fontfamily=MONO,
    )
    style_axes(ax)

    cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4], shrink=0.7, pad=0.02)
    cbar.set_label("optimal n_doubles", fontsize=10, fontfamily=MONO)
    cbar.ax.yaxis.label.set_fontfamily(MONO)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(MONO)


def plot_error_heatmap(
    ax: plt.Axes,
    best: pd.DataFrame,
    bond_lengths: list[float],
    noise_levels: list[float],
) -> None:
    """Continuous log10(error) heatmap with a chemical-accuracy contour."""
    nb = len(bond_lengths)
    nn = len(noise_levels)
    grid_err = np.full((nn, nb), np.nan)

    for i, p in enumerate(noise_levels):
        for j, bl in enumerate(bond_lengths):
            row = best[(best["bond_length"] == bl) & (best["noise"] == p)]
            if not row.empty:
                grid_err[i, j] = float(row["mitigated_error_mha"].iloc[0])

    # Clip the very-small noiseless errors so the colour map keeps useful
    # contrast in the noisy region. Anything below 0.001 mHa is "exact"
    # for our purposes.
    grid_err_clipped = np.clip(grid_err, 1e-3, None)
    log_err = np.log10(grid_err_clipped)
    vmin, vmax = log_err.min(), log_err.max()

    im = ax.imshow(log_err, aspect="auto", origin="lower",
                   cmap="viridis_r", vmin=vmin, vmax=vmax)

    for i in range(nn):
        for j in range(nb):
            err = grid_err[i, j]
            if np.isnan(err):
                continue
            normalized = (log_err[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            tcolor = "white" if normalized > 0.55 else "black"
            label = f"{err:.2g}" if err >= 0.01 else f"{err:.0e}"
            ax.text(j, i, f"{label} mHa",
                    ha="center", va="center",
                    fontsize=9, fontfamily=MONO, color=tcolor)

    # Chemical-accuracy contour at log10(1.6) ≈ 0.204. Only draw it when
    # the threshold actually falls inside the data range; otherwise the
    # call silently produces no line.
    chem_log = np.log10(CHEMICAL_ACCURACY_MHA)
    if log_err.min() < chem_log < log_err.max():
        X, Y = np.meshgrid(np.arange(nb), np.arange(nn))
        cs = ax.contour(X, Y, log_err,
                        levels=[chem_log], colors=["#cc0000"],
                        linestyles="--", linewidths=2.0)
        ax.clabel(cs, inline=True, fontsize=8,
                  fmt={chem_log: "1.6 mHa"}, inline_spacing=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    log_ticks: list[float] = []
    log_labels: list[str] = []
    for v in (0.001, 0.01, 0.1, 1.0, 10.0):
        lv = np.log10(v)
        if vmin - 1e-9 <= lv <= vmax + 1e-9:
            log_ticks.append(lv)
            log_labels.append(f"{v:g}")
    if log_ticks:
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels(log_labels)
    cbar.set_label("mitigated error (mHa)", fontsize=10, fontfamily=MONO)
    cbar.ax.yaxis.label.set_fontfamily(MONO)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(MONO)

    ax.set_xticks(range(nb))
    ax.set_xticklabels([f"{bl:.2f}" for bl in bond_lengths])
    ax.set_yticks(range(nn))
    ax.set_yticklabels(_format_y_ticklabels(noise_levels))
    ax.set_xlabel("bond length (A)", fontsize=11, fontfamily=MONO)
    ax.set_ylabel("noise (depolarizing p)", fontsize=11, fontfamily=MONO)
    ax.set_title(
        "best mitigated error\n"
        "dashed red: chemical accuracy boundary (1.6 mHa)",
        fontsize=12, fontweight="bold", fontfamily=MONO,
    )
    style_axes(ax)


def annotate_exact_energies(
    ax: plt.Axes, bond_lengths: list[float], exact_energies: list[float]
) -> None:
    """Print the exact ground-state energy below each column. The energy
    only depends on bond length (not on noise or n_doubles), so it sits
    along the bottom edge to give the reader a geometry reference."""
    for j, (bl, e) in enumerate(zip(bond_lengths, exact_energies)):
        ax.text(j, -0.78, f"E0={e:.4f} Ha",
                ha="center", va="top",
                fontsize=8, fontfamily=MONO, color="#666666",
                clip_on=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate phase-diagram plots.")
    p.add_argument("--molecule", type=str, default="lih",
                   help="Molecule key (default: lih)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.molecule)
    best = aggregate_best(df)

    bond_lengths = sorted(df["bond_length"].unique().tolist())
    noise_levels = sorted(df["noise"].unique().tolist())
    n_evals = len(df)
    total_wall_min = df["wall_time"].sum() / 60.0
    today = datetime.now().strftime("%Y-%m-%d")

    exact_energies = [
        float(df[df["bond_length"] == bl]["exact_energy"].iloc[0])
        for bl in bond_lengths
    ]
    molecule_upper = args.molecule.upper()

    # Plot 1 — optimal n_d standalone
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    fig.patch.set_facecolor(BG)
    plot_optimal_nd(ax, best, bond_lengths, noise_levels)
    annotate_exact_energies(ax, bond_lengths, exact_energies)
    fig.tight_layout()
    out1 = f"phase_diagram_optimal_nd_{args.molecule}.png"
    fig.savefig(out1, dpi=180, facecolor=BG, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig)

    # Plot 2 — error heatmap standalone
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    fig.patch.set_facecolor(BG)
    plot_error_heatmap(ax, best, bond_lengths, noise_levels)
    annotate_exact_energies(ax, bond_lengths, exact_energies)
    fig.tight_layout()
    out2 = f"phase_diagram_error_{args.molecule}.png"
    fig.savefig(out2, dpi=180, facecolor=BG, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig)

    # Plot 3 — combined 2-panel hero figure (16:9)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    plot_optimal_nd(axes[0], best, bond_lengths, noise_levels)
    annotate_exact_energies(axes[0], bond_lengths, exact_energies)
    plot_error_heatmap(axes[1], best, bond_lengths, noise_levels)
    annotate_exact_energies(axes[1], bond_lengths, exact_energies)
    fig.suptitle(
        f"{molecule_upper} noise-chemistry phase diagram",
        fontsize=17, fontweight="bold", fontfamily=MONO, y=0.97,
    )
    fig.text(
        0.5, 0.92,
        f"{n_evals} evaluations | {total_wall_min:.0f} min wall time | {today}",
        ha="center", fontsize=10, fontfamily=MONO, color="#666666",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    out3 = f"phase_diagram_combined_{args.molecule}.png"
    fig.savefig(out3, dpi=180, facecolor=BG, bbox_inches="tight")
    print(f"Saved {out3}")
    plt.close(fig)


if __name__ == "__main__":
    main()
