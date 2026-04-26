"""
plot_validation.py — Generate crossover plots from validation sweep data.

Reads validate_sweep_*.tsv and generates:
  - Images/validation_crossover_{molecule}.png (main result: noisy-optimized)
  - Images/validation_confound_{molecule}.png (confound check: fixed-params)

Usage:
    uv run --extra analysis plot_validation.py --molecule lih
    uv run --extra analysis plot_validation.py --molecule h2
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

CHEMICAL_ACCURACY_MHA = 1.6


def load_tsv(filepath: Path) -> list[dict[str, Any]]:
    """Load validation sweep TSV into list of dicts."""
    if not filepath.exists():
        print(f"Error: {filepath} not found. Run validate_sweep.py first.")
        sys.exit(1)
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({
                "variant": row["variant"],
                "n_singles": int(row["n_singles"]),
                "n_doubles": int(row["n_doubles"]),
                "n_params": int(row["n_params"]),
                "noise_level": float(row["noise_level"]),
                "ideal_error": float(row["ideal_error"]),
                "noisy_opt_mitigated_error": float(row["noisy_opt_mitigated_error"]),
                "fixed_mitigated_error": float(row["fixed_mitigated_error"]),
            })
    return rows


def make_crossover_plot(
    data: list[dict[str, Any]],
    error_key: str,
    title: str,
    output_file: str,
    molecule: str,
) -> None:
    """Generate a crossover plot: n_doubles vs mitigated error at each noise level."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    colors = {
        0.0: "#333333",
        0.001: "#5B9BD5",
        0.005: "#E06666",
        0.01: "#6AA84F",
    }
    markers = {0.0: "s", 0.001: "o", 0.005: "D", 0.01: "^"}

    noise_levels = sorted(set(d["noise_level"] for d in data))
    max_nd = max(d["n_doubles"] for d in data)

    for nl in noise_levels:
        subset = sorted(
            [d for d in data if d["noise_level"] == nl],
            key=lambda d: d["n_doubles"],
        )
        if not subset:
            continue

        x = [d["n_doubles"] for d in subset]
        y = [d[error_key] * 1000 for d in subset]  # convert to mHa

        label = "noiseless" if nl == 0.0 else f"p={nl}"
        ax.plot(
            x, y, f"{markers.get(nl, 'o')}-",
            color=colors.get(nl, "#999999"),
            label=label,
            linewidth=2,
            markersize=8,
        )

    # Chemical accuracy line
    ax.axhline(
        y=CHEMICAL_ACCURACY_MHA, color="#aaaaaa", linestyle="--",
        linewidth=0.8, alpha=0.6,
    )
    ax.text(
        max_nd - 0.1, CHEMICAL_ACCURACY_MHA * 1.15, "chem. acc.",
        color="#aaaaaa", fontsize=8, ha="right", fontfamily="monospace",
    )

    ax.set_xlabel("Number of double excitations", fontsize=11, fontfamily="monospace")
    ax.set_ylabel("ZNE-mitigated error (mHa)", fontsize=11, fontfamily="monospace")
    ax.set_yscale("log")
    ax.set_xticks(range(1, max_nd + 1))
    ax.set_title(title, fontsize=13, fontweight="bold", fontfamily="monospace")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_file, dpi=180, facecolor="#ffffff", bbox_inches="tight")
    print(f"Saved {output_file}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation crossover plots")
    parser.add_argument("--molecule", type=str, default="lih")
    args = parser.parse_args()

    molecule = args.molecule
    tsv_path = Path(f"validate_sweep_{molecule}.tsv")
    data = load_tsv(tsv_path)

    # Use zero_singles variant (matches the BO finding)
    variant_data = [d for d in data if d["variant"] == "zero_singles"]
    if not variant_data:
        variant_data = [d for d in data if d["variant"] == "all_singles"]

    # Plot 1: noisy-optimized (main result)
    make_crossover_plot(
        variant_data,
        error_key="noisy_opt_mitigated_error",
        title=f"{molecule.upper()}: Noisy-Optimized (Mode A)",
        output_file=f"Images/validation_crossover_{molecule}.png",
        molecule=molecule,
    )

    # Plot 2: fixed-params (confound check)
    make_crossover_plot(
        variant_data,
        error_key="fixed_mitigated_error",
        title=f"{molecule.upper()}: Fixed-Params Confound Check (Mode B)",
        output_file=f"Images/validation_confound_{molecule}.png",
        molecule=molecule,
    )


if __name__ == "__main__":
    main()
