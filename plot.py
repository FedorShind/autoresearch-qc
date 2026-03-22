"""
plot.py — Generate progress.png from results.tsv.

Usage: uv run plot.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CHEMICAL_ACCURACY_HA = 0.0016
RESULTS_FILE = Path("results.tsv")
OUTPUT_FILE = Path("progress.png")


def load_results() -> pd.DataFrame:
    """Load results.tsv into a DataFrame."""
    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found. Run some experiments first.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_FILE, sep="\t")
    df["experiment_num"] = range(1, len(df) + 1)
    return df


def plot_progress(df: pd.DataFrame) -> None:
    """Progress plot: energy_error vs experiment number (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by whether the experiment improved (was kept)
    colors = []
    best_so_far = float("inf")
    for _, row in df.iterrows():
        if row["energy_error"] < best_so_far:
            colors.append("#2ecc71")  # green — kept
            best_so_far = row["energy_error"]
        else:
            colors.append("#e74c3c")  # red — reverted

    ax.scatter(df["experiment_num"], df["energy_error"], c=colors, s=60, zorder=3)
    ax.set_yscale("log")
    ax.axhline(y=CHEMICAL_ACCURACY_HA, color="gray", linestyle="--", linewidth=1,
               label=f"Chemical accuracy ({CHEMICAL_ACCURACY_HA} Ha)")

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Energy Error (Ha, log scale)", fontsize=12)

    # Infer molecule from data if possible
    molecule = "molecule"
    if "tag" in df.columns and len(df) > 0:
        first_tag = str(df["tag"].iloc[0])
        if "/" in first_tag:
            molecule = first_tag.split("/")[-1]

    ax.set_title(f"autoresearch-qc: VQE Ansatz Discovery for {molecule}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"Saved {OUTPUT_FILE}")


def plot_complexity(df: pd.DataFrame) -> None:
    """Dual y-axis: n_params and circuit_depth vs experiment number."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Experiment #", fontsize=12)
    ax1.set_ylabel("n_params", color="#3498db", fontsize=12)
    ax1.plot(df["experiment_num"], df["n_params"], "o-", color="#3498db", label="n_params")
    ax1.tick_params(axis="y", labelcolor="#3498db")

    ax2 = ax1.twinx()
    ax2.set_ylabel("circuit_depth", color="#e67e22", fontsize=12)
    ax2.plot(df["experiment_num"], df["circuit_depth"], "s-", color="#e67e22", label="circuit_depth")
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    ax1.set_title("Circuit Complexity Over Time", fontsize=14)
    fig.tight_layout()
    fig.savefig("complexity.png", dpi=150)
    print("Saved complexity.png")


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics."""
    total = len(df)
    best_idx = df["energy_error"].idxmin()
    best = df.loc[best_idx]
    chem_acc_achieved = df["chemical_accuracy"].any()

    print(f"\n=== Summary ===")
    print(f"Total experiments: {total}")
    print(f"Best energy_error: {best['energy_error']:.8f} Ha ({best['energy_error'] * 1000:.4f} mHa)")
    print(f"Best n_params: {int(best['n_params'])}")
    print(f"Chemical accuracy achieved: {chem_acc_achieved}")
    if chem_acc_achieved:
        first_acc = df[df["chemical_accuracy"]].iloc[0]
        print(f"  First achieved at experiment #{int(first_acc['experiment_num'])}")


if __name__ == "__main__":
    df = load_results()
    plot_progress(df)
    plot_complexity(df)
    print_summary(df)
