# Reproducibility

Reproduction of the headline numbers from a clean clone, run on
2026-04-27 on the same machine class as the original (Windows 11,
Python 3.10 venv via uv, default.mixed CPU simulator).

## Method

Six checks against the three result reports, focused on the load-bearing
claims. Same hyperparameters, same scripts, same hardware class. Random
seeds specified per check. Each cell is re-run from a fresh state into
a separate output TSV (`phase_repro_lih.tsv`) so phase_scan's resume
logic does not reuse the original grid data.

## Results

### 1. Phase diagram headline cell

Source: `docs/phase_diagram_report.md`.

The single cell that dominates the phase-diagram conclusion: with all
four singles included and n_d=1, bl=2.5 A is the only point reaching
chemical accuracy at p=0.01. A second cell (bl=1.546, p=0.005) is
included as an anchor in case cell A reproduces by coincidence.

| cell                           | original   | reproduction | delta    | within 5% |
| ------------------------------ | ---------- | ------------ | -------- | --------- |
| bl=2.5, p=0.01, n_d=1          | 1.4828 mHa | 1.4828 mHa   | 0.0%     | yes       |
| bl=1.546, p=0.005, n_d=1       | 0.6627 mHa | 0.6627 mHa   | 0.0%     | yes       |

Both cells reproduce to four-decimal precision in mHa. Chemical-accuracy
classifications match (cell A: True, cell B: True).

Wall time per cell: ~100 s, matching the grid's average.

Reproduction command:

```
uv run --extra optimize phase_scan.py --molecule lih \
    --bond-lengths 2.5 --noise-levels 0.01 --n-doubles 1 \
    --output phase_repro_lih.tsv
```

### 2. v3 zero_singles finding

(pending)

### 3. ZNE overshoot artifact at p=0.001

(pending)

### 4. Agentic with seed 7

(pending)

### 5. Image paths and plotting

(pending)

### 6. Markdown link integrity

(pending)

## Summary

(pending)

## What this does NOT verify

This pass reproduces the headline numbers with a single fresh run per
check. It does NOT:

- Test variance across many seeds (the agentic check uses two: 42 from
  the original, 7 here).
- Verify the same numerical results on a different machine class.
- Re-run the full 60-cell phase diagram or the full validation sweeps.
- Test on different PennyLane / numpy versions.

For external publication those would be additional steps. For repo
confidence, this pass is enough.

## Findings outside the numerical comparison

**Setup gotcha (documented for future reproducers).** A bare `uv sync`
reverts to the base dependency set (no `analysis`, no `optimize`
extras). After sync, `phase_scan.py` and `validate_sweep.py` fail at
import time because they transitively import `skopt` (via
`optimize_noisy.py`), which lives in the `optimize` extra. The README's
`uv run phase_scan.py --molecule lih` example will fail post-sync. The
correct invocation is `uv run --extra optimize phase_scan.py ...`.
This is a packaging/documentation issue, not a scripting bug, and is
trivially worked around. No script was modified to avoid this in the
reproduction pass.
