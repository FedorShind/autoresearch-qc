# Molecule Definitions

Each `.yaml` file in this directory defines one molecule. The filename
(without `.yaml`) becomes the key passed to `--molecule`.

`prepare.py` loads all of these at import time via `load_molecules()`.

## Schema

```yaml
name: "LiH"                     # Display name (used in print output)
description: "..."              # Free-text, optional
symbols: ["Li", "H"]            # Atomic symbols, in coordinate order
charge: 0                       # Net molecular charge
multiplicity: 1                 # 2S+1 (singlet=1, triplet=3, ...)
active_electrons: 2             # Active-space electron count
active_orbitals: 3              # Active-space orbital count

# Either provide explicit `coordinates` for fixed geometry,
# OR provide a `geometry` block for parameterized geometry.

coordinates: [x1, y1, z1, x2, y2, z2, ...]   # Flat list, length 3*N, in Å

geometry:
  type: "diatomic"              # "diatomic" | "chain"
  default_bond_length: 1.546    # Used when --bond-length is omitted
  bond_length_range: [0.5, 4.0] # Validation bounds (inclusive)
```

Field naming note: the loader accepts `multiplicity` (preferred) or `mult`
(legacy); both map to the same internal field.

## Geometry types

**`diatomic`** — Two atoms along the z-axis. Coordinates resolved as
`[0, 0, 0, 0, 0, bond_length]`. Use for H₂, LiH, etc.

**`chain`** — N atoms along the z-axis at uniform spacing. The
`default_bond_length` becomes the inter-atom spacing. Use for H₄, H₆, etc.

**Fixed (no `geometry` block)** — Use the explicit `coordinates` list
unchanged. Passing `--bond-length` to a fixed-geometry molecule errors.

## Adding a new molecule

1. Create `molecules/<key>.yaml`. Use an existing file as a template.
2. Verify it builds: `uv run prepare.py --molecule <key>`. The exact
   energy and qubit count should print without errors.
3. Verify VQE works: `uv run circuit.py --molecule <key>`.
4. (Optional) Add a row to the README's Molecules table.

## Files

- `h2.yaml` — H₂, diatomic
- `lih.yaml` — LiH, diatomic
- `beh2.yaml` — BeH₂, fixed (linear)
- `h2o.yaml` — Water, fixed (bent)
- `h4_chain.yaml` — H₄ linear chain
