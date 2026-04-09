# autoresearch-qc

![results](Images/progress.png)

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for quantum computing. An AI agent iterates on a quantum circuit to minimize the ground-state energy error of molecules -- same loop, different domain.

140 experiments across 8 molecules. Chemical accuracy on all of them -- including strongly correlated systems and a hardware-constrained run with no chemistry-specific gates.

---

## Background

The Variational Quantum Eigensolver (VQE) approximates molecular ground-state energies using parameterized quantum circuits. Designing those circuits -- gate types, entanglement topology, parameter initialization, optimizer -- is the bottleneck. This project automates it.

An AI coding agent modifies the circuit code, runs a 5-minute optimization, checks the energy error, commits if improved, reverts if not, repeats. Same pattern as autoresearch, applied to quantum chemistry instead of LLM training.

**Energy error** = |computed − exact|, in Hartree. **Chemical accuracy** = error < 1.6 milliHartree. Below that, the results are useful for real chemistry.

## How it works

| autoresearch | autoresearch-qc |
|---|---|
| `train.py` -- GPT model + training | `circuit.py` -- quantum ansatz + VQE |
| `prepare.py` -- data prep + eval | `prepare.py` -- Hamiltonian + exact energy |
| `val_bpb` ↓ | `energy_error` ↓ |
| 5-min GPU budget | 5-min CPU budget |
| NVIDIA GPU | CPU only |

Three files: `prepare.py` (frozen evaluation harness), `circuit.py` (agent edits this), `program.md` (human edits this).

## Results

Two rounds of experiments. First round: four molecules at equilibrium geometry. Second round: stress tests on strongly correlated and constrained systems.

### Round 1 -- equilibrium molecules

| Molecule | Qubits | Baseline | Best | Params | Strategy |
|----------|--------|----------|------|--------|----------|
| H₂ | 4 | 0.005 mHa | ≈0 | 1 | Single DoubleExcitation gate |
| LiH | 6 | 145 mHa | 0.0001 mHa | 8 | UCCSD + Nesterov + zero init |
| BeH₂ | 8 | 420 mHa | 0.0007 mHa | 12 | Same recipe, first attempt |
| H₂O | 8 | 427 mHa | 0.0003 mHa | 12 | Same recipe, first attempt |

The agent discovered a universal strategy on LiH (experiment #4) that transferred to every subsequent molecule without modification: Hartree-Fock initial state → SingleExcitation + DoubleExcitation gates → zero parameter initialization → Nesterov momentum, step 0.4, convergence threshold 1e-8.

On BeH₂ and H₂O, this recipe hit chemical accuracy on the first experiment -- 420 mHa → 0.0007 mHa in one step.

### Round 2 -- stress tests

![stress test results](Images/progress_stress.png)

| Molecule | Qubits | Correlation | Baseline | Best | Params | What happened |
|----------|--------|-------------|----------|------|--------|---------------|
| H₄ chain | 8 | Strong | 311 mHa | 0.004 mHa | 78 | Needed 3-layer UCCSD. Single layer hit 0.08 mHa -- still chemical accuracy, but 3 layers improved 20x. |
| H₂ (3.0Å) | 4 | Strong | 0.62 mHa | ≈0 | 1 | Already at chemical accuracy from baseline. 4 qubits is too small to challenge anything. |
| LiH (3.0Å) | 6 | Strong | 117 mHa | 0.0001 mHa | 8 | UCCSD worked despite literature claims it fails at stretched geometries. At this active space size (3 orbitals, STO-3G), the problem is tractable. Key difference from equilibrium: doubles-only failed at 6.17 mHa -- singles became essential. |
| LiH (no chem gates) | 6 | Weak | 174 mHa | 0.30 mHa | 72 | Chemical accuracy with generic gates only. Required 4 layers, circular CNOT, triple rotation (RX+RY+RZ), small init ±0.1. Cost: 9x more parameters, 3000x worse accuracy vs UCCSD. |

### Full summary

| Molecule | Qubits | Type | Best error | Chem. acc? |
|----------|--------|------|-----------|------------|
| H₂ | 4 | Equilibrium | ≈0 mHa | ✓ |
| LiH | 6 | Equilibrium | 0.0001 mHa | ✓ |
| BeH₂ | 8 | Equilibrium | 0.0007 mHa | ✓ |
| H₂O | 8 | Equilibrium | 0.0003 mHa | ✓ |
| H₄ chain | 8 | Strong corr. | 0.004 mHa | ✓ |
| H₂ (3.0Å) | 4 | Stretched | ≈0 mHa | ✓ |
| LiH (3.0Å) | 6 | Stretched | 0.0001 mHa | ✓ |
| LiH (constrained) | 6 | No chem gates | 0.30 mHa | ✓ |

## Findings

**Ansatz architecture dominates.** Chemistry gates outperformed all hardware-efficient variants by 3--4 orders of magnitude. One architectural change on LiH improved accuracy by 10,000x. No optimizer or hyperparameter tuning came close.

**The recipe survives strong correlation.** UCCSD + Nesterov achieved chemical accuracy on all 8 molecules, including strongly correlated H₄ and stretched-geometry LiH. Multi-layer UCCSD (3 Trotter steps) helps at high precision -- single-layer UCCSD plateaus at 0.08 mHa on H₄, three layers reach 0.004 mHa.

**Generic gates can work, at a cost.** The constrained LiH run reached chemical accuracy (0.30 mHa) using only RX/RY/RZ + CNOT. But it needed 72 parameters vs 8 for UCCSD, and the accuracy was 3000x worse. The claim isn't "HEA fails on LiH" -- it's "HEA is 9x less efficient."

**The 4th double excitation is a cliff.** On both 8-qubit equilibrium molecules, dropping from 4 to 3 double excitations pushed error from ~0.07 mHa to ~3 mHa. Not gradual -- a discrete threshold.

**Optimizer choice barely matters.** Nesterov, Adam, GD, COBYLA all converge to the same precision given the right ansatz. Speed differs (Nesterov/GD are 2--3x faster), accuracy does not.

**Small active spaces hide the hard problem.** The literature says UCCSD fails on stretched LiH. At 6 qubits / 3 orbitals / STO-3G, it doesn't. The real challenge requires larger basis sets or 12+ qubit active spaces where the correlation structure overwhelms UCCSD's expressibility.

### The recipe

```
1. BasisState(hf_state)               # Hartree-Fock initial state
2. SingleExcitation(θ, wires)          # from qchem.excitations()
3. DoubleExcitation(θ, wires)          # from qchem.excitations()
4. params = zeros                      # initialize at identity
5. Nesterov, step=0.4, conv=1e-8      # tight convergence
```

For strongly correlated systems, repeat steps 2--3 with independent parameters (multi-layer UCCSD).

## Bayesian optimization

For systematic search over circuit configurations:

```bash
uv run optimize.py --molecule lih --n-trials 30
```

Ranks excitations by gradient importance, then uses GP-based Bayesian optimization to find the optimal subset and hyperparameters. Useful for finding the minimum circuit that achieves chemical accuracy.

## Noisy simulation (v3)

Real quantum hardware has gate errors. v3 adds depolarizing noise simulation and
ZNE (zero-noise extrapolation) error mitigation, so the agent faces a realistic
depth-noise tradeoff: deeper circuits are more expressive but accumulate more noise.

```bash
# Run with simulated noise
uv run noisy_circuit.py --molecule h2 --noise 0.005

# Output includes:
# - energy_error_noisy: raw error without mitigation
# - energy_error_mitigated: after ZNE extrapolation
# - improvement_factor: how much ZNE helped
```

Uses PennyLane's `default.mixed` density matrix simulator with `DepolarizingChannel`
inserted after each gate. ZNE runs the circuit at multiple noise levels (via circuit
folding) and extrapolates to the zero-noise limit.

The noisy agent edits `noisy_circuit.py` and follows `program_noisy.md`.

## Quick start

Python 3.10+, [uv](https://docs.astral.sh/uv/). No GPU.

```bash
git clone https://github.com/FedorShind/autoresearch-qc.git
cd autoresearch-qc
uv sync
uv run prepare.py       # verify setup
uv run circuit.py       # run baseline
```

## Running the agent

```
Read program.md and let's kick off a new experiment session.
```

Works with Claude Code, Codex, or any agent that can edit files and run shell commands.

## Molecules

| Molecule | Qubits | Difficulty | Notes |
|----------|--------|------------|-------|
| H₂ | 4 | Tutorial | Anything works |
| LiH | 6 | Easy | Generic ansatzes fail |
| BeH₂ | 8 | Medium | More excitation paths |
| H₂O | 8 | Medium | Classic benchmark |
| H₄ chain | 8 | Hard | Strongly correlated |
| H₂ (3.0Å) | 4 | Easy | Stretched geometry |
| LiH (3.0Å) | 6 | Medium | Bond-breaking regime |

Switch molecules with `--molecule`: `uv run circuit.py --molecule lih`

## Files

```
prepare.py        — Hamiltonian, exact energy, evaluation, noise model (frozen)
circuit.py        — ansatz + VQE loop (agent edits, noiseless mode)
noisy_circuit.py  — ansatz + noisy VQE + ZNE mitigation (agent edits, noisy mode)
program.md        — agent instructions, noiseless (human edits)
program_noisy.md  — agent instructions, noisy mode (human edits)
optimize.py       — Bayesian optimization tool
analysis.ipynb    — experiment analysis
plot.py           — chart generation
```

## Acknowledgments

Built on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [PennyLane](https://pennylane.ai/).

## License

MIT