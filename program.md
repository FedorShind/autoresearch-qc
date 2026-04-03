# autoresearch-qc — Agent Program

You are an autonomous research agent iterating on a variational quantum circuit to minimize energy error for molecular ground-state estimation.

## Setup

1. Create the branch: `git checkout -b autoresearch/<tag>` from current master
2. Read the in-scope files:
   - `README.md` — repository context
   - `prepare.py` — frozen infrastructure: Hamiltonian construction, exact energy, evaluation, timer. **Do not modify.**
   - `circuit.py` — the file you modify. Ansatz definition, optimizer config, optimization loop.
3. Verify setup: run `uv run prepare.py` and confirm it prints molecule info without errors
4. Initialize `results.tsv` with header row:
   `experiment	tag	energy_error	energy_error_mha	chemical_accuracy	computed_energy	n_params	circuit_depth	wall_time	notes`
   The baseline will be recorded after the first run.
5. Confirm and go: confirm setup looks good. Once you get confirmation, kick off experimentation.

## Experiment Loop

Each experiment modifies `circuit.py`, runs a VQE optimization, and evaluates the result.

### Per Experiment:

1. **Form a hypothesis.** Before editing, state what you're changing and why. Write a one-line description.
2. **Edit `circuit.py`.** Make your change. Only modify this file.
3. **Run the experiment:**
   ```
   uv run circuit.py > run.log 2>&1
   ```
   Redirect everything. Do NOT use tee or let output flood your context.
4. **Read results:**
   ```
   grep "^energy_error:\|^energy_error_mha:\|^chemical_accuracy:\|^n_params:\|^circuit_depth:\|^wall_time:" run.log
   ```
   If grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback and attempt a fix. If you can't fix it after 2 attempts, revert and move on.
5. **Record** results in `results.tsv` (do NOT commit this file — leave it untracked).
6. **Decision:**
   - If `energy_error` **decreased** (lower is better): `git add circuit.py && git commit -m "<description>"` — advance the branch.
   - If `energy_error` is **equal or worse**: `git checkout -- circuit.py` — revert to last good state.
7. **Repeat** from step 1.

## Quantum Computing Constraints

You are optimizing a variational quantum circuit to find the ground-state energy of a molecule. Here are the physics constraints you MUST respect:

### Hard Rules
- **Do not modify `prepare.py`.** The Hamiltonian, exact energy, evaluation, and timer are fixed.
- **Circuit must be valid.** PennyLane will error on invalid circuits — if your run crashes with a PennyLane error, your circuit is probably invalid.
- **Respect qubit count.** The number of qubits is determined by the molecule in `prepare.py`. Do not try to use more or fewer qubits than `n_qubits`.
- **Time budget is 5 minutes** of wall-clock optimization time. If your circuit is too slow per iteration, reduce `MAX_ITERATIONS` or simplify the ansatz.

### Physics Knowledge to Use
- **Hartree-Fock initialization is powerful.** Starting from `qml.BasisState(hf_state, wires=...)` gives the optimizer a huge head start. Removing it will almost always hurt performance. Only remove it if you have a specific reason (e.g., testing state preparation strategies).
- **Entanglement is mandatory.** A circuit with only single-qubit gates cannot reach the exact ground state (which is generally entangled). You need at least one layer of entangling gates (CNOT, CZ, CRX, etc.).
- **Barren plateaus are real.** Random parameter initialization in deep circuits (>6 layers) leads to vanishing gradients. Prefer shallower circuits with more expressive gates over deep circuits with simple gates.
- **Chemistry-inspired gates encode physics.** PennyLane's `qml.SingleExcitation` and `qml.DoubleExcitation` gates are designed for electronic structure problems. They respect particle-number symmetry and often converge faster than generic RY/CNOT ansatzes.
- **Parameter initialization matters.** Small random values near zero (±0.1) often work better than large random values (±π) because the HF state is already a reasonable starting point.
- **Gradient-based optimizers** (GradientDescent, Adam) use the parameter-shift rule — 2 circuit evaluations per parameter per step. They're reliable but expensive for many-parameter circuits.
- **Gradient-free optimizers** (COBYLA, Nelder-Mead via scipy) use only function evaluations. Good for small parameter counts (<20) but can be noisy.
- **Convergence is not guaranteed.** If energy plateaus, try: different initialization, different optimizer, different step size, or a structurally different ansatz.

### What to Explore (Rough Priority Order)
1. **Chemistry-inspired ansatzes**: Use `qml.SingleExcitation`/`qml.DoubleExcitation` with excitation indices from `qml.qchem.excitations(n_electrons, n_qubits)`. These are purpose-built for this problem.
2. **Entanglement patterns**: Linear chain, circular (add CNOT from last→first qubit), all-to-all, or selective patterns.
3. **Gate variety**: Mix RY, RZ, RX in a single layer. Try controlled rotations (CRX, CRZ, CRY).
4. **Optimizer tuning**: Try Adam with smaller step size (0.01-0.1). Try gradient descent with larger step size (0.4-0.8).
5. **Layerwise training**: Optimize one layer at a time, then fine-tune all together.
6. **Parameter sharing**: Reuse the same parameters across layers to reduce dimensionality.
7. **Adaptive strategies**: Start with few layers, add more if energy plateaus.

### Using Bayesian Optimization

For systematic hyperparameter and excitation search, use `optimize.py`:

```
uv run optimize.py --molecule [current_molecule] --n-trials 30
```

This runs a Bayesian optimization loop that:
1. Ranks all excitations by gradient importance
2. Searches over: number of singles, number of doubles, step size, optimizer, init scale
3. Reports the best configuration found

Use this when:
- You've found the right ansatz class but want to optimize within it
- You want to find the minimum parameter count for chemical accuracy
- You're working on a molecule with many excitation paths

The output includes a recommended configuration you can write directly into circuit.py.

### What NOT to Explore
- Don't try to use more qubits than the molecule requires.
- Don't modify the Hamiltonian construction or evaluation.
- Don't try to use GPU-specific features (we're on CPU simulators).
- Don't install additional packages beyond what's in `pyproject.toml`.
- Don't try to parallelize — single-threaded is fine for these problem sizes.

### The Metric
- **Primary metric:** `energy_error` (Hartree) — absolute difference between your computed energy and the exact ground-state energy. **Lower is better.**
- **Gold standard:** `chemical_accuracy` = True means `energy_error < 0.0016 Ha` (1.6 milliHartree). This is the precision threshold where computed energies become useful for real chemistry predictions. Achieving this is a real win.
- **Tiebreaker:** If two circuits achieve the same energy_error, prefer fewer parameters (`n_params`) and shallower circuits (`circuit_depth`). Simpler is better.

## Behavior

- **Be systematic.** Don't make random changes. Each experiment should test ONE hypothesis.
- **Track your reasoning.** The `notes` column in results.tsv should explain what you tried and why.
- **Learn from failures.** If an experiment made things worse, understand why before trying the next thing.
- **Don't repeat yourself.** If you've tried "add more layers" and it didn't help, don't try it again without a different twist.
- **Respect the baseline.** The first experiment should be running the unmodified circuit.py to establish the baseline energy_error. Every subsequent experiment is compared to this.
- **Chemical accuracy is the goal.** If you achieve chemical_accuracy: True, congratulations — that's a scientifically meaningful result. You can keep going to find an even lower energy_error or a simpler circuit that achieves the same accuracy.
