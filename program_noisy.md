# autoresearch-qc — Noisy Mode Agent Program

You are an autonomous research agent iterating on a variational quantum circuit to minimize energy error for molecular ground-state estimation **under simulated hardware noise**.

## What changed from noiseless mode

The device adds **depolarizing noise** after every gate. This means:
- Deeper circuits accumulate more error (each gate adds noise)
- The raw VQE result is biased away from the true ground state
- ZNE (zero-noise extrapolation) can partially correct this bias
- The optimal circuit under noise may be **different** from the optimal noiseless circuit

You modify `noisy_circuit.py` (not `circuit.py`).

## Setup

1. Create the branch: `git checkout -b autoresearch/noisy-<tag>` from current master
2. Read the in-scope files:
   - `README.md` — repository context
   - `prepare.py` — frozen infrastructure: Hamiltonian construction, exact energy, evaluation, timer, noise model. **Do not modify.**
   - `noisy_circuit.py` — the file you modify. Ansatz definition, optimizer config, noise/mitigation settings.
   - `molecules/*.yaml` — molecule definitions. One file per molecule. Add a new molecule by creating a YAML file (see `molecules/README.md` for the schema).
3. Verify setup: run `uv run noisy_circuit.py --molecule h2 --noise 0.005` and confirm it prints results without errors. Pass `--bond-length <Å>` on diatomics and chains to vary geometry without editing files.
4. Initialize `results.tsv` with header row:
   `experiment	tag	energy_error	energy_error_mha	chemical_accuracy	energy_error_noisy	energy_error_mitigated	energy_error_ideal	improvement_factor	noise_strength	mitigation_method	n_params	circuit_depth	wall_time	notes`
5. Confirm and go.

## Experiment Loop

Each experiment modifies `noisy_circuit.py`, runs a noisy VQE optimization, and evaluates the result.

### Per Experiment:

1. **Form a hypothesis.** State what you're changing and why. One-line description.
2. **Edit `noisy_circuit.py`.** Make your change. Only modify this file.
3. **Run the experiment:**
   ```
   uv run noisy_circuit.py > run.log 2>&1
   ```
4. **Read results:**
   ```
   grep "^energy_error:\|^energy_error_mha:\|^chemical_accuracy:\|^energy_error_noisy:\|^energy_error_mitigated:\|^energy_error_ideal:\|^improvement_factor:\|^mitigation_method:\|^noise_strength:\|^n_params:\|^circuit_depth:\|^wall_time:" run.log
   ```
   If grep output is empty, run `tail -n 50 run.log` to read the traceback.
5. **Record** results in `results.tsv`.
6. **Decision:**
   - If `energy_error` **decreased**: `git add noisy_circuit.py && git commit -m "<description>"`
   - If **equal or worse**: `git checkout -- noisy_circuit.py`
7. **Repeat** from step 1.

## What You Can Control

### Ansatz Configuration
- Ansatz architecture (gates, entanglement, depth)
- Parameter initialization
- Optimizer choice and hyperparameters
- Convergence strategy

### Mitigation Configuration
- `MITIGATION`: `"none"` (raw noisy) or `"zne"` (zero-noise extrapolation)
- `ZNE_SCALE_FACTORS`: Noise amplification levels (e.g., `[1, 2, 3]` or `[1, 1.5, 2, 2.5, 3]`)
- `ZNE_EXTRAPOLATION`: `"linear"`, `"polynomial"`, `"richardson"`, or `"exponential"`
- `ZNE_POLYNOMIAL_ORDER`: Degree for polynomial extrapolation
- `NOISE_STRENGTH`: Can override via CLI `--noise`

### The Depth-Noise Tradeoff

This is the core challenge. In noiseless mode, a 26-param UCCSD circuit is optimal. Under noise:
- 26 params = more gates = more noise channels = more accumulated error
- A 12-param circuit has worse *ideal* energy but may have better *mitigated* energy
- ZNE helps more on shallower circuits (less noise to extrapolate through)
- There is an optimal depth that balances expressibility vs noise resilience

Explore:
1. Full UCCSD with ZNE vs reduced UCCSD (fewer excitations) with ZNE
2. Whether chemistry gates are more noise-resilient than generic gates
3. Different ZNE scale factors (more factors = better extrapolation but slower)
4. Linear vs polynomial extrapolation (linear is safer for short circuits)

## Metrics

- **Primary:** `energy_error` — energy error of the active cost function (mitigated if ZNE is on). Lower is better.
- **Chemical accuracy:** `energy_error < 0.0016 Ha` (1.6 mHa). Achieving this under noise is a real win.
- **Reference:** `energy_error_noisy` — raw noisy error without mitigation.
- **Ideal:** `energy_error_ideal` — what the error would be without noise.
- **Improvement factor:** `energy_error_noisy / energy_error_mitigated` — how much ZNE helped. Higher is better.
- **Tiebreaker:** Fewer parameters, shallower circuit, higher improvement factor.

## What to Explore (Priority Order)

1. **Chemistry-inspired ansatzes under noise**: UCCSD with `SingleExcitation`/`DoubleExcitation`. These may be more noise-resilient because they are parameter-efficient.
2. **ZNE extrapolation methods**: Start with linear (safest). Try polynomial for longer circuits.
3. **ZNE scale factors**: `[1, 2, 3]` is the default. Try `[1, 1.5, 2, 2.5, 3]` for smoother extrapolation.
4. **Reduced excitation sets**: Drop the least important excitations to shorten the circuit.
5. **Noise-aware initialization**: Zero init may still work with chemistry gates under noise.
6. **Optimizer choice under noise**: Gradient-free optimizers (scipy COBYLA) may handle noisy gradients better than parameter-shift.

## What NOT to Explore

- Don't modify `prepare.py`.
- Don't try to remove or bypass the noise (it simulates real hardware).
- Don't increase circuit depth beyond 3 layers without checking noise impact first.
- Don't use more than 5 ZNE scale factors (diminishing returns, large slowdown).

## Physics Knowledge

All the noiseless physics knowledge from `program.md` still applies:
- Hartree-Fock initialization is powerful
- Entanglement is mandatory
- Barren plateaus are real (even worse under noise)
- Chemistry gates encode physics

Additional noise-specific knowledge:
- **Depolarizing noise** replaces the quantum state with the maximally mixed state with probability p per gate. More gates = more mixing = energy biased toward zero.
- **ZNE** runs the circuit at multiple artificially amplified noise levels, then extrapolates back to zero noise. It works best when noise scales approximately linearly with circuit depth.
- **Circuit folding** (U U^dag U) is how ZNE amplifies noise. Each fold roughly doubles the circuit depth and noise.
- **Short circuits benefit most from ZNE** because the noise-energy relationship is more linear.

## Performance Notes

- `default.mixed` is O(4^n) in memory vs O(2^n) for `default.qubit`. 8-qubit molecules will be noticeably slower.
- ZNE with 3 scale factors = 3x circuit evaluations per energy call. Combined with parameter-shift gradients (2 evals/param), expect ~3x slowdown vs noiseless.
- If the 5-minute budget is too tight, reduce `MAX_ITERATIONS` or use gradient-free optimization.
