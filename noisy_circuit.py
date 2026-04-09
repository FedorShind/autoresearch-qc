"""
noisy_circuit.py — Noisy VQE with error mitigation.

This is the file the agent edits when running in noisy mode.
Same structure as circuit.py, but adds:
1. Noise simulation via default.mixed + DepolarizingChannel
2. Error mitigation via ZNE (zero-noise extrapolation)
3. Comparison of raw noisy vs mitigated vs ideal results

Usage:
    uv run noisy_circuit.py                              # default molecule + noise
    uv run noisy_circuit.py --molecule lih --noise 0.005
"""

import argparse

import pennylane as qml
from pennylane import noise as qml_noise
from pennylane import numpy as pnp
import numpy as np
from prepare import (
    MOLECULE, MOLECULES, TIME_BUDGET_SECONDS, CHEMICAL_ACCURACY_HA,
    build_hamiltonian, compute_exact_energy, evaluate, TimeBudget,
    build_device, get_zne_config,
)

# ============================================================
# CLI ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser(description="Noisy VQE with error mitigation")
parser.add_argument("--molecule", default=MOLECULE, choices=list(MOLECULES.keys()),
                    help="Molecule key (default: %(default)s)")
parser.add_argument("--noise", type=float, default=0.005, help="Noise strength (default: 0.005)")
args = parser.parse_args()

# ============================================================
# NOISE & MITIGATION CONFIGURATION (agent edits these)
# ============================================================
NOISE_STRENGTH = args.noise
MITIGATION = "zne"                   # Options: "none", "zne"
ZNE_SCALE_FACTORS = [1, 2, 3]       # noise amplification levels
ZNE_EXTRAPOLATION = "linear"         # "linear", "polynomial", "richardson", "exponential"
ZNE_POLYNOMIAL_ORDER = 2             # for polynomial extrapolation

# ============================================================
# ANSATZ CONFIGURATION (agent edits these)
# ============================================================
N_LAYERS = 2
MAX_ITERATIONS = 200
STEP_SIZE = 0.4
CONVERGENCE_THRESHOLD = 1e-6
OPTIMIZER = "gradient_descent"  # options: gradient_descent, adam, nesterov

# ============================================================
# BUILD PROBLEM
# ============================================================
molecule_key = args.molecule
hamiltonian, n_qubits, n_electrons, hf_state = build_hamiltonian(molecule_key)
exact_energy = compute_exact_energy(hamiltonian, n_qubits)

# Noisy device (default.mixed + depolarizing noise model)
dev_noisy = build_device(n_qubits, noise_strength=NOISE_STRENGTH)

# Ideal device (default.qubit, for reference comparison)
dev_ideal = qml.device("default.qubit", wires=n_qubits)

# ============================================================
# ANSATZ DEFINITION (agent edits this function)
# ============================================================
def ansatz(params, wires):
    """
    Variational ansatz circuit.

    Current: Hardware-efficient ansatz (same as circuit.py baseline).
    The agent should experiment with different architectures, especially
    considering the depth-noise tradeoff: deeper circuits are more
    expressive but accumulate more noise under simulation.
    """
    qml.BasisState(hf_state, wires=wires)

    n_wires = len(wires)
    for layer in range(N_LAYERS):
        for i in range(n_wires):
            qml.RY(params[layer, i], wires=wires[i])
        for i in range(n_wires - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


# ============================================================
# COST FUNCTIONS
# ============================================================

# 1. Ideal — noiseless reference (for comparison only)
@qml.qnode(dev_ideal)
def cost_fn_ideal(params):
    ansatz(params, wires=range(n_qubits))
    return qml.expval(hamiltonian)


# 2. Noisy — raw noisy evaluation (no mitigation)
@qml.qnode(dev_noisy)
def cost_fn_noisy(params):
    ansatz(params, wires=range(n_qubits))
    return qml.expval(hamiltonian)


# 3. ZNE-mitigated — extrapolated to zero noise
if MITIGATION == "zne":
    zne_config = get_zne_config(
        scale_factors=tuple(ZNE_SCALE_FACTORS),
        extrapolation=ZNE_EXTRAPOLATION,
        polynomial_order=ZNE_POLYNOMIAL_ORDER,
    )
    cost_fn_zne = qml_noise.mitigate_with_zne(cost_fn_noisy, **zne_config)
else:
    cost_fn_zne = None

# Active cost function — what the optimizer uses
cost_fn_active = cost_fn_zne if cost_fn_zne is not None else cost_fn_noisy


# ============================================================
# OPTIMIZATION LOOP (agent edits this)
# ============================================================
def run_optimization():
    """Run the VQE optimization within the time budget."""

    # Initialize parameters
    n_params_per_layer = n_qubits
    params = pnp.random.uniform(-np.pi, np.pi, (N_LAYERS, n_params_per_layer),
                                requires_grad=True)

    # Select optimizer
    if OPTIMIZER == "gradient_descent":
        opt = qml.GradientDescentOptimizer(stepsize=STEP_SIZE)
    elif OPTIMIZER == "adam":
        opt = qml.AdamOptimizer(stepsize=STEP_SIZE)
    elif OPTIMIZER == "nesterov":
        opt = qml.NesterovMomentumOptimizer(stepsize=STEP_SIZE)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    best_energy = float("inf")
    best_params = params
    prev_energy = float("inf")

    with TimeBudget(TIME_BUDGET_SECONDS) as budget:
        for step in range(MAX_ITERATIONS):
            if budget.expired:
                break

            params, energy = opt.step_and_cost(cost_fn_active, params)
            energy = float(energy)

            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()

            # Convergence check
            if abs(energy - prev_energy) < CONVERGENCE_THRESHOLD:
                break

            prev_energy = energy

    return best_energy, budget.wall_time, best_params


# ============================================================
# MAIN — Run and report
# ============================================================
if __name__ == "__main__":
    best_energy, wall_time, final_params = run_optimization()

    # Evaluate on all three cost functions for comparison
    noisy_energy = float(cost_fn_noisy(final_params))
    ideal_energy = float(cost_fn_ideal(final_params))
    if cost_fn_zne is not None:
        mitigated_energy = float(cost_fn_zne(final_params))
    else:
        mitigated_energy = noisy_energy

    # Compute errors
    noisy_result = evaluate(noisy_energy, exact_energy)
    ideal_result = evaluate(ideal_energy, exact_energy)
    mitigated_result = evaluate(mitigated_energy, exact_energy)

    # Primary result is from the active cost function
    active_result = mitigated_result if MITIGATION == "zne" else noisy_result

    # Improvement factor: how much ZNE helped vs raw noisy
    noisy_err = noisy_result["energy_error"]
    mitigated_err = mitigated_result["energy_error"]
    improvement = noisy_err / mitigated_err if mitigated_err > 1e-12 else float("inf")

    # --- Machine-readable output (agent greps these) ---
    print(f"energy_error: {active_result['energy_error']:.8f}")
    print(f"energy_error_mha: {active_result['energy_error_mha']:.4f}")
    print(f"chemical_accuracy: {active_result['chemical_accuracy']}")
    print(f"computed_energy: {active_result['computed_energy']:.8f}")
    print(f"exact_energy: {active_result['exact_energy']:.8f}")
    print(f"energy_error_noisy: {noisy_result['energy_error']:.8f}")
    print(f"energy_error_mitigated: {mitigated_result['energy_error']:.8f}")
    print(f"energy_error_ideal: {ideal_result['energy_error']:.8f}")
    print(f"mitigation_method: {MITIGATION}")
    print(f"noise_strength: {NOISE_STRENGTH}")
    print(f"improvement_factor: {improvement:.2f}")
    print(f"n_params: {final_params.size}")
    print(f"circuit_depth: {N_LAYERS}")
    print(f"wall_time: {wall_time:.1f}")
