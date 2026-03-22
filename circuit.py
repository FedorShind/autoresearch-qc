"""
circuit.py — VQE ansatz and optimization loop.
This is the ONLY file the agent modifies.

The agent can change:
- Ansatz architecture (gates, entanglement, depth)
- Parameter initialization
- Optimizer choice and hyperparameters
- Convergence strategy
- Anything else in this file

The agent CANNOT modify prepare.py.
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from prepare import (
    MOLECULE, TIME_BUDGET_SECONDS, CHEMICAL_ACCURACY_HA,
    build_hamiltonian, compute_exact_energy, evaluate, TimeBudget
)

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
hamiltonian, n_qubits, n_electrons, hf_state = build_hamiltonian(MOLECULE)
exact_energy = compute_exact_energy(hamiltonian, n_qubits)
dev = qml.device("default.qubit", wires=n_qubits)

# ============================================================
# ANSATZ DEFINITION (agent edits this function)
# ============================================================
def ansatz(params, wires):
    """
    Variational ansatz circuit.

    Current: Hardware-efficient ansatz
    - Prepares Hartree-Fock state
    - N_LAYERS of: RY rotations on all qubits + linear CNOT chain

    The agent should experiment with:
    - Different gate choices (RX, RZ, CRX, CRZ, etc.)
    - Different entanglement patterns (circular, all-to-all)
    - Chemistry-inspired gates (SingleExcitation, DoubleExcitation)
    - Depth (more/fewer layers)
    - Whether to use HF state initialization
    """
    # Prepare Hartree-Fock initial state
    qml.BasisState(hf_state, wires=wires)

    n_wires = len(wires)
    for layer in range(N_LAYERS):
        # Single-qubit rotations
        for i in range(n_wires):
            qml.RY(params[layer, i], wires=wires[i])
        # Entangling layer: linear CNOT chain
        for i in range(n_wires - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


# ============================================================
# COST FUNCTION
# ============================================================
@qml.qnode(dev)
def cost_fn(params):
    ansatz(params, wires=range(n_qubits))
    return qml.expval(hamiltonian)


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
    prev_energy = float("inf")

    with TimeBudget(TIME_BUDGET_SECONDS) as budget:
        for step in range(MAX_ITERATIONS):
            if budget.expired:
                break

            params, energy = opt.step_and_cost(cost_fn, params)
            energy = float(energy)

            if energy < best_energy:
                best_energy = energy

            # Convergence check
            if abs(energy - prev_energy) < CONVERGENCE_THRESHOLD:
                break

            prev_energy = energy

    return best_energy, budget.wall_time, params


# ============================================================
# MAIN — Run and report (agent does not change output format)
# ============================================================
if __name__ == "__main__":
    best_energy, wall_time, final_params = run_optimization()
    results = evaluate(best_energy, exact_energy)

    # --- Machine-readable output (agent greps these) ---
    print(f"energy_error: {results['energy_error']:.8f}")
    print(f"energy_error_mha: {results['energy_error_mha']:.4f}")
    print(f"chemical_accuracy: {results['chemical_accuracy']}")
    print(f"computed_energy: {results['computed_energy']:.8f}")
    print(f"exact_energy: {results['exact_energy']:.8f}")
    print(f"n_params: {final_params.size}")
    print(f"circuit_depth: {N_LAYERS}")
    print(f"wall_time: {wall_time:.1f}")
