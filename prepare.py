"""
prepare.py — Frozen evaluation infrastructure for autoresearch-qc.

Builds molecular Hamiltonians, computes exact ground-state energies,
and provides evaluation utilities. The agent NEVER modifies this file.
"""

import time
from typing import Any

import numpy as np
import pennylane as qml

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
MOLECULE = "h2"  # Default — human changes this to level up
TIME_BUDGET_SECONDS = 300  # 5-minute wall-clock budget for VQE optimization
CHEMICAL_ACCURACY_HA = 0.0016  # 1.6 milliHartree threshold

# ============================================================
# MOLECULE DEFINITIONS
# ============================================================
MOLECULES: dict[str, dict[str, Any]] = {
    "h2": {
        "name": "H2",
        "symbols": ["H", "H"],
        "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.735]),
        "charge": 0,
        "mult": 1,
        "active_electrons": 2,
        "active_orbitals": 2,
    },
    "lih": {
        "name": "LiH",
        "symbols": ["Li", "H"],
        "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.546]),
        "charge": 0,
        "mult": 1,
        "active_electrons": 2,
        "active_orbitals": 3,
    },
    "beh2": {
        "name": "BeH2",
        "symbols": ["Be", "H", "H"],
        "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.326, 0.0, 0.0, -1.326]),
        "charge": 0,
        "mult": 1,
        "active_electrons": 4,
        "active_orbitals": 4,
    },
    "h2o": {
        "name": "Water",
        "symbols": ["O", "H", "H"],
        "coordinates": np.array(
            [0.0, 0.0, 0.1173, 0.0, 0.7572, -0.4692, 0.0, -0.7572, -0.4692]
        ),
        "charge": 0,
        "mult": 1,
        "active_electrons": 4,
        "active_orbitals": 4,
    },
    "h4_chain": {
        "name": "H4 chain",
        "symbols": ["H", "H", "H", "H"],
        "coordinates": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]
        ),
        "charge": 0,
        "mult": 1,
        "active_electrons": 4,
        "active_orbitals": 4,
    },
}


def build_hamiltonian(
    molecule_key: str,
) -> tuple[qml.operation.Operator, int, int, np.ndarray]:
    """Build the qubit Hamiltonian for a molecule.

    Uses Jordan-Wigner mapping, STO-3G basis, and active space reduction.

    Args:
        molecule_key: Key into the MOLECULES dict (e.g., "h2", "lih").

    Returns:
        Tuple of (hamiltonian, n_qubits, n_electrons, hf_state) where:
        - hamiltonian: qubit Hamiltonian as a PennyLane Operator
        - n_qubits: number of qubits (= 2 * active_orbitals)
        - n_electrons: number of active electrons
        - hf_state: Hartree-Fock occupation state vector
    """
    config = MOLECULES[molecule_key]

    mol = qml.qchem.Molecule(
        symbols=config["symbols"],
        coordinates=config["coordinates"],
        charge=config["charge"],
        mult=config["mult"],
        basis_name="sto-3g",
        unit="angstrom",
    )

    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
        mol,
        active_electrons=config["active_electrons"],
        active_orbitals=config["active_orbitals"],
        mapping="jordan_wigner",
    )

    n_electrons = config["active_electrons"]
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)

    return hamiltonian, n_qubits, n_electrons, hf_state


def compute_exact_energy(hamiltonian: qml.operation.Operator, n_qubits: int) -> float:
    """Compute the exact ground-state energy via full diagonalization.

    Args:
        hamiltonian: qubit Hamiltonian operator.
        n_qubits: number of qubits.

    Returns:
        Minimum eigenvalue (exact ground-state energy) in Hartree.
    """
    mat = qml.matrix(hamiltonian, wire_order=range(n_qubits))
    eigenvalues = np.linalg.eigvalsh(mat)
    return float(eigenvalues[0])


def evaluate(computed_energy: float, exact_energy: float) -> dict[str, Any]:
    """Evaluate a VQE result against the exact ground-state energy.

    Args:
        computed_energy: energy from VQE optimization (Hartree).
        exact_energy: exact ground-state energy (Hartree).

    Returns:
        Dict with energy_error, energy_error_mha, chemical_accuracy,
        computed_energy, and exact_energy.
    """
    error = abs(computed_energy - exact_energy)
    return {
        "energy_error": error,
        "energy_error_mha": error * 1000,
        "chemical_accuracy": error < CHEMICAL_ACCURACY_HA,
        "computed_energy": computed_energy,
        "exact_energy": exact_energy,
    }


class TimeBudget:
    """Context manager for wall-clock time budgeting.

    Does NOT forcefully kill — the optimization loop must check
    `budget.expired` or `budget.time_remaining()` periodically.
    """

    def __init__(self, budget_seconds: float) -> None:
        self.budget_seconds = budget_seconds
        self._start: float = 0.0
        self.wall_time: float = 0.0

    def __enter__(self) -> "TimeBudget":
        self._start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.wall_time = time.time() - self._start

    def time_remaining(self) -> float:
        """Seconds remaining in the budget."""
        return max(0.0, self.budget_seconds - (time.time() - self._start))

    @property
    def expired(self) -> bool:
        """Whether the time budget has been exhausted."""
        return time.time() - self._start >= self.budget_seconds


if __name__ == "__main__":
    config = MOLECULES[MOLECULE]
    hamiltonian, n_qubits, n_electrons, hf_state = build_hamiltonian(MOLECULE)
    exact_energy = compute_exact_energy(hamiltonian, n_qubits)

    print("=== autoresearch-qc prepare ===")
    print(f"molecule: {config['name']}")
    print(f"n_qubits: {n_qubits}")
    print(f"n_electrons: {n_electrons}")
    print(f"exact_energy: {exact_energy:.6f} Ha")
    print(f"chemical_accuracy_target: {CHEMICAL_ACCURACY_HA} Ha (1.6 mHa)")
    print(f"time_budget: {TIME_BUDGET_SECONDS}s")
    print("Ready for experiments.")
