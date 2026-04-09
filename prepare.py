"""
prepare.py — Frozen evaluation infrastructure for autoresearch-qc.

Builds molecular Hamiltonians, computes exact ground-state energies,
and provides evaluation utilities. The agent NEVER modifies this file.
"""

import argparse
import time
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import noise as qml_noise

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
MOLECULE = "h2"  # Default — human changes this to level up
TIME_BUDGET_SECONDS = 300  # 5-minute wall-clock budget for VQE optimization
CHEMICAL_ACCURACY_HA = 0.0016  # 1.6 milliHartree threshold

# Noise simulation (v3) — opt-in, 0.0 = noiseless (backward compatible)
NOISE_STRENGTH = 0.0  # Depolarizing probability per gate (realistic: 0.001–0.01)
NOISE_TYPE = "depolarizing"  # Only "depolarizing" supported currently

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
    "h2_stretched": {
        "name": "H2 (stretched, 3.0 A)",
        "symbols": ["H", "H"],
        "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0]),
        "charge": 0,
        "mult": 1,
        "active_electrons": 2,
        "active_orbitals": 2,
    },
    "lih_stretched": {
        "name": "LiH (stretched, 3.0 A)",
        "symbols": ["Li", "H"],
        "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0]),
        "charge": 0,
        "mult": 1,
        "active_electrons": 2,
        "active_orbitals": 3,
    },
}


def build_device(
    n_qubits: int,
    *,
    noise_strength: float = 0.0,
    noise_type: str = "depolarizing",
) -> Any:
    """Build a PennyLane device, noisy or noiseless depending on config.

    When noise_strength > 0, uses default.mixed (density matrix simulator)
    with a NoiseModel that inserts DepolarizingChannel after each gate.
    When noise_strength == 0, uses default.qubit (statevector, faster).

    Noise is attached at the device level (not in the circuit) so that
    qml.noise.fold_global can fold the clean circuit for ZNE.

    Args:
        n_qubits: Number of qubits.
        noise_strength: Depolarizing probability per gate. 0.0 = noiseless.
        noise_type: Type of noise channel. Only "depolarizing" supported.

    Returns:
        A PennyLane device (possibly wrapped with a noise model).
    """
    if noise_strength <= 0:
        return qml.device("default.qubit", wires=n_qubits)

    if noise_type != "depolarizing":
        raise ValueError(f"Unknown noise type: {noise_type!r}. Only 'depolarizing' supported.")

    noise_model = qml_noise.NoiseModel({
        qml_noise.op_in([
            qml.RX, qml.RY, qml.RZ,
            qml.CNOT, qml.CZ,
            qml.SingleExcitation, qml.DoubleExcitation,
            qml.Hadamard, qml.PauliX,
        ]): qml_noise.partial_wires(qml.DepolarizingChannel, noise_strength),
    })
    dev = qml.device("default.mixed", wires=n_qubits)
    return qml_noise.add_noise(dev, noise_model)


def get_zne_config(
    scale_factors: tuple[float, ...] = (1.0, 2.0, 3.0),
    extrapolation: str = "polynomial",
    polynomial_order: int = 2,
) -> dict[str, Any]:
    """Build ZNE configuration for qml.noise.mitigate_with_zne.

    Returns a dict that can be unpacked directly:
        mitigated_qnode = qml.noise.mitigate_with_zne(qnode, **get_zne_config())

    Args:
        scale_factors: Noise amplification levels for circuit folding.
        extrapolation: Extrapolation method — "polynomial", "richardson",
            or "exponential".
        polynomial_order: Polynomial degree (only used when
            extrapolation="polynomial").

    Returns:
        Dict with scale_factors, folding, and extrapolate keys.
    """
    extrapolate_fns = {
        "polynomial": qml_noise.poly_extrapolate,
        "richardson": qml_noise.richardson_extrapolate,
        "exponential": qml_noise.exponential_extrapolate,
    }
    if extrapolation not in extrapolate_fns:
        raise ValueError(
            f"Unknown extrapolation: {extrapolation!r}. "
            f"Options: {list(extrapolate_fns.keys())}"
        )

    extrapolate = extrapolate_fns[extrapolation]
    if extrapolation == "polynomial":
        extrapolate = lambda x, y, _fn=extrapolate: _fn(x, y, order=polynomial_order)

    return {
        "scale_factors": list(scale_factors),
        "folding": qml_noise.fold_global,
        "extrapolate": extrapolate,
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
    parser = argparse.ArgumentParser(description="autoresearch-qc: prepare molecular Hamiltonians")
    parser.add_argument("--molecule", default=MOLECULE, choices=list(MOLECULES.keys()),
                        help="Molecule to simulate (default: %(default)s)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Depolarizing noise strength per gate (0.0 = noiseless)")
    args = parser.parse_args()

    molecule_key = args.molecule
    noise_strength = args.noise

    config = MOLECULES[molecule_key]
    hamiltonian, n_qubits, n_electrons, hf_state = build_hamiltonian(molecule_key)
    exact_energy = compute_exact_energy(hamiltonian, n_qubits)

    print("=== autoresearch-qc prepare ===")
    print(f"molecule: {config['name']}")
    print(f"n_qubits: {n_qubits}")
    print(f"n_electrons: {n_electrons}")
    print(f"exact_energy: {exact_energy:.6f} Ha")
    print(f"chemical_accuracy_target: {CHEMICAL_ACCURACY_HA} Ha (1.6 mHa)")
    print(f"time_budget: {TIME_BUDGET_SECONDS}s")
    if noise_strength > 0:
        dev = build_device(n_qubits, noise_strength=noise_strength)
        print(f"noise_strength: {noise_strength}")
        print(f"noise_type: {NOISE_TYPE}")
        print(f"device: default.mixed")
    else:
        print(f"device: default.qubit")
    print("Ready for experiments.")
