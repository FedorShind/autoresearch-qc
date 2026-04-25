"""
prepare.py — Frozen evaluation infrastructure for autoresearch-qc.

Builds molecular Hamiltonians, computes exact ground-state energies,
and provides evaluation utilities. The agent NEVER modifies this file.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import yaml
from pennylane.boolean_fn import BooleanFn
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
# MOLECULE DEFINITIONS — loaded from molecules/*.yaml
# ============================================================
def load_molecules() -> dict[str, dict[str, Any]]:
    """Load all molecule definitions from molecules/*.yaml.

    Returns a dict mapping filename stem (e.g., "lih") to molecule config.
    Translates the YAML field "multiplicity" to "mult" so build_hamiltonian
    can keep reading config["mult"] unchanged.
    """
    molecules_dir = Path(__file__).parent / "molecules"
    if not molecules_dir.exists():
        raise FileNotFoundError(f"Required directory not found: {molecules_dir}")

    out: dict[str, dict[str, Any]] = {}
    for yaml_file in sorted(molecules_dir.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse {yaml_file.name}: {e}") from e
        if "multiplicity" in config:
            config["mult"] = config.pop("multiplicity")
        out[yaml_file.stem] = config

    if not out:
        raise RuntimeError(f"No molecule YAML files found in {molecules_dir}")
    return out


MOLECULES: dict[str, dict[str, Any]] = load_molecules()


def molecule_choice(value: str) -> str:
    """argparse type validator with a friendly error for the removed
    "stretched" molecule keys (migrated to --bond-length 3.0)."""
    if value in ("h2_stretched", "lih_stretched"):
        base = value.removesuffix("_stretched")
        raise argparse.ArgumentTypeError(
            f"'--molecule {value}' was removed. "
            f"Use: --molecule {base} --bond-length 3.0"
        )
    if value not in MOLECULES:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a known molecule. "
            f"Available: {sorted(MOLECULES.keys())}"
        )
    return value


def _build_diatomic_coords(bond_length: float) -> np.ndarray:
    """Coordinates for a diatomic molecule along the z-axis (Å)."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, bond_length])


def _build_chain_coords(n_atoms: int, spacing: float) -> np.ndarray:
    """Coordinates for a linear chain of n_atoms along the z-axis (Å)."""
    coords = np.zeros(3 * n_atoms)
    for i in range(n_atoms):
        coords[3 * i + 2] = i * spacing
    return coords


def get_coordinates(
    config: dict[str, Any],
    bond_length: float | None = None,
) -> np.ndarray:
    """Resolve coordinates for a molecule, with optional bond_length override.

    For molecules with a "geometry" block (diatomic, chain), bond_length
    overrides the default. For molecules with explicit "coordinates",
    passing bond_length raises ValueError.

    Raises ValueError if bond_length is non-positive or outside the
    molecule's allowed bond_length_range.
    """
    geom = config.get("geometry")

    if bond_length is not None:
        if geom is None:
            raise ValueError(
                f"Molecule '{config['name']}' has fixed coordinates; "
                f"--bond-length is not supported."
            )
        if bond_length <= 0:
            raise ValueError(f"bond_length must be positive, got {bond_length}")
        bl_min, bl_max = geom["bond_length_range"]
        if not (bl_min <= bond_length <= bl_max):
            raise ValueError(
                f"bond_length {bond_length} outside allowed range "
                f"[{bl_min}, {bl_max}] for '{config['name']}'"
            )

    if geom is None:
        return np.array(config["coordinates"])

    if bond_length is None:
        bond_length = geom["default_bond_length"]

    if geom["type"] == "diatomic":
        return _build_diatomic_coords(bond_length)
    if geom["type"] == "chain":
        return _build_chain_coords(len(config["symbols"]), bond_length)
    raise ValueError(f"Unknown geometry type: {geom['type']}")


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

    # Match all gate operations (including adjoints from ZNE folding),
    # but not state preparation ops like BasisState/StatePrep.
    class _AnyGate(BooleanFn):
        def __init__(self) -> None:
            super().__init__(
                lambda op: not isinstance(op, (qml.BasisState, qml.StatePrep))
            )

    noise_model = qml_noise.NoiseModel({
        _AnyGate(): qml_noise.partial_wires(qml.DepolarizingChannel, noise_strength),
    })
    dev = qml.device("default.mixed", wires=n_qubits)
    return qml_noise.add_noise(dev, noise_model)


def get_zne_config(
    scale_factors: tuple[float, ...] = (1.0, 2.0, 3.0),
    extrapolation: str = "linear",
    polynomial_order: int = 2,
) -> dict[str, Any]:
    """Build ZNE configuration for qml.noise.mitigate_with_zne.

    Returns a dict that can be unpacked directly:
        mitigated_qnode = qml.noise.mitigate_with_zne(qnode, **get_zne_config())

    Args:
        scale_factors: Noise amplification levels for circuit folding.
        extrapolation: Extrapolation method — "linear", "polynomial",
            "richardson", or "exponential".
        polynomial_order: Polynomial degree (only used when
            extrapolation="polynomial").

    Returns:
        Dict with scale_factors, folding, and extrapolate keys.
    """
    extrapolate_fns = {
        "linear": qml_noise.poly_extrapolate,  # order=1 applied below
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
    if extrapolation == "linear":
        extrapolate = lambda x, y, _fn=extrapolate: _fn(x, y, order=1)
    elif extrapolation == "polynomial":
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
        coordinates=get_coordinates(config),
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
    parser.add_argument("--molecule", default=MOLECULE, type=molecule_choice,
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
