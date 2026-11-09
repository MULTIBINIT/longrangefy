"""Fixed-cell ASE composition of an original calculator and matched correction."""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic, minkowski_reduce


class CorrectedCalculator(Calculator):
    """Add a matched harmonic correction once to an unchanged ASE calculator.

    ``reference`` must be the undisplaced target supercell in its intended atom
    order. ``max_displacement`` is an explicit per-site bound in angstrom; it
    must be strictly less than half the nearest reference-site separation. A
    fixed reference assignment is retained for the calculator's lifetime.
    Only whole-cell energy and forces are available, not corrected stress.

    ASE constraints belong to the outer ``Atoms`` boundary: the original
    calculator is evaluated with ``apply_constraint=False`` so each restraint
    is applied exactly once by the consumer's ordinary getters.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, original, correction, reference, *, max_displacement):
        super().__init__()
        if not np.all(reference.pbc):
            raise ValueError('the reference must have three-dimensional periodicity')
        if correction.metadata.get('species') is None:
            raise ValueError('correction must record reference species')
        if not np.isfinite(max_displacement) or max_displacement <= 0:
            raise ValueError('max_displacement must be positive and finite')
        mapping = correction.map_sites(reference.cell.array, reference.positions,
                                       reference.get_chemical_symbols())
        self._reference = mapping.positions_canonical[mapping.canonical_of_target]
        self._cell = reference.cell.array.copy()
        self._numbers = reference.numbers.copy()
        self._mapping = mapping
        self._correction = correction
        self._original = original
        self._max_displacement = float(max_displacement)
        reduced, _ = minkowski_reduce(self._cell, pbc=True)
        nearest = float(np.min(np.linalg.norm(reduced, axis=1)))
        for i in range(len(self._reference)):
            if i + 1 < len(self._reference):
                _, distances = find_mic(self._reference[i + 1:] - self._reference[i], self._cell, pbc=True)
                nearest = min(nearest, float(np.min(distances)))
        if 2 * self._max_displacement >= nearest:
            raise ValueError('max_displacement must be less than half the nearest reference-site distance')

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        if (not np.all(atoms.pbc) or not np.array_equal(atoms.numbers, self._numbers)
                or not np.allclose(atoms.cell.array, self._cell, rtol=0, atol=1e-10)):
            raise ValueError('changed cell, periodicity or species is outside the fixed-cell contract')
        displacement, lengths = find_mic(atoms.positions - self._reference, self._cell, pbc=True)
        if not np.isfinite(displacement).all() or np.any(lengths > self._max_displacement):
            raise ValueError('displacement exceeds the fixed reference-site mapping bound')
        energy, forces = self._correction.evaluate(displacement.ravel(), self._mapping)
        original_atoms = atoms.copy()
        original_atoms.calc = self._original
        original_energy = original_atoms.get_potential_energy(apply_constraint=False)
        original_forces = original_atoms.get_forces(apply_constraint=False)
        self.results = {'energy': float(original_energy + energy),
                        'forces': original_forces + forces.reshape(-1, 3)}
