"""Optional ASE integration of the variable-cell matched correction.

This module requires the ``ase`` extra.  The array core in
:mod:`longrangefy.variable_cell` never imports it.

Two adapters are provided:

``VariableCellCorrectedCalculator``
    Composes an unchanged ASE calculator with the frozen variable-cell
    correction and advertises corrected ``energy``, ``forces`` and ``stress``.
    The cell may change between evaluations; atom count, species, periodicity
    and the fixed reference-site assignment may not.  Stress follows the ASE
    affine chain, i.e. the exact conjugate to ``set_cell(scale_atoms=True)`` /
    ``UnitCellFilter``.  If the original calculator does not advertise stress,
    the composition does not advertise it either; no zero is fabricated.

``AseCellResponseProvider``
    A cell-capable force/energy/stress evaluator for the **original** model,
    used with the optional base-response diagnostics in
    :mod:`longrangefy.variable_cell` (``measure_internal_strain_response``,
    ``measure_clamped_ion_elasticity``).  The fixed-cell
    ``DeepMDForceProvider`` deliberately rejects cell changes; this provider is
    the variable-cell evaluation seam.

The reproducible DeepMD example is ``examples/deepmd_variable.py``.
Optional base-response diagnostics characterize the original model; the
approved induced-dipole correction adds no elastic or internal-strain blocks.

Limitations: frozen second-order (Design F) blocks; no per-configuration
dipolar re-evaluation (Design R) or upstream geometry-dependent dipole model;
no strain-dependent IFCs / phonon-strain; no strain dependence of ``Z*`` or
``epsilon_inf``; no permanent polarization; the correction preserves the base
model's elastic/internal-strain response but does not add missing long-range
elastic/piezoelectric physics.
Supported cell changes are symmetric strain increments about the reference
orientation; rigid rotations of the cell relative to that orientation are not
part of the declared path and are rejected by the correction core.  The
reported stress is the symmetric work conjugate, with the skew part reported
as a diagnostic.  The adapter passes its per-site displacement bound into the
core evaluation, so a bounded state in which an atom lies within the bound of
two deformed reference sites, or an assigned site has more than one periodic
image inside the bound, is rejected instead of silently re-assigned.
No production supported strain range is asserted (approved decision D4:
derive it from real-workflow validation).  ``max_abs_strain`` is an optional
caller-declared validation bound; the finite-strain checks here are synthetic
construction tests, not a material-accuracy or NPT claim.  A real DeepMD run,
Ewald convergence study, and finite-strain validation must be performed before
advertising relaxation or NPT workflows.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic, minkowski_reduce

from .variable_cell import VariableCellCorrection, tensor_to_voigt_stress

__all__ = [
    "AseVariableError",
    "VariableCellCorrectedCalculator",
    "AseCellResponseProvider",
]


class AseVariableError(ValueError):
    """Unsupported variable-cell ASE composition state or evaluator output."""


def _check_cell(cell) -> np.ndarray:
    array = np.asarray(cell, dtype=float)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise AseVariableError("cell must be a finite (3,3) row-vector lattice")
    if abs(float(np.linalg.det(array))) < 1e-8:
        raise AseVariableError("cell is singular or near-singular")
    return array


class VariableCellCorrectedCalculator(Calculator):
    """Add a frozen variable-cell matched correction to an ASE calculator.

    ``reference`` must be the undisplaced reference target supercell in any
    atom order; the correspondence is resolved once at construction and then
    frozen.  ``max_displacement`` is an explicit per-site Cartesian bound in
    angstrom, checked on every evaluation.  ``max_abs_strain`` optionally adds
    an adapter-level validation bound; the correction also enforces any bound
    it was constructed with.  Only the correction is added: the original
    calculator receives no parameter changes.

    ASE constraints belong to the outer ``Atoms`` boundary; the original
    calculator is evaluated with ``apply_constraint=False`` so each restraint
    is applied exactly once by the consumer's ordinary getters.
    """

    implemented_properties = ["energy", "forces", "stress"]
    #: This adapter reports the ASE-affine conjugate only; the core exposes
    #: the independent-coordinate chain explicitly and they must not be mixed.
    stress_chain = "affine"

    def __init__(self, original, correction: VariableCellCorrection, reference,
                 *, max_displacement: float, max_abs_strain=None):
        super().__init__()
        if not np.all(reference.pbc):
            raise AseVariableError("the reference must have three-dimensional periodicity")
        if correction.species is None:
            raise AseVariableError("correction must record reference species")
        if not np.isfinite(max_displacement) or max_displacement <= 0.0:
            raise AseVariableError("max_displacement must be positive and finite")
        if max_abs_strain is not None and (
                not np.isfinite(max_abs_strain) or max_abs_strain <= 0.0):
            raise AseVariableError("max_abs_strain must be positive and finite or None")

        mapping = correction.map_reference_sites(
            reference.cell.array, reference.positions,
            reference.get_chemical_symbols())
        self._correction = correction
        self._original = original
        self.last_evaluation = None
        self._canonical_of_target = np.array(mapping.canonical_of_target, dtype=int)
        self._reference_target_positions = (
            mapping.positions_canonical[self._canonical_of_target])
        self._numbers = np.array(reference.numbers, dtype=int, copy=True)
        self._natom = len(reference)
        self._max_displacement = float(max_displacement)
        self._max_abs_strain = (None if max_abs_strain is None
                                else float(max_abs_strain))
        # Do not advertise a property the underlying calculator cannot supply.
        supported = ["energy", "forces"]
        if "stress" in getattr(original, "implemented_properties", ()):
            supported.append("stress")
        self.implemented_properties = supported
        self._check_reference_branch()

    def _check_reference_branch(self) -> None:
        reduced, _ = minkowski_reduce(self._correction.reference_cell, pbc=True)
        nearest = float(np.min(np.linalg.norm(reduced, axis=1)))
        reference = self._reference_target_positions
        for i in range(len(reference)):
            if i + 1 < len(reference):
                _, distances = find_mic(
                    reference[i + 1:] - reference[i],
                    self._correction.reference_cell, pbc=True)
                nearest = min(nearest, float(np.min(distances)))
        if 2 * self._max_displacement >= nearest:
            raise AseVariableError(
                "max_displacement must be less than half the nearest reference-site "
                "distance")

    def _check_state(self, atoms) -> None:
        if not np.all(atoms.pbc):
            raise AseVariableError(
                "changed periodicity is outside the three-dimensional variable-cell contract")
        if atoms.numbers.shape != self._numbers.shape or not np.array_equal(
                atoms.numbers, self._numbers):
            raise AseVariableError("changed species or atom count is unsupported")

    def _check_bounds(self, evaluation) -> None:
        diagnostics = evaluation.diagnostics
        displacement = float(diagnostics["max_displacement_angstrom"])
        if displacement > self._max_displacement:
            raise AseVariableError(
                f"displacement {displacement:.6g} angstrom exceeds the adapter bound "
                f"{self._max_displacement:.6g} angstrom")
        if self._max_abs_strain is not None:
            strain = float(diagnostics["max_abs_strain_voigt"])
            if strain > self._max_abs_strain:
                raise AseVariableError(
                    f"engineering Voigt strain {strain:.6g} exceeds the adapter bound "
                    f"{self._max_abs_strain:.6g}")

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        self._check_state(atoms)

        cell = atoms.cell.array
        positions_canonical = np.empty((self._natom, 3))
        positions_canonical[self._canonical_of_target] = atoms.positions
        evaluation = self._correction.evaluate(
            cell, positions_canonical, max_displacement=self._max_displacement)
        self.last_evaluation = evaluation
        self._check_bounds(evaluation)

        correction_forces = evaluation.forces.reshape(-1, 3)[self._canonical_of_target]

        original_atoms = atoms.copy()
        original_atoms.calc = self._original
        original_energy = original_atoms.get_potential_energy(apply_constraint=False)
        original_forces = original_atoms.get_forces(apply_constraint=False)
        results = {
            "energy": float(original_energy + evaluation.energy),
            "forces": original_forces + correction_forces,
        }
        if "stress" in self.implemented_properties and "stress" in properties:
            original_stress = original_atoms.get_stress(
                voigt=False, apply_constraint=False)
            results["stress"] = (
                tensor_to_voigt_stress(original_stress) + evaluation.stress_voigt)
        self.results = results


class AseCellResponseProvider:
    """Cell-capable original-model force/energy/stress evaluation.

    Unlike the fixed-cell ``DeepMDForceProvider``, this provider accepts any
    finite non-singular cell and evaluates the attached ASE calculator at the
    requested Cartesian positions.  It is intended for the optional
    base-response diagnostics (``measure_*``); it does not add a correction.
    Atom count, species and periodicity are frozen at construction.
    """

    def __init__(self, atoms, calculator=None):
        if not np.all(atoms.pbc):
            raise AseVariableError("the provider requires three-dimensional periodicity")
        self._atoms = atoms
        if calculator is not None:
            self._atoms.calc = calculator
        if self._atoms.calc is None:
            raise AseVariableError("an ASE calculator is required")
        self._natom = len(self._atoms)
        self._numbers = np.array(self._atoms.numbers, dtype=int, copy=True)
        self._pbc = np.array(self._atoms.pbc, dtype=bool, copy=True)

    def _set_state(self, cell, positions, fresh: bool) -> None:
        cell = _check_cell(cell)
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self._natom, 3):
            raise AseVariableError(
                f"positions must have shape ({self._natom},3), got {positions.shape}")
        if not np.isfinite(positions).all():
            raise AseVariableError("positions contain non-finite values")
        if not np.array_equal(self._atoms.pbc, self._pbc):
            raise AseVariableError("periodicity changed after provider construction")
        if not np.array_equal(self._atoms.numbers, self._numbers):
            raise AseVariableError("species changed after provider construction")
        self._atoms.set_cell(cell, scale_atoms=False, apply_constraint=False)
        self._atoms.set_positions(positions, apply_constraint=False)
        if fresh:
            self._atoms.calc.reset()

    def _forces(self) -> np.ndarray:
        forces = np.asarray(self._atoms.get_forces(apply_constraint=False), dtype=float)
        if forces.shape != (self._natom, 3) or not np.isfinite(forces).all():
            raise AseVariableError("calculator returned invalid forces")
        return forces.copy()

    def get_forces(self, cell, positions) -> np.ndarray:
        self._set_state(cell, positions, fresh=False)
        return self._forces()

    def get_fresh_forces(self, cell, positions) -> np.ndarray:
        self._set_state(cell, positions, fresh=True)
        return self._forces()

    def get_stress(self, cell, positions, voigt: bool = False) -> np.ndarray:
        self._set_state(cell, positions, fresh=False)
        stress = np.asarray(
            self._atoms.get_stress(voigt=voigt, apply_constraint=False), dtype=float)
        expected = (6,) if voigt else (3, 3)
        if stress.shape != expected or not np.isfinite(stress).all():
            raise AseVariableError("calculator returned invalid stress")
        return stress.copy()

    def get_energy(self, cell, positions) -> float:
        self._set_state(cell, positions, fresh=False)
        return float(self._atoms.get_potential_energy(apply_constraint=False))
