"""Optional ASE integration (R12): structure bridging and calculator caching.

Importing this module requires the ``ase`` extra.  The array core never
imports it.  Ordering is exactly :class:`~longrangefy.reference.SupercellMapping`
cell-major order, so finite-difference providers and correction matrices share
one convention.

Scope note: the fixed-cell stage advertises correction energy and forces only.
This adapter therefore exposes no stress pathway; adding a calculator that
advertises stress (e.g. a mixed composition) is the correction stage's
business and must not fabricate a zero stress.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .reference import ReferenceStructure, SupercellMapping

__all__ = [
    "AseIntegrationError",
    "reference_to_atoms",
    "supercell_to_atoms",
    "AseForceProvider",
]


class AseIntegrationError(RuntimeError):
    """Missing optional dependency or unsupported ASE-side use."""


def _require_ase():
    try:
        from ase import Atoms
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise AseIntegrationError(
            "ASE integration needs the 'ase' extra; the array core stays usable "
            "without ASE") from exc
    return Atoms


def supercell_to_atoms(matching: SupercellMapping):
    """ASE ``Atoms`` for the matching supercell in the documented cell-major order."""
    Atoms = _require_ase()
    return Atoms(
        symbols=list(matching.species_supercell()),
        scaled_positions=matching.frac_positions_supercell(),
        cell=matching.supercell_cell,
        pbc=True,
    )


def reference_to_atoms(reference: ReferenceStructure, reps: Sequence[int] = (1, 1, 1)):
    """ASE ``Atoms`` for a diagonal repetition of the reference (cell-major order)."""
    mapping = SupercellMapping(reference, tuple(reps))
    return supercell_to_atoms(mapping)


class AseForceProvider:
    """Forces from an ASE calculator on a fixed cell, species, and site order.

    Only positions may change between calls; ASE's calculator cache is
    invalidated by a position change, so repeated identical positions reuse the
    cached evaluation and stale results cannot leak (R12). Cell, PBC and species
    are checked on each call because the caller retains access to the atoms.

    Constraints on the caller's ``Atoms`` are retained for its own getters but
    bypassed here: positions are set and the raw calculator energy/forces are
    fetched with ``apply_constraint=False``.  A finite-difference sample must
    therefore measure the original unconstrained potential at exactly the
    requested displacements, not a projected or restraint-shifted surface.
    """

    def __init__(self, atoms, calculator=None, identity: Mapping[str, Any] | None = None):
        _require_ase()
        self._atoms = atoms
        if calculator is not None:
            self._atoms.calc = calculator
        self._natom = len(self._atoms)
        self._cell = self._atoms.cell.array.copy()
        self._pbc = self._atoms.pbc.copy()
        self._numbers = self._atoms.numbers.copy()
        self._identity = dict(identity or {})

    @property
    def atoms(self):
        return self._atoms

    @property
    def identity(self) -> Mapping[str, Any]:
        """Best-effort calculator identity record; never invented when absent."""
        if self._identity:
            return dict(self._identity)
        calculator = self._atoms.calc
        if calculator is None:
            return {}
        record: dict = {"calculator_class": type(calculator).__name__}
        todict = getattr(calculator, "todict", None)
        if callable(todict):
            try:
                record["calculator_parameters"] = todict()
            except Exception:  # pragma: no cover - calculator-specific
                pass
        return record

    def _set_positions(self, positions: np.ndarray) -> None:
        if (not np.array_equal(self._atoms.cell.array, self._cell)
                or not np.array_equal(self._atoms.pbc, self._pbc)
                or not np.array_equal(self._atoms.numbers, self._numbers)):
            raise AseIntegrationError("cell, periodicity or species changed")
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self._natom, 3):
            raise AseIntegrationError(
                f"positions must have shape ({self._natom}, 3), got {positions.shape}")
        if not np.isfinite(positions).all():
            raise AseIntegrationError("positions contain non-finite values")
        # exact positions; bypassing constraints keeps finite differences on the original surface
        self._atoms.set_positions(positions, apply_constraint=False)

    def _forces(self, positions: np.ndarray, fresh: bool) -> np.ndarray:
        self._set_positions(positions)
        if fresh:
            if self._atoms.calc is None:
                raise AseIntegrationError("no ASE calculator attached")
            self._atoms.calc.reset()
        forces = np.asarray(self._atoms.get_forces(apply_constraint=False), dtype=float)
        if not np.isfinite(forces).all():
            raise AseIntegrationError("ASE calculator returned non-finite forces")
        return forces

    def get_forces(self, positions: np.ndarray) -> np.ndarray:
        return self._forces(positions, fresh=False)

    def get_fresh_forces(self, positions: np.ndarray) -> np.ndarray:
        """Same positions, but from an uncached calculator evaluation.

        Repeated identical positions reuse the cached result through
        ``get_forces``; noise studies must use this method to observe the
        calculator's actual run-to-run variability.
        """
        return self._forces(positions, fresh=True)

    def get_energy(self, positions: np.ndarray) -> float:
        self._set_positions(positions)
        return float(self._atoms.get_potential_energy(apply_constraint=False))
