"""Explicit reference structures and diagonal matching supercells.

Package-wide conventions (fixed, see ``docs/method-specification.md``):

- Lattice vectors are the **rows** of ``cell`` (Angstrom); a fractional
  coordinate row vector maps to Cartesian by ``cart = frac @ cell``.
- Lengths Angstrom, energies eV per whole periodic cell, forces eV/Angstrom,
  unmassweighted IFCs eV/Angstrom^2.
- Supercell site order is **cell-major**: the outer loop walks supercell cell
  offsets ``(c1, c2, c3)`` and the inner loop the ``n_uc`` primitive sites,
  ``atom = ((c1*n2 + c2)*n3 + c3)*n_uc + site``.
- Wavevectors are given in fractional reciprocal coordinates, ``q = m/reps``
  with integer ``m``; phases use ``exp(+-2*pi*i*q.R)`` on integer cell indices.

The reference structure is an explicit declared input (R01/R02).  Nothing in
this module relaxes, wraps, or symmetrizes supplied positions silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np

__all__ = [
    "ReferenceStructureError",
    "SymmetryOperation",
    "ReferenceStructure",
    "SupercellMapping",
]


class ReferenceStructureError(ValueError):
    """Invalid reference structure, symmetry declaration, or supercell mapping."""


def _as_float_array(value: Any, name: str, shape: Tuple[int, ...]) -> np.ndarray:
    # Own a copy: the frozen dataclass must not alias (or mutate) caller arrays.
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape:
        raise ReferenceStructureError(
            f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ReferenceStructureError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SymmetryOperation:
    """One space-group operation in fractional coordinates.

    ``frac_new = frac @ rot.T + trans``  (row-vector convention, matching
    ``cart = frac @ cell``).  Forces (polar vectors) transform with the
    Cartesian matrix ``inv(cell) @ rot.T @ cell`` applied as ``F @ C``.
    """

    rot: np.ndarray
    trans: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rot",
                           _as_float_array(self.rot, "rot", (3, 3)))
        object.__setattr__(self, "trans",
                           _as_float_array(self.trans, "trans", (3,)))
        det = float(np.linalg.det(self.rot))
        if abs(det) < 1e-12:
            raise ReferenceStructureError("symmetry operation is singular")

    def cart_matrix(self, cell: np.ndarray) -> np.ndarray:
        """Cartesian polar-vector rotation ``C`` with ``F_cart' = F_cart @ C``."""
        cell = np.asarray(cell, dtype=float)
        return np.linalg.inv(cell) @ self.rot.T @ cell


@dataclass(frozen=True)
class ReferenceStructure:
    """Declared reference structure: lattice, sites, species, symmetry (R01).

    ``symmetry_label`` and ``symmetry_operations`` are the *declared* symmetry
    of the reference phase (which may be an unstable parent phase; R02).  No
    relaxation is performed here.  ``source`` records provenance of the
    structure itself.
    """

    cell: np.ndarray                    # (3, 3) lattice vectors as rows, Angstrom
    frac_positions: np.ndarray          # (n, 3) fractional site coordinates
    species: Tuple[str, ...]            # n species labels, same order as sites
    symmetry_label: str = ""
    symmetry_operations: Tuple[SymmetryOperation, ...] = ()
    source: str = ""
    #: Fixed full 3D periodicity; supplied values other than all-True are rejected.
    pbc: Tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        if tuple(self.pbc) != (True, True, True):
            raise ReferenceStructureError(
                "only fully 3D-periodic references are supported "
                f"(slabs/wires/molecules are out of scope), got pbc={self.pbc}")
        object.__setattr__(self, "pbc", (True, True, True))
        cell = _as_float_array(self.cell, "cell", (3, 3))
        frac = np.array(self.frac_positions, dtype=float, copy=True)
        if frac.ndim != 2 or frac.shape[1] != 3 or frac.shape[0] == 0:
            raise ReferenceStructureError(
                f"frac_positions must have shape (n, 3) with n > 0, got {frac.shape}")
        if not np.isfinite(frac).all():
            raise ReferenceStructureError("frac_positions contains non-finite values")
        det = float(np.linalg.det(cell))
        if abs(det) < 1e-8:
            raise ReferenceStructureError(
                f"cell is singular or near-singular (|det| = {abs(det):.3e} Angstrom^3)")
        species = tuple(str(s) for s in self.species)
        if len(species) != frac.shape[0]:
            raise ReferenceStructureError(
                f"{len(species)} species labels for {frac.shape[0]} sites")
        frac.setflags(write=False)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "frac_positions", frac)
        object.__setattr__(self, "species", species)

    @property
    def natom(self) -> int:
        return int(self.frac_positions.shape[0])

    @property
    def cart_positions(self) -> np.ndarray:
        """(n, 3) Cartesian site positions in Angstrom."""
        return self.frac_positions @ self.cell

    def to_dict(self) -> dict:
        return {
            "cell": self.cell.tolist(),
            "frac_positions": self.frac_positions.tolist(),
            "species": list(self.species),
            "symmetry_label": self.symmetry_label,
            "symmetry_operations": [
                {"rot": op.rot.tolist(), "trans": op.trans.tolist()}
                for op in self.symmetry_operations
            ],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceStructure":
        ops = tuple(
            SymmetryOperation(op["rot"], op["trans"]) for op in
            data.get("symmetry_operations", [])
        )
        return cls(
            cell=np.asarray(data["cell"], dtype=float),
            frac_positions=np.asarray(data["frac_positions"], dtype=float),
            species=tuple(data["species"]),
            symmetry_label=str(data.get("symmetry_label", "")),
            symmetry_operations=ops,
            source=str(data.get("source", "")),
        )


@dataclass(frozen=True)
class SupercellMapping:
    """Diagonal integer repetition of one primitive reference (R09).

    The matching/target supercell is ``diag(reps)`` repetitions of the
    reference lattice (skew primitive cells allowed).  Exposes the cell-major
    atom ordering, the commensurate unshifted q-grid, and index arithmetic
    shared by the finite-difference and IFC machinery.
    """

    reference: ReferenceStructure
    reps: Tuple[int, int, int]

    def __post_init__(self) -> None:
        # Validate before any coercion: floats, bools, and other non-integers
        # are rejected outright (diagonal repetitions are integers by contract).
        for value in self.reps:
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))):
                raise ReferenceStructureError(
                    f"reps must be three positive integers, got {self.reps!r}")
        reps = np.asarray(self.reps, dtype=int)
        if reps.shape != (3,) or np.any(reps <= 0):
            raise ReferenceStructureError(
                f"reps must be three positive integers, got {self.reps!r}")
        object.__setattr__(self, "reps", tuple(int(v) for v in reps))

    @property
    def n_uc(self) -> int:
        return self.reference.natom

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.reps))

    @property
    def natom_supercell(self) -> int:
        return self.n_uc * self.n_cells

    @property
    def supercell_cell(self) -> np.ndarray:
        """(3, 3) supercell lattice vectors as rows, Angstrom."""
        return np.diag(np.asarray(self.reps, dtype=float)) @ self.reference.cell

    def atom_index(self, cell_index, site: int) -> int:
        """Cell-major flat atom index for one (cell offset, site) pair."""
        c1, c2, c3 = (int(v) for v in cell_index)
        n1, n2, n3 = self.reps
        if not (0 <= c1 < n1 and 0 <= c2 < n2 and 0 <= c3 < n3):
            raise ReferenceStructureError(
                f"cell offset {(c1, c2, c3)} outside supercell reps {self.reps}")
        if not (0 <= site < self.n_uc):
            raise ReferenceStructureError(f"site {site} outside 0..{self.n_uc - 1}")
        return ((c1 * n2 + c2) * n3 + c3) * self.n_uc + site

    def cell_and_site(self, atom: int) -> Tuple[Tuple[int, int, int], int]:
        """Inverse of :meth:`atom_index`."""
        atom = int(atom)
        if not (0 <= atom < self.natom_supercell):
            raise ReferenceStructureError(f"atom index {atom} out of range")
        site = atom % self.n_uc
        cell = atom // self.n_uc
        n3 = self.reps[2]
        n2 = self.reps[1]
        c3 = cell % n3
        c2 = (cell // n3) % n2
        c1 = cell // (n2 * n3)
        return ((c1, c2, c3), site)

    def frac_positions_supercell(self) -> np.ndarray:
        """(N_m, 3) supercell-basis fractional positions, cell-major.

        ``(offset + reference_frac) / reps``, i.e. expressed in the supercell
        lattice basis (the form ASE/`Atoms` expects with ``supercell_cell``).
        """
        n1, n2, n3 = self.reps
        offsets = np.array(
            [(c1, c2, c3) for c1 in range(n1) for c2 in range(n2) for c3 in range(n3)],
            dtype=float)
        absolute = (offsets[:, None, :] + self.reference.frac_positions[None, :, :])
        return absolute.reshape(-1, 3) / np.asarray(self.reps, dtype=float)[None, :]

    def cart_positions_supercell(self) -> np.ndarray:
        """(N_m, 3) Cartesian positions (Angstrom), cell-major.

        ``(offset + reference_frac) @ primitive_cell``: offsets count *primitive*
        lattice vectors, so the primitive cell (not the supercell) maps them.
        """
        n1, n2, n3 = self.reps
        offsets = np.array(
            [(c1, c2, c3) for c1 in range(n1) for c2 in range(n2) for c3 in range(n3)],
            dtype=float)
        absolute = (offsets[:, None, :] + self.reference.frac_positions[None, :, :])
        return absolute.reshape(-1, 3) @ self.reference.cell

    def species_supercell(self) -> Tuple[str, ...]:
        return tuple(self.reference.species) * self.n_cells

    def q_grid(self) -> np.ndarray:
        """(nq, 3) fractional reciprocal coordinates of the unshifted grid.

        ``q = m / reps`` for ``m`` in ``[0, n)`` per direction: the full,
        unshifted set of wavevectors commensurate with the supercell (the
        matching grid of the method specification).
        """
        n1, n2, n3 = self.reps
        grid = [(m1, m2, m3)
                for m1 in range(n1) for m2 in range(n2) for m3 in range(n3)]
        return np.asarray(grid, dtype=float) / np.asarray(self.reps, dtype=float)[None, :]
