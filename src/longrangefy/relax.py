"""Explicit reference preparation: no automatic relaxation (R02).

Modes:

- ``"none"`` (default): never move the reference; report its residual forces.
- ``"symmetry_preserving"``: relax internal displacements at the **fixed**
  lattice while preserving the declared symmetry.  Forces are projected onto
  the symmetry-invariant subspace using the reference's declared operations
  (falling back to a recorded spglib detection only when no operations were
  declared and spglib is importable).  Symmetry-related sites then move
  together by construction.
- ``"unrestricted"``: free internal displacements, requiring explicit
  ``allow_symmetry_breaking=True`` because it may change the reference phase.
  The lattice also stays fixed: cell strain belongs to the variable-cell
  stage and is rejected here.

Residual forces at the final positions are always reported in the result.
Relaxation moves only within the reference's own displacement branch; it does
not change species, site count, or the lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from .reference import (
    ReferenceStructure,
    SymmetryOperation,
)

__all__ = [
    "ReferencePreparationError",
    "ReferencePreparation",
    "prepare_reference",
]

MODES = ("none", "symmetry_preserving", "unrestricted")


class ReferencePreparationError(ValueError):
    """Requested preparation is inconsistent with the declared reference."""


@dataclass(frozen=True)
class ReferencePreparation:
    """Outcome of reference preparation, including residual forces (R02)."""

    mode: str
    converged: bool
    n_steps: int
    max_residual_force_ev_per_angstrom: float
    rms_residual_force_ev_per_angstrom: float
    residual_force_history_ev_per_angstrom: Tuple[float, ...]
    symmetry_preserved: bool
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "converged": bool(self.converged),
            "n_steps": int(self.n_steps),
            "max_residual_force_ev_per_angstrom":
                float(self.max_residual_force_ev_per_angstrom),
            "rms_residual_force_ev_per_angstrom":
                float(self.rms_residual_force_ev_per_angstrom),
            "residual_force_history_ev_per_angstrom":
                [float(v) for v in self.residual_force_history_ev_per_angstrom],
            "symmetry_preserved": bool(self.symmetry_preserved),
            "warnings": list(self.warnings),
        }


def _minimum_image_cart(delta_frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    wrapped = delta_frac - np.round(delta_frac)
    return wrapped @ cell


def _site_permutation(
    operations: Tuple[SymmetryOperation, ...],
    frac_positions: np.ndarray,
    cell: np.ndarray,
    tolerance: float = 1.0e-3,
) -> np.ndarray:
    """(n_ops, n) permutation ``sigma[g, i]`` with ``x'_i(g) = x_{sigma[g, i]}``."""
    natom = frac_positions.shape[0]
    sigma = np.empty((len(operations), natom), dtype=int)
    for g, op in enumerate(operations):
        mapped = frac_positions @ op.rot.T + op.trans
        delta = _minimum_image_cart(mapped[:, None, :] - frac_positions[None, :, :], cell)
        distances = np.linalg.norm(delta, axis=2)             # (n, n)
        assigned = distances.argmin(axis=1)
        residuals = distances[np.arange(natom), assigned]
        if residuals.max() > tolerance:
            raise ReferencePreparationError(
                f"declared symmetry operation {g} does not map the reference sites "
                f"onto themselves (max residual {residuals.max():.3e} Angstrom > "
                f"{tolerance:.1e}); the declared symmetry is inconsistent with the sites")
        if len(set(assigned.tolist())) != natom:
            raise ReferencePreparationError(
                f"declared symmetry operation {g} does not permute the sites "
                "one-to-one (site coincidence or multiplicity mismatch)")
        sigma[g] = assigned
    return sigma


def _detect_operations(reference: ReferenceStructure, symprec: float) -> Tuple[SymmetryOperation, ...]:
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ReferencePreparationError(
            "symmetry_preserving preparation needs declared symmetry_operations "
            "on the reference, or the optional spglib package for detection; "
            "supply the operations explicitly to keep detection out of the loop"
        ) from exc
    unique_species = sorted(set(reference.species))
    numbers = [unique_species.index(s) + 1 for s in reference.species]
    lattice = reference.cell
    dataset = spglib.get_symmetry_dataset(
        (lattice, reference.frac_positions, numbers), symprec=symprec)
    if dataset is None:
        raise ReferencePreparationError(
            f"spglib could not determine symmetry within symprec={symprec}")
    operations = []
    for rot, trans in zip(dataset.rotations, dataset.translations):
        operations.append(SymmetryOperation(np.asarray(rot, dtype=float),
                                            np.asarray(trans, dtype=float)))
    return tuple(operations)


def _symmetrize_forces(
    forces: np.ndarray,
    operations: Tuple[SymmetryOperation, ...],
    sigma: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Group-average forces onto the symmetry-invariant subspace.

    ``F_sym[i] = (1/|G|) sum_g F[sigma_g(i)] @ C_g^{-1}`` where ``C_g`` is the
    Cartesian polar-vector rotation of operation ``g``.  For a
    symmetry-consistent configuration this returns the forces unchanged; away
    from it, the result is the equivariant projection whose gradient flow
    keeps symmetry-related sites moving together.
    """
    projected = np.zeros_like(forces)
    for g, op in enumerate(operations):
        cart_rot = op.cart_matrix(cell)                       # F' = F @ C
        projected += np.linalg.solve(cart_rot.T, forces[sigma[g]].T).T
    return projected / len(operations)


def prepare_reference(
    reference: ReferenceStructure,
    get_forces: Callable[[np.ndarray], np.ndarray],
    mode: str = "none",
    *,
    fmax: float = 1.0e-3,
    max_steps: int = 200,
    step_size: float = 1.0e-3,
    allow_symmetry_breaking: bool = False,
    symprec: float = 1.0e-5,
    symmetry_tolerance: float = 1.0e-3,
) -> Tuple[ReferenceStructure, ReferencePreparation]:
    """Prepare (or explicitly not prepare) the reference structure (R02).

    ``get_forces`` maps primitive-cell Cartesian positions ``(n, 3)`` in
    Angstrom to forces ``(n, 3)`` in eV/Angstrom.  The lattice is fixed in
    every mode; unrestricted mode demands ``allow_symmetry_breaking=True``.
    Returns the (possibly relaxed) reference and the preparation record with
    residual forces; ``reference.source`` is not rewritten.
    """
    if mode not in MODES:
        raise ReferencePreparationError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "unrestricted" and not allow_symmetry_breaking:
        raise ReferencePreparationError(
            "unrestricted preparation requires allow_symmetry_breaking=True: "
            "it can change the reference phase, which must be an explicit choice")
    if mode == "none":
        forces = _check_forces(get_forces, reference.frac_positions @ reference.cell,
                               reference.natom)
        return reference, ReferencePreparation(
            mode=mode,
            converged=bool(np.abs(forces).max() <= fmax),
            n_steps=0,
            max_residual_force_ev_per_angstrom=float(np.abs(forces).max()),
            rms_residual_force_ev_per_angstrom=float(np.sqrt((forces ** 2).mean())),
            residual_force_history_ev_per_angstrom=(float(np.abs(forces).max()),),
            symmetry_preserved=False,
        )

    warnings = []
    if mode == "symmetry_preserving":
        operations = reference.symmetry_operations
        if not operations:
            operations = _detect_operations(reference, symprec)
            warnings.append(
                f"symmetry operations detected with spglib symprec={symprec} "
                "because none were declared; the detected set is part of the record")
        sigma = _site_permutation(operations, reference.frac_positions,
                                  reference.cell, symmetry_tolerance)
    else:
        operations = ()
        sigma = np.zeros((0, reference.natom), dtype=int)
        warnings.append(
            "unrestricted preparation: forces were not symmetry-projected; "
            "the relaxed structure may break the declared reference symmetry")

    cell = reference.cell
    frac = reference.frac_positions.copy()
    step = float(step_size)
    history = []
    forces = _check_forces(get_forces, frac @ cell, reference.natom)
    projected = (_symmetrize_forces(forces, operations, sigma, cell)
                 if operations else forces)
    residual = float(np.abs(projected).max())
    history.append(residual)
    converged = residual <= fmax
    n_steps = 0
    while not converged and n_steps < max_steps:
        for _ in range(8):   # backtracking on the projected residual
            trial_frac = frac + (step * projected) @ np.linalg.inv(cell)
            trial_forces = _check_forces(get_forces, trial_frac @ cell, reference.natom)
            trial_projected = (_symmetrize_forces(trial_forces, operations, sigma, cell)
                               if operations else trial_forces)
            trial_residual = float(np.abs(trial_projected).max())
            if trial_residual < residual:
                frac, forces, projected, residual = (
                    trial_frac, trial_forces, trial_projected, trial_residual)
                step *= 1.2
                break
            step *= 0.5
        else:
            warnings.append(
                "step size underflow during reference preparation: no further "
                "decrease of the projected residual forces")
            break
        n_steps += 1
        history.append(residual)
        converged = residual <= fmax

    relaxed = ReferenceStructure(
        cell=reference.cell,
        frac_positions=frac,
        species=reference.species,
        symmetry_label=reference.symmetry_label,
        symmetry_operations=reference.symmetry_operations,
        source=reference.source,
        pbc=reference.pbc,
    )
    _, final_residual_max, final_residual_rms = _residual(relaxed, get_forces)
    return relaxed, ReferencePreparation(
        mode=mode,
        converged=converged,
        n_steps=n_steps,
        max_residual_force_ev_per_angstrom=final_residual_max,
        rms_residual_force_ev_per_angstrom=final_residual_rms,
        residual_force_history_ev_per_angstrom=tuple(history),
        symmetry_preserved=(mode == "symmetry_preserving"),
        warnings=tuple(warnings),
    )


def _check_forces(get_forces, positions: np.ndarray, natom: int) -> np.ndarray:
    forces = np.asarray(get_forces(np.asarray(positions, dtype=float)), dtype=float)
    if forces.shape != (natom, 3):
        raise ReferencePreparationError(
            f"get_forces must return shape ({natom}, 3), got {forces.shape}")
    if not np.isfinite(forces).all():
        raise ReferencePreparationError("get_forces returned non-finite values")
    return forces


def _residual(reference: ReferenceStructure,
              get_forces: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, float, float]:
    forces = _check_forces(get_forces, reference.frac_positions @ reference.cell,
                           reference.natom)
    return forces, float(np.abs(forces).max()), float(np.sqrt((forces ** 2).mean()))
