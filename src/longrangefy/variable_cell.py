"""Variable-cell matched harmonic correction (ticket 09, approved Design F).

The fixed-cell construction supplies the displacement-channel block
``DeltaPhi_L`` on one target cell.  This module freezes that block at the
reference geometry and evaluates the joint second-order correction

    E_corr = 1/2 u^T DeltaPhi_L u + 1/2 u^T DeltaLambda_L eps_v
             + 1/2 V0 eps_v^T DeltaC eps_v,
    F_corr = -DeltaPhi_L u - 1/2 DeltaLambda_L eps_v,
    sigma  = sym(P^T M) / (det M V0),

with the reference target cell ``A0`` (rows, angstrom), current cell ``A``,
deformation ``M = inv(A0) @ A``, symmetric engineering strain
``eps_v = Voigt(sym(M - I))`` in ASE order ``[xx, yy, zz, yz, xz, xy]``,
reference target volume ``V0 = |det A0|``, and ``u`` measured to the
affinely moved reference sites with a provably complete nearest-image
search in the true Cartesian metric of the current cell (skew-safe; fractional
rounding is not used).

Two stress conjugates are available:

* ``"affine"`` (default, ASE calculator contract): the cell and every atom
  move affinely at fixed fractional coordinates.  This is the chain used by
  ``ase.Atoms.set_cell(scale_atoms=True)``, ``UnitCellFilter`` and
  ``FrechetCellFilter``.
* ``"cartesian"``: the cell changes at fixed Cartesian atomic positions,
  the chain of integrators that propagate Cartesian coordinates and the cell
  as independent variables (the pymultibinit-style chain).

The two differ exactly by the force-virial term; mixing them is an error.
Both are exact finite-strain transforms (no ``(1+eps)/V`` scaling).

Approved physical scope (design note `specs/lite/design-variable-cell-matching.md`,
Design F, decision D6):

* The specified induced-dipole branch has no linear term in ``u`` at zero
  displacement, so its Gamma-channel blocks vanish identically:
  ``DeltaLambda_L = 0`` and ``DeltaC = 0`` for the declared model class.
  The correction therefore **preserves the original model's own** clamped-ion
  elastic and internal-strain response and adds no long-range
  elastic/piezoelectric physics.  The latter is underdetermined by the
  supplied response (``Z*``, ``epsilon_inf``, displacement response) and is
  deliberately not claimed.
* ``DeltaLambda_L`` and ``DeltaC`` are nevertheless accepted as explicit
  arrays for model-class extension; they are not acquired implicitly.
* Correction blocks are frozen at the reference geometry (Design F).  The
  correction is exact to second order in the joint amplitudes ``(u, eps)``.
  Strain-dependent IFCs / phonon-strain (``dPhi/deps``) are a higher-order
  coupling and are not copied or required.  Relaxed-ion elasticity is not
  derived, substituted, or returned.  Design R (per-configuration
  re-evaluation of the dipolar branch) and upstream's geometry-dependent
  dipole model are not implemented.
* No production supported strain range is asserted (approved decision D4:
  derive it from real-workflow validation).  ``max_abs_strain`` is an optional
  caller-declared construction/validation bound, not a material-accuracy
  claim; the accompanying finite-strain tests exercise a synthetic
  construction range only.  Supported cell changes are symmetric strain increments about the
  reference orientation.  Rigid rotations of the cell relative to the
  reference are not part of the declared deformation path (no supported
  relaxer applies them) and are rejected by the polar rotation angle of the
  deformation gradient; the reported symmetric stress is the work conjugate
  to symmetric strain increments, with the skew part exposed as a diagnostic.
  For a correction constructed with a displacement bound, the fixed-site
  assignment must also remain unique at the deformed cell: an atom within the
  bound of two reference sites is rejected rather than silently exchanged, and
  an assigned site with more than one periodic image inside the bound is
  rejected rather than allowed to jump branches.

Matching preservation: on the matching supercell as target the correction
matrix is zero (to the fixed-cell construction residual) and the default
Gamma blocks are zero, so ``E_corr = F_corr = sigma_corr = 0`` for every
supported displacement and strain.  For a larger target, shared wavevectors
and tiled matching-periodic patterns retain the fixed-cell guarantee, bounded
by the dipolar evaluation residual.

Units: angstrom; eV per whole target cell; eV/angstrom; eV/angstrom**3
(Cauchy stress, full 3x3 in the core, ASE 6-vector at the adapter);
unmassweighted IFCs eV/angstrom**2; ``DeltaLambda_L`` eV/angstrom per unit
engineering strain; ``DeltaC`` eV/angstrom**3 referred to the unstrained
target-cell volume.

Provenance and reuse: the joint-harmonic energy, force and stress structure
and the strain/Voigt conventions are adapted from the user-authored
``pymultibinit/src/pymultibinit/pyeffpot/potential.py`` routines
``_evaluate_elastic``, ``_compute_strain``, ``_compute_du_delta``,
``_finalize_stress``, ``_strain_to_voigt`` and ``_voigt_to_tensor``.
``measure_internal_strain_response`` and
``measure_clamped_ion_elasticity`` adapt the user-authored
``atomchain/atomchain/ddb/finite_difference.py`` routines ``strain_atoms``,
``calculate_internal_strain_response`` and ``calculate_stress_response``
without their YAML cache/DDB writing.  The original upstream strain recovery
formula is *not* copied: it fails the zero-deformation test on skew cells;
this module uses ``M = inv(A0) @ A``.  Both upstream sources are authored by
the project user and reused with permission.  There is no runtime dependency
on pymultibinit, libabinit or atomchain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from .matched import (
    MatchedCorrection,
    SiteMapping,
    _closest_lattice_points,
    _nearest_images,
    _reduce_lattice,
)

__all__ = [
    "VariableCellError",
    "VariableCellEvaluation",
    "VariableCellCorrection",
    "InternalStrainResponse",
    "ClampedIonElasticity",
    "build_variable_correction",
    "deformation_gradient",
    "strain_voigt_from_cell",
    "tensor_to_voigt_strain",
    "tensor_to_voigt_stress",
    "voigt_to_tensor_strain",
    "voigt_to_tensor_stress",
    "measure_internal_strain_response",
    "measure_clamped_ion_elasticity",
]

#: Relative roundoff tolerance for supplied correction-matrix symmetry.
_MATRIX_SYMMETRY_TOL = 1e-10

#: Absolute determinant threshold below which a cell is treated as singular
#: (angstrom**3), matching the reference-structure contract.
_CELL_DET_TOL = 1e-8

#: Tolerance for the polar rotation angle of the deformation gradient (rad).
#: The fixed anisotropic correction is defined only for symmetric strain
#: increments about the reference orientation; a larger polar rotation is a
#: rigid rotation of the cell and is rejected rather than silently treated as
#: strain.  The value is above the rotation produced by composing a small
#: finite-difference symmetric increment with an existing finite strain (order
#: step*strain), so numerical stress derivatives remain valid; a rigid rotation
#: is orders of magnitude larger.
_ROTATION_TOL = 1e-5

#: Acceptance tolerance for the provably complete nearest-image search
#: (angstrom).  A small absolute value keeps the selected image exact; the
#: caller's displacement bound is checked separately on the resulting vector.
_IMAGE_TOL = 1e-7

_TINY = 1e-300


class VariableCellError(ValueError):
    """Invalid cell, displacement, block, or unsupported variable-cell use."""


# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------
def _as_cell(cell, name: str) -> np.ndarray:
    array = np.asarray(cell, dtype=float)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise VariableCellError(f"{name} must be a finite (3,3) row-vector lattice")
    if abs(float(np.linalg.det(array))) < _CELL_DET_TOL:
        raise VariableCellError(f"{name} is singular or near-singular")
    return array


def _lattice_images_within(cell, offset, radius: float, nearest: float):
    """All integer lattice translations within ``radius`` of ``offset``.

    ``offset`` is one Cartesian row vector, ``cell`` is the current row-vector
    lattice, and ``nearest`` is the exact nearest-image distance for
    ``offset`` (computed by the caller's nearest-image search).  The result
    ``(images, distances)`` gives every primitive-cell integer translation
    ``n`` with ``|offset - n @ cell| <= radius`` up to the shared roundoff
    tolerance.  The sphere enumeration is the same provably complete
    closest-vector machinery used by the fixed-cell mapping, so it is
    skew-cell exact and does not import ASE.  The caller's nearest distance is
    used only to skip the enumeration when no image can be inside the bound;
    the sphere itself is enumerated at the bound radius, so every competing
    image within that radius is included.
    """
    cell = np.asarray(cell, dtype=float)
    offset = np.asarray(offset, dtype=float).reshape(3)
    empty = np.empty((0, 3), dtype=int)
    if float(nearest) > float(radius) + _IMAGE_TOL:
        return empty, np.empty(0)
    try:
        reduce = _reduce_lattice(cell)
        reduced = reduce @ cell
        center = np.rint(offset @ np.linalg.inv(cell)).astype(int)
        residual = offset - center @ cell
        candidates = _closest_lattice_points(
            reduced, residual, float(radius) ** 2)
    except ValueError as exc:
        raise VariableCellError(
            "assigned-site periodic-image enumeration failed; the cell or "
            "displacement is outside the proved closest-vector domain") from exc
    if candidates.size == 0:
        return empty, np.empty(0)
    translations = candidates.astype(float) @ reduced
    distances = np.linalg.norm(residual - translations, axis=1)
    keep = distances <= float(radius) + _IMAGE_TOL
    images = center[None, :] + candidates @ reduce
    return images[keep], distances[keep]


def deformation_gradient(reference_cell, cell) -> np.ndarray:
    """Return ``M = inv(A_ref) @ A_def`` with row-vector cells (ASE convention).

    Only orientation-preserving symmetric strain increments about the
    reference orientation are supported.  The polar rotation factor of ``M``
    is a rigid rotation, which would require rotating the frozen correction
    blocks; it is rejected rather than silently discarded.  The polar angle is
    used instead of only the antisymmetric part because a 180-degree rotation
    is symmetric.
    """
    reference = _as_cell(reference_cell, "reference cell")
    current = _as_cell(cell, "current cell")
    if float(np.linalg.det(reference)) * float(np.linalg.det(current)) <= 0.0:
        raise VariableCellError(
            "current cell has an inverted/reflected orientation relative to the "
            "reference cell; only orientation-preserving deformations are supported"
        )
    m = np.linalg.inv(reference) @ current
    # A rigid rotation is the orthogonal factor of the polar decomposition;
    # checking only the antisymmetric part misses symmetric rotations (for
    # example 180 degrees), so use the polar angle directly.
    u_svd, _singular, vh_svd = np.linalg.svd(m)
    polar = u_svd @ vh_svd
    angle = float(np.arccos(np.clip(0.5 * (np.trace(polar) - 1.0), -1.0, 1.0)))
    if angle > _ROTATION_TOL:
        raise VariableCellError(
            "cell change contains a rigid rotation "
            f"(polar rotation angle {angle:.3e} rad); "
            "only symmetric strain increments about the reference orientation "
            "are supported")
    return m


def tensor_to_voigt_strain(strain) -> np.ndarray:
    """Symmetric strain tensor -> engineering Voigt ``[xx,yy,zz,yz,xz,xy]``."""
    s = np.asarray(strain, dtype=float)
    if s.shape != (3, 3) or not np.isfinite(s).all():
        raise VariableCellError("strain must be a finite (3,3) tensor")
    return np.array([s[0, 0], s[1, 1], s[2, 2],
                     s[1, 2] + s[2, 1], s[0, 2] + s[2, 0], s[0, 1] + s[1, 0]])


def tensor_to_voigt_stress(stress) -> np.ndarray:
    """Symmetric stress tensor -> true-shear Voigt ``[xx,yy,zz,yz,xz,xy]``."""
    s = np.asarray(stress, dtype=float)
    if s.shape != (3, 3) or not np.isfinite(s).all():
        raise VariableCellError("stress must be a finite (3,3) tensor")
    return np.array([s[0, 0], s[1, 1], s[2, 2],
                     0.5 * (s[1, 2] + s[2, 1]),
                     0.5 * (s[0, 2] + s[2, 0]),
                     0.5 * (s[0, 1] + s[1, 0])])


def voigt_to_tensor_strain(voigt) -> np.ndarray:
    """Engineering Voigt strain -> symmetric strain tensor."""
    v = np.asarray(voigt, dtype=float)
    if v.shape != (6,) or not np.isfinite(v).all():
        raise VariableCellError("strain Voigt vector must be finite with shape (6,)")
    return np.array([
        [v[0], 0.5 * v[5], 0.5 * v[4]],
        [0.5 * v[5], v[1], 0.5 * v[3]],
        [0.5 * v[4], 0.5 * v[3], v[2]],
    ])


def voigt_to_tensor_stress(voigt) -> np.ndarray:
    """True-shear Voigt stress -> symmetric stress tensor."""
    v = np.asarray(voigt, dtype=float)
    if v.shape != (6,) or not np.isfinite(v).all():
        raise VariableCellError("stress Voigt vector must be finite with shape (6,)")
    return np.array([
        [v[0], v[5], v[4]],
        [v[5], v[1], v[3]],
        [v[4], v[3], v[2]],
    ])


def strain_voigt_from_cell(reference_cell, cell) -> np.ndarray:
    """Engineering Voigt strain of ``A_def = A_ref (I + eps)``.

    The symmetric part is returned; a rigid rotation of the cell relative to
    the reference orientation is rejected by :func:`deformation_gradient`.
    """
    m = deformation_gradient(reference_cell, cell)
    return tensor_to_voigt_strain(0.5 * (m + m.T) - np.eye(3))


# ---------------------------------------------------------------------------
# evaluated state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VariableCellEvaluation:
    """Correction energy, forces and stress at one cell/position state.

    ``forces`` is in the same atom order as the positions passed to
    :meth:`VariableCellCorrection.evaluate` (canonical reference-site order).
    ``stress`` is the full symmetric 3x3 Cauchy tensor in eV/angstrom**3 with
    the ASE sign convention; ``stress_voigt`` is the ASE-ordered 6-vector.
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray
    stress_voigt: np.ndarray
    strain_voigt: np.ndarray
    deformation: np.ndarray
    displacements: np.ndarray
    diagnostics: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# correction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VariableCellCorrection:
    """Frozen joint-harmonic correction about one reference target cell.

    Construct directly for explicit synthetic blocks, or through
    :func:`build_variable_correction` to reuse a
    :class:`~longrangefy.matched.MatchedCorrection` unchanged.

    ``reference_frac`` are the reference sites in fractional coordinates of
    ``reference_cell`` (the target supercell at the reference geometry).
    ``correction_matrix`` is ``DeltaPhi_L`` in image-major/atom-minor target
    order.  ``lambda_correction`` is ``DeltaLambda_L`` with shape
    ``(natom, 3, 6)`` in eV/angstrom per unit engineering strain;
    ``elastic_correction`` is ``DeltaC`` with shape ``(6,6)`` in eV/angstrom**3
    referred to ``V0``.  Omitted blocks are identically zero, which is the
    approved specified-model-class result.  ``max_displacement`` is an
    optional per-site Cartesian bound; when set, an evaluation also requires
    the assigned reference-site image to be unique at the deformed cell,
    including its other periodic images.
    """

    reference_cell: np.ndarray
    reference_frac: np.ndarray
    correction_matrix: np.ndarray
    lambda_correction: Optional[np.ndarray] = None
    elastic_correction: Optional[np.ndarray] = None
    max_displacement: Optional[float] = None
    max_abs_strain: Optional[float] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    _matched: Optional[MatchedCorrection] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        cell = np.array(_as_cell(self.reference_cell, "reference cell"), copy=True)
        frac = np.array(self.reference_frac, dtype=float, copy=True)
        if frac.ndim != 2 or frac.shape[1] != 3 or frac.shape[0] == 0:
            raise VariableCellError(
                f"reference_frac must have shape (n,3) with n > 0, got {frac.shape}")
        if not np.isfinite(frac).all():
            raise VariableCellError("reference_frac contains non-finite values")
        natom = frac.shape[0]

        matrix = np.array(self.correction_matrix, dtype=float, copy=True)
        if matrix.shape != (3 * natom, 3 * natom):
            raise VariableCellError(
                f"correction_matrix must have shape ({3 * natom},{3 * natom}), "
                f"got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise VariableCellError("correction_matrix contains non-finite values")
        scale = max(float(np.abs(matrix).max()), _TINY)
        asymmetry = float(np.abs(matrix - matrix.T).max())
        if asymmetry > _MATRIX_SYMMETRY_TOL * scale:
            raise VariableCellError(
                "correction_matrix is not symmetric: a conservative quadratic "
                f"energy requires symmetry (max residual {asymmetry:.3e})")
        matrix = 0.5 * (matrix + matrix.T)

        if self.lambda_correction is None:
            coupling = np.zeros((natom, 3, 6))
        else:
            coupling = np.array(self.lambda_correction, dtype=float, copy=True)
            if coupling.shape != (natom, 3, 6):
                raise VariableCellError(
                    "lambda_correction must have shape (natom,3,6), "
                    f"got {coupling.shape}")
            if not np.isfinite(coupling).all():
                raise VariableCellError("lambda_correction contains non-finite values")

        if self.elastic_correction is None:
            elastic = np.zeros((6, 6))
            elastic_asymmetry = 0.0
        else:
            raw_elastic = np.array(self.elastic_correction, dtype=float, copy=True)
            if raw_elastic.shape != (6, 6):
                raise VariableCellError(
                    f"elastic_correction must have shape (6,6), got {raw_elastic.shape}")
            if not np.isfinite(raw_elastic).all():
                raise VariableCellError("elastic_correction contains non-finite values")
            elastic_asymmetry = float(np.abs(raw_elastic - raw_elastic.T).max())
            elastic = 0.5 * (raw_elastic + raw_elastic.T)

        for name, value in (("max_displacement", self.max_displacement),
                            ("max_abs_strain", self.max_abs_strain)):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise VariableCellError(f"{name} must be positive and finite or None")

        metadata = dict(self.metadata)
        metadata.setdefault(
            "correction_blocks",
            {
                "phi": "DeltaPhi_L reused from the fixed-cell matched construction",
                "lambda": "DeltaLambda_L supplied explicitly or identically zero",
                "elastic": "DeltaC supplied explicitly or identically zero",
            },
        )
        metadata.setdefault(
            "base_response",
            "the original potential retains its own clamped-ion elastic and "
            "internal-strain response; this correction adds no long-range "
            "elastic/piezoelectric content",
        )
        metadata.setdefault("joint_order", "second order in (u, strain); Design F frozen blocks")
        metadata.setdefault("stress_chain_default", "affine (ASE contract)")
        metadata.setdefault(
            "strain_contract",
            {
                "max_abs_engineering_voigt_strain": self.max_abs_strain,
                "status": (
                    "caller-declared construction/validation bound, or None; "
                    "no production supported range is asserted (design D4)"),
            },
        )
        metadata["elastic_correction_major_asymmetry"] = elastic_asymmetry
        metadata["correction_matrix_symmetry_residual"] = asymmetry

        frac.setflags(write=False)
        cell.setflags(write=False)
        matrix.setflags(write=False)
        coupling.setflags(write=False)
        elastic.setflags(write=False)
        object.__setattr__(self, "reference_cell", cell)
        object.__setattr__(self, "reference_frac", frac)
        object.__setattr__(self, "correction_matrix", matrix)
        object.__setattr__(self, "lambda_correction", coupling)
        object.__setattr__(self, "elastic_correction", elastic)
        object.__setattr__(self, "metadata", metadata)

    @property
    def natom(self) -> int:
        return int(self.reference_frac.shape[0])

    @property
    def reference_positions(self) -> np.ndarray:
        """Reference Cartesian positions in the target reference cell."""
        return self.reference_frac @ self.reference_cell

    @property
    def volume_reference(self) -> float:
        """Unstrained target-cell volume V0 in angstrom**3."""
        return abs(float(np.linalg.det(self.reference_cell)))

    @property
    def species(self):
        species = self.metadata.get("species")
        return tuple(species) if species is not None else None

    def map_reference_sites(self, cell, positions, species=None,
                            tol: float = 1e-6) -> SiteMapping:
        """Map unstrained reference atoms to canonical target sites.

        Only defined on the reference target cell; the correspondence is then
        frozen for the correction lifetime.  This delegates to the fixed-cell
        :meth:`~longrangefy.matched.MatchedCorrection.map_sites` validator, so
        atom permutations, species and ambiguous assignments are handled by
        the same rules.
        """
        if self._matched is None:
            raise VariableCellError(
                "reference-site mapping is unavailable for a correction built "
                "without a MatchedCorrection; pass reference_frac explicitly")
        current = _as_cell(cell, "reference-cell mapping cell")
        scale = max(float(np.abs(self.reference_cell).max()), 1.0)
        if not np.allclose(current, self.reference_cell, rtol=0.0, atol=1e-8 * scale):
            raise VariableCellError(
                "reference-site mapping is defined only at the unstrained "
                "reference target cell; map before applying strain")
        return self._matched.map_sites(current, positions, species=species, tol=tol)

    def _deformation(self, cell) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        current = _as_cell(cell, "current cell")
        m = deformation_gradient(self.reference_cell, current)
        strain = 0.5 * (m + m.T) - np.eye(3)
        eps_v = tensor_to_voigt_strain(strain)
        if self.max_abs_strain is not None:
            largest = float(np.abs(eps_v).max())
            if largest > self.max_abs_strain:
                raise VariableCellError(
                    f"engineering Voigt strain component {largest:.6g} exceeds the "
                    f"caller-declared validation bound {self.max_abs_strain:.6g}; "
                    "no production supported range is asserted (design D4)")
        return m, strain, eps_v

    def _positions(self, positions) -> np.ndarray:
        array = np.asarray(positions, dtype=float)
        if array.shape != (self.natom, 3):
            raise VariableCellError(
                f"positions must have shape ({self.natom},3), got {array.shape}")
        if not np.isfinite(array).all():
            raise VariableCellError("positions contain non-finite values")
        return array

    def _bounded_displacement(self, max_displacement) -> Optional[float]:
        if max_displacement is None:
            return self.max_displacement
        if not np.isfinite(max_displacement) or max_displacement <= 0.0:
            raise VariableCellError(
                "max_displacement must be positive and finite or None")
        if self.max_displacement is None:
            return float(max_displacement)
        return min(self.max_displacement, float(max_displacement))

    def _displacements_with_images(self, current, positions,
                                   max_displacement=None):
        """Return ``(u, selected_images)`` with true nearest periodic images.

        ``selected_images`` are integer coefficients of the current-cell
        lattice translation chosen for each assigned reference site.  The
        search is the matched construction's provably complete closest-vector
        routine, so a skew cell is measured in the true Cartesian metric
        rather than by rounding fractional coordinates.
        """
        bound = self._bounded_displacement(max_displacement)
        affine = self.reference_frac @ current
        raw = positions - affine
        if bound is None:
            try:
                _, images = _nearest_images(current, raw, _IMAGE_TOL)
            except ValueError as exc:
                raise VariableCellError(
                    "nearest periodic-image search failed for the current cell and "
                    "positions; the deformation is outside the proved closest-vector "
                    "domain") from exc
            return raw - images @ current, images

        n = self.natom
        displacement = np.empty_like(raw)
        selected_images = np.empty((n, 3), dtype=int)
        second = np.empty(n)
        for i in range(n):
            try:
                distances, images = _nearest_images(
                    current, positions[i] - affine, _IMAGE_TOL)
            except ValueError as exc:
                raise VariableCellError(
                    f"nearest periodic-image search failed for atom {i} in the "
                    "current cell; the deformation is outside the proved "
                    "closest-vector domain") from exc
            selected_images[i] = images[i]
            site_images, _site_distances = _lattice_images_within(
                current, raw[i], bound, distances[i])
            if site_images.shape[0] > 1:
                raise VariableCellError(
                    f"ambiguous periodic image for reference site {i}: "
                    f"{site_images.shape[0]} images lie within the displacement "
                    f"bound {bound:.6g} angstrom; the assigned-site branch is not "
                    "unique under this deformation")
            if site_images.shape[0] == 1:
                selected_images[i] = site_images[0]
            displacement[i] = raw[i] - selected_images[i] @ current
            other = np.delete(distances, i)
            second[i] = float(other.min()) if other.size else np.inf
        largest = float(np.linalg.norm(displacement, axis=1).max())
        if largest > bound:
            raise VariableCellError(
                f"displacement {largest:.6g} angstrom exceeds the declared "
                f"bound {bound:.6g} angstrom; the fixed reference-site assignment "
                "is outside its bounded branch")
        ambiguous = second <= bound
        if np.any(ambiguous):
            bad = np.where(ambiguous)[0]
            raise VariableCellError(
                "ambiguous reference-site assignment for atoms "
                f"{np.array2string(bad)}: two reference sites lie within the "
                f"displacement bound {bound:.6g} angstrom; the deformed affine "
                "site separation leaves the fixed-site branch exchanged or "
                "ambiguous under this deformation")
        return displacement, selected_images

    def displacements(self, cell, positions, *, max_displacement=None) -> np.ndarray:
        """Displacement to the nearest affine reference-site image.

        The minimum image is found in the true Cartesian metric of the current
        cell by the matched closest-vector search; this is exact for skew
        cells, unlike rounding fractional coordinates.  ``max_displacement``
        optionally adds a per-call bound; the stricter of it and the
        construction bound is enforced.  A bounded evaluation also rejects an
        atom that lies within the bound of two reference sites, because the
        fixed-site branch would be exchanged or ambiguous, and rejects an
        assigned site with more than one periodic image inside the bound.
        """
        current = _as_cell(cell, "current cell")
        positions = self._positions(positions)
        displacement, _images = self._displacements_with_images(
            current, positions, max_displacement)
        return displacement

    def strain(self, cell):
        """Return ``(M, strain_tensor, strain_voigt)`` for the current cell."""
        return self._deformation(cell)

    def evaluate(self, cell, positions, *, stress_chain: str = "affine",
                 max_displacement=None) -> VariableCellEvaluation:
        """Evaluate the correction energy, forces and Cauchy stress.

        ``positions`` must be in canonical reference-site order.  The ASE
        adapter reorders consumer atoms to this order.  ``stress_chain`` is
        ``"affine"`` (ASE contract, default) or ``"cartesian"`` (independent
        Cartesian positions and cell).  ``max_displacement`` optionally adds a
        per-call bounded-branch check; the stricter of it and the construction
        bound is enforced.
        """
        if stress_chain not in ("affine", "cartesian"):
            raise VariableCellError(
                f"stress_chain must be 'affine' or 'cartesian', got {stress_chain!r}")
        current = _as_cell(cell, "current cell")
        m, _strain, eps_v = self._deformation(current)
        positions = self._positions(positions)
        u, selected_images = self._displacements_with_images(
            current, positions, max_displacement)
        u_flat = u.reshape(-1)

        coupling_flat = np.asarray(self.lambda_correction).reshape(3 * self.natom, 6)
        coupling_u = coupling_flat @ eps_v
        phi_u = self.correction_matrix @ u_flat
        elastic_energy = 0.5 * self.volume_reference * (
            eps_v @ np.asarray(self.elastic_correction) @ eps_v)
        energy = 0.5 * u_flat @ phi_u + 0.5 * u_flat @ coupling_u + elastic_energy
        forces_flat = -(phi_u + 0.5 * coupling_u)
        forces = forces_flat.reshape(self.natom, 3)

        # Q = dE/d(eps_v) at fixed Cartesian positions (true-shear Voigt).
        q_voigt = (0.5 * (coupling_flat.T @ u_flat)
                   + self.volume_reference
                   * (np.asarray(self.elastic_correction) @ eps_v))
        # The reference-site term uses the selected periodic image, not the
        # home-cell site: u = X - (R + n A0) M, so dE/dX contributes
        # (R + n A0)^T F.  Without this the affine stress changes when an atom
        # is translated by a lattice vector even though E and F do not.
        selected_reference = (
            self.reference_positions + selected_images @ self.reference_cell)
        p_x = voigt_to_tensor_stress(q_voigt) + selected_reference.T @ forces

        volume_current = abs(float(np.linalg.det(current)))
        work_x = m.T @ p_x
        if stress_chain == "affine":
            # Fixed-fractional increment: dX = X W for the current Cartesian
            # positions X, so P_aff^T M = P_x^T M - F^T X.
            work = work_x - positions.T @ forces
        else:
            work = work_x
        stress = 0.5 * (work + work.T) / volume_current
        skew = 0.5 * (work - work.T) / volume_current
        skew_abs = float(np.abs(skew).max())
        stress_scale = max(float(np.abs(stress).max()), _TINY)
        diagnostics = {
            "stress_chain": stress_chain,
            "volume_current_angstrom3": volume_current,
            "volume_ratio": volume_current / self.volume_reference,
            "deformation_determinant": float(np.linalg.det(m)),
            "max_abs_strain_voigt": float(np.abs(eps_v).max()),
            "max_displacement_angstrom": float(np.linalg.norm(u, axis=1).max()),
            "stress_skew_abs_ev_per_angstrom3": skew_abs,
            "stress_skew_rel": skew_abs / stress_scale,
        }
        return VariableCellEvaluation(
            energy=float(energy),
            forces=forces,
            stress=stress,
            stress_voigt=tensor_to_voigt_stress(stress),
            strain_voigt=eps_v,
            deformation=m,
            displacements=u,
            diagnostics=diagnostics,
        )

    def energy(self, cell, positions, *, stress_chain: str = "affine") -> float:
        return self.evaluate(cell, positions, stress_chain=stress_chain).energy

    def forces(self, cell, positions, *, stress_chain: str = "affine") -> np.ndarray:
        return self.evaluate(cell, positions, stress_chain=stress_chain).forces

    def stress(self, cell, positions, *, stress_chain: str = "affine",
               voigt: bool = False) -> np.ndarray:
        evaluation = self.evaluate(cell, positions, stress_chain=stress_chain)
        return evaluation.stress_voigt.copy() if voigt else evaluation.stress.copy()


# ---------------------------------------------------------------------------
# build from the fixed-cell matched matrix
# ---------------------------------------------------------------------------
def _target_reference_geometry(matched: MatchedCorrection):
    reps = np.asarray(matched.target_reps, dtype=int)
    if reps.shape != (3,) or np.any(reps <= 0):
        raise VariableCellError(f"matched.target_reps is invalid: {matched.target_reps!r}")
    n_uc = len(matched.positions)
    nimg = int(np.prod(reps))
    offsets = np.array(list(np.ndindex(*[int(v) for v in reps])), dtype=float)
    frac_primitive = np.asarray(matched.positions) @ np.linalg.inv(matched.primitive_cell)
    absolute_frac = (offsets[:, None, :] + frac_primitive[None, :, :]).reshape(-1, 3)
    reference_frac = absolute_frac / reps[None, :]
    reference_cell = np.diag(reps.astype(float)) @ np.asarray(matched.primitive_cell)
    if matched.correction_matrix.shape != (3 * n_uc * nimg, 3 * n_uc * nimg):
        raise VariableCellError(
            "matched.correction_matrix does not match the target atom count "
            f"({3 * n_uc * nimg} expected, got {matched.correction_matrix.shape})")
    return reference_cell, reference_frac


def build_variable_correction(
    matched: MatchedCorrection,
    *,
    lambda_correction=None,
    elastic_correction=None,
    max_displacement: Optional[float] = None,
    max_abs_strain: Optional[float] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> VariableCellCorrection:
    """Freeze a matched fixed-cell matrix into a variable-cell correction.

    The target reference cell is ``diag(target_reps) @ primitive_cell`` and
    the reference sites are the canonical tiled sites of the matched
    construction, so the dense ``DeltaPhi_L`` keeps its ordering.  The optional
    ``lambda_correction`` ``(natom,3,6)`` and ``elastic_correction`` ``(6,6)``
    are explicit Gamma-channel blocks; omit them for the approved specified
    model class, where they are identically zero.  They are never acquired by
    this function from the base calculator.  ``max_abs_strain`` is an optional
    caller-declared validation bound; no production supported range is
    asserted (decision D4).
    """
    if not isinstance(matched, MatchedCorrection):
        raise TypeError("matched must be a longrangefy.matched.MatchedCorrection")
    reference_cell, reference_frac = _target_reference_geometry(matched)
    species = matched.metadata.get("species")
    species_target = None
    if species is not None:
        species = list(species)
        if len(species) != len(matched.positions):
            raise VariableCellError(
                "matched.metadata['species'] does not match the primitive site count")
        species_target = species * int(np.prod(matched.target_reps))
    matching = tuple(int(v) for v in matched.matching_reps)
    target = tuple(int(v) for v in matched.target_reps)
    correction_max = float(np.abs(matched.correction_matrix).max())
    base_metadata = {
        "species": species_target,
        "matching_reps": list(matching),
        "target_reps": list(target),
        "reference_cell": np.asarray(reference_cell).tolist(),
        "volume_reference_angstrom3": abs(float(np.linalg.det(reference_cell))),
        "matching_cell_exact": matching == target,
        "matched_correction_matrix_max_abs_ev_per_angstrom2": correction_max,
        "correction_blocks": {
            "phi": "DeltaPhi_L reused unchanged from MatchedCorrection.correction_matrix",
            "lambda": "DeltaLambda_L supplied explicitly; zero for the specified model class",
            "elastic": "DeltaC supplied explicitly; zero for the specified model class",
        },
        "base_response": (
            "the original potential's own clamped-ion elastic and internal-strain "
            "response is preserved; no long-range elastic/piezoelectric correction "
            "is added (underdetermined by the specified response inputs)"),
        "joint_order": (
            "second order in (u, strain); frozen Design F blocks; strain-dependent "
            "IFCs / phonon-strain are excluded"),
        "stress_chain_default": "affine (ASE calculator contract)",
        "response_units": (
            "angstrom, eV per whole target cell, eV/angstrom, eV/angstrom**3"),
        "provenance": {
            "source": "longrangefy.matched fixed-cell construction (unchanged)",
            "variable_cell_design": "specs/lite/design-variable-cell-matching.md (Design F)",
            "adapted_from": [
                "pymultibinit pyeffpot/potential.py: _evaluate_elastic, "
                "_compute_strain, _compute_du_delta, _finalize_stress, "
                "_strain_to_voigt, _voigt_to_tensor",
                "atomchain ddb/finite_difference.py: strain_atoms, "
                "calculate_stress_response, calculate_internal_strain_response",
            ],
            "runtime_dependencies": "numpy/scipy only; no pymultibinit, libabinit or atomchain",
        },
    }
    if metadata:
        base_metadata["caller_metadata"] = dict(metadata)
    return VariableCellCorrection(
        reference_cell=reference_cell,
        reference_frac=reference_frac,
        correction_matrix=matched.correction_matrix,
        lambda_correction=lambda_correction,
        elastic_correction=elastic_correction,
        max_displacement=max_displacement,
        max_abs_strain=max_abs_strain,
        metadata=base_metadata,
        _matched=matched,
    )


# ---------------------------------------------------------------------------
# optional base-response diagnostics (not part of the correction)
# ---------------------------------------------------------------------------
_STRAIN_STENCILS = {
    "central": ((1.0, 0.5), (-1.0, -0.5)),
    "central5": ((2.0, -1.0 / 12.0), (1.0, 8.0 / 12.0),
                 (-1.0, -8.0 / 12.0), (-2.0, 1.0 / 12.0)),
}


@dataclass(frozen=True)
class InternalStrainResponse:
    """Measured Gamma force-strain response of an original model (diagnostic).

    ``df_dstrain[i, mu, a] = dF_i,mu / d(eps_v[a])`` in eV/angstrom per unit
    engineering strain.  ``lambda_response = -2 df_dstrain`` is the package
    ``Lambda = -2 dF/deps`` convention used by the correction formulas.  The
    measured block belongs to the **original** model; the correction's
    ``DeltaLambda_L`` is zero for the approved specified model class.
    """

    df_dstrain: np.ndarray
    lambda_response: np.ndarray
    strain_amplitude: float
    difference: str
    n_force_calls: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ClampedIonElasticity:
    """Measured clamped-ion stress-strain response of an original model.

    ``elastic[a, b] = d(sigma_voigt[a]) / d(eps_v[b])`` in eV/angstrom**3,
    with the ASE true-shear Voigt convention.  ``elastic`` is the major-part
    symmetrization of ``raw_elastic``; the raw asymmetry is recorded.  This
    is a diagnostic of the original model, not an added correction block.
    """

    elastic: np.ndarray
    raw_elastic: np.ndarray
    major_asymmetry: float
    strain_amplitude: float
    difference: str
    n_stress_calls: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)


def _stencil_offsets(difference: str):
    if difference not in _STRAIN_STENCILS:
        raise VariableCellError(
            f"difference must be one of {sorted(_STRAIN_STENCILS)}, got {difference!r}")
    return _STRAIN_STENCILS[difference]


def _check_callable_output(value, shape, name: str):
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise VariableCellError(f"{name} must return shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise VariableCellError(f"{name} returned non-finite values")
    return array


def _stress_voigt(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (3, 3):
        return tensor_to_voigt_stress(array)
    if array.shape == (6,):
        if not np.isfinite(array).all():
            raise VariableCellError("stress callback returned non-finite values")
        return array
    raise VariableCellError(
        "stress callback must return a (3,3) tensor or a 6-vector in ASE "
        f"[xx,yy,zz,yz,xz,xy] order, got shape {array.shape}")


def measure_internal_strain_response(
    get_forces: Callable[[np.ndarray, np.ndarray], np.ndarray],
    reference_cell,
    reference_positions,
    *,
    strain_amplitude: float = 1e-3,
    difference: str = "central",
) -> InternalStrainResponse:
    """Measure ``dF/d(eps_v)`` of an original model by affine-strain FD.

    ``get_forces(cell, positions)`` must return the original model's forces
    ``(natom,3)`` in eV/angstrom at the supplied cell and Cartesian positions.
    For each engineering Voigt component the cell is strained affinely,
    ``A = A_ref (I + h e_a)``, and the sites move at fixed fractional
    coordinates.  This is the Gamma internal-strain route (atomchain
    ``calculate_internal_strain_response`` adapted without its cache/DDB
    writer).  It measures the base model only; the correction does not
    acquire ``Lambda`` implicitly.
    """
    cell = _as_cell(reference_cell, "reference cell")
    positions = np.asarray(reference_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise VariableCellError(
            f"reference_positions must have shape (n,3), got {positions.shape}")
    if not np.isfinite(positions).all():
        raise VariableCellError("reference_positions contains non-finite values")
    if not np.isfinite(strain_amplitude) or strain_amplitude <= 0.0:
        raise VariableCellError("strain_amplitude must be positive and finite")
    offsets = _stencil_offsets(difference)
    frac = positions @ np.linalg.inv(cell)
    natom = positions.shape[0]
    derivative = np.zeros((natom, 3, 6))
    calls = 0
    force_scale = 0.0
    for a in range(6):
        unit = voigt_to_tensor_strain(np.eye(6)[a])
        accumulated = np.zeros((natom, 3))
        for multiplier, weight in offsets:
            strained_cell = cell @ (np.eye(3) + multiplier * strain_amplitude * unit)
            strained_positions = frac @ strained_cell
            forces = _check_callable_output(
                get_forces(strained_cell, strained_positions), (natom, 3),
                "get_forces")
            accumulated += weight * forces
            force_scale = max(force_scale, float(np.abs(forces).max()))
            calls += 1
        derivative[:, :, a] = accumulated / strain_amplitude
    lambda_response = -2.0 * derivative
    force_sum_derivative = derivative.sum(axis=0)
    lambda_sum = lambda_response.sum(axis=0)
    diagnostics = {
        "max_abs_force_ev_per_angstrom": force_scale,
        "max_abs_df_dstrain_ev_per_angstrom": float(np.abs(derivative).max()),
        "max_abs_lambda_ev_per_angstrom": float(np.abs(lambda_response).max()),
        "force_sum_derivative_ev_per_angstrom": force_sum_derivative.tolist(),
        "lambda_neutrality_residual_ev_per_angstrom": float(np.abs(lambda_sum).max()),
        "note": (
            "base-model Gamma response; translation neutrality follows from "
            "force neutrality and is reported, never projected"),
    }
    return InternalStrainResponse(
        df_dstrain=derivative,
        lambda_response=lambda_response,
        strain_amplitude=float(strain_amplitude),
        difference=difference,
        n_force_calls=calls,
        diagnostics=diagnostics,
    )


def measure_clamped_ion_elasticity(
    get_stress: Callable[[np.ndarray, np.ndarray], np.ndarray],
    reference_cell,
    reference_positions,
    *,
    strain_amplitude: float = 1e-3,
    difference: str = "central",
) -> ClampedIonElasticity:
    """Measure clamped-ion ``C = d(sigma_v)/d(eps_v)`` of an original model.

    ``get_stress(cell, positions)`` must return the original model's Cauchy
    stress in the ASE sign convention (``dE = V sigma : d(eps)``), either as a
    ``(3,3)`` tensor or a true-shear Voigt 6-vector in ``[xx,yy,zz,yz,xz,xy]``
    order.  Sites move affinely at fixed fractional coordinates, i.e. the
    clamped-ion branch.  The major part is symmetrized explicitly and the raw
    asymmetry recorded (atomchain ``calculate_stress_response`` adapted
    without its cache/DDB writer).  This measures the base model only.
    """
    cell = _as_cell(reference_cell, "reference cell")
    positions = np.asarray(reference_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise VariableCellError(
            f"reference_positions must have shape (n,3), got {positions.shape}")
    if not np.isfinite(positions).all():
        raise VariableCellError("reference_positions contains non-finite values")
    if not np.isfinite(strain_amplitude) or strain_amplitude <= 0.0:
        raise VariableCellError("strain_amplitude must be positive and finite")
    offsets = _stencil_offsets(difference)
    frac = positions @ np.linalg.inv(cell)
    response = np.zeros((6, 6))
    calls = 0
    stress_scale = 0.0
    for a in range(6):
        unit = voigt_to_tensor_strain(np.eye(6)[a])
        accumulated = np.zeros(6)
        for multiplier, weight in offsets:
            strained_cell = cell @ (np.eye(3) + multiplier * strain_amplitude * unit)
            strained_positions = frac @ strained_cell
            voigt = _stress_voigt(get_stress(strained_cell, strained_positions))
            accumulated += weight * voigt
            stress_scale = max(stress_scale, float(np.abs(voigt).max()))
            calls += 1
        response[:, a] = accumulated / strain_amplitude
    symmetrized = 0.5 * (response + response.T)
    diagnostics = {
        "raw_major_asymmetry_ev_per_angstrom3": float(
            np.abs(response - response.T).max()),
        "max_abs_stress_ev_per_angstrom3": stress_scale,
        "note": (
            "base-model clamped-ion response; the correction's DeltaC is zero "
            "for the approved specified model class and is never inferred here"),
    }
    return ClampedIonElasticity(
        elastic=symmetrized,
        raw_elastic=response,
        major_asymmetry=float(np.abs(response - response.T).max()),
        strain_amplitude=float(strain_amplitude),
        difference=difference,
        n_stress_calls=calls,
        diagnostics=diagnostics,
    )
