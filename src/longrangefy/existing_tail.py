"""Matched replacement of a declared DPLR explicit electrostatic tail (ticket 11).

The model is the Deep Potential Long Range (DPLR) functional of Zhang et al.,
J. Chem. Phys. **156**, 124107 (2022), used through DeePMD-kit's
``dipole_charge`` modifier: Gaussian-smeared fixed charges (ions plus Wannier
centroids) evaluated in reciprocal space under the tin-foil convention,

    E_ele = 2 pi C/V sum_{k != 0, |k| <= k_max}
                exp(-k^2 / (4 beta^2)) / k^2 |S(k)|^2,
    S(k)  = sum_i q_i exp(-i k.r_i) + sum_n q_n exp(-i k.W_n),

with ``C = e^2/(4 pi eps_0)``.  The tail response used here is the Hessian of
this functional *including the Wannier-centroid response*.  A DPLR centroid is
predicted by the Deep Wannier network, so ``W = W(u)`` is not generally a
stationary point of the tail energy and the exact Hessian needs both
``dW/du`` and the curvature ``d^2W/du^2``:

    H = E_rr + E_rW (dW/du) + (dW/du)^T E_Wr
        + (dW/du)^T E_WW (dW/du) + g_W . d^2W/du^2,

with ``g_W = dE/dW``.  The final gradient-contraction term is evaluated from
the required declared centroid curvature; there is no stationary-centroid
shortcut.  A frozen centroid and a first-order-only claim are not accepted.

The approved replacement is Contract A of
``specs/lite/design-existing-tail-replacement.md``:

    DeltaPhi_A = (D_L - E_L) - I_Q[D - E]_L,
    E_corrected = E_model + 0.5 u^T DeltaPhi_A u,
    F_corrected = F_model - DeltaPhi_A u,

where ``E_L`` is the declared tail Hessian in the running cell, ``D_L`` is the
desired screened-dipolar response from :func:`longrangefy.dipole.dipole_ifc`,
and ``I_Q`` is the base package's sample-reproducing matching-grid interpolant
(:func:`longrangefy.matched.build_correction`).  The model is evaluated
normally and re-applies its own per-cell tail once; ``DeltaPhi_A`` carries the
pointwise ``-E`` explicitly, so the long-range response is replaced rather
than stacked.  On the matching grid ``DeltaPhi_A`` vanishes identically and
every matching-periodic (tiled) pattern is preserved.

The matching-cell fold is exact only for a translation-periodic response.
The builder measures ``max|H - fold_unfold(H)|`` for the matching and target
existing Hessians and rejects any residual above relative floating-point
roundoff; :func:`dense_to_blocks` is never allowed to silently average a
non-periodic response, and there is no tolerance override that would permit a
visibly nonpreserving matching-cell correction.

This module deliberately contains **no model-specific fallback**.  A usable
workflow must supply (a) the declared DPLR metadata and charge maps, (b) the
explicit reference ion and Wannier-centroid positions, and (c) a callable
returning the full declared Wannier-centroid Jacobian and curvature for every running cell.
Missing or incompatible declarations are rejected.

Runtime boundary (recorded, not hidden)
---------------------------------------
The DPLR runtime in DeePMD-kit is TensorFlow-only and its interfaces are
declared unstable upstream.  No production DPLR checkpoint (``ener.pb`` with
``modifier_attr/type = dipole_charge`` and its ``dw.pb`` Wannier model) is
available in this checkout, and the optional ASE ``DP`` calculator does not
plumb the modifier.  Real-model qualification therefore remains a documented
prerequisite; this module's operator is validated against the documented
functional and, when available, against DeePMD-kit's ``EwaldRecp`` runtime.
The Coulomb constant and the package's row-vector cell convention are pinned
in :data:`_COULOMB`.  This module's own energy/Hessian are validated against
energy finite differences on skew cells (the fractional-to-Cartesian transform
is not transpose-symmetric); separately, the runtime's *force* path on skew
cells must be re-qualified before claiming a skew DPLR workflow (see the
ticket-11 report).

Units: angstrom, eV (whole cell), eV/angstrom, eV/angstrom**2.  IFC axes are
atom-major Cartesian ``(i, mu, j, nu)``; site order is inherited from
:class:`longrangefy.matched.MatchedCorrection` (image-major, atom-minor).

Provenance: the DPLR functional follows the published paper and the DeePMD-kit
documentation; the reciprocal-space Hessian was derived here and validated
against force/energy finite differences and the ``EwaldRecp`` runtime.  The
matching/interpolation machinery is the user-authored base implementation in
:mod:`longrangefy.matched`; no pymultibinit or libabinit runtime dependency is
introduced.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dipole import dipole_ifc
from .ifc import blocks_to_dense, blocks_to_qmatrices, dense_to_blocks
from .matched import MatchedCorrection, SiteMapping, build_correction
from .reference import ReferenceStructure, SupercellMapping
from .response import ElectricalResponse

__all__ = [
    "ExistingTailError",
    "DPLRTailDeclaration",
    "WannierResponse",
    "gaussian_tail_energy",
    "gaussian_tail_gradient",
    "gaussian_tail_hessian",
    "DPLRTailHessian",
    "ExistingTailCorrection",
    "build_existing_tail_correction",
]

#: e**2 / (4 pi eps_0), eV angstrom (CODATA 2018; same constant as dipole.py).
_COULOMB = 14.3996454784255
#: Shape/metadata validation only; this is not a physical smoothing length.
_DEFAULT_WANNIER_TRANSLATION_TOL = 1.0e-6
#: Relative roundoff allowance for the exact matching/target fold/unfold identity.
_PERIODICITY_ROUNDOFF = 1.0e-10


class ExistingTailError(ValueError):
    """Invalid declared-tail metadata, response, or replacement construction."""


# --------------------------------------------------------------------------
# small validators
# --------------------------------------------------------------------------
def _require_text(value, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ExistingTailError(f"{name} is required and must be non-empty")
    return text


def _require_sha256(value, name: str) -> str:
    text = _require_text(value, name).lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ExistingTailError(f"{name} must be a 64-character hexadecimal SHA-256 digest")
    return text


def _require_positive_float(value, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ExistingTailError(f"{name} must be finite and positive")
    return number


def _as_reps(reps, name: str) -> Tuple[int, int, int]:
    arr = np.asarray(reps)
    if arr.shape != (3,) or np.any(arr != np.floor(arr)) or np.any(arr < 1):
        raise ExistingTailError(f"{name} must be three positive integers, got {reps!r}")
    return tuple(int(v) for v in arr)


def _grid_h(reps: Tuple[int, int, int]) -> np.ndarray:
    """Full unshifted grid indices in tiled C order (base-package convention)."""
    return np.stack(
        np.meshgrid(*[np.arange(k) for k in reps], indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _default_eta(cell: np.ndarray) -> float:
    """Shared Ewald parameter default, identical to dipole_ifc/matched._default_eta."""
    invcell = np.linalg.inv(cell)
    return float(np.pi * 100.0 / 33.0
                 * np.sqrt(1.69 * np.sum(invcell @ invcell.T) / np.sum(cell @ cell.T)))


def _is_tin_foil(value) -> bool:
    text = str(value).strip().lower()
    return text.startswith("tin-foil") or text.startswith("tin foil") or text == "conducting"


def _validate_cell(cell, name: str) -> np.ndarray:
    arr = np.asarray(cell, dtype=float)
    if arr.shape != (3, 3) or not np.isfinite(arr).all():
        raise ExistingTailError(f"{name} must be a finite 3x3 row-vector lattice")
    if abs(np.linalg.det(arr)) < 1e-12:
        raise ExistingTailError(f"{name} must be nonsingular")
    return np.array(arr, dtype=float, copy=True)


def _validate_positions(positions, name: str, n: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise ExistingTailError(f"{name} must be a finite (N, 3) Cartesian array")
    if n is not None and arr.shape[0] != n:
        raise ExistingTailError(f"{name} must carry {n} sites, got {arr.shape[0]}")
    if arr.shape[0] == 0:
        raise ExistingTailError(f"{name} must be non-empty")
    return arr


def _reciprocal_shells(cell: np.ndarray, k_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """Integer shells with ``|k| <= k_max`` and their ``m G m = k^2/(4 pi^2)``.

    ``k = m @ (2 pi inv(cell).T)``; a spherical cutoff is used so the declared
    ``k_max`` has the same meaning as the design note's convergence study.
    """
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    # |m @ reciprocal| >= sigma_min |m|, so this bound cannot miss a skew-cell
    # cancellation that a per-basis-vector bound would.
    sigma_min = float(np.linalg.svd(reciprocal, compute_uv=False).min())
    if sigma_min <= 1e-12:
        raise ExistingTailError("cell is too skew for the reciprocal shell enumeration")
    nmax = int(np.ceil(k_max / sigma_min)) + 1
    if nmax > 64:
        raise ExistingTailError(
            f"cell is too skew for the reciprocal shell enumeration at "
            f"reciprocal_cutoff={k_max} (need {nmax} shells)")
    axis = np.arange(-nmax, nmax + 1)
    m = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    k = m @ reciprocal
    kn = np.linalg.norm(k, axis=1)
    keep = (kn <= k_max) & (kn > 1e-12)
    if not np.any(keep):
        raise ExistingTailError(
            f"reciprocal_cutoff={k_max} selects no nonzero reciprocal vectors; "
            "declare a cutoff consistent with the model")
    return m[keep], (kn[keep] / (2.0 * np.pi)) ** 2


# --------------------------------------------------------------------------
# the declared DPLR functional and its exact Hessian
# --------------------------------------------------------------------------
def gaussian_tail_energy(
    cell,
    positions,
    charges,
    *,
    ewald_beta: float,
    reciprocal_cutoff: float,
) -> float:
    """DPLR reciprocal Gaussian-charge energy for one periodic cell.

    ``positions`` are Cartesian (n, 3) row vectors; ``charges`` are the fixed
    declared charges in the same order.  The sum omits ``k = 0`` (tin foil)
    and is truncated at the declared spherical ``reciprocal_cutoff`` (A^-1).
    """
    cell = _validate_cell(cell, "cell")
    positions = _validate_positions(positions, "positions")
    charges = np.asarray(charges, dtype=float).reshape(-1)
    if charges.shape[0] != positions.shape[0] or not np.isfinite(charges).all():
        raise ExistingTailError("charges must be one finite value per position")
    beta = _require_positive_float(ewald_beta, "ewald_beta")
    cutoff = _require_positive_float(reciprocal_cutoff, "reciprocal_cutoff")
    m, mgm = _reciprocal_shells(cell, cutoff)
    frac = positions @ np.linalg.inv(cell)
    structure = charges @ np.exp(-2j * np.pi * (frac @ m.T))
    weights = np.exp(-np.pi ** 2 * mgm / beta ** 2) / mgm
    volume = abs(np.linalg.det(cell))
    return float(_COULOMB / (2.0 * np.pi * volume)
                 * np.sum(weights * np.abs(structure) ** 2))


def gaussian_tail_gradient(
    cell,
    positions,
    charges,
    *,
    ewald_beta: float,
    reciprocal_cutoff: float,
) -> np.ndarray:
    """Cartesian gradient of :func:`gaussian_tail_energy` (eV/angstrom).

    Returns ``(n, 3)``.  The WC gradient is used by the full centroid chain
    rule: a predicted centroid is not generally a stationary point, so the
    Hessian needs the contraction ``g_W . d^2W/du^2``.
    """
    cell = _validate_cell(cell, "cell")
    positions = _validate_positions(positions, "positions")
    charges = np.asarray(charges, dtype=float).reshape(-1)
    if charges.shape[0] != positions.shape[0] or not np.isfinite(charges).all():
        raise ExistingTailError("charges must be one finite value per position")
    beta = _require_positive_float(ewald_beta, "ewald_beta")
    cutoff = _require_positive_float(reciprocal_cutoff, "reciprocal_cutoff")
    m, mgm = _reciprocal_shells(cell, cutoff)
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    k = m @ reciprocal
    frac = positions @ np.linalg.inv(cell)
    phases = np.exp(-2j * np.pi * (frac @ m.T))  # (n, nk), exp(-i k.r)
    structure = charges @ phases
    weights = np.exp(-np.pi ** 2 * mgm / beta ** 2) / mgm
    volume = abs(np.linalg.det(cell))
    prefactor = _COULOMB / (2.0 * np.pi * volume) * 2.0
    coefficients = (prefactor * weights[None, :] * charges[:, None]
                    * np.imag(np.conj(structure)[None, :] * phases))
    return coefficients @ k


def gaussian_tail_hessian(
    cell,
    positions,
    charges,
    *,
    ewald_beta: float,
    reciprocal_cutoff: float,
) -> np.ndarray:
    """Exact Cartesian Hessian of :func:`gaussian_tail_energy`.

    Returns ``(3n, 3n)`` eV/Angstrom**2 in atom-major Cartesian order.  The
    translation-invariant kernel gives exact acoustic row sums: the diagonal
    block is the negative sum of the off-diagonal blocks, so uniform
    translation of every charge is a no-op even for a truncated shell sum.
    """
    cell = _validate_cell(cell, "cell")
    positions = _validate_positions(positions, "positions")
    charges = np.asarray(charges, dtype=float).reshape(-1)
    if charges.shape[0] != positions.shape[0] or not np.isfinite(charges).all():
        raise ExistingTailError("charges must be one finite value per position")
    beta = _require_positive_float(ewald_beta, "ewald_beta")
    cutoff = _require_positive_float(reciprocal_cutoff, "reciprocal_cutoff")
    m, mgm = _reciprocal_shells(cell, cutoff)
    frac = positions @ np.linalg.inv(cell)
    volume = abs(np.linalg.det(cell))
    weights = np.exp(-np.pi ** 2 * mgm / beta ** 2) / mgm
    n = positions.shape[0]
    hessian = np.zeros((n, 3, n, 3), dtype=float)
    prefactor = _COULOMB / (2.0 * np.pi * volume) * 8.0 * np.pi ** 2
    m_outer = m[:, :, None] * m[:, None, :]
    for a in range(n):
        for b in range(a + 1, n):
            phase = 2.0 * np.pi * (m @ (frac[a] - frac[b]))
            block = prefactor * charges[a] * charges[b] * np.einsum(
                "m,mij->ij", weights * np.cos(phase), m_outer)
            hessian[a, :, b, :] = block
            hessian[b, :, a, :] = block.T
    for a in range(n):
        hessian[a, :, a, :] = -sum(
            hessian[a, :, b, :] for b in range(n) if b != a)
    inv_cell = np.linalg.inv(cell)
    # f = r @ inv_cell, so H_cart[M,N] = inv_cell[M,u] H_frac[u,v] inv_cell[N,v].
    # Transposing the index order silently breaks skew cells (it is invisible
    # only because an orthogonal cell has a symmetric inverse).
    cartesian = np.einsum("Mu,aubv,Nv->aMbN", inv_cell, hessian, inv_cell)
    return cartesian.reshape(3 * n, 3 * n)


# --------------------------------------------------------------------------
# declared metadata and Wannier response
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DPLRTailDeclaration:
    """Explicit DPLR tail declaration (eligibility is declared, never inferred).

    All fields are required except the neutrality tolerance.  The declaration
    records the model/DW identity, the functional parameters, the charge maps,
    the reference structure and the WC reference positions.  It does not
    accept a model without an explicit asymptotic tail: those models belong to
    the base fixed-cell workflow (``longrangefy.matched.build_correction``).
    Charge maps are copied into read-only mappings during validation, so
    mutating the caller's dicts (or the declaration) after construction cannot
    invalidate the neutrality check.
    """

    model_path: str
    model_sha256: str
    deepmd_version: str
    wannier_model_path: str
    wannier_model_sha256: str
    model_type: str
    modifier_type: str
    ewald_beta: float
    ewald_h: float
    reciprocal_cutoff: float
    boundary_convention: str
    regime: str
    regime_evidence: str
    per_cell_tail_confirmed: bool
    periodic_insulator: bool
    reference_cell: np.ndarray
    reference_positions: np.ndarray
    reference_species: Tuple[str, ...]
    wannier_positions: np.ndarray
    wannier_species: Tuple[str, ...]
    sys_charge_map: Mapping[str, float]
    model_charge_map: Mapping[str, float]
    neutrality_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", _require_text(self.model_path, "model_path"))
        object.__setattr__(self, "model_sha256",
                           _require_sha256(self.model_sha256, "model_sha256"))
        object.__setattr__(self, "deepmd_version",
                           _require_text(self.deepmd_version, "deepmd_version"))
        object.__setattr__(self, "wannier_model_path",
                           _require_text(self.wannier_model_path, "wannier_model_path"))
        object.__setattr__(self, "wannier_model_sha256",
                           _require_sha256(self.wannier_model_sha256, "wannier_model_sha256"))

        model_type = _require_text(self.model_type, "model_type").lower()
        if model_type not in ("dplr", "deepmd-dplr"):
            raise ExistingTailError(
                f"model_type must declare DPLR (got {self.model_type!r}); a model with no "
                "explicit asymptotic electrostatic tail belongs to the base matched "
                "fixed-cell workflow (longrangefy.matched.build_correction), not this module")
        object.__setattr__(self, "model_type", model_type)
        modifier = _require_text(self.modifier_type, "modifier_type").lower()
        if modifier != "dipole_charge":
            raise ExistingTailError(
                f"modifier_type must be 'dipole_charge' (got {self.modifier_type!r}); "
                "the DPLR tail declaration is required")
        object.__setattr__(self, "modifier_type", modifier)

        object.__setattr__(self, "ewald_beta",
                           _require_positive_float(self.ewald_beta, "ewald_beta"))
        object.__setattr__(self, "ewald_h",
                           _require_positive_float(self.ewald_h, "ewald_h"))
        object.__setattr__(self, "reciprocal_cutoff",
                           _require_positive_float(self.reciprocal_cutoff,
                                                   "reciprocal_cutoff"))
        if not _is_tin_foil(self.boundary_convention):
            raise ExistingTailError(
                "boundary_convention must declare tin-foil/conducting; the desired "
                "response and the DPLR functional must share it (no silent conversion)")
        object.__setattr__(self, "boundary_convention", str(self.boundary_convention).strip())

        regime = _require_text(self.regime, "regime").lower()
        if regime not in ("per-cell", "regime-b"):
            raise ExistingTailError(
                "regime must be 'per-cell' (regime B): the model must re-apply its "
                "declared tail in every cell; finite-range-extension behaviour is "
                "unsupported")
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "regime_evidence",
                           _require_text(self.regime_evidence, "regime_evidence"))
        if self.per_cell_tail_confirmed is not True:
            raise ExistingTailError(
                "per_cell_tail_confirmed must be True: Contract A's pointwise -E_L "
                "cancellation requires a runtime that re-applies the declared tail per cell")
        if self.periodic_insulator is not True:
            raise ExistingTailError(
                "periodic_insulator must be True; metallic screening and non-periodic "
                "systems are outside the DPLR replacement contract")

        cell = _validate_cell(self.reference_cell, "reference_cell")
        positions = _validate_positions(self.reference_positions, "reference_positions")
        if positions.shape[0] == 0:
            raise ExistingTailError("reference_positions must be non-empty")
        species = tuple(str(s) for s in self.reference_species)
        if len(species) != positions.shape[0]:
            raise ExistingTailError(
                f"reference_species has {len(species)} labels for {positions.shape[0]} sites")
        if any(not s.strip() for s in species):
            raise ExistingTailError("reference_species labels must be non-empty")
        cell.setflags(write=False)
        positions_copy = positions.copy()
        positions_copy.setflags(write=False)
        object.__setattr__(self, "reference_cell", cell)
        object.__setattr__(self, "reference_positions", positions_copy)
        object.__setattr__(self, "reference_species", species)

        wannier_species = tuple(str(s) for s in self.wannier_species)
        if not wannier_species:
            raise ExistingTailError(
                "wannier_species must declare at least one DPLR Wannier-centroid type; "
                "a DPLR tail always carries explicit centroids")
        if len(set(wannier_species)) != len(wannier_species):
            raise ExistingTailError("wannier_species must not contain duplicates")
        absent = sorted(set(wannier_species) - set(species))
        if absent:
            raise ExistingTailError(
                f"wannier_species {absent} do not occur in reference_species")
        object.__setattr__(self, "wannier_species", wannier_species)

        sys_map = self._validated_charge_map(self.sys_charge_map, "sys_charge_map")
        wc_map = self._validated_charge_map(self.model_charge_map, "model_charge_map")
        for label in species:
            if label not in sys_map:
                raise ExistingTailError(f"sys_charge_map has no ionic charge for species {label!r}")
        for label in wannier_species:
            if label not in wc_map:
                raise ExistingTailError(
                    f"model_charge_map has no Wannier-centroid charge for species {label!r}")
            if label not in sys_map:
                raise ExistingTailError(
                    f"wannier_species {label!r} is not present in sys_charge_map")
        object.__setattr__(self, "sys_charge_map", MappingProxyType(dict(sys_map)))
        object.__setattr__(self, "model_charge_map", MappingProxyType(dict(wc_map)))

        parents = self.wannier_parents
        if not parents:
            raise ExistingTailError(
                "wannier_species do not occur in reference_species; the declared DPLR "
                "reference must contain at least one WC-bearing ion")
        wannier_positions = _validate_positions(
            self.wannier_positions, "wannier_positions", n=len(parents)).copy()
        wannier_positions.setflags(write=False)
        object.__setattr__(self, "wannier_positions", wannier_positions)

        tolerance = float(self.neutrality_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ExistingTailError("neutrality_tolerance must be finite and nonnegative")
        object.__setattr__(self, "neutrality_tolerance", tolerance)

        ionic = sum(sys_map[s] for s in species)
        centroid = sum(wc_map[s] for s in species if s in wannier_species)
        residual = abs(ionic + centroid)
        if residual > tolerance:
            raise ExistingTailError(
                f"declared DPLR charges are not neutral: |sum q_ion + sum q_WC| = "
                f"{residual:.6e} e exceeds tolerance {tolerance:.1e} e; Ewald "
                "validity requires a neutral cell")

    @staticmethod
    def _validated_charge_map(value, name: str) -> Dict[str, float]:
        if not isinstance(value, MappingABC) or not value:
            raise ExistingTailError(f"{name} must be a non-empty mapping species -> charge")
        out: Dict[str, float] = {}
        for key, charge in value.items():
            label = _require_text(key, f"{name} key")
            number = float(charge)
            if not np.isfinite(number):
                raise ExistingTailError(f"{name}[{label!r}] is not finite")
            out[label] = number
        return out

    @property
    def natom(self) -> int:
        return int(np.asarray(self.reference_positions).shape[0])

    @property
    def wannier_parents(self) -> Tuple[int, ...]:
        """Reference ion indices owning a WC, in reference site order."""
        return tuple(i for i, s in enumerate(self.reference_species)
                     if s in self.wannier_species)

    @property
    def n_wannier(self) -> int:
        return len(self.wannier_parents)

    @property
    def wannier_offsets(self) -> np.ndarray:
        """Cartesian offsets ``W_n - r_parent`` for the reference WCs."""
        parents = self.wannier_parents
        return (np.asarray(self.wannier_positions)
                - np.asarray(self.reference_positions)[list(parents)])

    def to_dict(self) -> dict:
        """JSON-friendly declaration record (arrays as nested lists)."""
        return {
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "deepmd_version": self.deepmd_version,
            "wannier_model_path": self.wannier_model_path,
            "wannier_model_sha256": self.wannier_model_sha256,
            "model_type": self.model_type,
            "modifier_type": self.modifier_type,
            "ewald_beta": float(self.ewald_beta),
            "ewald_h": float(self.ewald_h),
            "reciprocal_cutoff": float(self.reciprocal_cutoff),
            "boundary_convention": self.boundary_convention,
            "regime": self.regime,
            "regime_evidence": self.regime_evidence,
            "per_cell_tail_confirmed": bool(self.per_cell_tail_confirmed),
            "periodic_insulator": bool(self.periodic_insulator),
            "reference_cell": np.asarray(self.reference_cell).tolist(),
            "reference_positions": np.asarray(self.reference_positions).tolist(),
            "reference_species": list(self.reference_species),
            "wannier_positions": np.asarray(self.wannier_positions).tolist(),
            "wannier_species": list(self.wannier_species),
            "sys_charge_map": dict(self.sys_charge_map),
            "model_charge_map": dict(self.model_charge_map),
            "neutrality_tolerance": float(self.neutrality_tolerance),
        }


@dataclass(frozen=True)
class WannierResponse:
    """Declared Wannier-centroid response about the reference.

    ``jacobian[n, mu, i, nu] = dW_{n,mu} / du_{i,nu}`` (Cartesian, angstrom per
    angstrom), where ``W_n`` is the *absolute* Wannier-centroid position.  It
    must therefore include the parent-ion identity term in addition to the
    Deep Wannier network's displacement response; otherwise a uniform
    translation would not move the centroid and
    :meth:`translation_covariance_residual` rejects it.  ``provenance`` and
    ``method`` identify how the model produced the response (for example
    autodiff or finite differences through the Deep Wannier network).  A
    zero/frozen Jacobian is rejected: the design requires the model's own
    centroid response, not a frozen-centroid shortcut.

    A DPLR centroid is predicted, not generally a stationary point of the
    tail energy, so the exact Hessian of ``E(u, W(u))`` requires
    ``sum_a (dE/dW_a) d^2W_a/du_i du_j`` in addition to the first-order
    Jacobian.  ``second_derivative`` is therefore a required field, with shape
    ``(n_wc, 3, n_ion, 3, n_ion, 3)`` (the full Cartesian curvature
    ``d^2W/du^2``).  There is no stationary-centroid shortcut: a bound on
    ``|dE/dW|`` alone does not bound the omitted term without a bound on the
    unknown curvature.  ``full_response=True`` certifies the first-order
    response is complete (not frozen/partial); it does not replace the
    required curvature.
    """

    jacobian: np.ndarray
    provenance: str
    method: str
    second_derivative: np.ndarray
    full_response: bool = True

    def __post_init__(self) -> None:
        jac = np.asarray(self.jacobian, dtype=float)
        if jac.ndim != 4 or jac.shape[1] != 3 or jac.shape[3] != 3 or jac.shape[0] == 0:
            raise ExistingTailError(
                "WannierResponse.jacobian must have shape (n_wc, 3, n_ion, 3), "
                f"got {jac.shape}")
        if not np.isfinite(jac).all():
            raise ExistingTailError("WannierResponse.jacobian contains non-finite values")
        if not np.any(jac):
            raise ExistingTailError(
                "WannierResponse.jacobian is zero: frozen-centroid response is rejected "
                "by the DPLR replacement contract")
        if self.full_response is not True:
            raise ExistingTailError(
                "WannierResponse.full_response must be True; partial/frozen centroid "
                "responses are not eligible")
        jac = jac.copy()
        jac.setflags(write=False)
        object.__setattr__(self, "jacobian", jac)
        object.__setattr__(self, "provenance", _require_text(self.provenance, "provenance"))
        object.__setattr__(self, "method", _require_text(self.method, "method"))
        object.__setattr__(self, "full_response", True)

        curvature = self.second_derivative
        if curvature is None:
            raise ExistingTailError(
                "WannierResponse.second_derivative is required: a first-order "
                "Jacobian alone does not determine the harmonic tail Hessian, "
                "because DPLR centroids are predicted and not generally stationary")
        expected = (jac.shape[0], 3, jac.shape[2], 3, jac.shape[2], 3)
        c_arr = np.asarray(curvature, dtype=float)
        if c_arr.shape != expected:
            raise ExistingTailError(
                "WannierResponse.second_derivative must have shape "
                f"{expected}, got {c_arr.shape}")
        if not np.isfinite(c_arr).all():
            raise ExistingTailError(
                "WannierResponse.second_derivative contains non-finite values")
        swap = c_arr.transpose(0, 1, 4, 5, 2, 3)
        asymmetry = float(np.abs(c_arr - swap).max())
        scale = max(float(np.abs(c_arr).max()), 1e-300)
        if asymmetry > 1.0e-8 * scale:
            raise ExistingTailError(
                "WannierResponse.second_derivative must be symmetric under "
                f"(i, nu) <-> (j, rho); residual {asymmetry:.3e}")
        c_arr = 0.5 * (c_arr + swap)
        c_arr.setflags(write=False)
        object.__setattr__(self, "second_derivative", c_arr)

    @property
    def max_abs(self) -> float:
        return float(np.abs(self.jacobian).max())

    @property
    def second_derivative_max_abs(self) -> float:
        return float(np.abs(self.second_derivative).max())

    def translation_covariance_residual(self) -> float:
        """Max deviation of ``sum_i dW_n/du_i`` from the identity.

        A translation-invariant model must translate every WC rigidly with a
        uniform translation of the ions.  The residual is a declared-response
        eligibility check, not a tolerance for a frozen centroid.
        """
        row = np.asarray(self.jacobian).sum(axis=2)  # (n_wc, 3, 3)
        expected = np.broadcast_to(np.eye(3), row.shape)
        return float(np.abs(row - expected).max())

    def translation_covariance_second_order_residual(self) -> float:
        """Max deviation of ``sum_j d^2W_{n,mu}/du_i du_j`` from zero.

        A uniform translation leaves the relative coordinates unchanged, so
        each first-ion row of the curvature must sum to zero over the second
        ion index.
        """
        row = np.asarray(self.second_derivative).sum(axis=4)
        return float(np.abs(row).max())

    def check_for(self, n_wc: int, n_ion: int,
                  translation_tolerance: float = _DEFAULT_WANNIER_TRANSLATION_TOL) -> None:
        """Validate shape, per-WC response and first/second-order covariance."""
        if self.jacobian.shape != (n_wc, 3, n_ion, 3):
            raise ExistingTailError(
                "WannierResponse.jacobian shape does not match the running cell: "
                f"expected {(n_wc, 3, n_ion, 3)}, got {self.jacobian.shape}")
        per_wc = np.abs(self.jacobian).reshape(n_wc, 3, -1).max(axis=(1, 2))
        if np.any(per_wc == 0.0):
            raise ExistingTailError(
                "WannierResponse has a WC with zero response: every centroid must carry "
                "its own declared response (frozen centroids are rejected)")
        tolerance = float(translation_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ExistingTailError("translation_tolerance must be finite and nonnegative")
        residual = self.translation_covariance_residual()
        if residual > tolerance:
            raise ExistingTailError(
                "WannierResponse violates translation covariance: max |sum_i dW_n/du_i - I| "
                f"= {residual:.3e} exceeds {tolerance:.3e}; a uniform ionic translation must "
                "translate every WC rigidly")
        expected = (n_wc, 3, n_ion, 3, n_ion, 3)
        if self.second_derivative.shape != expected:
            raise ExistingTailError(
                "WannierResponse.second_derivative shape does not match the running "
                f"cell: expected {expected}, got {self.second_derivative.shape}")
        residual2 = self.translation_covariance_second_order_residual()
        if residual2 > tolerance:
            raise ExistingTailError(
                "WannierResponse violates second-order translation covariance: "
                f"max |sum_j d^2W_n/du_i du_j| = {residual2:.3e} exceeds "
                f"{tolerance:.3e}")


# --------------------------------------------------------------------------
# declared tail operator for a running cell
# --------------------------------------------------------------------------
class DPLRTailHessian:
    """Hessian of the declared DPLR tail including the WC chain rule.

    The operator is cell-agnostic: the caller supplies the running cell, its
    ion/WC positions and parent map, and a :class:`WannierResponse` for that
    same cell.  The correction builder does this for the matching and target
    cells.  No charge, position, cutoff or response is inferred from forces.
    """

    def __init__(self, declaration: DPLRTailDeclaration) -> None:
        if not isinstance(declaration, DPLRTailDeclaration):
            raise ExistingTailError("declaration must be a DPLRTailDeclaration")
        self.declaration = declaration

    def charges(
        self,
        ion_species: Sequence[str],
        wannier_parents: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ionic, WC and concatenated charges for one running cell."""
        decl = self.declaration
        species = tuple(str(s) for s in ion_species)
        if any(s not in decl.sys_charge_map for s in species):
            missing = sorted({s for s in species if s not in decl.sys_charge_map})
            raise ExistingTailError(
                f"running cell has species without declared ionic charges: {missing}")
        parents = tuple(int(p) for p in wannier_parents)
        expected = {i for i, s in enumerate(species) if s in decl.wannier_species}
        if len(parents) != len(expected):
            raise ExistingTailError(
                "wannier_parents count does not match the WC-bearing sites of the "
                f"running cell: expected {len(expected)}, got {len(parents)}")
        for parent in parents:
            if parent < 0 or parent >= len(species):
                raise ExistingTailError(
                    f"wannier parent index {parent} outside the running ion list")
            if species[parent] not in decl.wannier_species:
                raise ExistingTailError(
                    f"wannier parent {parent} has species {species[parent]!r}, which does "
                    f"not declare a WC (wannier_species={decl.wannier_species})")
        if set(parents) != expected or len(set(parents)) != len(parents):
            raise ExistingTailError(
                "wannier_parents must cover every WC-bearing ion exactly once in the "
                f"running cell; expected {sorted(expected)}, got {sorted(parents)}")
        q_ion = np.array([decl.sys_charge_map[s] for s in species], dtype=float)
        q_wc = np.array([decl.model_charge_map[species[p]] for p in parents], dtype=float)
        return q_ion, q_wc, np.concatenate([q_ion, q_wc])

    def energy(
        self,
        cell,
        ion_positions,
        ion_species: Sequence[str],
        wannier_positions,
        wannier_parents: Sequence[int],
    ) -> float:
        """Declared tail energy of one cell (ions + WCs, fixed positions)."""
        cell = _validate_cell(cell, "cell")
        ions = _validate_positions(ion_positions, "ion_positions")
        wcs = _validate_positions(wannier_positions, "wannier_positions")
        _, _, charges = self.charges(ion_species, wannier_parents)
        if len(charges) != ions.shape[0] + wcs.shape[0]:
            raise ExistingTailError("charge/site count mismatch in the running cell")
        positions = np.concatenate([ions, wcs], axis=0)
        return gaussian_tail_energy(
            cell, positions, charges,
            ewald_beta=self.declaration.ewald_beta,
            reciprocal_cutoff=self.declaration.reciprocal_cutoff)

    def gradient(
        self,
        cell,
        ion_positions,
        ion_species: Sequence[str],
        wannier_positions,
        wannier_parents: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gradient of the declared tail wrt ion and WC Cartesian positions."""
        cell = _validate_cell(cell, "cell")
        ions = _validate_positions(ion_positions, "ion_positions")
        wcs = _validate_positions(wannier_positions, "wannier_positions")
        q_ion, q_wc, charges = self.charges(ion_species, wannier_parents)
        if len(q_ion) != ions.shape[0] or len(q_wc) != wcs.shape[0]:
            raise ExistingTailError("charge/site count mismatch in the running cell")
        positions = np.concatenate([ions, wcs], axis=0)
        grad = gaussian_tail_gradient(
            cell, positions, charges,
            ewald_beta=self.declaration.ewald_beta,
            reciprocal_cutoff=self.declaration.reciprocal_cutoff)
        n_ion = len(q_ion)
        return grad[:n_ion].copy(), grad[n_ion:].copy()

    def hessian(
        self,
        cell,
        ion_positions,
        ion_species: Sequence[str],
        wannier_positions,
        wannier_parents: Sequence[int],
        wannier_response: WannierResponse,
        *,
        translation_tolerance: float = _DEFAULT_WANNIER_TRANSLATION_TOL,
        report: Optional[Dict[str, object]] = None,
    ) -> np.ndarray:
        """Full ionic Hessian ``E_uu`` after eliminating the WC response.

        A predicted centroid satisfies ``W = W(u)`` but is not generally a
        stationary point of ``E_ele``.  The exact chain rule is
        ``H = E_rr + E_rW J + J^T E_Wr + J^T E_WW J + g_W . d^2W/du^2`` with
        ``g_W = dE/dW``.  The curvature term is evaluated from the required
        ``WannierResponse.second_derivative``; a first-order-only response is
        not representable by :class:`WannierResponse`.
        """
        cell = _validate_cell(cell, "cell")
        ions = _validate_positions(ion_positions, "ion_positions")
        wcs = _validate_positions(wannier_positions, "wannier_positions")
        q_ion, q_wc, charges = self.charges(ion_species, wannier_parents)
        if len(q_ion) != ions.shape[0] or len(q_wc) != wcs.shape[0]:
            raise ExistingTailError("charge/site count mismatch in the running cell")
        if not isinstance(wannier_response, WannierResponse):
            raise ExistingTailError(
                "a WannierResponse carrying provenance and method is required; raw "
                "arrays and None are not a declared full centroid response")
        wannier_response.check_for(len(q_wc), len(q_ion), translation_tolerance)
        _, grad_wc = self.gradient(
            cell, ions, ion_species, wcs, wannier_parents)
        c_flat = wannier_response.second_derivative.reshape(
            3 * len(q_wc), 3 * len(q_ion), 3 * len(q_ion))
        curvature_term = np.einsum("a,aij->ij", grad_wc.reshape(-1), c_flat)
        if report is not None:
            report["wc_gradient_max_abs"] = float(np.abs(grad_wc).max())
            report["curvature_max_abs"] = wannier_response.second_derivative_max_abs

        positions = np.concatenate([ions, wcs], axis=0)
        joint = gaussian_tail_hessian(
            cell, positions, charges,
            ewald_beta=self.declaration.ewald_beta,
            reciprocal_cutoff=self.declaration.reciprocal_cutoff)
        n_ion, n_wc = len(q_ion), len(q_wc)
        k_ii = joint.reshape(n_ion + n_wc, 3, n_ion + n_wc, 3)[:n_ion, :, :n_ion, :]
        k_iw = joint.reshape(n_ion + n_wc, 3, n_ion + n_wc, 3)[:n_ion, :, n_ion:, :]
        k_ww = joint.reshape(n_ion + n_wc, 3, n_ion + n_wc, 3)[n_ion:, :, n_ion:, :]
        k_ii = k_ii.reshape(3 * n_ion, 3 * n_ion)
        k_iw = k_iw.reshape(3 * n_ion, 3 * n_wc)
        k_ww = k_ww.reshape(3 * n_wc, 3 * n_wc)
        jac = wannier_response.jacobian.reshape(3 * n_wc, 3 * n_ion)
        hessian = (k_ii + k_iw @ jac + jac.T @ k_iw.T + jac.T @ k_ww @ jac
                   + curvature_term)
        asymmetry = float(np.abs(hessian - hessian.T).max())
        scale = max(float(np.abs(hessian).max()), 1e-300)
        if asymmetry > 1e-8 * scale:
            raise ExistingTailError(
                f"full WC response produced an asymmetric Hessian (residual {asymmetry:.3e})")
        return 0.5 * (hessian + hessian.T)


def _tile_system(
    declaration: DPLRTailDeclaration,
    reps: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Tile the declared reference into one diagonal repetition.

    WC order is the filtered ion order of the cell (one WC per ion whose
    species is in ``wannier_species``).  WC positions follow their parent ion's
    declared offset, so a lattice-vector image choice in the reference is
    preserved exactly.
    """
    cell = declaration.reference_cell
    ref = ReferenceStructure(
        cell=cell,
        frac_positions=declaration.reference_positions @ np.linalg.inv(cell),
        species=declaration.reference_species,
        source="DPLRTailDeclaration")
    mapping = SupercellMapping(ref, reps)
    ions = mapping.cart_positions_supercell()
    species = tuple(mapping.species_supercell())
    n_ion = ions.shape[0]
    parents_ref = declaration.wannier_parents
    offsets_ref = declaration.wannier_offsets
    site_to_wc = {parent: index for index, parent in enumerate(parents_ref)}
    wc_positions = []
    wc_parents = []
    for ion in range(n_ion):
        site = ion % declaration.natom
        label = species[ion]
        if label in declaration.wannier_species:
            if site not in site_to_wc:
                raise ExistingTailError(
                    "declaration is inconsistent: a WC-bearing species has no declared "
                    f"reference centroid at site {site}")
            wc_positions.append(ions[ion] + offsets_ref[site_to_wc[site]])
            wc_parents.append(ion)
    wc_array = np.asarray(wc_positions, dtype=float).reshape(-1, 3)
    return (np.diag(np.asarray(reps, dtype=float)) @ cell, ions, species,
            wc_array, np.asarray(wc_parents, dtype=int),
            tuple(species[p] for p in wc_parents))


# --------------------------------------------------------------------------
# assembled Contract A correction
# --------------------------------------------------------------------------
@dataclass
class ExistingTailCorrection:
    """Contract A replacement matrix plus the base correction carrier.

    ``correction_matrix`` is ``DeltaPhi_A``.  The desired and existing
    responses, their matching-grid interpolants and the two carriers are
    retained so callers can audit preservation and double counting without
    reconstructing anything.
    """

    carrier: MatchedCorrection
    existing_carrier: MatchedCorrection
    correction_matrix: np.ndarray
    desired_matrix: np.ndarray
    existing_matrix: np.ndarray
    interpolated_desired: np.ndarray
    interpolated_existing: np.ndarray
    diagnostics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.carrier.natom_target
        expected = (3 * n, 3 * n)
        for name in ("correction_matrix", "desired_matrix", "existing_matrix",
                     "interpolated_desired", "interpolated_existing"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != expected or not np.isfinite(arr).all():
                raise ExistingTailError(
                    f"{name} must be finite with shape {expected}, got {arr.shape}")
            object.__setattr__(self, name, arr)
        if self.carrier.target_reps != self.existing_carrier.target_reps:
            raise ExistingTailError("carrier target repetitions disagree")

    @property
    def natom_target(self) -> int:
        return self.carrier.natom_target

    @property
    def matching_reps(self) -> Tuple[int, int, int]:
        return self.carrier.matching_reps

    @property
    def target_reps(self) -> Tuple[int, int, int]:
        return self.carrier.target_reps

    @property
    def contract_b_matrix(self) -> np.ndarray:
        """``D_L - E_L`` (documented alternative; not used for correction)."""
        return self.desired_matrix - self.existing_matrix

    @property
    def interpolated_difference(self) -> np.ndarray:
        """``I_Q[D]_L - I_Q[E]_L = I_Q[D - E]_L`` (linear interpolation)."""
        return self.interpolated_desired - self.interpolated_existing

    @property
    def existing_interpolation_mismatch(self) -> np.ndarray:
        """``E_L - I_Q[E]_L``: vanishes on tiled patterns but acts off grid."""
        return self.existing_matrix - self.interpolated_existing

    def _check_u(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.shape[0] != 3 * self.natom_target:
            raise ValueError(
                f"expected {3 * self.natom_target} displacement components, "
                f"got {u.shape[0]}")
        return u

    def energy(self, u) -> float:
        """Correction energy (eV, whole target cell) in canonical site order."""
        u = self._check_u(u)
        return float(0.5 * u @ (self.correction_matrix @ u))

    def forces(self, u) -> np.ndarray:
        """Correction forces (eV/angstrom) in canonical site order."""
        return -(self.correction_matrix @ self._check_u(u))

    def evaluate(self, u, mapping: Optional[SiteMapping] = None) -> Tuple[float, np.ndarray]:
        """Correction energy and forces in the caller's atom order."""
        u = self._check_u(u)
        if mapping is None:
            return self.energy(u), self.forces(u)
        u_can = mapping.permute_u(u)
        f_can = -(self.correction_matrix @ u_can)
        energy = float(0.5 * u_can @ (self.correction_matrix @ u_can))
        return energy, mapping.unpermute_forces(f_can)

    def map_sites(self, cell, positions, species=None, tol: float = 1e-6) -> SiteMapping:
        """Delegate the base package's explicit site mapping and validation."""
        return self.carrier.map_sites(cell, positions, species=species, tol=tol)

    def matching_periodic_tiling(self, pattern) -> np.ndarray:
        """Tile a matching-periodic pattern (see ``MatchedCorrection``)."""
        return self.carrier.matching_periodic_tiling(pattern)

    def tiled_compatible(self) -> bool:
        return self.carrier.tiled_compatible()

    def desired_dynamical(self, q) -> np.ndarray:
        """Desired response ``D(q)`` (the base builder restores it exactly)."""
        return self.carrier.branch_dynamical(q, branch="split")

    def interpolated_desired_dynamical(self, q) -> np.ndarray:
        """Matching-grid interpolant ``I_Q[D](q)``."""
        return self.carrier.branch_dynamical(q, branch="unsplit")

    def interpolated_existing_dynamical(self, q) -> np.ndarray:
        """Matching-grid interpolant ``I_Q[E](q)``."""
        return self.existing_carrier.branch_dynamical(q, branch="unsplit")

    def interpolated_difference_dynamical(self, q) -> np.ndarray:
        """``I_Q[D - E](q)``."""
        return (self.interpolated_desired_dynamical(q)
                - self.interpolated_existing_dynamical(q))

    def composed_tail_dynamical(self, q) -> np.ndarray:
        """Response of ``E + DeltaPhi_A`` at ``q``.

        The model supplies its own ``E(q)``; Contract A adds
        ``(D - E) - I_Q[D - E]``, so the composed result is
        ``D(q) - I_Q[D - E](q)``.  This is the quantity whose small-q
        directional limit is the desired nonanalytic response.
        """
        return self.desired_dynamical(q) - self.interpolated_difference_dynamical(q)

    def double_counting_audit(self, pattern, generic) -> Dict[str, float]:
        """Actions probing the masking and blind-stacking failure modes.

        ``pattern`` is a primitive or matching-cell matching-periodic pattern;
        ``generic`` is a target-cell distortion with off-grid content.  The
        returned values are max-abs force-scale actions (eV/angstrom before
        the sign of ``-DeltaPhi u`` is applied).
        """
        tiled = self.matching_periodic_tiling(pattern)
        generic = self._check_u(generic)
        mismatch = self.existing_interpolation_mismatch
        return {
            "contract_a_tiled_action": float(np.abs(self.correction_matrix @ tiled).max()),
            "contract_b_tiled_action": float(np.abs(self.contract_b_matrix @ tiled).max()),
            "naive_desired_tiled_action": float(np.abs(self.desired_matrix @ tiled).max()),
            "existing_mismatch_tiled_action": float(np.abs(mismatch @ tiled).max()),
            "existing_mismatch_generic_action": float(np.abs(mismatch @ generic).max()),
            "contract_a_generic_action": float(np.abs(self.correction_matrix @ generic).max()),
        }


def _call_wannier_provider(
    provider: Callable,
    cell: np.ndarray,
    ion_positions: np.ndarray,
    ion_species: Tuple[str, ...],
    wannier_positions: np.ndarray,
    wannier_parents: np.ndarray,
    *,
    n_wc: int,
    n_ion: int,
    translation_tolerance: float,
) -> WannierResponse:
    if provider is None or not callable(provider):
        raise ExistingTailError(
            "get_wannier_response must be a callable returning a WannierResponse; "
            "the full ionic/Wannier-centroid response including the curvature "
            "d^2W/du^2 is required (frozen/first-order-only responses are not accepted)")
    response = provider(cell, ion_positions, ion_species, wannier_positions, wannier_parents)
    if response is None:
        raise ExistingTailError(
            "get_wannier_response returned None; a declared full Wannier-centroid "
            "response is required")
    if not isinstance(response, WannierResponse):
        raise ExistingTailError(
            "get_wannier_response must return a WannierResponse carrying its provenance, "
            "method and required second_derivative; raw arrays are not an eligible "
            "declared response")
    response.check_for(n_wc, n_ion, translation_tolerance)
    return response


def build_existing_tail_correction(
    declaration: DPLRTailDeclaration,
    response: ElectricalResponse,
    matching_reps,
    target_reps,
    get_wannier_response: Callable,
    *,
    ft_sign: int = 1,
    eta: Optional[float] = None,
    nreal: int = 5,
    nrecip: int = 8,
    wannier_translation_tolerance: float = _DEFAULT_WANNIER_TRANSLATION_TOL,
    phi0_symmetry_tol: Optional[float] = None,
) -> ExistingTailCorrection:
    """Build the approved Contract A existing-tail replacement.

    Parameters
    ----------
    declaration : validated DPLR tail metadata and reference positions.
    response : declared Born/dielectric response for the same reference
        (tin-foil boundary required).
    matching_reps, target_reps : positive diagonal integer repetitions of the
        declared primitive cell.
    get_wannier_response : callable
        ``(cell, ion_positions, ion_species, wannier_positions,
        wannier_parents) -> WannierResponse`` evaluated for the matching and
        target cells.  It must return the full centroid Jacobian and the
        required curvature ``d^2W/du^2`` for that cell; there is no fallback.
    ft_sign : Fourier synthesis sign of the base IFC data (+1 or -1).
    eta, nreal, nrecip : shared numerical Ewald settings for the desired
        response, matching ``longrangefy.matched.build_correction``.
    wannier_translation_tolerance : absolute bound on
        ``max|sum_i dW_n/du_i - I|`` for the declared response.
    phi0_symmetry_tol : optional declared symmetry tolerance forwarded to the
        base builder for the existing-tail samples.

    The matching and target existing Hessians must be translation periodic:
    ``max|H - fold_unfold(H)|`` is checked against a relative roundoff bound,
    and a larger residual is rejected rather than silently averaged by
    :func:`dense_to_blocks`.  There is no override: a non-periodic declared
    Wannier response cannot produce a matching-preserving Contract A
    correction.

    The returned correction is added exactly once to the unchanged original
    model, which must re-apply its own declared tail per cell.
    """
    if not isinstance(declaration, DPLRTailDeclaration):
        raise ExistingTailError("declaration must be a DPLRTailDeclaration")
    if not isinstance(response, ElectricalResponse):
        raise ExistingTailError("response must be an ElectricalResponse with provenance")
    try:
        response.check_natom(declaration.natom)
    except ValueError as exc:
        raise ExistingTailError(str(exc)) from exc
    if not _is_tin_foil(response.provenance.boundary_convention):
        raise ExistingTailError(
            "desired response boundary_convention must be tin-foil/conducting; "
            f"got {response.provenance.boundary_convention!r} (a mismatch is a rejection, "
            "not a silent conversion)")
    reps_m = _as_reps(matching_reps, "matching_reps")
    reps_l = _as_reps(target_reps, "target_reps")
    sign = int(ft_sign)
    if sign not in (1, -1):
        raise ExistingTailError("ft_sign must be +1 or -1")
    cell = declaration.reference_cell
    positions = declaration.reference_positions
    born = np.asarray(response.born_charges, dtype=float)
    dielectric = np.asarray(response.dielectric, dtype=float)
    shared_eta = float(eta) if eta is not None else _default_eta(cell)
    if not np.isfinite(shared_eta) or shared_eta <= 0.0:
        raise ExistingTailError("eta must be finite and positive")
    for name, value in (("nreal", nreal), ("nrecip", nrecip)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
                or value < 0:
            raise ExistingTailError(f"{name} must be a nonnegative integer")
    tolerance = float(wannier_translation_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ExistingTailError("wannier_translation_tolerance must be finite and nonnegative")

    tail = DPLRTailHessian(declaration)
    cell_m, ions_m, species_m, wc_m, parents_m, _ = _tile_system(declaration, reps_m)
    cell_l, ions_l, species_l, wc_l, parents_l, _ = _tile_system(declaration, reps_l)
    n_wc_m = len(parents_m)
    n_wc_l = len(parents_l)

    response_m = _call_wannier_provider(
        get_wannier_response, cell_m, ions_m, species_m, wc_m, parents_m,
        n_wc=n_wc_m, n_ion=ions_m.shape[0], translation_tolerance=tolerance)
    response_l = _call_wannier_provider(
        get_wannier_response, cell_l, ions_l, species_l, wc_l, parents_l,
        n_wc=n_wc_l, n_ion=ions_l.shape[0], translation_tolerance=tolerance)

    matching_report: Dict[str, object] = {}
    target_report: Dict[str, object] = {}
    existing_matching = tail.hessian(
        cell_m, ions_m, species_m, wc_m, parents_m, response_m,
        translation_tolerance=tolerance, report=matching_report)
    existing_target = tail.hessian(
        cell_l, ions_l, species_l, wc_l, parents_l, response_l,
        translation_tolerance=tolerance, report=target_report)

    # The fold is admissible only as a roundoff-level identity.  A
    # non-periodic Hessian is rejected, never silently projected.
    matching_blocks = dense_to_blocks(existing_matching, reps_m)
    matching_periodicity = float(np.abs(
        blocks_to_dense(matching_blocks, reps_m) - existing_matching).max())
    matching_scale = max(float(np.abs(existing_matching).max()), 1e-300)
    target_periodicity = float(np.abs(
        blocks_to_dense(dense_to_blocks(existing_target, reps_l), reps_l)
        - existing_target).max())
    target_scale = max(float(np.abs(existing_target).max()), 1e-300)
    matching_limit = _PERIODICITY_ROUNDOFF * matching_scale
    target_limit = _PERIODICITY_ROUNDOFF * target_scale
    if matching_periodicity > matching_limit:
        raise ExistingTailError(
            "non-periodic matching-cell existing-tail Hessian from the declared "
            f"Wannier response: max|H - fold_unfold(H)| = {matching_periodicity:.3e} "
            f"eV/angstrom**2 exceeds the roundoff bound {matching_limit:.3e}. "
            "dense_to_blocks would silently average the non-periodic part; the "
            "declared Wannier response must be translation periodic over the "
            "matching cell for Contract A to preserve matching-periodic patterns.")
    if target_periodicity > target_limit:
        raise ExistingTailError(
            "non-periodic target-cell existing-tail Hessian from the declared "
            f"Wannier response: max|H - fold_unfold(H)| = {target_periodicity:.3e} "
            f"eV/angstrom**2 exceeds the roundoff bound {target_limit:.3e}. "
            "dense_to_blocks would silently average the non-periodic part; the "
            "declared Wannier response must be translation periodic over the "
            "target cell for Contract A to preserve matching-periodic patterns.")
    existing_samples = blocks_to_qmatrices(matching_blocks)
    grid_h = _grid_h(reps_m)
    q_match = grid_h.astype(float) / np.asarray(reps_m, dtype=float)
    desired_samples = np.stack([
        dipole_ifc(cell, positions, born, dielectric, q, eta=shared_eta,
                   nreal=int(nreal), nrecip=int(nrecip),
                   neutrality_tolerance=response.neutrality_tolerance)
        for q in q_match
    ])
    if sign == -1:
        existing_samples = np.conj(existing_samples)
        desired_samples = np.conj(desired_samples)

    provenance = response.validation_report()["provenance"]
    desired_carrier = build_correction(
        cell, positions, born, dielectric, reps_m, desired_samples, grid_h, reps_l,
        ft_sign=sign, eta=shared_eta, nreal=int(nreal), nrecip=int(nrecip),
        neutrality_tolerance=response.neutrality_tolerance,
        species=declaration.reference_species,
        phi0_symmetry_tol=phi0_symmetry_tol,
        response_provenance=provenance)
    existing_carrier = build_correction(
        cell, positions, born, dielectric, reps_m, existing_samples, grid_h, reps_l,
        ft_sign=sign, eta=shared_eta, nreal=int(nreal), nrecip=int(nrecip),
        neutrality_tolerance=response.neutrality_tolerance,
        species=declaration.reference_species,
        phi0_symmetry_tol=phi0_symmetry_tol,
        response_provenance=provenance)

    desired_matrix = desired_carrier.split_matrix          # D_L
    interpolated_desired = desired_carrier.unsplit_matrix  # I_Q[D]
    interpolated_existing = existing_carrier.unsplit_matrix  # I_Q[E]
    correction = (desired_carrier.correction_matrix
                  - (existing_target - interpolated_existing))
    correction = 0.5 * (correction + correction.T)
    asymmetry = float(np.abs(correction - correction.T).max())
    contract_b = desired_matrix - existing_target
    difference = interpolated_desired - interpolated_existing
    algebra = float(np.abs(correction - (contract_b - difference)).max())
    asr = float(np.abs(correction.reshape(
        correction.shape[0] // 3, 3, correction.shape[0] // 3, 3).sum(axis=2)).max())

    diagnostics = {
        "correction_matrix_max_abs": float(np.abs(correction).max()),
        "desired_matrix_max_abs": float(np.abs(desired_matrix).max()),
        "existing_matrix_max_abs": float(np.abs(existing_target).max()),
        "interpolated_desired_max_abs": float(np.abs(interpolated_desired).max()),
        "interpolated_existing_max_abs": float(np.abs(interpolated_existing).max()),
        "contract_a_algebra_residual": algebra,
        "symmetry_residual": asymmetry,
        "asr_row_sum_max_abs": asr,
        "wannier_translation_covariance_matching": response_m.translation_covariance_residual(),
        "wannier_translation_covariance_target": response_l.translation_covariance_residual(),
        "wannier_jacobian_max_abs_matching": response_m.max_abs,
        "wannier_jacobian_max_abs_target": response_l.max_abs,
        "wannier_second_derivative_max_abs_matching": response_m.second_derivative_max_abs,
        "wannier_second_derivative_max_abs_target": response_l.second_derivative_max_abs,
        "wc_gradient_max_abs_matching": float(
            matching_report.get("wc_gradient_max_abs", 0.0)),
        "wc_gradient_max_abs_target": float(
            target_report.get("wc_gradient_max_abs", 0.0)),
        "ewald_eta": shared_eta,
        "matching_grid_points": int(np.prod(reps_m)),
        "matching_periodicity_residual": matching_periodicity,
        "matching_periodicity_scale": matching_scale,
        "matching_periodicity_limit": matching_limit,
        "target_periodicity_residual": target_periodicity,
        "target_periodicity_scale": target_scale,
        "target_periodicity_limit": target_limit,
    }
    if reps_m == reps_l:
        diagnostics["matching_cell_correction_max_abs"] = float(np.abs(correction).max())
    if existing_carrier.tiled_compatible():
        primitive_pattern = np.linspace(0.011, 0.023, 3 * declaration.natom)
        tiled_pattern = existing_carrier.matching_periodic_tiling(primitive_pattern)
        diagnostics["tiled_pattern_action_max_abs"] = float(
            np.abs(correction @ tiled_pattern).max())

    metadata = {
        "units": "angstrom, eV (whole cell), eV/angstrom, eV/angstrom**2",
        "ifc_axes": "(i, mu, j, nu) atom-major Cartesian, unmassweighted",
        "site_order": desired_carrier.metadata.get("site_order"),
        "fourier_convention": desired_carrier.metadata.get("fourier_convention"),
        "boundary": "tin-foil (conducting); shared by the DPLR tail and desired response",
        "contract": "A: DeltaPhi = (D_L - E_L) - I_Q[D - E]_L",
        "interpolation": desired_carrier.metadata.get("interpolation"),
        "ewald": {
            "eta": shared_eta,
            "nreal": int(nreal),
            "nrecip": int(nrecip),
            "neutrality_tolerance": float(response.neutrality_tolerance),
        },
        "existing_tail": declaration.to_dict(),
        "desired_response": response.validation_report(),
        "full_response_semantics": (
            "first-order complete (not frozen/partial); the required "
            "second_derivative supplies the exact centroid-curvature term"),
        "periodicity": {
            "matching_residual": matching_periodicity,
            "matching_limit": matching_limit,
            "target_residual": target_periodicity,
            "target_limit": target_limit,
        },
        "wannier_response": {
            "matching": {
                "provenance": response_m.provenance,
                "method": response_m.method,
                "second_derivative_max_abs": response_m.second_derivative_max_abs,
                "translation_covariance_residual":
                    response_m.translation_covariance_residual(),
                "second_order_translation_covariance_residual":
                    response_m.translation_covariance_second_order_residual(),
                "wc_gradient_max_abs": matching_report.get("wc_gradient_max_abs"),
            },
            "target": {
                "provenance": response_l.provenance,
                "method": response_l.method,
                "second_derivative_max_abs": response_l.second_derivative_max_abs,
                "translation_covariance_residual":
                    response_l.translation_covariance_residual(),
                "second_order_translation_covariance_residual":
                    response_l.translation_covariance_second_order_residual(),
                "wc_gradient_max_abs": target_report.get("wc_gradient_max_abs"),
            },
        },
        "ft_sign": sign,
        "model_specific_fallback": False,
    }
    return ExistingTailCorrection(
        carrier=desired_carrier,
        existing_carrier=existing_carrier,
        correction_matrix=correction,
        desired_matrix=desired_matrix,
        existing_matrix=existing_target,
        interpolated_desired=interpolated_desired,
        interpolated_existing=interpolated_existing,
        diagnostics=diagnostics,
        metadata=metadata,
    )
