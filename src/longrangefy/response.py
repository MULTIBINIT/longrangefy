"""Electrical response inputs: Born effective charges and dielectric tensor.

The imposed dipolar asymptotic response is exactly the one specified by these
externally supplied tensors (R03).  They are **never** inferred from forces:
no API here or elsewhere in the package derives them from a calculator.

Units and index conventions (recorded in :class:`ResponseProvenance`):

- ``Z[i, alpha, beta] = dP_alpha / du_beta`` with displacement ``u`` in
  Angstrom and dipole ``P`` in ``e * Angstrom`` (elementary charge units).
- ``epsilon[i, j]`` is the electronic (high-frequency) dielectric tensor,
  dimensionless, in Cartesian axes of the reference cell.
- The initial correction supports only a declared tin-foil/conducting
  boundary; incompatible conventions are rejected.

Validation reports the neutrality residual and dielectric eigenvalues with
results; incompatible tensors are rejected, never silently altered.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = [
    "ResponseValidationError",
    "ResponseProvenance",
    "ElectricalResponse",
]

#: Default neutrality tolerance on ``sum_i Z[i]`` in elementary charge (e).
DEFAULT_NEUTRALITY_TOLERANCE = 1.0e-4
#: Default tolerance for treating the supplied dielectric tensor as symmetric.
DEFAULT_DIELECTRIC_SYMMETRY_TOLERANCE = 1.0e-8
#: Smallest admissible dielectric eigenvalue (dimensionless).
DEFAULT_DIELECTRIC_MIN_EIGENVALUE = 1.0e-6


class ResponseValidationError(ValueError):
    """Malformed or physically incompatible electrical response tensors."""


@dataclass(frozen=True)
class ResponseProvenance:
    """Recorded source, conventions, and boundary of the supplied response (R03)."""

    source: str
    method: str
    reference_description: str
    index_convention: str = "Z[i, alpha, beta] = dP_alpha / du_beta; u in angstrom, P in e*angstrom"
    boundary_convention: str = "tin-foil (conducting)"
    gamma_handling: str = "conducting-boundary Gamma; no directional LO term substituted at q=0"
    extracted_by: str = ""

    def __post_init__(self) -> None:
        for name in ("source", "method", "reference_description"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ResponseValidationError(
                    f"ResponseProvenance.{name} must be a non-empty description")
        if not isinstance(self.boundary_convention, str) or self.boundary_convention.strip().lower() not in {
            "tin-foil", "tin foil", "conducting", "tin-foil (conducting)",
            "tin foil (conducting)", "conducting (tin-foil)", "conducting (tin foil)",
        }:
            raise ResponseValidationError(
                "boundary_convention must explicitly declare tin-foil/conducting; "
                "other electrical boundaries are unsupported")


@dataclass(frozen=True)
class ElectricalResponse:
    """Fixed Born effective charges and electronic dielectric tensor."""

    born_charges: np.ndarray            # (n, 3, 3) in elementary charge units
    dielectric: np.ndarray              # (3, 3) dimensionless, symmetric positive definite
    provenance: ResponseProvenance
    neutrality_tolerance: float = DEFAULT_NEUTRALITY_TOLERANCE
    symmetry_tolerance: float = DEFAULT_DIELECTRIC_SYMMETRY_TOLERANCE
    min_dielectric_eigenvalue: float = DEFAULT_DIELECTRIC_MIN_EIGENVALUE

    def __post_init__(self) -> None:
        # Own copies: frozen tensors must not alias (or mutate) caller arrays.
        born = np.array(self.born_charges, dtype=float, copy=True)
        if born.ndim != 3 or born.shape[1:] != (3, 3) or born.shape[0] == 0:
            raise ResponseValidationError(
                f"born_charges must have shape (n, 3, 3) with n > 0, got {born.shape}")
        eps = np.array(self.dielectric, dtype=float, copy=True)
        if eps.shape != (3, 3):
            raise ResponseValidationError(
                f"dielectric must have shape (3, 3), got {eps.shape}")
        if not np.isfinite(born).all():
            raise ResponseValidationError("born_charges contain non-finite values")
        if not np.isfinite(eps).all():
            raise ResponseValidationError("dielectric contains non-finite values")
        asymmetry = float(np.abs(eps - eps.T).max())
        if asymmetry > self.symmetry_tolerance:
            raise ResponseValidationError(
                f"dielectric tensor is not symmetric within "
                f"{self.symmetry_tolerance:.1e}: max|eps - eps^T| = {asymmetry:.3e}; "
                "supply a symmetric tensor instead of relying on silent symmetrization")
        eigenvalues = np.linalg.eigvalsh(0.5 * (eps + eps.T))
        if float(eigenvalues.min()) <= self.min_dielectric_eigenvalue:
            raise ResponseValidationError(
                f"dielectric tensor is not positive definite (min eigenvalue "
                f"{eigenvalues.min():.6f} <= {self.min_dielectric_eigenvalue:.1e})")
        neutrality = np.abs(born.sum(axis=0)).max()
        if neutrality > self.neutrality_tolerance:
            raise ResponseValidationError(
                f"Born charges violate charge neutrality: max |sum_i Z[i]| = "
                f"{neutrality:.6f} e exceeds tolerance {self.neutrality_tolerance:.1e} e; "
                "report or repair the tensors at the source, do not use them as supplied")
        born.setflags(write=False)
        eps.setflags(write=False)
        object.__setattr__(self, "born_charges", born)
        object.__setattr__(self, "dielectric", eps)

    @property
    def natom(self) -> int:
        return int(self.born_charges.shape[0])

    def validation_report(self) -> dict:
        """Declared tolerance checks and measured residuals (R13)."""
        neutrality = self.born_charges.sum(axis=0)
        eigenvalues = np.linalg.eigvalsh(self.dielectric)
        return {
            "natom": self.natom,
            "born_units": "elementary charge (P in e*angstrom, u in angstrom)",
            "dielectric_units": "dimensionless electronic (high-frequency) response",
            "max_neutrality_residual_e": float(np.abs(neutrality).max()),
            "neutrality_residual_matrix_e": neutrality.tolist(),
            "neutrality_tolerance_e": float(self.neutrality_tolerance),
            "dielectric_symmetry_residual": float(np.abs(self.dielectric - self.dielectric.T).max()),
            "dielectric_symmetry_tolerance": float(self.symmetry_tolerance),
            "dielectric_eigenvalues": eigenvalues.tolist(),
            "dielectric_min_eigenvalue_threshold": float(self.min_dielectric_eigenvalue),
            "provenance": {
                "source": self.provenance.source,
                "method": self.provenance.method,
                "reference_description": self.provenance.reference_description,
                "index_convention": self.provenance.index_convention,
                "boundary_convention": self.provenance.boundary_convention,
                "gamma_handling": self.provenance.gamma_handling,
                "extracted_by": self.provenance.extracted_by,
            },
        }

    def check_natom(self, natom: int) -> None:
        """Fail explicitly when the response does not match the reference sites."""
        if natom != self.natom:
            raise ResponseValidationError(
                f"response supplies {self.natom} Born charge tensors but the "
                f"reference structure has {natom} sites; the response must "
                "correspond to the declared reference (R03)")
