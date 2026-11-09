"""Workflow: original-model reference harmonics on the matching grid (ticket 01).

Builds, from one conservative force callable and the declared reference:

1. residual forces of the original model at the reference,
2. the dense matching-supercell Hessian by force finite differences,
3. translation-resolved IFC blocks on the matching-cell R range,
4. matching-grid q matrices ``Phi_0(q)`` (the input of the correction's
   common interpolation operator),
5. Wigner-Seitz image weights and the on-grid ``I_Q`` sample-reproducing
   identity check,
6. recorded conventions and diagnostics with units.

Electrical response tensors are validated and carried as provenance here;
their dipolar matrices belong to the correction stage, which must reuse
``Phi_0(q)`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .finitediff import FiniteDifferenceResult, STENCILS, estimate_force_noise, finite_difference_hessian
from .ifc import (
    SupercellIFC,
    apply_asr_simple,
    apply_asr_weighted,
    blocks_asr_row_sums,
    blocks_to_qmatrices,
    dense_to_blocks,
    extend_blocks,
    fold_ws_weights,
    interpolation_rpoints,
    qmatrices_to_blocks,
    wigner_seitz_weights,
)
from .reference import ReferenceStructure, SupercellMapping
from .response import ElectricalResponse

__all__ = ["HarmonicsError", "ReferenceHarmonics", "build_reference_harmonics"]

#: Accepted acoustic-sum-rule conventions (recorded, applied identically to
#: both branches downstream).
ASR_OPTIONS = ("none", "weighted", "simple")


class HarmonicsError(ValueError):
    """Invalid harmonics workflow inputs or failed internal identity checks."""


def _conventions(asr: str, stencil: str, displacement: float, ws_toldist: float) -> dict:
    return {
        "length_unit": "angstrom",
        "energy_unit": "eV (whole periodic cell)",
        "force_unit": "eV/angstrom",
        "ifc_unit": "eV/angstrom^2 (unmassweighted)",
        "site_order": "cell-major; flat index = ((c1*n2+c2)*n3+c3)*n_uc + site",
        "hessian_index": "atom-major Cartesian; flat = 3*atom + cartesian",
        "lattice": "lattice vectors are rows of cell; cartesian = fractional @ cell",
        "fourier_forward": "Phi(q) = sum_R exp(+2*pi*i*q.R) Phi(R); q fractional reciprocal; no Bloch site phases",
        "fourier_inverse": "Phi(R) = (1/nq) sum_q exp(-2*pi*i*q.R) Phi(q)",
        "image_weights": "ABINIT wght9 (brav=1) Wigner-Seitz boundary weights, folded into extended IFCs",
        "ws_toldist_angstrom2": float(ws_toldist),
        "asr": str(asr),
        "finite_difference_stencil": str(stencil),
        "finite_difference_displacement_angstrom": float(displacement),
    }


@dataclass(frozen=True)
class ReferenceHarmonics:
    """Original-model harmonic data on the matching grid (R01/R04/R05)."""

    reference: ReferenceStructure
    response: ElectricalResponse
    matching: SupercellMapping
    dense_hessian: np.ndarray            # (3*N_m, 3*N_m) eV/angstrom^2
    blocks: SupercellIFC                 # matching-cell R range
    q_matrices: np.ndarray               # (nq, n_uc, 3, n_uc, 3) complex
    grid_h: np.ndarray                   # (nq, 3) integer q = h / reps
    ws_rpoints: np.ndarray               # (n_ws, 3) interpolation-range R points
    ws_weights: np.ndarray               # (n_uc, n_uc, n_ws) WS weights on that range
    folded_ws_weights: np.ndarray        # (n_uc, n_uc, nrpt) folded onto the matching domain
    conventions: Mapping[str, object]
    calculator_identity: Mapping[str, object]
    diagnostics: Mapping[str, object]
    finite_difference: FiniteDifferenceResult

    @property
    def natom_supercell(self) -> int:
        return int(self.dense_hessian.shape[0] // 3)


def build_reference_harmonics(
    reference: ReferenceStructure,
    response: ElectricalResponse,
    matching_reps,
    get_forces: Callable[[np.ndarray], np.ndarray],
    *,
    displacement: float = 0.01,
    stencil: str = "central",
    asr: str = "none",
    force_noise_repeats: int = 2,
    fresh_get_forces: Callable[[np.ndarray], np.ndarray] | None = None,
    ws_toldist: float = 1.0e-8,
    calculator_identity: Mapping[str, object] | None = None,
) -> ReferenceHarmonics:
    """Build the original potential's matching-grid harmonics (spec section 5).

    ``get_forces`` receives Cartesian positions ``(N_m, 3)`` of the matching
    supercell in cell-major order and returns forces ``(N_m, 3)`` in
    eV/Angstrom; the cell, species, and site assignment are fixed by
    ``reference`` and ``matching_reps``.  No relaxation, projection, or
    symmetrization is applied beyond the explicitly requested ``asr``
    convention, which is recorded.

    ``fresh_get_forces`` is the provider's uncached evaluation callback
    (``AseForceProvider.get_fresh_forces`` / ``DeepMDForceProvider.``
    ``get_fresh_forces``).  Pass it for caching providers so the force-noise
    study measures fresh runtime variability instead of one cached result.
    """
    if asr not in ASR_OPTIONS:
        raise HarmonicsError(f"asr must be one of {ASR_OPTIONS}, got {asr!r}")
    if stencil not in STENCILS:
        raise HarmonicsError(f"stencil must be one of {sorted(STENCILS)}, got {stencil!r}")
    response.check_natom(reference.natom)
    matching = SupercellMapping(reference, matching_reps)
    positions = matching.cart_positions_supercell()

    # Residual forces of the unchanged original model at the reference (R02).
    # Own the array: later force calls may reuse a provider output buffer.
    forces0 = np.array(get_forces(positions), dtype=float, copy=True)
    if forces0.shape != (matching.natom_supercell, 3) or not np.isfinite(forces0).all():
        raise HarmonicsError(
            "get_forces must return finite forces of shape "
            f"({matching.natom_supercell}, 3) at the reference positions")

    noise = estimate_force_noise(
        get_forces, positions, n_repeats=force_noise_repeats,
        fresh_forces=fresh_get_forces)
    fd = finite_difference_hessian(
        get_forces, positions, displacement=displacement, stencil=stencil)

    blocks = dense_to_blocks(fd.hessian, matching.reps)
    ws_rpoints = interpolation_rpoints(matching.reps)
    ws_weights = wigner_seitz_weights(
        ws_rpoints, reference.cell,
        reference.cart_positions, matching.reps, toldist=ws_toldist)
    folded_weights = fold_ws_weights(ws_weights, ws_rpoints, matching.reps)
    if asr == "weighted":
        blocks = apply_asr_weighted(blocks, folded_weights)
    elif asr == "simple":
        blocks = apply_asr_simple(blocks)

    q_matrices = blocks_to_qmatrices(blocks)
    grid_h = np.asarray(
        [(m1, m2, m3)
         for m1 in range(matching.reps[0])
         for m2 in range(matching.reps[1])
         for m3 in range(matching.reps[2])], dtype=int)

    # Round trip and on-grid I_Q identity checks (recorded, asserted).
    roundtrip = qmatrices_to_blocks(q_matrices, matching.reps)
    q_roundtrip_max = float(np.abs(roundtrip.blocks - blocks.blocks).max())
    extended = extend_blocks(blocks, ws_weights, ws_rpoints, matching.reps)
    q_extended = blocks_to_qmatrices(extended)
    iq_identity_max = float(np.abs(q_extended - q_matrices).max())
    identity_tolerance = 1e-8  # floating-point round trip of O(1) eV/angstrom^2 data
    if q_roundtrip_max > identity_tolerance or iq_identity_max > identity_tolerance:
        raise HarmonicsError(
            "matching-grid identity failed: q round-trip deviation "
            f"{q_roundtrip_max:.3e}, I_Q sample-reproduction deviation "
            f"{iq_identity_max:.3e} (tolerance {identity_tolerance:.1e}); "
            "the common interpolation operator must reproduce its input samples")

    diagnostics = {
        "residual_forces_max_ev_per_angstrom": float(np.abs(forces0).max()),
        "residual_forces_rms_ev_per_angstrom": float(np.sqrt((forces0 ** 2).mean())),
        "force_noise": noise,
        "force_noise_fresh_evaluations": noise["fresh_evaluations"],
        "reciprocity_abs_ev_per_angstrom2": fd.diagnostics["reciprocity_abs_ev_per_angstrom2"],
        "reciprocity_rel": fd.diagnostics["reciprocity_rel"],
        "asr_row_sum_abs_ev_per_angstrom2": blocks_asr_row_sums(blocks),
        "asr_row_sum_weighted_abs_ev_per_angstrom2":
            blocks_asr_row_sums(blocks, folded_weights),
        "asr_convention_applied": asr,
        "q_roundtrip_max_abs_ev_per_angstrom2": q_roundtrip_max,
        "iq_on_grid_identity_max_abs_ev_per_angstrom2": iq_identity_max,
        "iq_identity_tolerance": identity_tolerance,
        "n_force_calls": fd.n_force_calls,
        "matching_grid_qpoints": int(q_matrices.shape[0]),
        "reference_symmetry_label": reference.symmetry_label,
        "reference_source": reference.source,
        "response_provenance": response.validation_report()["provenance"],
    }
    return ReferenceHarmonics(
        reference=reference,
        response=response,
        matching=matching,
        dense_hessian=fd.hessian,
        blocks=blocks,
        q_matrices=q_matrices,
        grid_h=grid_h,
        ws_rpoints=ws_rpoints,
        ws_weights=ws_weights,
        folded_ws_weights=folded_weights,
        conventions=_conventions(asr, stencil, displacement, ws_toldist),
        calculator_identity=dict(calculator_identity or {}),
        diagnostics=diagnostics,
        finite_difference=fd,
    )
