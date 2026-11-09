"""Ticket 11 behaviour tests: declared DPLR existing-tail replacement (Contract A).

Scope: the independent array core in :mod:`longrangefy.existing_tail` plus the
base :mod:`longrangefy.matched` interpolation/correction machinery.  Nothing
here imports ASE or DeepMD except the optional runtime cross-check, which is
skipped when the TensorFlow DPLR runtime is absent.  The fixtures are
synthetic declared DPLR tails; they are not a production NaCl model and imply
nothing about model accuracy.

The tests assert the identities the design note predeclares:
* the DPLR functional and its exact Hessian (energy/force finite differences,
  symmetry, acoustic sum rule, translation invariance);
* the full ionic + Wannier-centroid response including the curvature
  ``d^2W/du^2`` gradient contraction (a frozen or first-order-only centroid is
  rejected and is demonstrably a different Hessian);
* Contract A algebra ``(D - E) - I_Q[D - E]`` and matching-grid/tiled
  preservation, with the Contract B and naive-stack contrasts;
* the masking matrix ``E - I_Q[E]`` (zero on tiled patterns, nonzero off grid);
* the fold/unfold periodicity identity for the existing Hessian, with a
  non-periodic Wannier response rejected beyond numerical roundoff (there is
  no tolerance override that would permit a nonpreserving matching cell);
* metadata rejection, missing/partial response rejection, immutable copied
  charge maps and the unchanged base no-tail workflow.

Tolerances are numerical-identity tolerances, not material tolerances.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from longrangefy.existing_tail import (
    DPLRTailDeclaration,
    DPLRTailHessian,
    ExistingTailError,
    WannierResponse,
    build_existing_tail_correction,
    gaussian_tail_energy,
    gaussian_tail_gradient,
    gaussian_tail_hessian,
)
from longrangefy.matched import build_correction
from longrangefy.reference import ReferenceStructure, SupercellMapping
from longrangefy.response import (
    ElectricalResponse,
    ResponseProvenance,
    ResponseValidationError,
)


# --------------------------------------------------------------------------
# declared synthetic DPLR fixture
# --------------------------------------------------------------------------
def _sha(char: str) -> str:
    return char * 64


def _zero_curvature(n_wc: int, n_ion: int) -> np.ndarray:
    return np.zeros((n_wc, 3, n_ion, 3, n_ion, 3))


def _declaration(**overrides) -> DPLRTailDeclaration:
    cell = np.diag([4.0, 4.0, 4.0])
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    positions = frac @ cell
    wc_positions = positions + np.array([[0.12, 0.0, 0.0], [-0.12, 0.0, 0.0]])
    base = dict(
        model_path="/models/nacl_dplr.pb",
        model_sha256=_sha("a"),
        deepmd_version="3.1.3",
        wannier_model_path="/models/naacl_dw.pb",
        wannier_model_sha256=_sha("b"),
        model_type="dplr",
        modifier_type="dipole_charge",
        ewald_beta=0.4,
        ewald_h=1.0,
        reciprocal_cutoff=6.0,
        boundary_convention="tin-foil (conducting)",
        regime="per-cell",
        regime_evidence=(
            "LAMMPS pppm/dplr and the DeepPot modifier path re-apply the declared "
            "electrostatic functional in every cell"),
        per_cell_tail_confirmed=True,
        periodic_insulator=True,
        reference_cell=cell,
        reference_positions=positions,
        reference_species=("Na", "Cl"),
        wannier_positions=wc_positions,
        wannier_species=("Na", "Cl"),
        sys_charge_map={"Na": 1.0, "Cl": -1.0},
        model_charge_map={"Na": -1.0, "Cl": 1.0},
        neutrality_tolerance=1.0e-6,
    )
    base.update(overrides)
    return DPLRTailDeclaration(**base)


def _response() -> ElectricalResponse:
    born = np.array([np.eye(3) * 1.1, -np.eye(3) * 1.1])
    provenance = ResponseProvenance(
        source="synthetic ticket-11 fixture",
        method="declared test tensors (not a first-principles or DeepMD response)",
        reference_description="two-site cubic reference; Born charges and dielectric are stand-ins",
        boundary_convention="tin-foil (conducting)",
    )
    return ElectricalResponse(
        born_charges=born,
        dielectric=np.eye(3) * 2.0,
        provenance=provenance,
        neutrality_tolerance=1.0e-6,
    )


def _provider(weight: float = 0.0, curvature: float = 0.0,
              curvature_all: bool = False):
    """Declared local WC response plus an optional relative-coordinate curvature.

    ``curvature`` adds a symmetric second derivative from
    ``0.25*kappa*sum_b (u_parent - u_b)^2`` along x.  It is translation
    covariant (the row sum over the second ion index is zero) and has a
    nonzero tail-gradient contraction, so it exercises the term that a
    first-order-only chain rule omits.  ``curvature_all=True`` applies the
    same relative-coordinate curvature to every WC, giving a translation
    periodic response for supercell reps; the default applies it only to WC 0,
    which is intentionally non-periodic when the matching cell is repeated.
    """

    def provider(cell, ion_positions, ion_species, wannier_positions, wannier_parents):
        n_ion = len(ion_positions)
        n_wc = len(wannier_positions)
        jacobian = np.zeros((n_wc, 3, n_ion, 3))
        second = np.zeros((n_wc, 3, n_ion, 3, n_ion, 3))
        for wc, parent in enumerate(wannier_parents):
            parent = int(parent)
            other = (parent + 1) % n_ion
            if other == parent:
                jacobian[wc, :, parent, :] = np.eye(3)
            else:
                jacobian[wc, :, parent, :] = (1.0 - weight) * np.eye(3)
                jacobian[wc, :, other, :] = weight * np.eye(3)
            # Curvature on one centroid only: the two-site fixture is
            # inversion-symmetric, so identical curvature on both centroids
            # would cancel the gradient contraction exactly.  curvature_all
            # is used only for the supercell periodicity check.
            if curvature and (curvature_all or wc == 0):
                for partner in range(n_ion):
                    if partner == parent:
                        continue
                    second[wc, 0, parent, 0, parent, 0] += 0.5 * curvature
                    second[wc, 0, partner, 0, partner, 0] += 0.5 * curvature
                    second[wc, 0, parent, 0, partner, 0] -= 0.5 * curvature
                    second[wc, 0, partner, 0, parent, 0] -= 0.5 * curvature
        method = "linearised WC response"
        if curvature:
            method += " plus a declared relative-coordinate quadratic term"
        return WannierResponse(
            jacobian=jacobian,
            second_derivative=second,
            provenance="synthetic test provider (deterministic local response)",
            method=method,
        )

    return provider


def _build(declaration=None, response=None, matching=(2, 1, 1), target=(4, 1, 1),
           provider=None, **kwargs):
    return build_existing_tail_correction(
        declaration if declaration is not None else _declaration(),
        response if response is not None else _response(),
        matching,
        target,
        provider if provider is not None else _provider(),
        **kwargs,
    )


def _tiled_geometry(declaration, reps):
    cell = declaration.reference_cell
    reference = ReferenceStructure(
        cell=cell,
        frac_positions=declaration.reference_positions @ np.linalg.inv(cell),
        species=declaration.reference_species,
    )
    mapping = SupercellMapping(reference, reps)
    return mapping.supercell_cell, mapping.cart_positions_supercell(), mapping.species_supercell()


# --------------------------------------------------------------------------
# declared DPLR functional and Hessian
# --------------------------------------------------------------------------
def test_tail_energy_matches_deepmd_runtime_box_convention():
    """Cross-check the declared functional against DeePMD-kit's EwaldRecp.

    The runtime box is row-major in this package's row-vector cell convention;
    the skew case is included so a silent transpose cannot pass.
    """
    pytest.importorskip("tensorflow")
    ewald_recp = pytest.importorskip("deepmd.tf.infer.ewald_recp")
    cells = [
        np.diag([4.0, 3.7, 4.2]),
        np.array([[4.0, 0.3, 0.0], [0.0, 3.7, 0.2], [0.1, 0.0, 4.2]]),
    ]
    frac = np.array([[0.0, 0.0, 0.0], [0.31, 0.52, 0.19], [0.72, 0.18, 0.61]])
    charges = np.array([1.0, -1.3, 0.3])
    beta = 0.45
    runtime = ewald_recp.EwaldRecp(0.5, beta)
    for cell in cells:
        positions = frac @ cell
        ours = gaussian_tail_energy(
            cell, positions, charges, ewald_beta=beta, reciprocal_cutoff=4.0)
        reference = runtime.eval(
            positions.reshape(1, -1), charges.reshape(1, -1), cell.reshape(1, -1))[0][0]
        assert_allclose(ours, reference, rtol=1e-6, atol=1e-8)


def test_tail_hessian_matches_energy_finite_differences_and_invariants():
    # A skew cell is deliberate: an orthogonal cell cannot detect a
    # fractional/Cartesian index transpose in the Hessian transform.
    cell = np.array([[4.0, 0.3, 0.0], [0.0, 3.7, 0.2], [0.1, 0.0, 4.2]])
    frac = np.array([[0.0, 0.0, 0.0], [0.31, 0.52, 0.19], [0.72, 0.18, 0.61]])
    charges = np.array([1.0, -1.3, 0.3])
    beta, cutoff = 0.45, 4.0
    positions = frac @ cell
    hessian = gaussian_tail_hessian(
        cell, positions, charges, ewald_beta=beta, reciprocal_cutoff=cutoff)

    assert hessian.shape == (9, 9)
    assert_allclose(hessian, hessian.T, atol=1e-10)
    blocks = hessian.reshape(3, 3, 3, 3)
    assert_allclose(blocks.sum(axis=2), 0.0, atol=1e-10)  # exact ASR by construction

    # Mixed Cartesian finite differences of the energy reproduce the Hessian.
    step = 1e-4
    finite = np.zeros_like(hessian)

    def energy(r):
        return gaussian_tail_energy(
            cell, r, charges, ewald_beta=beta, reciprocal_cutoff=cutoff)

    for a in range(3):
        for mu in range(3):
            for b in range(3):
                for nu in range(3):
                    pp = positions.copy()
                    pm = positions.copy()
                    mp = positions.copy()
                    mm = positions.copy()
                    pp[a, mu] += step
                    pp[b, nu] += step
                    pm[a, mu] += step
                    pm[b, nu] -= step
                    mp[a, mu] -= step
                    mp[b, nu] += step
                    mm[a, mu] -= step
                    mm[b, nu] -= step
                    finite[3 * a + mu, 3 * b + nu] = (
                        energy(pp) - energy(pm) - energy(mp) + energy(mm)) / (4.0 * step ** 2)
    assert_allclose(hessian, finite, rtol=1e-5, atol=2e-6)

    # A global non-lattice translation changes every phase uniformly and
    # therefore leaves the Hessian unchanged.
    shifted = gaussian_tail_hessian(
        cell, positions + np.array([0.123, -0.045, 0.067]), charges,
        ewald_beta=beta, reciprocal_cutoff=cutoff)
    assert_allclose(shifted, hessian, atol=1e-10)


def test_tail_cutoff_convergence():
    cell = np.diag([4.0, 4.0, 4.0])
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    positions = frac @ cell
    charges = np.array([1.0, -1.0])
    coarse = gaussian_tail_hessian(
        cell, positions, charges, ewald_beta=0.4, reciprocal_cutoff=4.0)
    fine = gaussian_tail_hessian(
        cell, positions, charges, ewald_beta=0.4, reciprocal_cutoff=6.0)
    assert_allclose(coarse, fine, rtol=1e-6, atol=1e-5)


def test_tail_input_validation():
    cell = np.diag([4.0, 4.0, 4.0])
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    charges = np.array([1.0, -1.0])
    with pytest.raises(ExistingTailError, match="one finite value per position"):
        gaussian_tail_energy(cell, positions, charges[:-1],
                             ewald_beta=0.4, reciprocal_cutoff=4.0)
    with pytest.raises(ExistingTailError, match="ewald_beta"):
        gaussian_tail_energy(cell, positions, charges,
                             ewald_beta=0.0, reciprocal_cutoff=4.0)
    with pytest.raises(ExistingTailError, match="selects no"):
        gaussian_tail_energy(cell, positions, charges,
                             ewald_beta=0.4, reciprocal_cutoff=1e-9)
    with pytest.raises(ExistingTailError, match="nonsingular"):
        gaussian_tail_energy(np.zeros((3, 3)), positions, charges,
                             ewald_beta=0.4, reciprocal_cutoff=4.0)
    bad = positions.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ExistingTailError, match="finite"):
        gaussian_tail_energy(cell, bad, charges,
                             ewald_beta=0.4, reciprocal_cutoff=4.0)


# --------------------------------------------------------------------------
# metadata and declared response
# --------------------------------------------------------------------------
def test_declaration_records_identity_and_maps():
    declaration = _declaration()
    assert declaration.natom == 2
    assert declaration.n_wannier == 2
    assert declaration.wannier_parents == (0, 1)
    assert_allclose(declaration.wannier_offsets[0], [0.12, 0.0, 0.0])
    record = declaration.to_dict()
    json.dumps(record)
    assert record["model_type"] == "dplr"
    assert record["modifier_type"] == "dipole_charge"
    assert record["sys_charge_map"] == {"Na": 1.0, "Cl": -1.0}
    assert record["model_charge_map"] == {"Na": -1.0, "Cl": 1.0}


def test_declaration_charge_maps_are_immutable_and_copied():
    sys_map = {"Na": 1.0, "Cl": -1.0}
    wc_map = {"Na": -1.0, "Cl": 1.0}
    declaration = _declaration(sys_charge_map=sys_map, model_charge_map=wc_map)
    with pytest.raises(TypeError):
        declaration.sys_charge_map["Na"] = 0.0
    with pytest.raises(TypeError):
        declaration.model_charge_map["Na"] = 0.0
    # Post-construction mutation of the caller's mappings is also disconnected.
    sys_map["Na"] = 0.0
    wc_map["Na"] = 0.0
    assert declaration.sys_charge_map["Na"] == 1.0
    assert declaration.model_charge_map["Na"] == -1.0
    assert declaration.to_dict()["sys_charge_map"] == {"Na": 1.0, "Cl": -1.0}
    assert declaration.to_dict()["model_charge_map"] == {"Na": -1.0, "Cl": 1.0}


@pytest.mark.parametrize("overrides, match", [
    (dict(model_type="short-range"), "base matched fixed-cell workflow"),
    (dict(modifier_type="none"), "dipole_charge"),
    (dict(boundary_convention="vacuum"), "tin-foil"),
    (dict(regime="finite-range"), "per-cell"),
    (dict(per_cell_tail_confirmed=False), "per_cell_tail_confirmed"),
    (dict(regime_evidence=""), "regime_evidence"),
    (dict(periodic_insulator=False), "periodic_insulator"),
    (dict(ewald_beta=0.0), "ewald_beta"),
    (dict(ewald_h=0.0), "ewald_h"),
    (dict(reciprocal_cutoff=0.0), "reciprocal_cutoff"),
    (dict(model_sha256="not-a-sha"), "SHA-256"),
    (dict(sys_charge_map={"Na": 1.0}), "sys_charge_map"),
    (dict(model_charge_map={"Na": -1.0}), "model_charge_map"),
    (dict(sys_charge_map={"Na": 1.0, "Cl": -0.8}), "not neutral"),
    (dict(wannier_species=()), "wannier_species"),
    (dict(wannier_species=("K",)), "do not occur"),
    (dict(wannier_positions=np.zeros((1, 3))), "wannier_positions"),
    (dict(reference_cell=np.zeros((3, 3))), "nonsingular"),
    (dict(reference_species=("Na",)), "reference_species"),
    (dict(reference_positions=np.zeros((3, 3))), "reference_species"),
    (dict(neutrality_tolerance=-1.0), "neutrality_tolerance"),
])
def test_declaration_rejects_missing_or_incompatible_metadata(overrides, match):
    with pytest.raises(ExistingTailError, match=match):
        _declaration(**overrides)


def test_wannier_response_rejects_frozen_and_incomplete_responses():
    good = np.zeros((1, 3, 1, 3))
    good[0, :, 0, :] = np.eye(3)
    response = WannierResponse(
        jacobian=good, provenance="test", method="test linear response",
        second_derivative=_zero_curvature(1, 1))
    assert response.translation_covariance_residual() == 0.0

    with pytest.raises(ExistingTailError, match="zero|frozen"):
        WannierResponse(jacobian=np.zeros((1, 3, 1, 3)),
                        provenance="test", method="frozen",
                        second_derivative=_zero_curvature(1, 1))
    with pytest.raises(ExistingTailError, match="full_response"):
        WannierResponse(jacobian=good, provenance="test", method="partial",
                        second_derivative=_zero_curvature(1, 1),
                        full_response=False)
    with pytest.raises(ExistingTailError, match="provenance"):
        WannierResponse(jacobian=good, provenance="", method="test",
                        second_derivative=_zero_curvature(1, 1))
    with pytest.raises(ExistingTailError, match="shape"):
        WannierResponse(jacobian=np.zeros((1, 3, 1, 2)),
                        provenance="test", method="test",
                        second_derivative=_zero_curvature(1, 1))
    bad = good.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ExistingTailError, match="non-finite"):
        WannierResponse(jacobian=bad, provenance="test", method="test",
                        second_derivative=_zero_curvature(1, 1))

    # Translation covariance is an eligibility check, not a fitting tolerance.
    bad_covariance = good * 2.0
    response = WannierResponse(
        jacobian=bad_covariance, provenance="test", method="test",
        second_derivative=_zero_curvature(1, 1))
    with pytest.raises(ExistingTailError, match="translation covariance"):
        response.check_for(1, 1)

    # Every WC must carry a response; a partially frozen response is rejected.
    two_wc = np.zeros((2, 3, 1, 3))
    two_wc[1, :, 0, :] = np.eye(3)
    partial = WannierResponse(
        jacobian=two_wc, provenance="test", method="test",
        second_derivative=_zero_curvature(2, 1))
    with pytest.raises(ExistingTailError, match="zero response"):
        partial.check_for(2, 1)


def test_tail_hessian_uses_full_wannier_chain_rule():
    declaration = _declaration()
    cell = declaration.reference_cell
    ions = declaration.reference_positions
    species = declaration.reference_species
    wcs = declaration.wannier_positions
    parents = declaration.wannier_parents
    tail = DPLRTailHessian(declaration)
    response = _provider(weight=0.02, curvature=0.03)(cell, ions, species, wcs, parents)
    hessian = tail.hessian(cell, ions, species, wcs, parents, response)

    # Frozen-centroid evaluation is a different operator; it must not be
    # mistaken for the full response.
    q_ion, _, _ = tail.charges(species, parents)
    frozen = gaussian_tail_hessian(
        cell, ions, q_ion,
        ewald_beta=declaration.ewald_beta,
        reciprocal_cutoff=declaration.reciprocal_cutoff)
    assert np.max(np.abs(hessian - frozen)) > 1e-3

    # A predicted centroid is not generally stationary: the gradient
    # contraction g_W . d^2W/du^2 is part of the Hessian.
    _, grad_wc = tail.gradient(cell, ions, species, wcs, parents)
    assert np.max(np.abs(grad_wc)) > 1e-6
    n_ion = ions.shape[0]
    jac_flat = response.jacobian.reshape(3 * len(wcs), 3 * n_ion)
    c_flat = response.second_derivative.reshape(3 * len(wcs), 3 * n_ion, 3 * n_ion)
    curvature_term = np.einsum("a,aij->ij", grad_wc.reshape(-1), c_flat)
    assert np.max(np.abs(curvature_term)) > 1e-6

    def energy(u):
        r = ions + u.reshape(n_ion, 3)
        w = (wcs + (jac_flat @ u).reshape(-1, 3)
             + 0.5 * np.einsum("aij,i,j->a", c_flat, u, u).reshape(-1, 3))
        return tail.energy(cell, r, species, w, parents)

    step = 1e-4
    finite = np.zeros_like(hessian)
    u0 = np.zeros(3 * n_ion)
    for a in range(3 * n_ion):
        for b in range(3 * n_ion):
            pp = u0.copy()
            pm = u0.copy()
            mp = u0.copy()
            mm = u0.copy()
            pp[a] += step
            pp[b] += step
            pm[a] += step
            pm[b] -= step
            mp[a] -= step
            mp[b] += step
            mm[a] -= step
            mm[b] -= step
            finite[a, b] = (energy(pp) - energy(pm) - energy(mp) + energy(mm)) / (4 * step ** 2)
    assert_allclose(hessian, finite, rtol=1e-5, atol=2e-6)
    # The first-order-only Hessian is observably different; this is not a
    # tolerance-level distinction.
    assert np.max(np.abs((hessian - curvature_term) - finite)) > 1e-5
    assert_allclose(hessian, hessian.T, atol=1e-10)
    assert_allclose(hessian.reshape(n_ion, 3, n_ion, 3).sum(axis=2), 0.0, atol=1e-10)


def test_tail_gradient_matches_energy_finite_differences():
    cell = np.array([[4.0, 0.3, 0.0], [0.0, 3.7, 0.2], [0.1, 0.0, 4.2]])
    frac = np.array([[0.0, 0.0, 0.0], [0.31, 0.52, 0.19], [0.72, 0.18, 0.61]])
    charges = np.array([1.0, -1.3, 0.3])
    positions = frac @ cell
    beta, cutoff = 0.45, 4.0
    gradient = gaussian_tail_gradient(
        cell, positions, charges, ewald_beta=beta, reciprocal_cutoff=cutoff)
    step = 1e-6
    finite = np.zeros_like(gradient)
    for a in range(3):
        for mu in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[a, mu] += step
            minus[a, mu] -= step
            finite[a, mu] = (
                gaussian_tail_energy(
                    cell, plus, charges, ewald_beta=beta, reciprocal_cutoff=cutoff)
                - gaussian_tail_energy(
                    cell, minus, charges, ewald_beta=beta, reciprocal_cutoff=cutoff)
            ) / (2.0 * step)
    assert_allclose(gradient, finite, rtol=1e-6, atol=1e-7)
    assert_allclose(gradient.sum(axis=0), 0.0, atol=1e-10)


def test_wannier_response_second_order_requirements():
    declaration = _declaration()
    cell = declaration.reference_cell
    ions = declaration.reference_positions
    species = declaration.reference_species
    wcs = declaration.wannier_positions
    parents = declaration.wannier_parents
    base = _provider()(cell, ions, species, wcs, parents)
    n_wc, n_ion = base.jacobian.shape[0], base.jacobian.shape[2]

    with pytest.raises(TypeError):
        WannierResponse(
            jacobian=base.jacobian,
            provenance="test",
            method="missing curvature")
    with pytest.raises(ExistingTailError, match="second_derivative"):
        WannierResponse(
            jacobian=base.jacobian,
            provenance="test",
            method="null curvature",
            second_derivative=None)

    with pytest.raises(ExistingTailError, match="shape"):
        WannierResponse(
            jacobian=base.jacobian,
            provenance="test",
            method="bad curvature shape",
            second_derivative=np.zeros((n_wc, 3, n_ion, 3)))

    asymmetric = np.zeros((n_wc, 3, n_ion, 3, n_ion, 3))
    asymmetric[0, 0, 0, 0, 1, 0] = 1.0
    with pytest.raises(ExistingTailError, match="symmetric"):
        WannierResponse(
            jacobian=base.jacobian,
            provenance="test",
            method="non-symmetric curvature",
            second_derivative=asymmetric)

    bad_covariance = np.zeros((n_wc, 3, n_ion, 3, n_ion, 3))
    bad_covariance[0, 0, 0, 0, 0, 0] = 1.0
    response = WannierResponse(
        jacobian=base.jacobian,
        provenance="test",
        method="translation-covariance-violating curvature",
        second_derivative=bad_covariance)
    with pytest.raises(ExistingTailError, match="second-order translation"):
        response.check_for(n_wc, n_ion)


# --------------------------------------------------------------------------
# Contract A construction, preservation and double-counting audit
# --------------------------------------------------------------------------
def test_contract_a_algebra_and_matching_preservation():
    correction = _build()

    assert correction.matching_reps == (2, 1, 1)
    assert correction.target_reps == (4, 1, 1)
    contract_b = correction.contract_b_matrix
    interpolated = correction.interpolated_difference
    assert_allclose(correction.correction_matrix,
                    contract_b - interpolated, atol=1e-9)
    assert_allclose(correction.correction_matrix,
                    correction.correction_matrix.T, atol=1e-9)
    rows = correction.correction_matrix.reshape(
        correction.natom_target, 3, correction.natom_target, 3).sum(axis=2)
    assert_allclose(rows, 0.0, atol=1e-7)

    pattern = np.linspace(0.011, 0.023, 3 * 2)
    tiled = correction.matching_periodic_tiling(pattern)
    generic = np.cos(np.arange(3 * correction.natom_target) + 0.3)

    # Contract A is interstitial: it annihilates every matching-periodic
    # pattern.  Contract B and the naive stack do not.
    assert_allclose(correction.correction_matrix @ tiled, 0.0, atol=1e-7)
    assert np.max(np.abs(contract_b @ tiled)) > 1e-8
    assert np.max(np.abs(correction.desired_matrix @ tiled)) > 1e-8

    # Masking test: removing the existing tail only through its interpolant
    # vanishes on tiled patterns yet acts on a generic target distortion.
    mismatch = correction.existing_interpolation_mismatch
    assert_allclose(mismatch @ tiled, 0.0, atol=1e-7)
    assert np.max(np.abs(mismatch @ generic)) > 1e-8
    assert np.max(np.abs(correction.correction_matrix @ generic)) > 1e-8

    audit = correction.double_counting_audit(pattern, generic)
    assert audit["contract_a_tiled_action"] < 1e-7
    assert audit["contract_b_tiled_action"] > 1e-8
    assert audit["naive_desired_tiled_action"] > 1e-8
    assert audit["existing_mismatch_tiled_action"] < 1e-7
    assert audit["existing_mismatch_generic_action"] > 1e-8
    assert audit["contract_a_generic_action"] > 1e-8


def test_contract_a_energy_force_consistency_and_site_mapping():
    correction = _build()
    u = 0.02 * np.sin(np.arange(3 * correction.natom_target) + 0.4)
    energy = correction.energy(u)
    forces = correction.forces(u)
    step = 1e-6
    for index in (0, 4, 11, 3 * correction.natom_target - 1):
        shifted = u.copy()
        shifted[index] += step
        plus = correction.energy(shifted)
        shifted[index] -= 2 * step
        minus = correction.energy(shifted)
        assert_allclose(-forces[index], (plus - minus) / (2 * step), atol=1e-7)

    cell, positions, species = _tiled_geometry(_declaration(), correction.target_reps)
    mapping = correction.map_sites(cell, positions, species=species)
    mapped_energy, mapped_forces = correction.evaluate(u, mapping)
    assert_allclose(mapped_energy, energy, atol=1e-12)
    assert_allclose(mapped_forces, forces, atol=1e-12)


def test_contract_a_dynamical_response_and_desired_asymptote():
    correction = _build()
    q0 = np.zeros(3)
    composed0 = correction.composed_tail_dynamical(q0)
    desired0 = correction.desired_dynamical(q0)
    difference0 = correction.interpolated_difference_dynamical(q0)
    assert_allclose(composed0, desired0 - difference0, atol=1e-10)
    assert_allclose(composed0, composed0.transpose(2, 3, 0, 1).conj(), atol=1e-10)

    # At a small q the smooth interpolant is essentially its Gamma value, so
    # the change of the composed tail response is the desired directional
    # nonanalytic response: the existing asymptote has been replaced.
    qx = np.array([1e-6, 0.0, 0.0])
    qy = np.array([0.0, 1e-6, 0.0])
    for q in (qx, qy):
        composed = correction.composed_tail_dynamical(q)
        desired = correction.desired_dynamical(q)
        assert_allclose(composed - composed0, desired - desired0, atol=1e-4)


def test_matching_cell_target_is_exactly_unchanged():
    matching = _build(matching=(2, 1, 1), target=(2, 1, 1))
    assert np.max(np.abs(matching.correction_matrix)) < 1e-9
    assert matching.diagnostics["matching_cell_correction_max_abs"] < 1e-9


def test_nonperiodic_wannier_response_is_rejected():
    with pytest.raises(ExistingTailError, match="non-periodic matching-cell"):
        _build(matching=(2, 1, 1), target=(2, 1, 1),
               provider=_provider(curvature=0.03))


def test_nonperiodic_target_response_is_rejected():
    # The matching cell is the primitive cell (trivially periodic), so this
    # exercises the independent target-cell fold/unfold check.
    with pytest.raises(ExistingTailError, match="non-periodic target-cell"):
        _build(matching=(1, 1, 1), target=(2, 1, 1),
               provider=_provider(curvature=0.03))


def test_periodic_nonzero_curvature_preserves_the_matching_cell():
    correction = _build(
        matching=(2, 1, 1), target=(2, 1, 1),
        provider=_provider(curvature=0.03, curvature_all=True))
    diagnostics = correction.diagnostics
    assert diagnostics["wannier_second_derivative_max_abs_matching"] > 0.0
    assert diagnostics["matching_periodicity_residual"] <= \
        diagnostics["matching_periodicity_limit"]
    assert diagnostics["target_periodicity_residual"] <= \
        diagnostics["target_periodicity_limit"]
    assert diagnostics["matching_cell_correction_max_abs"] < 1e-9
    assert np.max(np.abs(correction.correction_matrix)) < 1e-9
    periodicity = correction.metadata["periodicity"]
    assert np.isfinite(periodicity["matching_residual"])
    assert periodicity["matching_residual"] <= periodicity["matching_limit"]
    assert periodicity["target_residual"] <= periodicity["target_limit"]


def test_fourier_sign_convention_is_consistent():
    positive = _build(ft_sign=1)
    negative = _build(ft_sign=-1)
    assert_allclose(negative.correction_matrix, positive.correction_matrix, atol=1e-8)
    pattern = np.linspace(0.011, 0.023, 6)
    tiled = negative.matching_periodic_tiling(pattern)
    assert_allclose(negative.correction_matrix @ tiled, 0.0, atol=1e-7)


def test_incompatible_target_does_not_claim_tiled_preservation():
    correction = _build(matching=(2, 1, 1), target=(3, 1, 1))
    assert correction.tiled_compatible() is False
    with pytest.raises(ValueError, match="not multiples"):
        correction.matching_periodic_tiling(np.zeros(6))


# --------------------------------------------------------------------------
# rejection paths and the no-tail boundary
# --------------------------------------------------------------------------
def test_builder_requires_a_declared_full_wannier_response():
    declaration, response = _declaration(), _response()
    with pytest.raises(ExistingTailError, match="callable"):
        build_existing_tail_correction(
            declaration, response, (1, 1, 1), (1, 1, 1), None)
    with pytest.raises(ExistingTailError, match="returned None"):
        build_existing_tail_correction(
            declaration, response, (1, 1, 1), (1, 1, 1),
            lambda *args: None)
    with pytest.raises(ExistingTailError, match="WannierResponse"):
        build_existing_tail_correction(
            declaration, response, (1, 1, 1), (1, 1, 1),
            lambda *args: np.eye(3))
    with pytest.raises(ExistingTailError, match="translation covariance"):
        build_existing_tail_correction(
            declaration, response, (1, 1, 1), (1, 1, 1),
            lambda cell, ions, species, wcs, parents: WannierResponse(
                jacobian=np.broadcast_to(
                    np.eye(3)[None, :, None, :], (2, 3, 2, 3)),
                provenance="bad",
                method="violates uniform-translation covariance",
                second_derivative=_zero_curvature(2, 2)))


def test_builder_rejects_incompatible_response_and_settings():
    declaration = _declaration()
    with pytest.raises(ResponseValidationError):
        ResponseProvenance(
            source="test", method="test", reference_description="test",
            boundary_convention="vacuum")

    three_site = ElectricalResponse(
        born_charges=np.array([np.eye(3), -np.eye(3), np.zeros((3, 3))]),
        dielectric=np.eye(3) * 2.0,
        provenance=_response().provenance,
        neutrality_tolerance=1e-6)
    with pytest.raises(ExistingTailError, match="2 sites|3 Born"):
        _build(declaration=declaration, response=three_site)

    with pytest.raises(ExistingTailError, match="matching_reps"):
        _build(matching=(2.5, 1, 1))
    with pytest.raises(ExistingTailError, match="target_reps"):
        _build(target=(0, 1, 1))
    with pytest.raises(ExistingTailError, match="ft_sign"):
        _build(ft_sign=0)
    with pytest.raises(ExistingTailError, match="eta"):
        _build(eta=0.0)
    with pytest.raises(ExistingTailError, match="nreal"):
        _build(nreal=-1)
    with pytest.raises(ExistingTailError, match="wannier_translation_tolerance"):
        _build(wannier_translation_tolerance=-1.0)


def test_no_explicit_tail_points_to_the_base_workflow():
    with pytest.raises(ExistingTailError, match="base matched fixed-cell workflow"):
        _declaration(model_type="standard ener (no explicit tail)")

    # The original no-tail construction still works through the base builder.
    cell = np.diag([4.0, 4.0, 4.0])
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    positions = frac @ cell
    born = np.array([np.eye(3) * 1.1, -np.eye(3) * 1.1])
    correction = build_correction(
        cell, positions, born, np.eye(3) * 2.0,
        (1, 1, 1), np.zeros((1, 2, 3, 2, 3), dtype=complex),
        np.array([[0, 0, 0]]), (1, 1, 1),
        eta=0.3, nreal=2, nrecip=3, neutrality_tolerance=1e-6)
    assert correction.correction_matrix.shape == (6, 6)
    assert np.isfinite(correction.correction_matrix).all()
