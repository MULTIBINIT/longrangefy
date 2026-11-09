"""End-to-end reference harmonics on a synthetic conservative model (R04/R05/R13)."""

import numpy as np
import pytest

from conftest import (
    SKEW_CELL,
    SKEW_FRAC,
    SKEW_SPECIES,
    AnharmonicModel,
    HarmonicModel,
    _spring_hessian,
)
from longrangefy import (
    ElectricalResponse,
    ReferenceStructure,
    ResponseProvenance,
    SupercellMapping,
    build_reference_harmonics,
)


def make_reference():
    return ReferenceStructure(cell=SKEW_CELL, frac_positions=SKEW_FRAC,
                              species=SKEW_SPECIES, symmetry_label="P-1 (test)",
                              source="synthetic skew fixture")


def make_response(natom=2):
    born = np.zeros((natom, 3, 3))
    born[0] = np.eye(3) * 2.0
    born[1] = -np.eye(3) * 2.0
    return ElectricalResponse(
        born_charges=born, dielectric=9.0 * np.eye(3),
        provenance=ResponseProvenance(
            source="tests", method="analytic test values",
            reference_description="synthetic skew fixture"))


def supercell_positions(reference, reps):
    mapping = SupercellMapping(reference, reps)
    return mapping.cart_positions_supercell()


def test_workflow_recovers_analytic_matching_grid_matrices():
    reference = make_reference()
    reps = (2, 1, 2)
    n_m = 4 * reference.natom
    rng = np.random.default_rng(21)
    hessian = _spring_hessian(n_m, rng)
    # Make the dense matrix exactly translation-invariant so its matching-grid
    # matrices have an independent analytic reference from the blocks.
    from conftest import random_blocks
    from longrangefy import SupercellIFC, blocks_to_dense, blocks_to_qmatrices
    rpoints = np.array([(a, b, c) for a in range(2) for b in range(1) for c in range(2)])
    blocks = SupercellIFC(reps=reps, rpoints=rpoints,
                          blocks=random_blocks(rng, reps, reference.natom))
    hessian = blocks_to_dense(blocks, reps)
    model = HarmonicModel(hessian, supercell_positions(reference, reps))

    harmonics = build_reference_harmonics(
        reference, make_response(), reps, model.get_forces,
        displacement=0.01, asr="none", force_noise_repeats=2,
        calculator_identity={"test_model": "analytic harmonic"})

    np.testing.assert_allclose(harmonics.dense_hessian, hessian, atol=1e-9)
    np.testing.assert_allclose(harmonics.blocks.blocks, blocks.blocks, atol=1e-9)
    analytic_q = blocks_to_qmatrices(blocks)
    np.testing.assert_allclose(harmonics.q_matrices, analytic_q, atol=1e-9)

    assert harmonics.grid_h.shape == (4, 3)
    q = harmonics.matching.q_grid()
    np.testing.assert_allclose(q, harmonics.grid_h / np.asarray(reps)[None, :])
    assert harmonics.ws_weights.shape[0] == 2 and harmonics.ws_weights.shape[1] == 2
    assert harmonics.ws_rpoints.shape[1] == 3
    assert harmonics.folded_ws_weights.shape == (2, 2, 4)


def test_workflow_diagnostics_and_metadata_are_complete():
    reference = make_reference()
    reps = (1, 1, 1)
    rng = np.random.default_rng(22)
    model = HarmonicModel(_spring_hessian(reference.natom, rng),
                          supercell_positions(reference, reps))
    harmonics = build_reference_harmonics(
        reference, make_response(), reps, model.get_forces,
        displacement=0.005, stencil="central", asr="simple",
        calculator_identity={"test_model": "analytic harmonic"})

    conventions = harmonics.conventions
    assert conventions["ifc_unit"] == "eV/angstrom^2 (unmassweighted)"
    assert conventions["asr"] == "simple"
    assert conventions["finite_difference_displacement_angstrom"] == 0.005
    assert "exp(+2*pi*i*q.R)" in conventions["fourier_forward"]

    diagnostics = harmonics.diagnostics
    assert diagnostics["residual_forces_max_ev_per_angstrom"] == 0.0
    assert diagnostics["n_force_calls"] == 2 * 3 * reference.natom
    assert diagnostics["force_noise"]["max_repeat_deviation_ev_per_angstrom"] == 0.0
    assert diagnostics["reciprocity_abs_ev_per_angstrom2"] < 1e-10
    assert diagnostics["asr_convention_applied"] == "simple"
    assert diagnostics["asr_row_sum_abs_ev_per_angstrom2"] < 1e-10
    assert diagnostics["iq_on_grid_identity_max_abs_ev_per_angstrom2"] < 1e-8
    assert diagnostics["q_roundtrip_max_abs_ev_per_angstrom2"] < 1e-8
    assert diagnostics["reference_symmetry_label"] == "P-1 (test)"
    assert diagnostics["response_provenance"]["source"] == "tests"
    assert harmonics.calculator_identity == {"test_model": "analytic harmonic"}


def test_workflow_preserves_unstable_reference_and_reports_residuals():
    # Anharmonic model displaced off stationarity keeps nonzero residual forces
    # without any automatic relaxation or reference movement.
    reference = make_reference()
    reps = (1, 1, 1)
    rng = np.random.default_rng(23)
    model = AnharmonicModel(_spring_hessian(reference.natom, rng),
                            supercell_positions(reference, reps), k4=4.0)
    shifted_positions = supercell_positions(reference, reps).copy()
    shifted_positions[0] += np.array([0.02, -0.01, 0.03])
    model.reference = shifted_positions.copy()

    harmonics = build_reference_harmonics(
        reference, make_response(), reps, model.get_forces,
        displacement=0.01, force_noise_repeats=2)
    assert harmonics.diagnostics["residual_forces_max_ev_per_angstrom"] > 1e-3
    # Reference sites themselves were not moved by the harmonics build.
    np.testing.assert_allclose(harmonics.reference.frac_positions,
                               reference.frac_positions, atol=0)


def test_asr_weighted_workflow_zeroes_weighted_row_sums():
    reference = make_reference()
    reps = (2, 1, 1)
    rng = np.random.default_rng(24)
    from conftest import random_blocks
    from longrangefy import SupercellIFC, blocks_to_dense
    rpoints = np.array([(a, 0, 0) for a in range(2)])
    blocks = SupercellIFC(reps=reps, rpoints=rpoints,
                          blocks=random_blocks(rng, reps, reference.natom))
    hessian = blocks_to_dense(blocks, reps)
    model = HarmonicModel(hessian, supercell_positions(reference, reps))
    harmonics = build_reference_harmonics(
        reference, make_response(), reps, model.get_forces,
        displacement=0.01, asr="weighted", force_noise_repeats=2)
    from longrangefy import blocks_asr_row_sums
    assert blocks_asr_row_sums(harmonics.blocks, harmonics.folded_ws_weights) < 1e-10


def test_validation_of_workflow_inputs():
    reference = make_reference()
    rng = np.random.default_rng(25)
    model = HarmonicModel(_spring_hessian(reference.natom, rng),
                          supercell_positions(reference, (1, 1, 1)))
    with pytest.raises(ValueError, match="asr must be"):
        build_reference_harmonics(reference, make_response(), (1, 1, 1),
                                  model.get_forces, asr="symmetric")
    with pytest.raises(ValueError, match="stencil"):
        build_reference_harmonics(reference, make_response(), (1, 1, 1),
                                  model.get_forces, stencil="backward")

    wrong = make_response(natom=3)
    with pytest.raises(ValueError, match="Born charge tensors"):
        build_reference_harmonics(reference, wrong, (1, 1, 1), model.get_forces)

    def bad_forces(positions_):
        return np.zeros((3, 3))

    with pytest.raises(ValueError, match="shape"):
        build_reference_harmonics(reference, make_response(), (1, 1, 1), bad_forces)
