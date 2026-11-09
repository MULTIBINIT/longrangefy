"""Force finite differences: known Hessian, anharmonic convergence, noise (R04/R13)."""

import numpy as np
import pytest

from conftest import SKEW_CELL, SKEW_FRAC, AnharmonicModel, HarmonicModel, _spring_hessian
from longrangefy import (
    estimate_force_noise,
    finite_difference_hessian,
    hessian_diagnostics,
    step_convergence,
)


def positions(natom=2):
    return SKEW_FRAC[:natom] @ SKEW_CELL


def quadratic_model(seed=7, natom=2):
    rng = np.random.default_rng(seed)
    return HarmonicModel(_spring_hessian(natom, rng), positions(natom))


def test_known_hessian_recovered_exactly_from_quadratic_model():
    model = quadratic_model()
    result = finite_difference_hessian(model.get_forces, model.reference,
                                      displacement=0.017, stencil="central")
    np.testing.assert_allclose(result.hessian, model.hessian, atol=1e-10)
    assert result.n_force_calls == 2 * 3 * model.reference.shape[0]
    assert result.displacement == 0.017
    assert result.stencil == "central"
    # The quadratic model is conservative, so reciprocity is exact.  Acoustic
    # row sums require translational invariance, which this random SPD fixture
    # deliberately does not impose; the diagnostic must still measure the
    # fixture truthfully instead of assuming zero.
    diagnostics = result.diagnostics
    assert diagnostics["reciprocity_abs_ev_per_angstrom2"] < 1e-10
    natom = model.reference.shape[0]
    expected_row_sum = float(
        np.abs(model.hessian.reshape(natom, 3, natom, 3).sum(axis=2)).max())
    assert diagnostics["asr_row_sum_abs_ev_per_angstrom2"] == pytest.approx(
        expected_row_sum, rel=1e-12)


def test_known_hessian_recovered_with_central5_stencil():
    model = quadratic_model()
    result = finite_difference_hessian(model.get_forces, model.reference,
                                      displacement=0.02, stencil="central5")
    np.testing.assert_allclose(result.hessian, model.hessian, atol=1e-10)
    assert result.n_force_calls == 4 * 3 * model.reference.shape[0]


def test_anharmonic_error_scales_as_expected_for_central_stencil():
    rng = np.random.default_rng(3)
    model = AnharmonicModel(_spring_hessian(2, rng), positions(2), k4=8.0)
    steps = (0.08, 0.04, 0.02, 0.01)
    studies = step_convergence(model.get_forces, model.reference, steps,
                               stencil="central", reference_hessian=model.hessian)
    errors = [study.max_abs_deviation_ev_per_angstrom2 for study in studies]
    # Central differences are second order; the quartic contamination shrinks ~4x.
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse / 2.5
    # Successive-change diagnostics also decrease.
    changes = [s.max_abs_step_change_ev_per_angstrom2 for s in studies[1:]]
    assert all(b < a for a, b in zip(changes, changes[1:]))


def test_anharmonic_central5_is_more_accurate_than_central_at_large_step():
    rng = np.random.default_rng(11)
    model = AnharmonicModel(_spring_hessian(2, rng), positions(2), k4=15.0)
    step = 0.1
    central = finite_difference_hessian(model.get_forces, model.reference, step, "central")
    central5 = finite_difference_hessian(model.get_forces, model.reference, step, "central5")
    error_central = np.abs(central.hessian - model.hessian).max()
    error_central5 = np.abs(central5.hessian - model.hessian).max()
    assert error_central5 < error_central / 10


def test_force_noise_estimation_reports_repeat_spread():
    rng = np.random.default_rng(5)
    reference = positions(2)

    def noisy(positions_):
        return rng.normal(scale=1e-5, size=reference.shape)

    noise = estimate_force_noise(noisy, reference, n_repeats=16)
    assert noise["n_repeats"] == 16
    assert 0 < noise["max_repeat_deviation_ev_per_angstrom"] < 1e-3

    def deterministic(positions_):
        return np.zeros_like(reference)

    assert estimate_force_noise(deterministic, reference, n_repeats=3)[
        "max_repeat_deviation_ev_per_angstrom"] == 0.0


def test_non_conservative_forces_flagged_by_reciprocity():
    def wild(positions_):
        rng = np.random.default_rng(int(abs(positions_).sum() * 1e9) % 2 ** 31)
        return rng.normal(size=positions_.shape)

    result = finite_difference_hessian(wild, positions(2), displacement=0.01)
    assert result.diagnostics["reciprocity_rel"] > 0.1


def test_input_validation():
    model = quadratic_model()
    with pytest.raises(ValueError, match="stencil"):
        finite_difference_hessian(model.get_forces, model.reference, stencil="forward")
    with pytest.raises(ValueError, match="positive"):
        finite_difference_hessian(model.get_forces, model.reference, displacement=0.0)
    with pytest.raises(ValueError, match="positive"):
        step_convergence(model.get_forces, model.reference, (0.1, -0.1))
    with pytest.raises(ValueError, match="shape"):
        finite_difference_hessian(lambda p: np.zeros((3, 3)), model.reference)
    with pytest.raises(ValueError, match="non-finite"):
        finite_difference_hessian(lambda p: np.full(p.shape, np.nan), model.reference)
    with pytest.raises(ValueError, match="non-empty"):
        step_convergence(model.get_forces, model.reference, ())


def test_hessian_diagnostics_detects_asymmetry():
    hessian = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    diagnostics = hessian_diagnostics(hessian)
    assert diagnostics["reciprocity_abs_ev_per_angstrom2"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="square"):
        hessian_diagnostics(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="3N"):
        hessian_diagnostics(np.zeros((4, 4)))
