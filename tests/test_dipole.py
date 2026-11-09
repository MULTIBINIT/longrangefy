"""Physical invariants of the independent dipolar IFC evaluator."""

import numpy as np
import pytest

from longrangefy.dipole import dipole_ifc


def test_ewald_splitting_and_reciprocity_in_skew_cell():
    cell = np.array([[4.0, .5, .2], [0, 4.4, .3], [.1, 0, 5.]])
    positions = np.array([[.1, .2, .1], [1.4, 1.7, 1.3]])
    born = np.array([np.diag([2., 1.3, .7]), -np.diag([2., 1.3, .7])])
    epsilon = np.array([[3., .2, 0], [.2, 4., .1], [0, .1, 2.]])
    args = (cell, positions, born, epsilon, [.17, .13, .09])
    first = dipole_ifc(*args, eta=.05, nreal=12, nrecip=10).reshape(6, 6)
    second = dipole_ifc(*args, eta=.25, nreal=12, nrecip=10).reshape(6, 6)
    np.testing.assert_allclose(first, second, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(first, first.conj().T, atol=1e-10, rtol=1e-10)


def test_conducting_gamma_is_not_directional_limit():
    cell = np.eye(3) * 4
    positions = np.array([[0., 0., 0.], [2., 2., 2.]])
    born = np.array([np.eye(3), -np.eye(3)])
    epsilon = np.eye(3) * 3
    args = (cell, positions, born, epsilon)
    gamma = dipole_ifc(*args, [0, 0, 0], eta=.15, nreal=7, nrecip=9)
    near = dipole_ifc(*args, [1e-8, 0, 0], eta=.15, nreal=7, nrecip=9)
    mode = np.einsum('iam,a->im', born, [1., 0., 0.])
    # Standard nonanalytic dipole term in eV/angstrom**2; volume is 64 A**3.
    expected = (14.3996454784255 * 4 * np.pi / (64 * 3)
                * np.einsum('im,jn->imjn', mode, mode))
    np.testing.assert_allclose(near - gamma, expected, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(gamma.sum(axis=2), 0, atol=1e-10)


def test_invalid_dielectric_and_non_neutral_born_are_rejected():
    args = (np.eye(3)*4, np.array([[0., 0., 0.], [2., 2., 2.]]))
    born = np.array([np.eye(3), -np.eye(3)])
    with pytest.raises(ValueError, match='positive definite'):
        dipole_ifc(*args, born, np.diag([1., 1., 0.]), [0, 0, 0])
    with pytest.raises(ValueError, match='neutrality'):
        dipole_ifc(*args, born + .1, np.eye(3), [0, 0, 0])
