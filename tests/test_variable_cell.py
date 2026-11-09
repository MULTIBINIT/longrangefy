"""Behavior tests for the variable-cell matched correction (ticket 09).

Scope: the array core (``longrangefy.variable_cell``) plus the optional ASE
adapters.  The synthetic fixtures use a skew reference cell and explicit
conservative blocks; no DeepMD model, pymultibinit or libabinit is imported or
needed.  Tolerances follow the identities being checked:

* structural matching/reference identities are asserted at floating-point
  roundoff (1e-12 in the block units);
* force and stress finite differences of the quadratic correction are asserted
  at the central-difference noise floor of the exercised blocks;
* the base-response FD measurements are linear in strain, so the recovered
  blocks are asserted to roundoff plus callback-roundoff;
* ASE composition tests use a real ASE calculator and compare both the exact
  component sum and independent energy finite differences.

The synthetic response tensors are construction fixtures, not material data.
No test here establishes predictive accuracy, a DeepMD workflow, convergence
of Ewald sums beyond the declared settings, or NPT support.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from longrangefy.matched import build_correction
from longrangefy.variable_cell import (
    VariableCellCorrection,
    VariableCellError,
    build_variable_correction,
    deformation_gradient,
    measure_clamped_ion_elasticity,
    measure_internal_strain_response,
    strain_voigt_from_cell,
    tensor_to_voigt_strain,
    tensor_to_voigt_stress,
    voigt_to_tensor_strain,
    voigt_to_tensor_stress,
)

ROUNDOFF = 1e-12
CHEAP = dict(eta=0.3, nreal=3, nrecip=4)
#: Explicit test-only construction bound; the package asserts no production
#: strain range (design D4).  Used only to test the rejection path.
TEST_STRAIN_BOUND = 0.02

SKEW_CELL = np.array([
    [4.1, 0.0, 0.0],
    [0.7, 4.6, 0.0],
    [0.4, -1.1, 5.0],
])
SKEW_FRAC = np.array([
    [0.0, 0.0, 0.0],
    [0.31, 0.42, 0.57],
])
SKEW_POSITIONS = SKEW_FRAC @ SKEW_CELL


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _joint_blocks(rng: np.random.Generator, natom: int = 2):
    """Random conservative blocks; Lambda is translation-neutral."""
    raw = rng.normal(size=(3 * natom, 3 * natom))
    phi = raw.T @ raw
    lam = rng.normal(size=(natom, 3, 6))
    lam[0] = -lam[1:].sum(axis=0)
    elastic = _symmetric(rng.normal(size=(6, 6)))
    return phi, lam, elastic


def _half_strain(voigt_index: int) -> np.ndarray:
    unit = np.zeros(6)
    unit[voigt_index] = 1.0
    return voigt_to_tensor_strain(unit)


def _affine_at(base_cell: np.ndarray, base_positions: np.ndarray,
               target_cell: np.ndarray) -> np.ndarray:
    return base_positions @ np.linalg.inv(base_cell) @ target_cell


# ---------------------------------------------------------------------------
# conventions and reference preservation
# ---------------------------------------------------------------------------
def test_skew_cell_strain_and_voigt_conventions():
    """Zero strain, symmetric strain and the low-symmetry shear order."""
    assert_allclose(deformation_gradient(SKEW_CELL, SKEW_CELL), np.eye(3),
                    atol=ROUNDOFF)
    assert_allclose(strain_voigt_from_cell(SKEW_CELL, SKEW_CELL), np.zeros(6),
                    atol=ROUNDOFF)

    strain = np.array([
        [0.010, 0.003, -0.002],
        [0.003, -0.004, 0.0015],
        [-0.002, 0.0015, 0.007],
    ])
    deformed = SKEW_CELL @ (np.eye(3) + strain)
    assert_allclose(deformation_gradient(SKEW_CELL, deformed),
                    np.eye(3) + strain, atol=ROUNDOFF)
    assert_allclose(strain_voigt_from_cell(SKEW_CELL, deformed),
                    tensor_to_voigt_strain(strain), atol=ROUNDOFF)

    # A generic tensor with distinct shear components survives the two Voigt
    # conventions; a shear-order transposition is measurable here.
    generic = np.array([[1.0, 0.2, 0.3],
                        [0.4, 2.0, 0.5],
                        [0.6, 0.7, 3.0]])
    symmetric = _symmetric(generic)
    assert_allclose(voigt_to_tensor_strain(tensor_to_voigt_strain(generic)),
                    symmetric, atol=ROUNDOFF)
    assert_allclose(voigt_to_tensor_stress(tensor_to_voigt_stress(generic)),
                    symmetric, atol=ROUNDOFF)

    engineering = np.array([0.0, 0.0, 0.0, 0.011, 0.022, 0.033])
    tensor = voigt_to_tensor_strain(engineering)
    assert_allclose(tensor[1, 2], 0.0055, atol=ROUNDOFF)
    assert_allclose(tensor[0, 2], 0.011, atol=ROUNDOFF)
    assert_allclose(tensor[0, 1], 0.0165, atol=ROUNDOFF)
    assert_allclose(tensor_to_voigt_strain(tensor), engineering, atol=ROUNDOFF)

    true_shear = voigt_to_tensor_stress(engineering)
    assert_allclose(true_shear[1, 2], 0.011, atol=ROUNDOFF)
    assert_allclose(true_shear[0, 2], 0.022, atol=ROUNDOFF)
    assert_allclose(true_shear[0, 1], 0.033, atol=ROUNDOFF)
    assert_allclose(tensor_to_voigt_stress(true_shear), engineering, atol=ROUNDOFF)


def test_reference_and_matching_cell_corrections_are_exactly_zero():
    """A zero block set gives E = F = sigma = 0 at every supported state."""
    correction = VariableCellCorrection(
        SKEW_CELL, SKEW_FRAC, np.zeros((6, 6)), max_abs_strain=None)
    reference = correction.evaluate(SKEW_CELL, SKEW_POSITIONS)
    assert reference.energy == 0.0
    assert_allclose(reference.forces, 0.0, atol=0.0)
    assert_allclose(reference.stress, 0.0, atol=0.0)

    rng = np.random.default_rng(3)
    strain = np.array([0.012, -0.008, 0.006, 0.009, -0.007, 0.004])
    cell = SKEW_CELL @ (np.eye(3) + voigt_to_tensor_strain(strain))
    positions = SKEW_FRAC @ cell + rng.normal(size=(2, 3), scale=0.01)
    for chain in ("affine", "cartesian"):
        evaluation = correction.evaluate(cell, positions, stress_chain=chain)
        assert evaluation.energy == 0.0
        assert_allclose(evaluation.forces, 0.0, atol=0.0)
        assert_allclose(evaluation.stress, 0.0, atol=0.0)


def test_displacement_block_adds_no_elastic_or_internal_strain_content():
    """At affine reference sites, Phi alone adds zero force and zero stress.

    This is the conservative sense in which the correction preserves the base
    model's own clamped-ion elastic and internal-strain response: a pure
    displacement channel has no linear force or stress at u = 0, at any strain.
    """
    rng = np.random.default_rng(5)
    phi, _, _ = _joint_blocks(rng)
    correction = VariableCellCorrection(
        SKEW_CELL, SKEW_FRAC, phi, max_abs_strain=None)
    for strain in (np.zeros(6),
                   np.array([0.012, -0.007, 0.005, 0.008, -0.004, 0.003])):
        cell = SKEW_CELL @ (np.eye(3) + voigt_to_tensor_strain(strain))
        evaluation = correction.evaluate(cell, SKEW_FRAC @ cell)
        assert abs(evaluation.energy) < ROUNDOFF
        assert np.abs(evaluation.forces).max() < ROUNDOFF
        assert np.abs(evaluation.stress).max() < ROUNDOFF


# ---------------------------------------------------------------------------
# joint-block force/stress conjugacy
# ---------------------------------------------------------------------------
def test_joint_blocks_force_and_stress_finite_differences():
    """Forces and both stress chains are exact energy derivatives."""
    rng = np.random.default_rng(11)
    phi, coupling, elastic = _joint_blocks(rng)
    correction = VariableCellCorrection(
        SKEW_CELL, SKEW_FRAC, phi, lambda_correction=coupling,
        elastic_correction=elastic, max_abs_strain=None)

    base_strain = np.array([0.008, -0.005, 0.004, 0.006, -0.003, 0.002])
    base_cell = SKEW_CELL @ (np.eye(3) + voigt_to_tensor_strain(base_strain))
    base_positions = (SKEW_FRAC @ base_cell
                      + rng.normal(size=(2, 3), scale=0.01))
    evaluation = correction.evaluate(base_cell, base_positions)
    step = 1.0e-6

    # Force = -dE/du at fixed cell.
    for atom in range(2):
        for direction in range(3):
            plus = base_positions.copy()
            minus = base_positions.copy()
            plus[atom, direction] += step
            minus[atom, direction] -= step
            derivative = -(correction.energy(base_cell, plus)
                           - correction.energy(base_cell, minus)) / (2.0 * step)
            assert_allclose(derivative, evaluation.forces[atom, direction],
                            atol=2.0e-9, rtol=1.0e-8)

    volume = abs(float(np.linalg.det(base_cell)))
    # ASE affine chain: cell and atoms deform at fixed fractional coordinates.
    for voigt_index in range(6):
        unit = _half_strain(voigt_index)
        plus_cell = base_cell @ (np.eye(3) + step * unit)
        minus_cell = base_cell @ (np.eye(3) - step * unit)
        plus_positions = _affine_at(base_cell, base_positions, plus_cell)
        minus_positions = _affine_at(base_cell, base_positions, minus_cell)
        derivative = (correction.energy(plus_cell, plus_positions)
                      - correction.energy(minus_cell, minus_positions)) / (2.0 * step)
        assert_allclose(derivative, volume * evaluation.stress_voigt[voigt_index],
                        atol=2.0e-8, rtol=1.0e-8)

    # Independent-coordinate chain: cell changes at fixed Cartesian positions.
    cartesian = correction.evaluate(
        base_cell, base_positions, stress_chain="cartesian")
    for voigt_index in range(6):
        unit = _half_strain(voigt_index)
        plus_cell = base_cell @ (np.eye(3) + step * unit)
        minus_cell = base_cell @ (np.eye(3) - step * unit)
        derivative = (correction.energy(plus_cell, base_positions)
                      - correction.energy(minus_cell, base_positions)) / (2.0 * step)
        assert_allclose(derivative, volume * cartesian.stress_voigt[voigt_index],
                        atol=2.0e-8, rtol=1.0e-8)

    # The chains differ exactly by the force-virial term.
    forces = correction.forces(base_cell, base_positions)
    virial = _symmetric(base_positions.T @ forces) / volume
    assert_allclose(cartesian.stress - evaluation.stress, virial,
                    atol=2.0e-10, rtol=1.0e-8)


def test_energy_fd_along_mixed_displacement_strain_path():
    """dE/dt = V sigma_x : W - F . (X W + v) on a mixed path at t = 0."""
    rng = np.random.default_rng(19)
    phi, coupling, elastic = _joint_blocks(rng)
    correction = VariableCellCorrection(
        SKEW_CELL, SKEW_FRAC, phi, lambda_correction=coupling,
        elastic_correction=elastic, max_abs_strain=None)

    base_strain = np.array([0.007, -0.004, 0.003, 0.005, -0.002, 0.001])
    base_cell = SKEW_CELL @ (np.eye(3) + voigt_to_tensor_strain(base_strain))
    base_positions = (SKEW_FRAC @ base_cell
                      + rng.normal(size=(2, 3), scale=0.008))
    extra_velocity = rng.normal(size=(2, 3), scale=0.004)
    cartesian = correction.evaluate(
        base_cell, base_positions, stress_chain="cartesian")
    forces = cartesian.forces
    volume = abs(float(np.linalg.det(base_cell)))
    step = 1.0e-6

    for voigt_index in range(6):
        unit = _half_strain(voigt_index)
        plus_cell = base_cell @ (np.eye(3) + step * unit)
        minus_cell = base_cell @ (np.eye(3) - step * unit)
        plus_positions = (_affine_at(base_cell, base_positions, plus_cell)
                          + step * extra_velocity)
        minus_positions = (_affine_at(base_cell, base_positions, minus_cell)
                           - step * extra_velocity)
        derivative = (correction.energy(plus_cell, plus_positions)
                      - correction.energy(minus_cell, minus_positions)) / (2.0 * step)
        position_velocity = base_positions @ unit + extra_velocity
        force_term = float(np.sum(forces * position_velocity))
        expected = volume * cartesian.stress_voigt[voigt_index] - force_term
        assert_allclose(derivative, expected, atol=2.0e-8, rtol=1.0e-8)


@pytest.mark.parametrize("magnitude", [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02])
def test_affine_stress_finite_strain_sweep(magnitude):
    """The exact affine stress conjugate holds across a finite-strain sweep."""
    rng = np.random.default_rng(13)
    phi, coupling, elastic = _joint_blocks(rng)
    correction = VariableCellCorrection(
        SKEW_CELL, SKEW_FRAC, phi, lambda_correction=coupling,
        elastic_correction=elastic, max_abs_strain=None)
    pattern = np.array([1.0, -0.6, 0.4, 0.5, -0.3, 0.2])
    base_cell = SKEW_CELL @ (
        np.eye(3) + voigt_to_tensor_strain(magnitude * pattern))
    base_positions = (SKEW_FRAC @ base_cell
                      + rng.normal(size=(2, 3), scale=0.01))
    stress = correction.stress(base_cell, base_positions, voigt=True)
    volume = abs(float(np.linalg.det(base_cell)))
    step = 1.0e-6
    for voigt_index in range(6):
        unit = _half_strain(voigt_index)
        plus_cell = base_cell @ (np.eye(3) + step * unit)
        minus_cell = base_cell @ (np.eye(3) - step * unit)
        plus_positions = _affine_at(base_cell, base_positions, plus_cell)
        minus_positions = _affine_at(base_cell, base_positions, minus_cell)
        derivative = (correction.energy(plus_cell, plus_positions)
                      - correction.energy(minus_cell, minus_positions)) / (2.0 * step)
        assert_allclose(derivative, volume * stress[voigt_index],
                        atol=2.0e-8, rtol=1.0e-8)


def test_mixed_force_strain_derivative_recovers_bilinear_block():
    """dF/d(eps) at u = 0 equals -Lambda/2 for every Voigt component."""
    rng = np.random.default_rng(17)
    phi, coupling, elastic = _joint_blocks(rng)
    correction = VariableCellCorrection(
        SKEW_CELL, SKEW_FRAC, phi, lambda_correction=coupling,
        elastic_correction=elastic, max_abs_strain=None)
    step = 1.0e-6
    reference_positions = SKEW_FRAC @ SKEW_CELL

    for voigt_index in range(6):
        unit = _half_strain(voigt_index)
        plus_cell = SKEW_CELL @ (np.eye(3) + step * unit)
        minus_cell = SKEW_CELL @ (np.eye(3) - step * unit)
        plus_forces = correction.forces(plus_cell, SKEW_FRAC @ plus_cell)
        minus_forces = correction.forces(minus_cell, SKEW_FRAC @ minus_cell)
        derivative = (plus_forces - minus_forces) / (2.0 * step)
        assert_allclose(derivative, -0.5 * coupling[:, :, voigt_index],
                        atol=2.0e-9, rtol=1.0e-8)
    # The reference position itself is untouched by the strain sweep.
    assert_allclose(correction.forces(SKEW_CELL, reference_positions), 0.0,
                    atol=2.0e-8)


# ---------------------------------------------------------------------------
# review regressions: image selection, rotation, skew MIC, branch uniqueness
# ---------------------------------------------------------------------------
def test_affine_stress_is_invariant_under_periodic_image_translation():
    """E, F and the affine stress are unchanged by wrapping one atom by A.

    The selected reference image must enter the affine virial term; using only
    the home-cell reference site changes the stress while leaving E and F
    unchanged.
    """
    cell = 4.0 * np.eye(3)
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    phi = np.zeros((6, 6))
    # Translation-neutral xx pair block: E = 1/2 (u0_x - u1_x)^2.
    phi[0, 0] = phi[3, 3] = 1.0
    phi[0, 3] = phi[3, 0] = -1.0
    correction = VariableCellCorrection(cell, frac, phi, max_abs_strain=None)

    positions = np.array([[-0.01, 0.0, 0.0], [2.0, 0.0, 0.0]])
    shifted = positions.copy()
    shifted[0] += np.array([4.0, 0.0, 0.0])
    direct = correction.evaluate(cell, positions)
    wrapped = correction.evaluate(cell, shifted)

    assert abs(direct.stress).max() > 1.0e-8
    assert_allclose(wrapped.energy, direct.energy, atol=ROUNDOFF)
    assert_allclose(wrapped.forces, direct.forces, atol=ROUNDOFF)
    assert_allclose(wrapped.stress, direct.stress, atol=ROUNDOFF, rtol=0.0)


def test_rigid_cell_rotation_is_rejected():
    """The frozen anisotropic blocks cannot be silently rotated with the cell."""
    theta = np.deg2rad(10.0)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    cell = 4.0 * np.eye(3)
    rotated = rotation @ cell
    correction = VariableCellCorrection(
        cell, np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        np.zeros((6, 6)), max_abs_strain=TEST_STRAIN_BOUND)

    # The symmetric part is below the strain bound; only the rotation guard
    # can reject this unsupported path.
    with pytest.raises(VariableCellError, match="rotation"):
        correction.evaluate(rotated, correction.reference_frac @ rotated)
    with pytest.raises(VariableCellError, match="rotation"):
        strain_voigt_from_cell(cell, rotated)

    # A 180-degree rotation is symmetric, so an antisymmetric-part check alone
    # would miss it; the polar-angle guard rejects it as well.
    half_turn = np.diag([-1.0, -1.0, 1.0]) @ cell
    with pytest.raises(VariableCellError, match="rotation"):
        strain_voigt_from_cell(cell, half_turn)


def test_skew_cell_nearest_image_uses_true_cartesian_metric():
    """Rounding fractional coordinates fails for a highly skew cell."""
    cell = np.array([
        [1.0, 0.0, 0.0],
        [10.0, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ])
    correction = VariableCellCorrection(
        cell, np.zeros((1, 3)), np.zeros((3, 3)), max_displacement=0.04)
    positions = np.array([[0.0, 0.01, 0.0]])

    displacement = correction.displacements(cell, positions)
    assert_allclose(displacement, positions, atol=ROUNDOFF)
    assert np.linalg.norm(displacement[0]) < 0.02


def test_deformed_bounded_assignment_rejects_exchanged_sites():
    """A bound that overlaps two deformed sites cannot define a unique branch."""
    cell = 4.0 * np.eye(3)
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    correction = VariableCellCorrection(
        cell, frac, np.zeros((6, 6)), max_displacement=0.99)
    compressed = cell @ (np.eye(3) + np.diag([-0.02, 0.0, 0.0]))

    # Exact affine sites are unambiguous even though the two bound balls
    # overlap: neither atom lies within the bound of the other site.
    correction.evaluate(compressed, frac @ compressed)

    ambiguous = np.array([[0.985, 0.0, 0.0], [0.975, 0.0, 0.0]])
    with pytest.raises(VariableCellError, match="ambiguous"):
        correction.evaluate(compressed, ambiguous)


def test_deformed_bounded_assignment_rejects_competing_periodic_image():
    """A site's own periodic images can compete even with one reference site."""
    cell = np.diag([4.0, 10.0, 10.0])
    frac = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    correction = VariableCellCorrection(
        cell, frac, np.zeros((6, 6)), max_displacement=1.99)
    compressed = cell @ (np.eye(3) + np.diag([-0.02, 0.0, 0.0]))

    # The affine sites are unique even though the bound exceeds half the
    # shortest lattice vector: exact image enumeration avoids over-rejection.
    correction.evaluate(compressed, frac @ compressed)

    # x = 1.96 is 1.96 angstrom from both images (0 and 3.92) of site 0.
    ambiguous = np.array([[1.96, 0.0, 0.0], [0.0, 5.0, 0.0]])
    with pytest.raises(VariableCellError, match="periodic image"):
        correction.evaluate(compressed, ambiguous)


# ---------------------------------------------------------------------------
# reusing the fixed-cell matched matrix
# ---------------------------------------------------------------------------
def _matched_fixture(target_reps, matching_reps=(2, 1, 1), cell=None,
                     primitive_positions=None, species=("Na", "Cl")):
    cell = SKEW_CELL if cell is None else np.asarray(cell, dtype=float)
    positions = (SKEW_POSITIONS if primitive_positions is None
                 else np.asarray(primitive_positions, dtype=float))
    nq = int(np.prod(matching_reps))
    grid_h = np.array(list(np.ndindex(*matching_reps)), dtype=int)
    phi0 = np.zeros((nq, len(positions), 3, len(positions), 3), dtype=complex)
    born = np.array([np.eye(3), -np.eye(3)])
    dielectric = np.eye(3) * 3.0
    return build_correction(
        cell, positions, born, dielectric, matching_reps, phi0, grid_h,
        target_reps, species=list(species), **CHEAP)


def test_matching_cell_preservation_under_finite_strain():
    """The matching target keeps zero correction at every supported strain."""
    matched = _matched_fixture((2, 1, 1), matching_reps=(2, 1, 1))
    assert np.abs(matched.correction_matrix).max() < ROUNDOFF
    correction = build_variable_correction(matched, max_displacement=0.4)
    assert correction.metadata["matching_cell_exact"]

    rng = np.random.default_rng(23)
    strain = np.array([0.011, -0.007, 0.005, 0.008, -0.006, 0.003])
    cell = correction.reference_cell @ (
        np.eye(3) + voigt_to_tensor_strain(strain))
    displacement = rng.normal(size=(correction.natom, 3), scale=0.01)
    positions = correction.reference_frac @ cell + displacement
    for chain in ("affine", "cartesian"):
        evaluation = correction.evaluate(cell, positions, stress_chain=chain)
        assert abs(evaluation.energy) < ROUNDOFF
        assert np.abs(evaluation.forces).max() < ROUNDOFF
        assert np.abs(evaluation.stress).max() < ROUNDOFF


def test_tiled_matching_patterns_are_preserved_under_strain():
    """Tiled matching-periodic patterns stay in the correction null space."""
    matched = _matched_fixture((4, 1, 1), matching_reps=(2, 1, 1))
    assert matched.tiled_compatible()
    correction = build_variable_correction(matched, max_displacement=0.4)

    rng = np.random.default_rng(29)
    primitive_pattern = rng.normal(size=3 * len(matched.positions), scale=0.005)
    tiled = matched.matching_periodic_tiling(primitive_pattern)

    strain = np.array([0.010, -0.006, 0.004, 0.007, -0.005, 0.002])
    cell = correction.reference_cell @ (
        np.eye(3) + voigt_to_tensor_strain(strain))
    positions = correction.reference_frac @ cell + tiled.reshape(-1, 3)
    for chain in ("affine", "cartesian"):
        evaluation = correction.evaluate(cell, positions, stress_chain=chain)
        assert abs(evaluation.energy) < ROUNDOFF
        assert np.abs(evaluation.forces).max() < ROUNDOFF

    # A generic displacement is genuinely corrected, so the null-space check
    # is not vacuous.
    generic = rng.normal(size=tiled.shape, scale=0.005)
    generic_positions = correction.reference_frac @ cell + generic.reshape(-1, 3)
    assert abs(correction.energy(cell, generic_positions)) > 1.0e-8


def test_build_reuses_matched_energy_and_forces():
    """At the reference cell the variable cell object is the matched one."""
    matched = _matched_fixture((3, 1, 1), matching_reps=(2, 1, 1))
    assert np.abs(matched.correction_matrix).max() > 1.0e-8
    correction = build_variable_correction(matched, max_displacement=0.4)
    assert not correction.metadata["matching_cell_exact"]
    assert correction.natom == 6
    assert_allclose(correction.reference_cell,
                    np.diag([3.0, 1.0, 1.0]) @ SKEW_CELL, atol=ROUNDOFF)
    assert_allclose(correction.volume_reference,
                    abs(np.linalg.det(correction.reference_cell)), atol=ROUNDOFF)

    rng = np.random.default_rng(31)
    displacement = rng.normal(size=(correction.natom, 3), scale=0.005)
    positions = correction.reference_frac @ correction.reference_cell + displacement
    evaluation = correction.evaluate(correction.reference_cell, positions)
    flat = displacement.reshape(-1)
    assert_allclose(evaluation.energy, matched.energy(flat), atol=ROUNDOFF)
    assert_allclose(evaluation.forces.reshape(-1), matched.forces(flat),
                    atol=ROUNDOFF)


# ---------------------------------------------------------------------------
# optional base-response diagnostics
# ---------------------------------------------------------------------------
def test_internal_strain_measurement_recovers_bilinear_block():
    """Affine force FD returns the package Lambda = -2 dF/deps convention."""
    rng = np.random.default_rng(37)
    phi, coupling, _ = _joint_blocks(rng)
    coupling_flat = coupling.reshape(6, 6)
    frac = SKEW_FRAC.copy()

    def get_forces(cell, positions):
        m = np.linalg.inv(SKEW_CELL) @ cell
        eps_v = tensor_to_voigt_strain(0.5 * (m + m.T) - np.eye(3))
        u = np.asarray(positions) - frac @ cell
        forces = -(phi @ u.reshape(-1) + 0.5 * (coupling_flat @ eps_v))
        return forces.reshape(-1, 3)

    for difference, expected_calls in (("central", 12), ("central5", 24)):
        measured = measure_internal_strain_response(
            get_forces, SKEW_CELL, SKEW_POSITIONS,
            strain_amplitude=1.0e-4, difference=difference)
        assert measured.difference == difference
        assert measured.n_force_calls == expected_calls
        assert_allclose(measured.df_dstrain, -0.5 * coupling,
                        atol=2.0e-8, rtol=1.0e-8)
        assert_allclose(measured.lambda_response, coupling,
                        atol=4.0e-8, rtol=1.0e-8)
        assert measured.diagnostics[
            "lambda_neutrality_residual_ev_per_angstrom"] < 1.0e-8


def test_clamped_ion_elasticity_measurement_recovers_stress_response():
    """Affine stress FD returns the raw C and its explicit major symmetrization."""
    rng = np.random.default_rng(41)
    raw = rng.normal(size=(6, 6))

    def get_stress(cell, positions):
        m = np.linalg.inv(SKEW_CELL) @ cell
        eps_v = tensor_to_voigt_strain(0.5 * (m + m.T) - np.eye(3))
        return voigt_to_tensor_stress(raw @ eps_v)

    for difference, expected_calls in (("central", 12), ("central5", 24)):
        measured = measure_clamped_ion_elasticity(
            get_stress, SKEW_CELL, SKEW_POSITIONS,
            strain_amplitude=1.0e-4, difference=difference)
        assert measured.difference == difference
        assert measured.n_stress_calls == expected_calls
        assert_allclose(measured.raw_elastic, raw, atol=2.0e-8, rtol=1.0e-8)
        assert_allclose(measured.elastic, _symmetric(raw),
                        atol=2.0e-8, rtol=1.0e-8)
        assert_allclose(measured.major_asymmetry,
                        np.abs(raw - raw.T).max(), atol=2.0e-8)


# ---------------------------------------------------------------------------
# ASE adapter
# ---------------------------------------------------------------------------
def _ase_fixture(*, max_abs_strain=None):
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    from longrangefy.ase_variable import VariableCellCorrectedCalculator

    matched = _matched_fixture((3, 1, 1), matching_reps=(2, 1, 1),
                               species=("Ar", "Ar"))
    correction = build_variable_correction(matched, max_displacement=0.4)
    atoms = Atoms("Ar6", positions=correction.reference_positions,
                  cell=correction.reference_cell, pbc=True)
    atoms.calc = VariableCellCorrectedCalculator(
        LennardJones(), correction, atoms, max_displacement=0.4,
        max_abs_strain=max_abs_strain)
    atoms.positions[0] += np.array([0.04, -0.03, 0.02])
    return atoms, correction


def test_ase_composed_energy_force_stress_and_finite_differences():
    """The ASE adapter adds exactly the correction and passes energy FDs."""
    from ase.calculators.lj import LennardJones

    atoms, correction = _ase_fixture()
    assert np.abs(correction.correction_matrix).max() > 1.0e-8
    base_cell = atoms.cell.array.copy()
    base_positions = atoms.positions.copy()
    evaluation = correction.evaluate(base_cell, base_positions)

    original_atoms = atoms.copy()
    original_atoms.calc = LennardJones()
    assert_allclose(atoms.get_potential_energy(),
                    original_atoms.get_potential_energy() + evaluation.energy,
                    atol=1.0e-10)
    assert_allclose(atoms.get_forces(),
                    original_atoms.get_forces() + evaluation.forces,
                    atol=1.0e-10)
    assert_allclose(atoms.get_stress(),
                    original_atoms.get_stress() + evaluation.stress_voigt,
                    atol=1.0e-10)

    def energy_at(cell, positions):
        atoms.set_cell(cell, scale_atoms=False)
        atoms.set_positions(positions, apply_constraint=False)
        return atoms.get_potential_energy()

    step = 1.0e-5
    force = atoms.get_forces()[1, 2]
    plus = base_positions.copy()
    minus = base_positions.copy()
    plus[1, 2] += step
    minus[1, 2] -= step
    assert_allclose(-(energy_at(base_cell, plus) - energy_at(base_cell, minus))
                    / (2.0 * step), force, atol=1.0e-6, rtol=1.0e-6)

    atoms.set_cell(base_cell, scale_atoms=False)
    atoms.set_positions(base_positions, apply_constraint=False)
    stress = atoms.get_stress()
    volume = abs(float(np.linalg.det(base_cell)))
    for voigt_index in range(6):
        unit = _half_strain(voigt_index)
        plus_cell = base_cell @ (np.eye(3) + step * unit)
        minus_cell = base_cell @ (np.eye(3) - step * unit)
        derivative = (
            energy_at(plus_cell, _affine_at(base_cell, base_positions, plus_cell))
            - energy_at(minus_cell, _affine_at(base_cell, base_positions, minus_cell))
        ) / (2.0 * step)
        assert_allclose(derivative, volume * stress[voigt_index],
                        atol=2.0e-5, rtol=1.0e-5)


def test_ase_adapter_rejects_unsupported_changes_and_bounds():
    from longrangefy.ase_variable import AseVariableError

    atoms, _correction = _ase_fixture()

    atoms.pbc = [True, True, False]
    with pytest.raises(AseVariableError, match="periodicity"):
        atoms.get_potential_energy()

    atoms, _correction = _ase_fixture()
    atoms.numbers[0] = 10  # Ne, not Ar
    with pytest.raises(AseVariableError, match="species"):
        atoms.get_potential_energy()

    atoms, _correction = _ase_fixture()
    atoms.positions[0] += np.array([0.5, 0.0, 0.0])
    with pytest.raises((AseVariableError, VariableCellError), match="displacement"):
        atoms.get_forces()

    atoms, _correction = _ase_fixture(max_abs_strain=TEST_STRAIN_BOUND)
    base_cell = atoms.cell.array.copy()
    strain = np.zeros(6)
    strain[0] = 2.0 * TEST_STRAIN_BOUND
    atoms.set_cell(base_cell @ (np.eye(3) + voigt_to_tensor_strain(strain)),
                   scale_atoms=True)
    with pytest.raises((AseVariableError, VariableCellError), match="strain"):
        atoms.get_potential_energy()




def test_ase_adapter_does_not_advertise_unavailable_stress():
    """An energy/force-only original calculator yields no fabricated stress."""
    from ase import Atoms
    from ase.calculators.calculator import (
        Calculator, PropertyNotImplementedError, all_changes)

    from longrangefy.ase_variable import VariableCellCorrectedCalculator

    class EnergyOnly(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(self.atoms), 3)),
            }

    matched = _matched_fixture((3, 1, 1), matching_reps=(2, 1, 1),
                               species=("Ar", "Ar"))
    correction = build_variable_correction(matched, max_displacement=0.4)
    atoms = Atoms("Ar6", positions=correction.reference_positions,
                  cell=correction.reference_cell, pbc=True)
    calculator = VariableCellCorrectedCalculator(
        EnergyOnly(), correction, atoms, max_displacement=0.4)
    assert "stress" not in calculator.implemented_properties
    atoms.calc = calculator
    atoms.get_potential_energy()
    atoms.get_forces()
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()


def test_ase_cell_response_provider_matches_direct_calculator():
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    from longrangefy.ase_variable import AseCellResponseProvider, AseVariableError

    cell = np.diag([4.0, 5.0, 6.0])
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 2.0, 2.5]])
    atoms = Atoms("Ar2", positions=positions, cell=cell, pbc=True)
    atoms.calc = LennardJones()
    provider = AseCellResponseProvider(atoms)

    strain = np.array([0.004, -0.003, 0.002, 0.001, -0.001, 0.002])
    strained_cell = cell @ (np.eye(3) + voigt_to_tensor_strain(strain))
    strained_positions = _affine_at(cell, positions, strained_cell)

    direct = Atoms("Ar2", positions=strained_positions, cell=strained_cell, pbc=True)
    direct.calc = LennardJones()
    assert_allclose(provider.get_forces(strained_cell, strained_positions),
                    direct.get_forces(), atol=ROUNDOFF)
    assert_allclose(provider.get_stress(strained_cell, strained_positions),
                    direct.get_stress(voigt=False), atol=ROUNDOFF)
    assert provider.get_energy(strained_cell, strained_positions) == pytest.approx(
        direct.get_potential_energy(), abs=1.0e-12)
    with pytest.raises(AseVariableError, match="positions"):
        provider.get_forces(strained_cell, np.zeros((1, 3)))


# ---------------------------------------------------------------------------
# validation and independence
# ---------------------------------------------------------------------------
def test_validation_rejects_unsupported_inputs():
    zeros = np.zeros((6, 6))
    with pytest.raises(VariableCellError, match="singular"):
        VariableCellCorrection(np.zeros((3, 3)), SKEW_FRAC, zeros)
    with pytest.raises(VariableCellError, match="inverted"):
        deformed = SKEW_CELL.copy()
        deformed[0] *= -1.0
        correction = VariableCellCorrection(SKEW_CELL, SKEW_FRAC, zeros,
                                            max_abs_strain=None)
        correction.evaluate(deformed, SKEW_FRAC @ deformed)
    asymmetric = np.zeros((6, 6))
    asymmetric[0, 1] = 1.0
    with pytest.raises(VariableCellError, match="symmetric"):
        VariableCellCorrection(SKEW_CELL, SKEW_FRAC, asymmetric)
    with pytest.raises(VariableCellError, match="shape"):
        VariableCellCorrection(SKEW_CELL, SKEW_FRAC, zeros,
                               lambda_correction=np.zeros((2, 6)))
    with pytest.raises(VariableCellError, match="shape"):
        VariableCellCorrection(SKEW_CELL, SKEW_FRAC, zeros,
                               elastic_correction=np.zeros((3, 3)))
    with pytest.raises(VariableCellError, match="max_displacement"):
        VariableCellCorrection(SKEW_CELL, SKEW_FRAC, zeros,
                               max_displacement=-1.0)

    correction = VariableCellCorrection(SKEW_CELL, SKEW_FRAC, zeros,
                                        max_abs_strain=TEST_STRAIN_BOUND)
    strain = np.zeros(6)
    strain[0] = 2.0 * TEST_STRAIN_BOUND
    with pytest.raises(VariableCellError, match="strain"):
        correction.evaluate(SKEW_CELL @ (np.eye(3) + voigt_to_tensor_strain(strain)),
                            SKEW_POSITIONS)
    with pytest.raises(VariableCellError, match="stress_chain"):
        correction.evaluate(SKEW_CELL, SKEW_POSITIONS, stress_chain="mixed")

    bounded = VariableCellCorrection(SKEW_CELL, SKEW_FRAC, zeros,
                                     max_displacement=0.01, max_abs_strain=None)
    with pytest.raises(VariableCellError, match="displacement"):
        bounded.evaluate(SKEW_CELL, SKEW_POSITIONS + np.array([[0.02, 0.0, 0.0],
                                                               [0.0, 0.0, 0.0]]))

    with pytest.raises(VariableCellError, match="difference"):
        measure_internal_strain_response(
            lambda cell, positions: np.zeros((2, 3)), SKEW_CELL, SKEW_POSITIONS,
            difference="forward")
    with pytest.raises(VariableCellError, match="shape"):
        measure_clamped_ion_elasticity(
            lambda cell, positions: np.zeros((4, 4)), SKEW_CELL, SKEW_POSITIONS)


def test_variable_cell_core_imports_without_optional_packages():
    """Array-core evaluation works with ASE/DeepMD/pymultibinit blocked."""
    src = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        BLOCKED = ("ase", "deepmd", "pymultibinit", "abipy", "libabinit")

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in BLOCKED:
                    raise ImportError("blocked optional package: " + fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        sys.path.insert(0, {str(src)!r})
        import numpy as np
        from longrangefy.variable_cell import VariableCellCorrection

        cell = np.array([[4.0, 0.0, 0.0], [0.5, 4.5, 0.0], [0.2, -0.7, 5.0]])
        frac = np.zeros((1, 3))
        correction = VariableCellCorrection(
            cell, frac, np.zeros((3, 3)), max_abs_strain=None)
        strain = np.zeros(6)
        strain[0] = 0.01
        from longrangefy.variable_cell import voigt_to_tensor_strain
        deformed = cell @ (np.eye(3) + voigt_to_tensor_strain(strain))
        evaluation = correction.evaluate(deformed, frac @ deformed)
        assert evaluation.energy == 0.0
        assert not any(name in sys.modules for name in BLOCKED)
        print("ok")
        """
    )
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
