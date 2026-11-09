"""Behavior tests for the matched fixed-cell correction (tickets 02 and 03).

Scope: the array core only (``longrangefy.matched`` plus the parent
``longrangefy.dipole``).  Nothing here imports ASE, DeepMD, pymultibinit or
libabinit.  The fixtures are deterministic synthetic conservative models with
a known finite-range harmonic kernel; the SrTiO3 geometry fixture uses
nominal ionic Born charges and a scalar dielectric explicitly as *synthetic
response stand-ins* - they are not a first-principles or DeepMD response and
imply nothing about the supplied production model.

Tolerances are chosen from the identities being checked, not fitted to a
particular result:
  * structural identities (matching cancellation, counterterm equivalence,
    tiled preservation, permutation invariance) hold to floating-point
    roundoff, so they are asserted at 1e-12 eV / eV/angstrom / eV/angstrom^2;
  * Ewald stability is asserted only with the tightened summation settings
    below (eta=0.25 A^-2, 6 real / 10 reciprocal shells), where varying eta by
    factors 0.6 and 1.5 changed the correction by at most 1.1e-7 in exercised
    runs; the threshold used here (1e-6) leaves headroom for platform
    roundoff while still rejecting a non-converged split.
  * Site mapping is asserted against an angstrom displacement bound
    (``tol``), not a fractional residue or a cell-scaled tolerance; the
    disagreement between the two metrics is measurable (a 0.3 A displacement
    in a skew cell is 0.1 in primitive fractional coordinates).
  * Symmetry repair of the input samples is exercised in both directions: the
    default accepts only roundoff-level violation and refuses anything larger
    unless the caller declares an absolute ``phi0_symmetry_tol``; there is no
    built-in material-specific tolerance.
  * Image selection is checked against an independent brute-force enumeration
    (nearest distance and multiplicity per pair/residue class, including a
    pathological near-degenerate skew basis whose true nearest images lie far
    outside a fixed +/-2 window), not merely against its own identities.
  * Off-grid branch matrices are checked for Hermiticity and real diagonals,
    against the analytic Fourier interpolant of a nearest-neighbour spring,
    and against the assembled correction's q-blocks on the target grid.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from longrangefy.matched import (
    build_correction,
    _closest_lattice_points,
    _fold_to_target,
    _grid_h,
    _pair_ws_weights,
)

TIGHT = dict(eta=0.25, nreal=6, nrecip=10)
CHEAP = dict(eta=0.3, nreal=3, nrecip=4)
ROUNDOFF = 1e-12


# --------------------------------------------------------------------------
# deterministic synthetic models
# --------------------------------------------------------------------------
def _reciprocal_kernel(reps, natom, rng, scale=1.0):
    """Random real finite-range kernel with kappa(-T) = kappa(T)^T."""
    m = reps
    kernel = np.zeros((*m, natom, 3, natom, 3))
    done = set()
    for t in [tuple(x) for x in _grid_h(m)]:
        if t in done:
            continue
        block = rng.normal(size=(natom, 3, natom, 3)) * scale
        neg = tuple((-np.array(t)) % np.array(m))
        if neg == t:
            kernel[t] = 0.5 * (block + block.transpose(2, 3, 0, 1))
        else:
            kernel[t] = block
            kernel[neg] = block.transpose(2, 3, 0, 1)
            done.add(neg)
        done.add(t)
    return kernel


def _grid_matrices(kernel, reps, sign=1):
    """Phi(q) = sum_T kappa(T) exp(sign*2i pi q.T) on the full grid."""
    h = _grid_h(reps)
    q = h / np.array(reps, dtype=float)
    t_idx = _grid_h(reps).astype(float)
    phase = np.exp(sign * 2j * np.pi * q @ t_idx.T)
    natom = kernel.shape[3]
    return np.einsum("t...,qt->q...", kernel.reshape(-1, natom, 3, natom, 3), phase)


def _bruteforce_ws(cell, positions, reps, shells=8):
    """Independent WS weights from a wide original-coordinate window.

    Re-derives the pair displacement (``tau_j - tau_i + T``), the residue
    classes and the per-class nearest/equidistant classification from
    scratch, so the production search is checked against an independent
    implementation rather than against its own identities.
    """
    m = np.array(reps)
    n = len(positions)
    nrho = int(np.prod(m))
    grid = np.array(list(np.ndindex(*[2 * shells + 1] * 3))) - shells
    cart = grid.astype(float) @ cell
    classes = ((grid[:, 0] % m[0]) * m[1] + grid[:, 1] % m[1]) * m[2] + grid[:, 2] % m[2]
    distances = np.zeros((n, n, len(grid)))
    nearest = np.full((n, n, nrho), np.inf)
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(positions[j] - positions[i] + cart, axis=1)
            np.minimum.at(nearest[i, j], classes, distances[i, j])
    weights = np.zeros((n, n, len(grid)))
    for i in range(n):
        for j in range(n):
            within = distances[i, j] <= nearest[i, j][classes] + 1e-6
            counts = np.bincount(classes[within], minlength=nrho)
            weights[i, j] = np.where(within, 1.0 / np.maximum(counts[classes], 1), 0.0)
    return grid, distances, classes, weights


def _assert_ws_matches_bruteforce(cell, positions, reps, shells=8, atol=1e-5):
    """Production WS weights reproduce an independent brute-force enumeration."""
    cand, weights, rho, residual = _pair_ws_weights(cell, positions, reps)
    assert residual == 0.0
    m = np.array(reps)
    n = len(positions)
    nrho = int(np.prod(m))
    delta = positions[None, :, :] - positions[:, None, :]
    cart = cand.astype(float) @ cell
    dprod = np.linalg.norm(delta[:, :, None, :] + cart[None, None, :, :], axis=3)
    _grid_b, dbrute, classes_b, weights_b = _bruteforce_ws(cell, positions, reps, shells=shells)
    for i in range(n):
        for j in range(n):
            for cls in range(nrho):
                prod = rho == cls
                brute = classes_b == cls
                assert prod.any() and brute.any()
                assert_allclose(dprod[i, j, prod].min(), dbrute[i, j, brute].min(), atol=atol)
                assert_allclose(weights[i, j, prod].sum(), 1.0, atol=1e-12)
                assert_allclose(weights_b[i, j, brute].sum(), 1.0, atol=1e-12)
                assert_allclose(weights[i, j, prod].max(), weights_b[i, j, brute].max(), atol=1e-6)


def _class_candidates(cand, weights, rho, i, j, cls):
    """Nonzero-weight candidates and weights of one pair and residue class."""
    mask = (rho == cls) & (weights[i, j] > 0)
    return cand[mask], weights[i, j][mask]


def _orthogonal_model(seed=7):
    cell = np.array([[3.0, 0.0, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 3.4]])
    frac = np.array([[0.0, 0.0, 0.0], [0.4, 0.3, 0.25]])
    positions = frac @ cell
    born = np.zeros((2, 3, 3))
    born[0] = np.eye(3) * 2.0
    born[1] = -born[0]
    dielectric = np.eye(3) * 4.0
    reps = (2, 2, 2)
    rng = np.random.default_rng(seed)
    phi0 = _grid_matrices(_reciprocal_kernel(reps, 2, rng), reps)
    return dict(cell=cell, positions=positions, born=born, dielectric=dielectric,
                reps=reps, grid_h=_grid_h(reps), phi0=phi0)


def _skew_model(seed=13):
    cell = np.array([[3.0, 0.0, 0.0], [0.7, 3.2, 0.0], [0.4, -0.5, 3.6]])
    frac = np.array([[0.0, 0.0, 0.0], [0.33, 0.21, 0.47]])
    positions = frac @ cell
    born = np.zeros((2, 3, 3))
    born[0] = np.eye(3) * 1.5
    born[1] = -born[0]
    reps = (2, 2, 2)
    rng = np.random.default_rng(seed)
    phi0 = _grid_matrices(_reciprocal_kernel(reps, 2, rng), reps)
    return dict(cell=cell, positions=positions, born=born, dielectric=np.eye(3) * 3.0,
                reps=reps, grid_h=_grid_h(reps), phi0=phi0)


def _sto_model(seed=21):
    """Cubic SrTiO3 geometry with synthetic (nominal ionic) response tensors."""
    a = 3.905
    cell = np.diag([a, a, a])
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    positions = frac @ cell
    born = np.zeros((5, 3, 3))
    born[0] = np.eye(3) * 2.0   # Sr
    born[1] = np.eye(3) * 4.0   # Ti
    for k in range(2, 5):
        born[k] = np.eye(3) * (-2.0)  # O
    reps = (2, 2, 2)
    rng = np.random.default_rng(seed)
    phi0 = _grid_matrices(_reciprocal_kernel(reps, 5, rng), reps)
    return dict(cell=cell, positions=positions, born=born, dielectric=np.eye(3) * 5.0,
                reps=reps, grid_h=_grid_h(reps), phi0=phi0,
                species=["Sr", "Ti", "O", "O", "O"])


def _canonical_target(model, target_reps):
    """Tiled target geometry in canonical (image-major, atom-minor) order."""
    cell_t = np.diag(target_reps).astype(float) @ model["cell"]
    frac = model["positions"] @ np.linalg.inv(model["cell"])
    positions = np.array([(frac[s] + np.array(t)) @ model["cell"]
                          for t in np.ndindex(*target_reps) for s in range(len(frac))])
    species = None
    if model.get("species") is not None:
        species = model["species"] * int(np.prod(target_reps))
    return cell_t, positions, species


# --------------------------------------------------------------------------
# built corrections (module scope: construction is the expensive part)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ortho():
    return _orthogonal_model()


@pytest.fixture(scope="module")
def correction_equal(ortho):
    return build_correction(ortho["cell"], ortho["positions"], ortho["born"],
                            ortho["dielectric"], ortho["reps"], ortho["phi0"],
                            ortho["grid_h"], ortho["reps"], **TIGHT)


@pytest.fixture(scope="module")
def correction_larger(ortho):
    return build_correction(ortho["cell"], ortho["positions"], ortho["born"],
                            ortho["dielectric"], ortho["reps"], ortho["phi0"],
                            ortho["grid_h"], (2, 2, 4), **TIGHT)


@pytest.fixture(scope="module")
def correction_smaller(ortho):
    return build_correction(ortho["cell"], ortho["positions"], ortho["born"],
                            ortho["dielectric"], ortho["reps"], ortho["phi0"],
                            ortho["grid_h"], (1, 1, 1), **CHEAP)


@pytest.fixture(scope="module")
def correction_skew():
    model = _skew_model()
    return build_correction(model["cell"], model["positions"], model["born"],
                            model["dielectric"], model["reps"], model["phi0"],
                            model["grid_h"], (2, 4, 2), **TIGHT), model


@pytest.fixture(scope="module")
def sto_correction():
    model = _sto_model()
    return build_correction(model["cell"], model["positions"], model["born"],
                            model["dielectric"], model["reps"], model["phi0"],
                            model["grid_h"], (4, 4, 2), species=model["species"],
                            **CHEAP), model


@pytest.fixture(scope="module")
def incompatible(ortho):
    return build_correction(ortho["cell"], ortho["positions"], ortho["born"],
                            ortho["dielectric"], ortho["reps"], ortho["phi0"],
                            ortho["grid_h"], (2, 2, 3), **CHEAP)


# --------------------------------------------------------------------------
# matching-grid preservation (ticket 02, 03)
# --------------------------------------------------------------------------
def test_matching_cell_correction_vanishes(correction_equal):
    """Equal matching/target cell: every correction energy/force is zero."""
    mc = correction_equal
    assert np.abs(mc.correction_matrix).max() < ROUNDOFF
    assert np.abs(mc.split_matrix - mc.unsplit_matrix).max() < ROUNDOFF
    assert mc.diagnostics["shared_wavevector_count"] == 8
    assert mc.diagnostics["counterterm_max_abs_difference"] < ROUNDOFF
    assert mc.diagnostics["shared_grid_max_branch_difference"] < ROUNDOFF
    assert mc.diagnostics["gamma_uniform_displacement_force_residual"] < ROUNDOFF
    rng = np.random.default_rng(5)
    u = rng.normal(size=3 * mc.natom_target, scale=0.02)
    assert abs(mc.energy(u)) < ROUNDOFF
    assert np.abs(mc.forces(u)).max() < ROUNDOFF


def test_both_branches_are_explicit_and_equivalent_to_counterterm(correction_larger):
    """Split minus unsplit is built explicitly and matches D_L - I_Q[D]."""
    mc = correction_larger
    natom_l = mc.natom_target
    assert mc.unsplit_matrix.shape == (3 * natom_l, 3 * natom_l)
    assert mc.split_matrix.shape == (3 * natom_l, 3 * natom_l)
    assert_allclose(mc.correction_matrix, mc.split_matrix - mc.unsplit_matrix, atol=ROUNDOFF)
    assert_allclose(mc.correction_matrix, mc.counterterm_matrix(), atol=ROUNDOFF)
    assert np.abs(mc.correction_matrix).max() > 1e-3  # new wavevectors exist
    assert_allclose(mc.unsplit_matrix, mc.unsplit_matrix.T, atol=ROUNDOFF)
    assert_allclose(mc.split_matrix, mc.split_matrix.T, atol=ROUNDOFF)


def test_shared_wavevector_branch_agreement_and_off_grid_change(correction_larger, ortho):
    """Branch matrices agree on every shared grid point and differ off-grid."""
    mc = correction_larger
    assert mc.diagnostics["shared_wavevector_count"] == 8
    assert mc.diagnostics["shared_grid_max_branch_difference"] < ROUNDOFF
    # unsplit branch reproduces the input samples on the matching grid
    for k, h in enumerate(ortho["grid_h"]):
        q = h / np.array(ortho["reps"], dtype=float)
        assert np.abs(mc.branch_dynamical(q, "unsplit") - ortho["phi0"][k]).max() < 1e-10
        # split and unsplit agree on the matching grid
        split = mc.branch_dynamical(q, "split")
        unsplit = mc.branch_dynamical(q, "unsplit")
        assert np.abs(split - unsplit).max() < ROUNDOFF
    # ... and the correction is genuinely nonzero at an unsampled wavevector
    off = np.array([0.25, 0.25, 0.25])
    assert np.abs(mc.branch_dynamical(off, "difference")).max() > 1e-2


def test_tiled_matching_patterns_are_preserved(correction_larger):
    """Matching-periodic distortions in a compatible larger cell cancel."""
    mc = correction_larger
    assert mc.tiled_compatible()
    rng = np.random.default_rng(9)
    n = len(mc.positions)
    n_match = n * int(np.prod(mc.matching_reps))
    primitive_pattern = rng.normal(size=3 * n, scale=0.01)
    u = mc.matching_periodic_tiling(primitive_pattern)
    assert abs(mc.energy(u)) < ROUNDOFF
    assert np.abs(mc.forces(u)).max() < ROUNDOFF
    # the same pattern declared on the matching supercell tiles identically
    constant = np.tile(primitive_pattern.reshape(n, 3), (n_match // n, 1)).reshape(-1)
    assert np.array_equal(mc.matching_periodic_tiling(constant), u)
    # a genuinely matching-supercell-periodic pattern is preserved as well
    supercell_pattern = rng.normal(size=3 * n_match, scale=0.01)
    u_super = mc.matching_periodic_tiling(supercell_pattern)
    assert u_super.size == 3 * mc.natom_target
    assert abs(mc.energy(u_super)) < ROUNDOFF
    assert np.abs(mc.forces(u_super)).max() < ROUNDOFF
    # the tiled pattern really varies between matching images, not only cells
    assert np.abs(u_super.reshape(-1, 3 * n)[1] - u_super.reshape(-1, 3 * n)[0]).max() > 0
    with pytest.raises(ValueError, match="3 components per primitive atom"):
        mc.matching_periodic_tiling(np.zeros(7))


def test_smaller_target_is_exactly_preserved(correction_smaller):
    """A target whose wavevectors are all shared carries a zero correction."""
    mc = correction_smaller
    assert mc.diagnostics["shared_wavevector_count"] == 1  # Gamma only
    assert np.abs(mc.correction_matrix).max() < ROUNDOFF
    rng = np.random.default_rng(11)
    u = rng.normal(size=3 * mc.natom_target, scale=0.02)
    assert abs(mc.energy(u)) < ROUNDOFF
    assert np.abs(mc.forces(u)).max() < ROUNDOFF


def test_incompatible_target_makes_no_tiling_claim(incompatible):
    """Non-multiple targets keep shared-wavevector matching only."""
    mc = incompatible
    assert not mc.tiled_compatible()
    assert mc.diagnostics["shared_wavevector_count"] == 4
    assert np.abs(mc.correction_matrix).max() > 1e-3
    with pytest.raises(ValueError, match="not multiples"):
        mc.matching_periodic_tiling(np.zeros(3 * len(mc.positions)))


# --------------------------------------------------------------------------
# quadratic energy/forces, invariances, Ewald convergence
# --------------------------------------------------------------------------
def test_energy_forces_are_consistent(correction_larger):
    """Correction forces are minus the derivative of the quadratic energy."""
    mc = correction_larger
    rng = np.random.default_rng(17)
    u = rng.normal(size=3 * mc.natom_target, scale=0.01)
    forces = mc.forces(u)
    assert_allclose(forces, -(mc.correction_matrix @ u), atol=ROUNDOFF)
    step = 1e-6
    fd = np.array([-(mc.energy(u + step * e) - mc.energy(u - step * e)) / (2 * step)
                   for e in np.eye(len(u))])
    assert np.abs(fd - forces).max() < 1e-8


def test_uniform_displacement_produces_no_correction_force(correction_larger):
    """Acoustic behaviour: a rigid shift has no correction force."""
    mc = correction_larger
    u = np.tile([0.01, -0.02, 0.03], mc.natom_target)
    assert np.abs(mc.forces(u)).max() < ROUNDOFF
    assert mc.diagnostics["gamma_uniform_displacement_force_residual"] < ROUNDOFF


def test_ewald_splitting_is_numerical(correction_larger):
    """Varying the Ewald parameter must not change the correction."""
    mc = correction_larger
    report = mc.eta_sensitivity((0.6, 1.5))
    assert set(report) == {"max_abs_change_eta_x0.6", "max_abs_change_eta_x1.5"}
    assert max(report.values()) < 1e-6


def test_skew_cell_preservation_and_tiling(correction_skew):
    """Skew primitive lattices keep the same matching guarantees."""
    mc, model = correction_skew
    assert np.linalg.det(model["cell"]) > 0
    assert mc.diagnostics["shared_wavevector_count"] == 8
    assert mc.diagnostics["counterterm_max_abs_difference"] < ROUNDOFF
    assert mc.diagnostics["shared_grid_max_branch_difference"] < ROUNDOFF
    assert mc.diagnostics["gamma_uniform_displacement_force_residual"] < ROUNDOFF
    assert mc.metadata["weight_class_sum_residual"] == 0.0
    rng = np.random.default_rng(19)
    pattern = rng.normal(size=3 * len(mc.positions), scale=0.01)
    u = mc.matching_periodic_tiling(pattern)
    assert abs(mc.energy(u)) < ROUNDOFF
    assert np.abs(mc.forces(u)).max() < ROUNDOFF
    assert max(mc.eta_sensitivity((0.6, 1.5)).values()) < 1e-6


def test_fourier_sign_convention_is_explicit():
    """ft_sign=-1 accepts conjugate-synthesised samples with the same result."""
    model = _orthogonal_model()
    reps = (2, 2, 4)
    phi_neg = _grid_matrices(
        _reciprocal_kernel(model["reps"], 2, np.random.default_rng(7)), model["reps"], sign=-1)
    a = build_correction(model["cell"], model["positions"], model["born"], model["dielectric"],
                         model["reps"], model["phi0"], model["grid_h"], reps, **CHEAP)
    b = build_correction(model["cell"], model["positions"], model["born"], model["dielectric"],
                         model["reps"], phi_neg, model["grid_h"], reps, ft_sign=-1, **CHEAP)
    assert_allclose(a.correction_matrix, b.correction_matrix, atol=ROUNDOFF)


def test_branch_interpolation_is_physical():
    """Arbitrary-q branches interpolate the original kernel, not a folded one."""
    cell = np.eye(3) * 4.0
    positions = np.zeros((1, 3))
    born = np.zeros((1, 3, 3))  # neutral: no dipolar response
    reps = (2, 1, 1)
    phi0 = np.zeros((2, 1, 3, 1, 3), dtype=complex)
    phi0[1, 0, 0, 0, 0] = 4.0  # Phi_xx(0) = 0, Phi_xx(1/2) = 4 eV/angstrom**2
    mc = build_correction(cell, positions, born, np.eye(3) * 2.0, reps, phi0,
                          _grid_h(reps), (4, 1, 1), eta=0.3, nreal=2, nrecip=3)
    root2 = np.sqrt(2.0)
    for q, value in (([0.0, 0.0, 0.0], 0.0), ([0.5, 0.0, 0.0], 4.0),
                     ([0.25, 0.0, 0.0], 2.0), ([0.125, 0.0, 0.0], 2.0 - root2)):
        block = mc.branch_dynamical(q, "unsplit")[0, 0, 0, 0]
        # a folded-kernel synthesis returns 2 - i*sqrt(2) at q = 1/8 here
        assert block.imag == pytest.approx(0.0, abs=1e-12)
        assert block.real == pytest.approx(value, abs=1e-12)
    assert mc.branch_dynamical([0.125, 0.0, 0.0], "difference")[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-12)


def test_branch_matrices_are_hermitian_off_grid(correction_larger):
    """Off-grid branch matrices are Hermitian with real diagonal elements."""
    mc = correction_larger
    n = len(mc.positions)
    for q in ([0.125, 0.125, 0.125], [0.1, -0.2, 0.3], [0.0625, 0.375, -0.125]):
        for branch in ("unsplit", "split", "difference"):
            block = mc.branch_dynamical(q, branch)
            assert np.abs(block - block.transpose(2, 3, 0, 1).conj()).max() < ROUNDOFF
            diagonal = np.array([block[i, mu, i, mu] for i in range(n) for mu in range(3)])
            assert np.abs(diagonal.imag).max() < ROUNDOFF


def test_branch_equals_dense_matrix_on_target_grid(correction_larger):
    """On the target grid the interpolant equals the assembled correction block."""
    mc = correction_larger
    n = len(mc.positions)
    reps = mc.target_reps
    matrix = mc.correction_matrix
    for h in _grid_h(reps):
        q = h / np.array(reps, dtype=float)
        block = np.zeros((n, 3, n, 3), dtype=complex)
        for t in np.ndindex(*reps):
            image = (t[0] * reps[1] + t[1]) * reps[2] + t[2]
            columns = (3 * image * n, 3 * (image + 1) * n)
            block = block + matrix[: 3 * n, columns[0]:columns[1]].reshape(n, 3, n, 3) * np.exp(
                2j * np.pi * (np.asarray(q) @ np.array(t)))
        assert np.abs(mc.branch_dynamical(q, "difference") - block).max() < 1e-10


def test_eta_sensitivity_preserves_fourier_convention():
    """The convergence diagnostic rebuilds through the same sign convention."""
    model = _orthogonal_model()
    phi_neg = _grid_matrices(_reciprocal_kernel(model["reps"], 2, np.random.default_rng(7)),
                             model["reps"], sign=-1)
    mc = build_correction(model["cell"], model["positions"], model["born"], model["dielectric"],
                          model["reps"], phi_neg, model["grid_h"], (2, 2, 4), ft_sign=-1, **CHEAP)
    assert mc.eta_sensitivity((1.0,))["max_abs_change_eta_x1"] < ROUNDOFF


# --------------------------------------------------------------------------
# Wigner-Seitz image weights
# --------------------------------------------------------------------------
def test_ws_image_weights_multiplicities():
    """Simple-cubic superlattice: image weights are the WS multiplicities."""
    cell = np.eye(3) * 4.0
    cand, weights, rho, residual = _pair_ws_weights(cell, np.zeros((1, 3)), (2, 2, 2))
    assert residual == 0.0
    expected = {(0, 0, 0): 1.0, (1, 0, 0): 0.5, (1, 1, 0): 0.25, (1, 1, 1): 0.125}
    for t, value in expected.items():
        cls = (t[0] * 2 + t[1]) * 2 + t[2]
        nonzero = weights[0, 0][rho == cls]
        nonzero = nonzero[nonzero > 0]
        assert_allclose(weights[0, 0][rho == cls].sum(), 1.0, atol=1e-12)
        assert_allclose(nonzero, value, atol=1e-12)


def test_ws_image_weights_know_their_nearest_image():
    """Reviewer regressions: pair-offset sign and class-preserving wrapping."""
    cell = np.eye(3) * 4.0
    # sign of the pair offset: tau_j - tau_i + T must pick T = -1 (3 A), not +1 (5 A)
    cand, weights, rho, residual = _pair_ws_weights(
        cell, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), (2, 1, 1))
    assert residual == 0.0
    chosen, w = _class_candidates(cand, weights, rho, 0, 1, 1)
    assert chosen.tolist() == [[-1, 0, 0]]
    assert_allclose(w, 1.0, atol=1e-12)
    pair = np.array([1.0, 0.0, 0.0])  # tau_j - tau_i
    assert_allclose(chosen.astype(float) @ cell, [[-4.0, 0.0, 0.0]], atol=1e-12)
    assert np.linalg.norm(pair + chosen[0].astype(float) @ cell) == pytest.approx(3.0)
    assert np.linalg.norm(pair + np.array([4.0, 0.0, 0.0])) == pytest.approx(5.0)
    # wrapping must not move images between residue classes
    cand, weights, rho, residual = _pair_ws_weights(
        cell, np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]]), (2, 1, 1))
    assert residual == 0.0
    chosen0, _ = _class_candidates(cand, weights, rho, 0, 1, 0)
    chosen1, _ = _class_candidates(cand, weights, rho, 0, 1, 1)
    assert chosen0.tolist() == [[0, 0, 0]]  # 3.5 A stays in class 0
    assert chosen1.tolist() == [[-1, 0, 0]]  # 0.5 A is the class-1 image
    # pathological skew basis: the true nearest images are far outside a fixed
    # +/-2 window of primitive translations
    pathological = np.array([[1.0, 0.0, 0.0], [10.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    cand, weights, rho, residual = _pair_ws_weights(
        pathological, np.zeros((1, 3)), (1, 2, 1))
    assert residual == 0.0
    chosen, w = _class_candidates(cand, weights, rho, 0, 0, 1)
    assert sorted(chosen.tolist()) == [[-10, 1, 0], [10, -1, 0]]
    assert_allclose(w, 0.5, atol=1e-12)
    distances = np.linalg.norm(chosen.astype(float) @ pathological, axis=1)
    assert_allclose(distances, 0.1, atol=1e-9)


def test_ws_true_translation_labels_survive_target_folding():
    """Parent regression: the unwrapped pair offset keeps its true label."""
    cell = np.eye(3) * 4.0
    positions = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    cand, weights, rho, residual = _pair_ws_weights(cell, positions, (1, 1, 1))
    assert residual == 0.0
    chosen, w = _class_candidates(cand, weights, rho, 0, 1, 0)
    assert chosen.tolist() == [[-1, 0, 0]]
    assert_allclose(w, 1.0, atol=1e-12)
    assert np.linalg.norm(positions[1] - positions[0] + chosen[0] @ cell) == pytest.approx(0.5)
    # the true label (not a superlattice-wrapped stand-in) selects the target
    # image: the coupling folded into a (4,1,1) target lands in residue 3
    kernel = np.zeros((1, 1, 1, 2, 3, 2, 3))
    kernel[0, 0, 0, 0, 0, 1, 0] = 7.0
    folded = _fold_to_target(weights, kernel, rho, cand, (4, 1, 1))
    assert folded[3, 0, 0, 0, 0, 1, 0] == pytest.approx(7.0)
    assert folded[0, 0, 0, 0, 0, 1, 0] == pytest.approx(0.0)


def test_ws_long_diagonal_repetitions_are_exact():
    """A (64,1,1) repetition resolves every class, including far images."""
    cell = np.eye(3) * 4.0
    cand, weights, rho, residual = _pair_ws_weights(cell, np.zeros((1, 3)), (64, 1, 1))
    assert residual == 0.0
    assert len(np.unique(rho)) == 64
    for cls in range(64):
        assert_allclose(weights[0, 0][rho == cls].sum(), 1.0, atol=1e-12)
    # class 32 has exactly two equidistant images at +-32 a
    chosen, w = _class_candidates(cand, weights, rho, 0, 0, 32)
    assert sorted(chosen.tolist()) == [[-32, 0, 0], [32, 0, 0]]
    assert_allclose(w, 0.5, atol=1e-12)
    assert_allclose(np.linalg.norm(chosen.astype(float) @ cell, axis=1), 128.0, atol=1e-9)
    chosen, w = _class_candidates(cand, weights, rho, 0, 0, 1)
    assert chosen.tolist() == [[1, 0, 0]]
    assert_allclose(w, 1.0, atol=1e-12)
    chosen, w = _class_candidates(cand, weights, rho, 0, 0, 63)
    assert chosen.tolist() == [[-1, 0, 0]]
    assert_allclose(w, 1.0, atol=1e-12)
    # end-to-end: a long diagonal target must construct instead of raising
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    born = np.zeros((2, 3, 3))
    born[0] = np.eye(3) * 1.5
    born[1] = -born[0]
    reps_m = (1, 1, 1)
    mc = build_correction(cell, positions, born, np.eye(3) * 4.0, reps_m,
                          np.zeros((1, 2, 3, 2, 3)), _grid_h(reps_m), (64, 1, 1),
                          eta=0.3, nreal=2, nrecip=3)
    assert mc.natom_target == 128
    assert mc.metadata["weight_class_sum_residual"] == 0.0
    correction = mc.correction_matrix
    assert np.isfinite(correction).all()
    assert np.abs(correction).max() > 0.0
    assert_allclose(correction, mc.counterterm_matrix(), atol=1e-12)
    # the shared-wavevector guarantee survives the long target: a matching-
    # periodic (primitive-cell) pattern has zero correction energy
    pattern = np.array([0.01, 0.0, 0.0, -0.01, 0.0, 0.004])
    tiled = mc.matching_periodic_tiling(pattern)
    assert np.abs(correction @ tiled).max() < 1e-12 * max(np.abs(correction).max(), 1.0)


def test_ws_image_weights_match_independent_bruteforce():
    """Cubic, skew and pathological cells agree with an independent enumeration."""
    cubic = np.eye(3) * 4.0
    _assert_ws_matches_bruteforce(cubic, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), (2, 2, 2))
    skew = np.array([[3.0, 0.0, 0.0], [0.7, 3.2, 0.0], [0.4, -0.5, 3.6]])
    positions = np.array([[0.0, 0.0, 0.0], [0.33, 0.21, 0.47]]) @ skew
    for reps in ((2, 2, 2), (2, 4, 2)):
        _assert_ws_matches_bruteforce(skew, positions, reps)
    pathological = np.array([[1.0, 0.0, 0.0], [10.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    _assert_ws_matches_bruteforce(pathological, np.zeros((1, 3)), (1, 2, 1), shells=14)


# --------------------------------------------------------------------------
# site mapping, permutations and periodic images
# --------------------------------------------------------------------------
def test_site_mapping_permutation_and_image_shift(sto_correction):
    """Arbitrary atom order and periodic-image choices give identical results."""
    mc, model = sto_correction
    target_reps = (4, 4, 2)
    cell_t, positions, species = _canonical_target(model, target_reps)
    identity = mc.map_sites(cell_t, positions, species=species)
    assert np.array_equal(identity.canonical_of_target, np.arange(len(positions)))
    assert np.abs(identity.positions_canonical - positions).max() < 1e-12
    rng = np.random.default_rng(23)
    perm = rng.permutation(len(positions))
    shifts = rng.integers(-1, 2, size=(len(positions), 3))
    permuted = mc.map_sites(cell_t, positions[perm] + shifts @ cell_t,
                            species=[species[k] for k in perm])
    assert np.array_equal(permuted.canonical_of_target, perm)
    u = rng.normal(size=3 * len(positions), scale=0.005)
    energy, forces = mc.evaluate(u)
    energy_p, forces_p = mc.evaluate(u.reshape(-1, 3)[perm].reshape(-1), mapping=permuted)
    forces_back = np.empty_like(forces_p)
    forces_back.reshape(-1, 3)[perm] = forces_p.reshape(-1, 3)
    assert abs(energy - energy_p) < ROUNDOFF
    assert np.abs(forces - forces_back).max() < ROUNDOFF


def test_site_mapping_rejections(sto_correction):
    """Unsupported geometry, species and assignments fail with diagnostics."""
    mc, model = sto_correction
    target_reps = (4, 4, 2)
    cell_t, positions, species = _canonical_target(model, target_reps)
    n_l = len(positions)
    with pytest.raises(ValueError, match="unstrained diagonal repetition"):
        mc.map_sites(cell_t * 1.001, positions)
    with pytest.raises(ValueError, match="atom-count changes"):
        mc.map_sites(cell_t, positions[:-1])
    with pytest.raises(ValueError, match="species"):
        mc.map_sites(cell_t, positions, species=["Ti", "Sr", "O", "O", "O"] * (n_l // 5))
    with pytest.raises(ValueError, match="do not coincide"):
        mc.map_sites(cell_t, positions + np.array([0.3, 0.0, 0.0]))
    with pytest.raises(ValueError, match="same canonical site"):
        mc.map_sites(cell_t, np.vstack([positions[:-1], positions[0]]))
    with pytest.raises(ValueError, match="displacement components"):
        mc.energy(np.zeros(3 * n_l - 1))
    with pytest.raises(ValueError, match="positive finite"):
        mc.map_sites(cell_t, positions, tol=0.0)


def test_single_site_reference_mapping():
    """One-site references map without the pairwise ambiguity machinery."""
    cell = np.diag([3.0, 3.2, 3.4])
    positions = np.zeros((1, 3))
    born = np.zeros((1, 3, 3))  # charge neutral: no dipolar response at all
    reps = (2, 2, 2)
    phi0 = _grid_matrices(_reciprocal_kernel(reps, 1, np.random.default_rng(31)), reps)
    mc = build_correction(cell, positions, born, np.eye(3) * 4.0, reps, phi0,
                          _grid_h(reps), reps, **CHEAP)
    cell_t = np.diag(reps).astype(float) @ cell
    target = np.array([np.array(t) @ cell for t in np.ndindex(*reps)])
    mapping = mc.map_sites(cell_t, target)
    assert np.array_equal(mapping.canonical_of_target, np.arange(len(target)))
    assert np.abs(mapping.positions_canonical - target).max() == 0.0
    u = np.random.default_rng(32).normal(size=3 * len(target), scale=0.01)
    assert mc.energy(u) == 0.0
    assert np.abs(mc.forces(u)).max() == 0.0


def test_mapping_distances_are_cartesian(sto_correction):
    """The accepted deviation is an angstrom displacement, not a fractional residue."""
    mc, model = sto_correction
    cell_t, positions, species = _canonical_target(model, (4, 4, 2))
    shift = np.array([5e-6, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"Cartesian residual 5\.000e-06"):
        mc.map_sites(cell_t, positions + shift)
    accepted = mc.map_sites(cell_t, positions + shift, species=species, tol=1e-5)
    assert np.array_equal(accepted.canonical_of_target, np.arange(len(positions)))
    assert np.array_equal(accepted.species_canonical, species)


def test_mapping_distances_are_cartesian_on_skew_cells(correction_skew):
    """Skew cells report the true Cartesian residual, not the fractional one."""
    mc, model = correction_skew
    cell_t, positions, _ = _canonical_target(model, (2, 4, 2))
    # 0.3 angstrom along x is 0.1 in primitive fractional coordinates here
    with pytest.raises(ValueError, match=r"Cartesian residual 3\.000e-01"):
        mc.map_sites(cell_t, positions + np.array([0.3, 0.0, 0.0]))


def test_mapping_provable_search_on_pathological_skew():
    """Reviewer regression: skew-cell images are resolved in the true metric."""
    cell = np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    target = (3, 1, 1)
    reps_m = (1, 1, 1)
    mc = build_correction(cell, np.zeros((1, 3)), np.zeros((1, 3, 3)), np.eye(3) * 2.0,
                          reps_m, np.zeros((1, 1, 3, 1, 3)), _grid_h(reps_m), target,
                          eta=0.4, nreal=2, nrecip=3)
    cell_t, positions, _ = _canonical_target(dict(cell=cell, positions=np.zeros((1, 3))), target)
    displaced = positions.copy()
    displaced[0] += np.array([-0.02, 0.049, 0.0])  # 0.05292 A from the nearest site image
    mapping = mc.map_sites(cell_t, displaced, tol=0.053)
    assert np.array_equal(mapping.canonical_of_target, np.arange(3))
    assert np.array_equal(mapping.images, np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    assert np.abs(mapping.positions_canonical - positions).max() < 1e-12
    with pytest.raises(ValueError, match="Cartesian residual"):
        mc.map_sites(cell_t, displaced, tol=0.05)


def test_ambiguous_site_assignment_rejected(correction_larger, ortho):
    """A tolerance spanning two reference sites is ambiguous, not a nearest pick."""
    mc = correction_larger
    cell_t, positions, _ = _canonical_target(ortho, (2, 2, 4))
    sites = ortho["positions"]
    separation = np.linalg.norm(sites[1] - sites[0])
    with pytest.raises(ValueError, match="ambiguous site assignment"):
        mc.map_sites(cell_t, positions, tol=1.5 * separation)


# --------------------------------------------------------------------------
# input validation and recorded metadata
# --------------------------------------------------------------------------
def test_neutrality_tolerance_is_declared_and_propagated(ortho):
    """ElectricalResponse's declared neutrality tolerance is what is accepted."""
    born = ortho["born"].copy()
    born[0] += 5e-5 * np.eye(3)  # net charge 5e-5 per component
    assert np.abs(born.sum(axis=0)).max() == pytest.approx(5e-5)

    def build(**kw):
        return build_correction(ortho["cell"], ortho["positions"], born, ortho["dielectric"],
                                ortho["reps"], ortho["phi0"], ortho["grid_h"], ortho["reps"],
                                **{**CHEAP, **kw})

    # the default is the ElectricalResponse-compatible bound, not dipole_ifc's 1e-8
    mc = build()
    assert mc.metadata["ewald"]["neutrality_tolerance"] == pytest.approx(1e-4)
    assert mc._neutrality_tolerance == pytest.approx(1e-4)
    assert_allclose(mc.born, born, atol=0)  # validity bound, not a renormalisation
    # declared data stays valid for every later re-evaluation
    assert np.isfinite(mc._dipole_at(np.array([0.1, 0.0, 0.0]))).all()
    mc._d_target_samples = None  # force the re-evaluation path, not the stored samples
    assert np.isfinite(mc._dipole_target_grid()).all()
    assert np.isfinite(mc.branch_dynamical(np.array([0.25, 0.0, 0.0]), "difference")).all()
    assert np.isfinite(mc.eta_sensitivity((0.5,))["max_abs_change_eta_x0.5"])
    # a stricter declaration refuses the same data instead of repairing it
    with pytest.raises(ValueError, match="charge neutrality"):
        build(neutrality_tolerance=1e-8)
    # the caller passes the response's own declared tolerance explicitly
    tighter = build(neutrality_tolerance=5e-5)
    assert tighter.metadata["ewald"]["neutrality_tolerance"] == pytest.approx(5e-5)


def test_malformed_inputs_are_rejected(ortho):
    reps = ortho["reps"]
    cell, positions = ortho["cell"], ortho["positions"]
    born, dielectric = ortho["born"], ortho["dielectric"]
    phi0, grid_h = ortho["phi0"], ortho["grid_h"]

    with pytest.raises(ValueError, match="positive integers"):
        build_correction(cell, positions, born, dielectric, (2, 0, 2), phi0, grid_h, reps)
    with pytest.raises(ValueError, match="phi0_grid must be finite"):
        build_correction(cell, positions, born, dielectric, reps, phi0[:-1], grid_h, reps)
    with pytest.raises(ValueError, match="exactly once"):
        build_correction(cell, positions, born, dielectric, reps, phi0,
                         np.zeros((8, 3), dtype=int), reps)
    with pytest.raises(ValueError, match="nonsingular"):
        build_correction(np.zeros((3, 3)), positions, born, dielectric, reps, phi0, grid_h, reps)
    bad = phi0.copy()
    bad[:, 0, 0, 1, 1] += 100.0  # break the reciprocal-kernel symmetries
    with pytest.raises(ValueError, match="reciprocal-kernel symmetries"):
        build_correction(cell, positions, born, dielectric, reps, bad, grid_h, reps)
    bad_born = born.copy()
    bad_born[1] = np.eye(3) * 3.0  # not neutral
    with pytest.raises(ValueError, match="charge neutrality"):
        build_correction(cell, positions, bad_born, dielectric, reps, phi0, grid_h, reps)
    with pytest.raises(ValueError, match="symmetric positive definite"):
        build_correction(cell, positions, born, np.diag([1.0, 1.0, -1.0]), reps, phi0, grid_h, reps)
    with pytest.raises(ValueError, match="ft_sign"):
        build_correction(cell, positions, born, dielectric, reps, phi0, grid_h, reps, ft_sign=2)
    with pytest.raises(ValueError, match="eta"):
        build_correction(cell, positions, born, dielectric, reps, phi0, grid_h, reps, eta=-1.0)


def test_phi0_symmetrization_requires_explicit_tolerance(ortho):
    """No built-in tolerance: roundoff passes, larger violations need a declaration."""
    reps = ortho["reps"]
    policy_shells = dict(eta=0.3, nreal=2, nrecip=3)  # policy is Ewald independent

    def build(phi0, **kw):
        return build_correction(ortho["cell"], ortho["positions"], ortho["born"],
                                ortho["dielectric"], reps, phi0, ortho["grid_h"], reps,
                                **{**policy_shells, **kw})

    # genuine no-projection default: roundoff-level samples are used as given
    plain = build(ortho["phi0"])
    record = plain.metadata["phi0_symmetrization"]
    assert record["declared_tolerance"] is None
    assert record["mode"].startswith("roundoff-only")
    assert record["accepted_violation_max_abs"] <= record["roundoff_limit"]
    assert record["applied_max_abs_change"] < ROUNDOFF
    assert np.abs(plain.phi0_samples - ortho["phi0"]).max() < ROUNDOFF
    assert plain.diagnostics["phi0_sample_reproduction_max_abs"] < ROUNDOFF
    # a finite-difference-sized violation is refused, not silently repaired
    noisy = ortho["phi0"].copy()
    noisy[:, 0, 0, 1, 1] += 1e-3
    with pytest.raises(ValueError, match="phi0_symmetry_tol"):
        build(noisy)
    # the caller's declared tolerance is what admits the projection
    accepted = build(noisy, phi0_symmetry_tol=1e-2)
    record = accepted.metadata["phi0_symmetrization"]
    assert record["mode"].startswith("explicit tolerance")
    assert record["declared_tolerance"] == pytest.approx(1e-2)
    assert record["accepted_violation_max_abs"] == pytest.approx(1e-3, abs=1e-12)
    assert record["applied_max_abs_change"] == pytest.approx(5e-4, abs=1e-12)
    assert accepted.diagnostics["phi0_sample_reproduction_max_abs"] < ROUNDOFF
    # a tighter declaration refuses the same data
    with pytest.raises(ValueError, match="phi0_symmetry_tol"):
        build(noisy, phi0_symmetry_tol=1e-5)
    # a declared tolerance is a bound, not a blanket permission
    bad = ortho["phi0"].copy()
    bad[:, 0, 0, 1, 1] += 1e-1
    with pytest.raises(ValueError, match="reciprocal-kernel symmetries"):
        build(bad, phi0_symmetry_tol=1e-2)
    with pytest.raises(ValueError, match="positive finite"):
        build(ortho["phi0"], phi0_symmetry_tol=0.0)


def test_conventions_and_provenance_are_recorded(correction_equal):
    mc = correction_equal
    meta = mc.metadata
    for key in ("units", "ifc_axes", "site_order", "fourier_convention", "boundary",
                "interpolation", "weight_class_sum_residual", "ewald", "asr"):
        assert key in meta
    assert "conducting" in meta["boundary"]
    assert meta["ewald"]["eta"] == mc._eta
    assert meta["response_provenance"] is None
    assert meta["phi0_symmetrization"]["mode"].startswith("roundoff-only")
    assert meta["phi0_symmetrization"]["declared_tolerance"] is None
    assert meta["phi0_symmetrization"]["applied_max_abs_change"] < ROUNDOFF
    assert mc.phi0_samples is not None
    assert mc.diagnostics["phi0_sample_reproduction_max_abs"] < ROUNDOFF
    # supplied response tensors are preserved, not silently repaired
    assert np.array_equal(mc.born, correction_equal.born)
    assert np.array_equal(mc.dielectric, correction_equal.dielectric)
    # diagnostics are finite numbers
    for value in mc.diagnostics.values():
        assert np.isfinite(value)
    assert mc.diagnostics["shared_wavevector_count"] == 8
    # provenance passthrough (recorded verbatim, including tolerance status)
    provenance = {"source": "synthetic test fixture", "reference": "cubic model",
                  "phi0_symmetry_tol": 1e-2,
                  "phi0_symmetry_tol_status": "empirically estimated in-sample threshold, not predeclared"}
    mc2 = build_correction(mc.primitive_cell, mc.positions, mc.born, mc.dielectric,
                           mc.matching_reps, _grid_matrices(
                               _reciprocal_kernel(mc.matching_reps, 2, np.random.default_rng(7)),
                               mc.matching_reps),
                           mc.grid_h, mc.target_reps, response_provenance=provenance, **CHEAP)
    assert mc2.metadata["response_provenance"] == provenance


def test_ws_lattice_ties_are_reciprocal():
    """Near-equal distances within tol are ties, not a new minimum."""
    basis = np.array([[1.0, 0.0, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target = np.array([2.10000001, 4.0, 0.0])  # 0.49999999 vs 0.50000001 from two images
    positive = sorted(map(tuple, _closest_lattice_points(basis, target, 1e-7).tolist()))
    negative = sorted(map(tuple, _closest_lattice_points(basis, -target, 1e-7).tolist()))
    assert positive == [(0, 4, 0), (1, 4, 0)]
    assert negative == [(-1, -4, 0), (0, -4, 0)]


def test_mapping_long_orthogonal_target_is_not_distance_limited():
    """A (64,1,1) target spans 63 cells; each image search is centered per offset."""
    cell = np.eye(3) * 4.0
    rng = np.random.default_rng(3)
    frac = rng.random((5, 3)) * 0.5
    positions = frac @ cell
    born = np.zeros((5, 3, 3))
    born[0] = np.eye(3)
    born[1] = -np.eye(3)
    mc = build_correction(cell, positions, born, np.eye(3) * 4.0, (1, 1, 1),
                          np.zeros((1, 5, 3, 5, 3)), _grid_h((1, 1, 1)), (64, 1, 1),
                          eta=0.3, nreal=2, nrecip=3)
    cell_t = np.diag((64, 1, 1)).astype(float) @ cell
    positions_t = np.array([(frac[s] + np.array(t)) @ cell
                            for t in np.ndindex(64, 1, 1) for s in range(5)])
    mapping = mc.map_sites(cell_t, positions_t, tol=1e-6)
    assert np.array_equal(mapping.canonical_of_target, np.arange(320))
    assert np.array_equal(mapping.images, np.array(
        [[t[0], t[1], t[2]] for t in np.ndindex(64, 1, 1) for _ in range(5)]))
