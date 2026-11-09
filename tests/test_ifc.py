"""Dense <-> resolved mapping, Fourier conventions, WS weights, ASR (ticket criterion 3)."""

import numpy as np
import pytest

from conftest import SKEW_CELL, SKEW_FRAC, random_blocks
from longrangefy import (
    IfcMappingError,
    SupercellIFC,
    apply_asr_simple,
    apply_asr_weighted,
    blocks_asr_row_sums,
    blocks_to_dense,
    blocks_to_qmatrices,
    dense_to_blocks,
    extend_blocks,
    fold_ws_weights,
    interpolation_rpoints,
    qmatrices_to_blocks,
    wigner_seitz_weights,
)


def make_blocks(seed=1, reps=(2, 1, 3), n_uc=2):
    rng = np.random.default_rng(seed)
    rpoints = np.array([(r1, r2, r3)
                        for r1 in range(reps[0])
                        for r2 in range(reps[1])
                        for r3 in range(reps[2])], dtype=int)
    return SupercellIFC(reps=reps, rpoints=rpoints,
                        blocks=random_blocks(rng, reps, n_uc))


def test_dense_to_blocks_to_dense_roundtrip_skew_cell():
    blocks = make_blocks()
    dense = blocks_to_dense(blocks, blocks.reps)
    recovered = dense_to_blocks(dense, blocks.reps)
    np.testing.assert_allclose(recovered.blocks, blocks.blocks, atol=1e-12)
    # And the dense matrix itself is recovered from a fresh fold (sign/order).
    np.testing.assert_allclose(blocks_to_dense(recovered, blocks.reps), dense, atol=1e-12)
    # Translation invariance of the reconstructed dense matrix, explicitly checked:
    # entry (i@C, j@C+R) must not depend on C.
    n1, n2, n3 = blocks.reps
    n_uc = blocks.n_uc
    d = dense.reshape(n1, n2, n3, n_uc, 3, n1, n2, n3, n_uc, 3)
    r1, r2, r3 = blocks.rpoints[3]
    values = [d[c1, c2, c3, :, :, (c1 + r1) % n1, (c2 + r2) % n2, (c3 + r3) % n3]
              for c1 in range(n1) for c2 in range(n2) for c3 in range(n3)]
    np.testing.assert_allclose(np.stack(values), np.broadcast_to(values[0],
                               (len(values), n_uc, 3, n_uc, 3)), atol=1e-12)


def test_dense_hessian_symmetry_and_hermitian_q_matrices():
    blocks = make_blocks(seed=2)
    dense = blocks_to_dense(blocks, blocks.reps)
    np.testing.assert_allclose(dense, dense.T, atol=1e-12)
    q = blocks_to_qmatrices(blocks)
    for qmatrix in q:
        np.testing.assert_allclose(qmatrix, qmatrix.conj().transpose(2, 3, 0, 1),
                                   atol=1e-12)


def test_q_transform_roundtrip_and_convention_sign():
    blocks = make_blocks(seed=3)
    q = blocks_to_qmatrices(blocks)
    recovered = qmatrices_to_blocks(q, blocks.reps)
    np.testing.assert_allclose(recovered.blocks, blocks.blocks, atol=1e-12)
    # Explicit convention check: Phi(q)[i,a,j,b] = sum_R exp(+2*pi*i*q.R) Phi(R)
    q_point = np.array([0.5, 0.0, 1.0 / 3.0])
    reference = np.einsum(
        "R,Riajb->iajb",
        np.exp(2j * np.pi * (blocks.rpoints @ q_point)), blocks.blocks)
    index = int(np.flatnonzero(
        np.all(np.isclose(blocks.rpoints / np.asarray(blocks.reps)[None, :], q_point),
               axis=1))[0])
    np.testing.assert_allclose(q[index], reference, atol=1e-12)


def test_atom_permutation_consistency():
    blocks = make_blocks(seed=4)
    dense = blocks_to_dense(blocks, blocks.reps)
    permutation = np.array([1, 0])          # swap the two sites
    perm_blocks = SupercellIFC(
        reps=blocks.reps, rpoints=blocks.rpoints,
        blocks=blocks.blocks[:, permutation][:, :, :, permutation])
    perm_dense = blocks_to_dense(perm_blocks, blocks.reps)
    n_cells = np.prod(blocks.reps)
    n_uc = blocks.n_uc
    permuted_atoms = []
    for cell in range(n_cells):
        permuted_atoms.extend(cell * n_uc + permutation)
    permuted_atoms = np.asarray(permuted_atoms)
    flat = 3 * permuted_atoms[:, None] + np.arange(3)[None, :]
    index = flat.reshape(-1)
    expected = dense[np.ix_(index, index)]
    np.testing.assert_allclose(perm_dense, expected, atol=1e-12)


def test_cell_axis_permutation_consistency():
    # Swap the n1/n2 axes of reps and the corresponding cell rows: same physics.
    blocks = make_blocks(seed=5, reps=(2, 1, 3))
    q = blocks_to_qmatrices(blocks)
    permuted_reps = (1, 2, 3)
    permuted_rpoints = blocks.rpoints[:, [1, 0, 2]]
    permuted_blocks = SupercellIFC(reps=permuted_reps, rpoints=permuted_rpoints,
                                   blocks=blocks.blocks)
    q_permuted = blocks_to_qmatrices(permuted_blocks)
    # q components permute with the reciprocal axes: (m2/n2, m1/n1, m3/n3)
    old_grid = np.asarray([(a, b, c) for a in range(2) for b in range(1)
                           for c in range(3)]) / np.array([2.0, 1.0, 3.0])
    new_grid = permuted_rpoints / np.asarray(permuted_reps)[None, :]
    for i, point in enumerate(old_grid):
        target = point[[1, 0, 2]]
        j = int(np.flatnonzero(np.all(np.isclose(new_grid, target), axis=1))[0])
        np.testing.assert_allclose(q_permuted[j], q[i], atol=1e-12)


def test_plain_replication_is_exact_inverse_only():
    blocks = make_blocks(seed=6, reps=(2, 2, 2))
    with pytest.raises(IfcMappingError, match="extend_blocks"):
        blocks_to_dense(blocks, (4, 2, 2))


def test_weighted_extension_to_larger_target_is_finite_range_and_symmetric():
    blocks = make_blocks(seed=6, reps=(2, 2, 2))
    ws_rpoints, weights, _ = weights_for(blocks)
    extended = extend_blocks(blocks, weights, ws_rpoints, (4, 2, 2))
    larger = blocks_to_dense(extended, (4, 2, 2))
    n_uc = blocks.n_uc

    def atom(cell, site=0):
        c1, c2, c3 = cell
        return ((c1 * 2 + c2) * 2 + c3) * n_uc + site

    def entry(a, b):
        return larger[3 * a:3 * a + 3, 3 * b:3 * b + 3]

    target_index = {tuple(int(v) for v in r): i
                    for i, r in enumerate(extended.rpoints)}
    # One primitive cell along axis 0: the class-1 value.
    np.testing.assert_allclose(
        entry(atom((0, 0, 0)), atom((1, 0, 0))),
        extended.blocks[target_index[(1, 0, 0)], 0, :, 0, :], atol=1e-12)
    # The negative representative (-1 = 3 mod 4) carries the same class value.
    np.testing.assert_allclose(
        entry(atom((0, 0, 0)), atom((3, 0, 0))),
        extended.blocks[target_index[(3, 0, 0)], 0, :, 0, :], atol=1e-12)
    # For a diagonal site pair the WS weight is symmetric under R -> -R, so the
    # +1 and -1 classes coincide there; off-diagonal pairs need not (the weight
    # depends on the site separation), which the dense symmetry below captures.
    np.testing.assert_allclose(extended.blocks[target_index[(3, 0, 0)], 0, :, 0, :],
                               extended.blocks[target_index[(1, 0, 0)], 0, :, 0, :],
                               atol=1e-12)
    # Two cells along axis 0 is outside the interpolation support: exactly zero.
    np.testing.assert_allclose(entry(atom((0, 0, 0)), atom((2, 0, 0))), 0.0,
                               atol=1e-15)
    np.testing.assert_allclose(larger, larger.T, atol=1e-12)


def weights_for(blocks):
    ws_rpoints = interpolation_rpoints(blocks.reps)
    ws_weights = wigner_seitz_weights(
        ws_rpoints, SKEW_CELL, SKEW_FRAC @ SKEW_CELL, blocks.reps, toldist=1e-8)
    folded = fold_ws_weights(ws_weights, ws_rpoints, blocks.reps)
    return ws_rpoints, ws_weights, folded


def test_wigner_seitz_weight_sum_rule_and_boundary_fractions():
    reps = (2, 2, 2)
    positions = SKEW_FRAC @ SKEW_CELL
    rpoints = interpolation_rpoints(reps)
    weights = wigner_seitz_weights(rpoints, SKEW_CELL, positions, reps, toldist=1e-8)
    assert weights.shape == (2, 2, rpoints.shape[0])
    # Per-pair sum rule: weights over the interpolation range count nq images.
    np.testing.assert_allclose(weights.sum(axis=2), 8.0, atol=1e-8)
    # The WS cell of the q-superlattice is small: most range points get zero
    # weight, boundary points get fractional weights, and exactly one point per
    # fold class is counted (folded weights sum to nq as well).
    assert np.count_nonzero(weights[0, 0]) < rpoints.shape[0]
    assert weights.max() <= 1.0 + 1e-12
    assert weights.min() >= 0.0
    folded = fold_ws_weights(weights, rpoints, reps)
    assert folded.shape == (2, 2, 8)
    np.testing.assert_allclose(folded.sum(axis=2), 8.0, atol=1e-8)
    # Cubic 2x2x2 folding distributes exactly one image per class (symmetry).
    np.testing.assert_allclose(folded, 1.0, atol=1e-10)


def test_iq_extension_is_sample_reproducing_on_matching_grid():
    """The central algebraic property: I_Q[Phi](q in Q) = Phi(q)."""
    for seed, reps in ((7, (2, 2, 2)), (8, (3, 1, 2)), (9, (1, 1, 1)),
                       (13, (2, 2, 1))):
        blocks = make_blocks(seed=seed, reps=reps)
        ws_rpoints, weights, folded = weights_for(blocks)
        extended = extend_blocks(blocks, weights, ws_rpoints, reps)
        np.testing.assert_allclose(
            blocks_to_qmatrices(extended), blocks_to_qmatrices(blocks), atol=1e-10)
        # The real-space extension is the folded-weight scaling of the blocks.
        np.testing.assert_allclose(
            extended.blocks,
            blocks.blocks * np.transpose(folded, (2, 0, 1))[:, :, None, :, None],
            atol=1e-12)


def test_iq_extension_folds_into_larger_target():
    blocks = make_blocks(seed=10, reps=(2, 2, 2))
    ws_rpoints, weights, _ = weights_for(blocks)
    extended = extend_blocks(blocks, weights, ws_rpoints, (4, 2, 2))
    assert extended.reps == (4, 2, 2)
    assert extended.nrpt == 16
    # Shared matching-grid wavevectors are still reproduced exactly in the
    # larger target: the same q samples, in the target's own block ordering.
    q_matching = blocks_to_qmatrices(blocks)
    q_target = blocks_to_qmatrices(extended)
    matching_grid = blocks.rpoints / np.asarray(blocks.reps)[None, :]
    target_grid = extended.rpoints / np.asarray(extended.reps)[None, :]
    for i, q in enumerate(matching_grid):
        j = int(np.flatnonzero(np.all(np.isclose(target_grid, q), axis=1))[0])
        np.testing.assert_allclose(q_target[j], q_matching[i], atol=1e-10)
    # The target dense matrix is symmetric and has the finite-range support of
    # the declared extension (zero beyond the interpolation range).
    dense = blocks_to_dense(extended, (4, 2, 2))
    np.testing.assert_allclose(dense, dense.T, atol=1e-10)


def test_asr_conventions_zero_their_own_row_sums():
    blocks = make_blocks(seed=11, reps=(2, 2, 2))
    _, _, folded = weights_for(blocks)
    assert blocks_asr_row_sums(blocks) > 1e-6
    # For this 2x2x2 fixture the folded weights are exactly one, so the
    # weighted and simple conventions coincide; scale the weights to make the
    # conventions distinct and check that each zeroes only its own row sums.
    assert np.allclose(folded, 1.0, atol=1e-10)
    scaled = folded * (1.0 + 0.25 * np.arange(folded.shape[2])[None, None, :])
    weighted = apply_asr_weighted(blocks, scaled)
    assert blocks_asr_row_sums(weighted, scaled) < 1e-10
    assert blocks_asr_row_sums(weighted) > 1e-6          # simple convention not zeroed
    assert blocks_asr_row_sums(weighted, folded) > 1e-6  # unit weights: same as simple
    simple = apply_asr_simple(blocks)
    assert blocks_asr_row_sums(simple) < 1e-10
    assert blocks_asr_row_sums(simple, scaled) > 1e-6    # scaled convention not zeroed
    # With unit folded weights both corrections coincide exactly.
    np.testing.assert_allclose(apply_asr_weighted(blocks, folded).blocks,
                               simple.blocks, atol=1e-12)
    # ASR correction touches only the R = 0 self block.
    changed = np.abs(simple.blocks - blocks.blocks).max(axis=(1, 2, 3, 4))
    izero = int(np.argmin(np.abs(blocks.rpoints).sum(axis=1)))
    assert changed[izero] > 0.0
    np.testing.assert_allclose(np.delete(changed, izero), 0.0, atol=1e-15)


def test_shape_and_mismatch_validation():
    blocks = make_blocks(seed=12)
    with pytest.raises(IfcMappingError, match="does not match"):
        dense_to_blocks(np.zeros((7, 7)), blocks.reps)
    with pytest.raises(IfcMappingError, match="shape"):
        blocks_to_qmatrices(SupercellIFC(reps=(2, 2, 2), rpoints=np.zeros((8, 3), int),
                                         blocks=np.zeros((8, 2, 3, 2, 2))))
    with pytest.raises(IfcMappingError, match="ws_weights"):
        extend_blocks(blocks, np.ones((2, 2, 2)), interpolation_rpoints(blocks.reps),
                      blocks.reps)
    with pytest.raises(IfcMappingError, match="shape"):
        qmatrices_to_blocks(np.zeros((5, 2, 3, 2, 3), complex), (2, 1, 3))
