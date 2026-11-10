"""Dense <-> translation-resolved IFC mapping, Fourier transforms, WS weights.

Conventions (shared by both harmonic branches; recorded in metadata):

- ``blocks[irpt, i, a, j, b]`` = ``Phi_R``: force constant coupling site ``i``
  Cartesian ``a`` of the home cell to site ``j`` Cartesian ``b`` of the cell
  displaced by integer ``R = rpoints[irpt]`` (primitive-cell indices),
  eV/Angstrom^2, real for a conservative real Hessian.
- ``rpoints`` are the folded fundamental range ``R in [0, n)`` per axis with
  ``irpt = (r1*n2 + r2)*n3 + r3`` (matches ``SupercellMapping.atom_index``).
- Fourier transforms on the matching grid:
  ``Phi(q) = sum_R exp(+2*pi*i*q.R) Phi(R)`` (forward, no Bloch site phases),
  ``Phi(R) = (1/nq) sum_q exp(-2*pi*i*q.R) Phi(q)`` (inverse).
- ``Phi_alias(R)`` obtained from matching-grid samples on any representative
  of ``R`` is representative-independent (it is the aliased image sum of the
  underlying supercell data); hence a single fundamental range suffices.
- Wigner-Seitz image weights follow ABINIT ``wght9`` (``brav = 1`` branch,
  adapted from pymultibinit ``pyeffpot/supercell_builder.py``):
  1 inside the WS cell of the q-grid superlattice, ``1/(k+1)`` when the
  point lies on ``k`` boundary planes, 0 outside.  ``sum_R w = nq`` per atom
  pair is enforced (checked, not assumed).

The extension operator ``I_Q`` is realised as :func:`extend_blocks`: every
point of the interpolation range (``interpolation_rpoints``, ABINIT
``bigbx9`` style, covering the Wigner-Seitz cell of the q-superlattice) is
weighted with ``w(i, j, R)`` and folded onto the target cell.  On the matching
grid this operator is exactly sample-reproducing (the Fourier transform of the
WS weights vanishes at every nonzero reciprocal vector of the q-superlattice),
which is the matching-grid identity ``I_Q[Phi](q in Q) = Phi(q)`` used by the
correction construction.  :func:`blocks_to_dense` is deliberately only the
exact inverse of :func:`dense_to_blocks` for the sampled cell; target cells go
through :func:`extend_blocks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "IfcMappingError",
    "SupercellIFC",
    "dense_to_blocks",
    "blocks_to_dense",
    "blocks_to_qmatrices",
    "qmatrices_to_blocks",
    "wigner_seitz_weights",
    "interpolation_rpoints",
    "fold_ws_weights",
    "extend_blocks",
    "apply_asr_weighted",
    "apply_asr_simple",
    "blocks_asr_row_sums",
]


class IfcMappingError(ValueError):
    """Inconsistent IFC shapes, grids, or weight bookkeeping."""


@dataclass(frozen=True)
class SupercellIFC:
    """Translation-resolved IFC blocks of a diagonal supercell sample.

    ``blocks`` is real; the Fourier dual carries the complex phase.  ``units``
    and ``order`` are recorded rather than assumed.
    """

    reps: Tuple[int, int, int]
    rpoints: np.ndarray                 # (nrpt, 3) integer, fundamental range
    blocks: np.ndarray                  # (nrpt, n_uc, 3, n_uc, 3) eV/Angstrom^2
    units: str = "eV/angstrom^2"
    order: str = "blocks[irpt, i, a, j, b]; rpoints integer primitive-cell indices"

    def __post_init__(self) -> None:
        reps = tuple(int(v) for v in self.reps)
        rpoints = np.asarray(self.rpoints, dtype=int)
        blocks = np.asarray(self.blocks, dtype=float)
        n_uc = blocks.shape[1]
        expected = (int(np.prod(reps)), n_uc, 3, n_uc, 3)
        if blocks.shape != expected:
            raise IfcMappingError(
                f"blocks must have shape {expected} for reps {reps}, got {blocks.shape}")
        if rpoints.shape != (expected[0], 3):
            raise IfcMappingError(
                f"rpoints must have shape ({expected[0]}, 3), got {rpoints.shape}")
        if not np.isfinite(blocks).all():
            raise IfcMappingError("blocks contain non-finite values")
        object.__setattr__(self, "reps", reps)
        object.__setattr__(self, "rpoints", rpoints)
        object.__setattr__(self, "blocks", blocks)

    @property
    def n_uc(self) -> int:
        return int(self.blocks.shape[1])

    @property
    def nrpt(self) -> int:
        return int(self.blocks.shape[0])


def _rpoints_from_reps(reps: Tuple[int, int, int]) -> np.ndarray:
    """Integer triples of the fundamental range in the documented fold order."""
    n1, n2, n3 = (int(v) for v in reps)
    grid = [(r1, r2, r3)
            for r1 in range(n1) for r2 in range(n2) for r3 in range(n3)]
    return np.asarray(grid, dtype=int)


def _cell_meshgrid(reps: Tuple[int, int, int]) -> Tuple[np.ndarray, ...]:
    n1, n2, n3 = (int(v) for v in reps)
    return np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")


def _flat_index(cell: np.ndarray, site: np.ndarray | int, reps, n_uc: int) -> np.ndarray:
    n1, n2, n3 = (int(v) for v in reps)
    return ((cell[..., 0] * n2 + cell[..., 1]) * n3 + cell[..., 2]) * n_uc + site


def dense_to_blocks(dense: np.ndarray, reps: Tuple[int, int, int]) -> SupercellIFC:
    """Fold a dense (3*N_m, 3*N_m) supercell Hessian into translation-resolved blocks.

    Each block is the average over all supercell home-cell origins, so exact
    translation invariance maps to itself and finite-difference noise averages
    deterministically over the documented fold order.
    """
    dense = np.asarray(dense, dtype=float)
    n1, n2, n3 = (int(v) for v in reps)
    n_cells = n1 * n2 * n3
    n_uc = dense.shape[0] // (3 * n_cells)
    if dense.shape != (3 * n_cells * n_uc, 3 * n_cells * n_uc):
        raise IfcMappingError(
            f"dense shape {dense.shape} does not match reps {reps} and n_uc {n_uc}")
    d = dense.reshape(n1, n2, n3, n_uc, 3, n1, n2, n3, n_uc, 3)
    c1, c2, c3 = _cell_meshgrid(reps)
    blocks = np.zeros((n_cells, n_uc, 3, n_uc, 3), dtype=float)
    rpoints = _rpoints_from_reps(reps)
    for irpt, (r1, r2, r3) in enumerate(rpoints):
        partner = (np.stack([
            (c1 + r1) % n1, (c2 + r2) % n2, (c3 + r3) % n3], axis=-1).astype(int))
        pair = d[c1, c2, c3, :, :, partner[..., 0], partner[..., 1], partner[..., 2], :, :]
        blocks[irpt] = pair.mean(axis=(0, 1, 2))
    return SupercellIFC(reps=reps, rpoints=rpoints, blocks=blocks)


def blocks_to_dense(
    blocks: SupercellIFC,
    target_reps: Tuple[int, int, int],
) -> np.ndarray:
    """Exact inverse of :func:`dense_to_blocks` for the same supercell.

    The fold/unfold pair is defined for one supercell: every sampled block
    ``R`` is placed once at each home cell of that same cell.  Extensions to a
    *different* target cell are the job of :func:`extend_blocks`, which folds
    the weighted interpolation range (including negative representatives) and
    is the operator both harmonic branches must share.
    """
    if tuple(int(v) for v in target_reps) != tuple(int(v) for v in blocks.reps):
        raise IfcMappingError(
            "blocks_to_dense reproduces the sampled supercell only "
            f"(target {tuple(target_reps)} != sampled {blocks.reps}); use "
            "extend_blocks for a different target cell")
    n_uc = blocks.n_uc
    t1, t2, t3 = (int(v) for v in target_reps)
    n_target = t1 * t2 * t3
    dense = np.zeros((3 * n_target * n_uc, 3 * n_target * n_uc), dtype=float)
    view = dense.reshape(t1, t2, t3, n_uc, 3, t1, t2, t3, n_uc, 3)
    c1, c2, c3 = _cell_meshgrid(target_reps)
    r1 = blocks.rpoints[:, 0]
    r2 = blocks.rpoints[:, 1]
    r3 = blocks.rpoints[:, 2]
    for irpt in range(blocks.nrpt):
        partner = (np.stack([
            (c1 + int(r1[irpt])) % t1,
            (c2 + int(r2[irpt])) % t2,
            (c3 + int(r3[irpt])) % t3], axis=-1).astype(int))
        view[c1, c2, c3, :, :, partner[..., 0], partner[..., 1], partner[..., 2], :, :] += \
            blocks.blocks[irpt]
    return dense


def blocks_to_qmatrices(blocks: SupercellIFC) -> np.ndarray:
    """Forward transform: complex (nq, n_uc, 3, n_uc, 3) on the matching grid."""
    n1, n2, n3 = blocks.reps
    qgrid = np.asarray([(m1, m2, m3)
                        for m1 in range(n1) for m2 in range(n2) for m3 in range(n3)],
                       dtype=float) / np.asarray(blocks.reps, dtype=float)[None, :]
    phase = np.exp(2j * np.pi * (qgrid @ blocks.rpoints.T))   # (nq, nrpt)
    return np.einsum("qR,Riajb->qiajb", phase, blocks.blocks)


def qmatrices_to_blocks(
    qmatrices: np.ndarray,
    reps: Tuple[int, int, int],
) -> SupercellIFC:
    """Inverse transform of matching-grid matrices onto the fundamental range."""
    qmatrices = np.asarray(qmatrices, dtype=complex)
    n1, n2, n3 = (int(v) for v in reps)
    n_q = n1 * n2 * n3
    n_uc = qmatrices.shape[1]
    if qmatrices.shape != (n_q, n_uc, 3, n_uc, 3):
        raise IfcMappingError(
            f"qmatrices must have shape ({n_q}, {n_uc}, 3, {n_uc}, 3) for reps {reps}, "
            f"got {qmatrices.shape}")
    rpoints = _rpoints_from_reps(reps)
    qgrid = np.asarray([(m1, m2, m3)
                        for m1 in range(n1) for m2 in range(n2) for m3 in range(n3)],
                       dtype=float) / np.asarray(reps, dtype=float)[None, :]
    phase = np.exp(-2j * np.pi * (rpoints.astype(float) @ qgrid.T))   # (nrpt, nq)
    blocks = np.real(np.einsum("Rq,qiajb->Riajb", phase, qmatrices)) / n_q
    return SupercellIFC(reps=reps, rpoints=rpoints, blocks=blocks)


def wigner_seitz_weights(
    rpoints: np.ndarray,
    primitive_cell: np.ndarray,
    cart_positions: np.ndarray,
    grid_reps: Tuple[int, int, int],
    toldist: float = 1.0e-8,
) -> np.ndarray:
    """WS image weights ``w(i, j, R)`` (ABINIT ``wght9`` brav=1 adaptation).

    ``rpoints`` are integer *primitive*-cell indices; ``primitive_cell`` is the
    primitive lattice basis (rows, Angstrom); ``grid_reps`` are the q-grid /
    supercell repetitions.  Boundary points of the WS cell are
    ``(k * grid_reps) @ primitive_cell`` with ``k in [-2, 2]^3 \\ 0``, and
    ``R_cart = rpoints @ primitive_cell``.  Weight ``1`` inside, ``1/(k+1)`` on
    ``k`` boundary planes, ``0`` outside.  Raises when the per-pair weight sum
    rule ``sum_R w = prod(grid_reps)`` is violated (recorded upstream
    behaviour, kept strict here).

    Adapted from pymultibinit ``pyeffpot/supercell_builder.py``
    ``_compute_rpoint_weights`` (user-authored code, reused with permission).
    """
    rpoints = np.asarray(rpoints, dtype=int)
    primitive_cell = np.asarray(primitive_cell, dtype=float)
    positions = np.asarray(cart_positions, dtype=float)
    n_uc = positions.shape[0]
    n_q = int(np.prod(grid_reps))
    reps = np.asarray(grid_reps, dtype=float)

    ptws = []
    ptws_normsq = []
    for ii in range(-2, 3):
        for jj in range(-2, 3):
            for kk in range(-2, 3):
                coeff = np.array([ii, jj, kk], dtype=float) * reps
                pp = coeff @ primitive_cell
                normsq = float(pp @ pp)
                if normsq > 1e-12:
                    ptws.append(pp)
                    ptws_normsq.append(0.5 * normsq)
    ptws = np.asarray(ptws)
    ptws_normsq = np.asarray(ptws_normsq)

    r_cart = rpoints.astype(float) @ primitive_cell   # (nrpt, 3)
    weights = np.zeros((n_uc, n_uc, rpoints.shape[0]), dtype=float)
    for ia in range(n_uc):
        for ib in range(n_uc):
            rdiff = positions[ib] - positions[ia] + r_cart
            proj = rdiff @ ptws.T
            diff = proj - ptws_normsq[None, :]
            outside = np.any(diff > toldist, axis=1)
            equidistant = np.sum(np.abs(diff) <= toldist, axis=1)
            weights[ia, ib] = np.where(outside, 0.0, 1.0 / (1.0 + equidistant))
            total = weights[ia, ib].sum()
            if abs(total - n_q) > 1e-6:
                raise IfcMappingError(
                    f"WS weight sum rule violated for pair ({ia},{ib}): "
                    f"sum = {total:.6f}, expected {n_q}")
    return weights


def interpolation_rpoints(grid_reps: Tuple[int, int, int], nqshft: int = 1) -> np.ndarray:
    """Integer R range of the common finite-range extension operator.

    Follows ABINIT ``bigbx9`` (``brav = 1``) with buffer 1:
    ``lim_i = (n_i + 1) * lqshft + 1`` and ``R in [-lim, lim]`` per axis.
    The range must contain the Wigner-Seitz cell of the q-grid superlattice;
    :func:`wigner_seitz_weights` enforces the resulting sum rule, so an
    insufficient range is an explicit error rather than silent truncation.
    """
    lqshft = 1 if nqshft == 1 else 2
    lim = [(int(n) + 1) * lqshft + 1 for n in grid_reps]
    points = [(r1, r2, r3)
              for r1 in range(-lim[0], lim[0] + 1)
              for r2 in range(-lim[1], lim[1] + 1)
              for r3 in range(-lim[2], lim[2] + 1)]
    return np.asarray(points, dtype=int)


def fold_ws_weights(
    ws_weights: np.ndarray,
    ws_rpoints: np.ndarray,
    reps: Tuple[int, int, int],
) -> np.ndarray:
    """Fold interpolation-range weights onto the matching fundamental domain.

    ``sum_{R == D mod reps} w(i, j, R)`` for every class ``D``; the result
    multiplies the translation-resolved blocks (which already represent the
    aliased image sum and are representative-independent).
    """
    weights = np.asarray(ws_weights, dtype=float)
    ws_rpoints = np.asarray(ws_rpoints, dtype=int)
    n_uc = weights.shape[0]
    nrpt = int(np.prod(reps))
    if weights.shape != (n_uc, n_uc, ws_rpoints.shape[0]):
        raise IfcMappingError(
            f"ws_weights shape {weights.shape} does not match rpoints "
            f"length {ws_rpoints.shape[0]}")
    t1, t2, t3 = (int(v) for v in reps)
    folded = np.zeros((n_uc, n_uc, nrpt), dtype=float)
    for irpt, r in enumerate(ws_rpoints):
        d = ((int(r[0]) % t1) * t2 + (int(r[1]) % t2)) * t3 + (int(r[2]) % t3)
        folded[:, :, d] += weights[:, :, irpt]
    return folded


def extend_blocks(
    blocks: SupercellIFC,
    ws_weights: np.ndarray,
    ws_rpoints: np.ndarray,
    target_reps: Tuple[int, int, int],
) -> SupercellIFC:
    """The common extension ``I_Q`` in real space, folded to a target cell.

    ``Phi_ext(D) = sum_{R in interpolation range, R mod target = D}
    w(i, j, R) Phi_alias(R mod matching reps)``.  Contributions of the
    interpolation range with zero weight are absent; target classes reached by
    no nonzero-weight representative are zero (the finite-range support of the
    declared extension), not an error.  Evaluating the Fourier dual of the
    result on the matching grid reproduces the input samples exactly
    (sample-reproducing interpolation); the harmonics workflow asserts that
    identity instead of assuming it.
    """
    weights = np.asarray(ws_weights, dtype=float)
    ws_rpoints = np.asarray(ws_rpoints, dtype=int)
    if weights.shape != (blocks.n_uc, blocks.n_uc, ws_rpoints.shape[0]):
        raise IfcMappingError(
            f"ws_weights shape {weights.shape} does not match rpoints "
            f"length {ws_rpoints.shape[0]}")
    m1, m2, m3 = (int(v) for v in blocks.reps)
    t1, t2, t3 = (int(v) for v in target_reps)
    n_uc = blocks.n_uc
    n_target = t1 * t2 * t3
    folded = np.zeros((n_target, n_uc, 3, n_uc, 3), dtype=float)
    folded_points = np.empty((n_target, 3), dtype=int)
    for d in range(n_target):
        d1, d2, d3 = (d // (t2 * t3), (d // t3) % t2, d % t3)
        folded_points[d] = (d1, d2, d3)
    for irpt, r in enumerate(ws_rpoints):
        matching = (((int(r[0]) % m1) * m2 + (int(r[1]) % m2)) * m3
                    + (int(r[2]) % m3))
        target = (((int(r[0]) % t1) * t2 + (int(r[1]) % t2)) * t3
                  + (int(r[2]) % t3))
        contribution = (blocks.blocks[matching]
                        * weights[:, :, irpt][:, None, :, None])
        if np.any(contribution):
            folded[target] += contribution
    return SupercellIFC(reps=target_reps, rpoints=folded_points, blocks=folded)


def blocks_asr_row_sums(blocks: SupercellIFC, ws_weights: np.ndarray | None = None) -> float:
    """Max acoustic residual over (i, a, b): ``sum_{j,R} Phi_R(i,a,j,b)``.

    Translational invariance requires the full 3x3 block sum over ``(j, R)`` to
    vanish; this is the block-level counterpart of the dense Hessian row sums.
    With ``ws_weights`` (folded onto the matching domain) the weighted variant
    used by ``apply_asr_weighted`` is measured instead.
    """
    if ws_weights is None:
        return float(np.abs(blocks.blocks.sum(axis=(0, 3))).max())
    weights = np.asarray(ws_weights, dtype=float)
    if weights.shape != (blocks.n_uc, blocks.n_uc, blocks.nrpt):
        raise IfcMappingError(
            f"ws_weights shape {weights.shape} does not match blocks "
            f"({blocks.n_uc}, {blocks.n_uc}, {blocks.nrpt})")
    weighted = blocks.blocks * np.transpose(weights, (2, 0, 1))[:, :, None, :, None]
    return float(np.abs(weighted.sum(axis=(0, 3))).max())


def _apply_asr(blocks: SupercellIFC, weights: np.ndarray) -> SupercellIFC:
    """Enforce reciprocity and the selected Cartesian sum rule together."""
    if weights.shape != (blocks.n_uc, blocks.n_uc, blocks.nrpt):
        raise IfcMappingError(
            f"ws_weights shape {weights.shape} does not match blocks "
            f"({blocks.n_uc}, {blocks.n_uc}, {blocks.nrpt})")
    if not np.isfinite(weights).all():
        raise IfcMappingError("ASR weights must be finite")

    reps = np.asarray(blocks.reps)
    indices = {tuple(r % reps): i for i, r in enumerate(blocks.rpoints)}
    if len(indices) != blocks.nrpt or (0, 0, 0) not in indices:
        raise IfcMappingError("rpoints must contain each matching-grid class once")
    inverse = np.array([indices[tuple((-r) % reps)] for r in blocks.rpoints])
    if not np.allclose(weights, weights.transpose(1, 0, 2)[:, :, inverse],
                       rtol=1e-12, atol=0):
        raise IfcMappingError("ASR weights must respect reciprocal site/image pairs")

    # The user-selected ASR also symmetrizes the sampled IFCs, before either
    # branch is constructed. This avoids concealing mismatched branch data.
    reciprocal = blocks.blocks[inverse].transpose(0, 3, 4, 1, 2)
    new_blocks = 0.5 * (blocks.blocks + reciprocal)
    row_sums = (new_blocks * weights.transpose(2, 0, 1)[:, :, None, :, None]
                ).sum(axis=(0, 3))
    symmetric = 0.5 * (row_sums + row_sums.transpose(0, 2, 1))
    antisymmetric = -0.5 * (row_sums - row_sums.transpose(0, 2, 1))
    izero = indices[(0, 0, 0)]
    for ia in range(blocks.n_uc):
        self_weights = weights[ia, ia]
        if self_weights[izero] != 0:
            new_blocks[izero, ia, :, ia, :] -= symmetric[ia] / self_weights[izero]
        else:
            norm = self_weights @ self_weights
            if norm == 0:
                raise IfcMappingError("ASR requires nonzero weights for every self pair")
            new_blocks[:, ia, :, ia, :] -= (
                self_weights[:, None, None] * symmetric[ia] / norm)
        for ib in range(ia + 1, blocks.n_uc):
            pair_weights = weights[ia, ib]
            norm = pair_weights @ pair_weights
            if norm == 0:
                raise IfcMappingError("ASR requires nonzero weights for every site pair")
            transfer = (antisymmetric[ia] - antisymmetric[ib]) / blocks.n_uc
            delta = pair_weights[:, None, None] * transfer / norm
            new_blocks[:, ia, :, ib, :] += delta
            new_blocks[inverse, ib, :, ia, :] += delta.transpose(0, 2, 1)
    return SupercellIFC(reps=blocks.reps, rpoints=blocks.rpoints, blocks=new_blocks,
                        units=blocks.units, order=blocks.order)


def apply_asr_weighted(
    blocks: SupercellIFC,
    ws_weights: np.ndarray,
) -> SupercellIFC:
    """Enforce weighted Cartesian ASR and reciprocity on the matching grid.

    Adapted from pymultibinit's weighted onsite ASR; reciprocal pair transfers
    also remove the antisymmetric row residual.
    """
    return _apply_asr(blocks, np.asarray(ws_weights, dtype=float))


def apply_asr_simple(blocks: SupercellIFC) -> SupercellIFC:
    """Enforce unweighted Cartesian ASR and reciprocity on the matching grid."""
    return _apply_asr(blocks, np.ones((blocks.n_uc, blocks.n_uc, blocks.nrpt)))
