"""Matched fixed-cell harmonic long-range correction (tickets 02 and 03).

Constructs, for one primitive reference cell and one explicit matching
supercell, the correction matrix

    DeltaPhi_L = Phi_split_L - Phi_unsplit_L
               = [I_Q[(Phi_0 - D)|_Q] + D_L] - I_Q[Phi_0|_Q],

where ``I_Q`` is a single common linear, sample-reproducing interpolation
operator (Fourier q->R on the matching grid with Wigner-Seitz image weights,
folded into the target cell) shared by both branches, and ``D`` is the
analytic screened dipolar response from :func:`longrangefy.dipole.dipole_ifc`
(conducting boundary: the Gamma macroscopic zero mode is omitted on both the
matching and the target grids).  The correction adds, exactly once, to the
unchanged original potential:

    E_corrected = E_original + 0.5 u^T DeltaPhi_L u
    F_corrected = F_original - DeltaPhi_L u

Conventions (all recorded in ``MatchedCorrection.metadata``)
------------------------------------------------------------
* Cells: (3, 3) row vectors in angstrom; positions Cartesian angstrom.
* IFCs: unmassweighted eV/angstrom**2, atom-major axes (i, mu, j, nu);
  energies eV per whole target cell; forces eV/angstrom.
* Wavevectors: fractional coordinates w.r.t. the primitive reciprocal
  lattice; the matching grid is the full, unshifted set q = h / reps_m.
* Fourier convention: Phi(q) = sum_T kappa(T) exp(+2i pi q . T); the inverse
  q->R transform divides by the number of grid points.  ``ft_sign=-1``
  selects data produced with the opposite synthesis sign.
* Site order of the (3N_L, 3N_L) matrices and of displacement/force vectors:
  image-major, atom-minor, ``index = site + natom * (iz + nz * (iy + ny * ix))``
  with image indices in tiled order [0, l) per axis (identical to the
  SupercellMapping.atom_index formula of the reference/IFC module).
* The dipole matrices sampled from ``dipole_ifc`` are translation-resolved in
  the same convention: their grid Fourier kernels are real to roundoff and
  coincide with Ewald Gamma blocks (verified numerically; the q-matrices are
  periodic under reciprocal-lattice shifts).  No site-phase stripping is
  applied or needed.  Both grids' dipolar samples use the primitive cell at
  the grid's own physical wavevectors; the Ewald matrix is
  periodicity-independent at convergence, so this is the target-periodic
  response evaluated at the target wavevectors and the shared-wavevector
  samples coincide exactly across grids.
* Wigner-Seitz image weights: for every atom pair (i, j) and lattice
  translation T of site j relative to site i (pair offset
  ``tau_j - tau_i + T``, the ``ifc.wigner_seitz_weights`` convention), the
  weight is 1 / (number of equidistant periodic images) when no strictly
  nearer image of the same residue class exists and 0 otherwise.  The pair
  offset is used unwrapped, so the translation label is the true one and
  target folding selects the correct image of a larger cell.  Nearest images
  per residue class are the exact closest-vector solutions of the coset
  ``c0 + (mZ)^3`` (LLL-reduced superlattice, sphere enumeration), so skew
  cells and long diagonal repetitions are supported by construction rather
  than by a fixed tolerance or window.  Per residue class the weights sum to
  exactly one; this identity is what makes ``I_Q`` reproduce its input
  samples on shared wavevectors, and class-sum residuals are reported and
  enforced.
* Branch matrices: ``branch_dynamical(q)`` evaluates the WS-weighted Fourier
  interpolant of the image-resolved kernels at any wavevector (it reproduces
  the matching-grid samples there, equals the assembled matrices' q-blocks on
  the target grid, and interpolates the original response in between), with
  the analytic dipolar matrix restored as ``D(q) - I(D)(q)``.
* Symmetrization policy: building a real-space kernel needs the Hermitian
  part of the matching-grid samples.  That projection is applied explicitly
  and its size is recorded (``phi0_symmetrization`` metadata; the accepted
  samples are available as ``MatchedCorrection.phi0_samples``); the matching
  guarantee is verified against those *accepted* samples
  (``phi0_sample_reproduction_max_abs``).  By default only roundoff-level
  violation is tolerated; anything larger is rejected rather than repaired,
  and is accepted only when the caller declares an absolute
  ``phi0_symmetry_tol`` (eV/angstrom**2).  There is deliberately no built-in
  material-specific tolerance.  No ASR projection is applied.
* Site mapping: target atoms must coincide with reference sites to within
  ``tol`` angstrom in the Cartesian metric (true nearest periodic image, found
  by a provably complete lattice search: LLL-reduced basis, dual-basis bound,
  explicit raise instead of a bounded window), exactly once each; diffusion,
  exchanges and ambiguous assignments are rejected.  A displacement, not a
  fractional coordinate, is the quantity compared to the tolerance, so the
  accepted deviation does not grow with the cell size.

Supported geometry: positive diagonal integer repetitions of one primitive
lattice that may be skew.  General integer supercell transforms, partial
periodicity, and cell strain are rejected.  Matching is guaranteed on the
shared wavevectors (those matching-grid points that are also
target-periodic); for tiled-compatible targets (repetitions divisible by the
matching repetitions) this covers every matching-grid wavevector and every
tiled matching-periodic displacement pattern.

Provenance: q->R transform, Wigner-Seitz weight and translation-fold logic
adapted from pymultibinit ``pyeffpot/supercell_builder.py`` (``_ftifc_q2r``,
``_compute_rpoint_weights``, ``_find_bound_supercell`` and the union fold),
rewritten for the matched correction; no runtime dependency on pymultibinit
or libabinit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .dipole import dipole_ifc

# Initial superlattice shells searched around the fundamental zone for
# equidistant or nearer periodic images when building Wigner-Seitz weights.
# _pair_ws_weights additionally starts from a window that provably covers every
# residue class, so this is only a floor, not a correctness bound.
_WS_SEARCH_INIT = 2
# Maximum number of window-growth iterations before giving up (raises).  The
# growth is multiplicative and the residue classes are covered by construction,
# so reaching this limit means offsets or the lattice are outside the proved
# domain -- never a silent approximation.
_WS_SEARCH_ITERATIONS = 200
# Hard resource bound on the equidistant-image multiplicity per residue class.
# Exceeding it raises instead of silently truncating the image sum.
_WS_MAX_CANDIDATES = 200_000
# Boundary-degeneracy tolerance for image distances (angstrom).
_WS_TOL = 1e-7
# Work bound (offsets x candidates) for the nearest-image search used by
# map_sites; exceeding it raises instead of silently restricting the search.
_IMAGE_SEARCH_MAX_WORK = 5_000_000
# Relative tolerance above which Hermitian/symmetry violations are rejected
# instead of only reported.
_HERM_TOL = 1e-6
# Relative bound that classifies a matching-grid input as already real and
# reciprocal up to floating-point roundoff (a few orders above double
# precision FFT noise, orders below any physical or finite-difference noise).
# A larger violation is only accepted when the caller declares an explicit
# absolute tolerance: the admissible noise level belongs to the data producer,
# not to this construction.
_PHI0_ROUNDOFF = 1e-10


def _as_reps(reps: Sequence[int], name: str) -> Tuple[int, int, int]:
    arr = np.asarray(reps, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr != np.floor(arr)) or np.any(arr < 1):
        raise ValueError(f"{name} must be three positive integers, got {reps!r}")
    return (int(arr[0]), int(arr[1]), int(arr[2]))


def _validate_cell(cell, name: str) -> np.ndarray:
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError(f"{name} must be a finite 3x3 row-vector lattice in angstrom")
    if abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError(f"{name} must be nonsingular")
    return cell


def _default_eta(cell: np.ndarray) -> float:
    """Ewald parameter default: the same formula dipole_ifc applies per cell.

    One shared value is used for every dipole evaluation (matching and
    target) so subtraction and restoration share a single numerical split.
    """
    invcell = np.linalg.inv(cell)
    return float(np.pi * 100.0 / 33.0 * np.sqrt(1.69 * np.sum(invcell @ invcell.T) / np.sum(cell @ cell.T)))


def _grid_h(reps: Tuple[int, int, int]) -> np.ndarray:
    """Full unshifted grid indices in tiled C order (h1*n2+h2)*n3+h3."""
    return np.stack(np.meshgrid(*[np.arange(k) for k in reps], indexing="ij"), axis=-1).reshape(-1, 3)


def _dipole_grid(cell_super, positions, born, dielectric, q_list, eta, nreal, nrecip,
                 neutrality_tolerance) -> np.ndarray:
    return np.stack([
        dipole_ifc(cell_super, positions, born, dielectric, q, eta=eta, nreal=nreal, nrecip=nrecip,
                   neutrality_tolerance=neutrality_tolerance)
        for q in q_list
    ])


def _q_to_r(matrices: np.ndarray, reps: Tuple[int, int, int], ft_sign: int) -> np.ndarray:
    """Inverse Fourier transform of grid matrices to tiled translation kernels.

    ``matrices`` is ordered like :func:`_grid_h`.  Returns the tiled
    (m1, m2, m3, ...) complex kernel; for ``ft_sign=-1`` the kernel of data
    synthesised with exp(-2i pi q.T) (index flip of the base transform).
    """
    m = reps
    tiled = np.asarray(matrices).reshape(m[0], m[1], m[2], *matrices.shape[1:])
    kernel = np.fft.fftn(tiled, axes=(0, 1, 2)) / (m[0] * m[1] * m[2])
    if ft_sign == -1:
        neg = [(np.arange(k) * 0 + (k - np.arange(k)) % k) for k in m]
        kernel = kernel[np.ix_(neg[0], neg[1], neg[2])]
    return kernel


def _reduce_lattice(basis: np.ndarray) -> np.ndarray:
    """Unimodular integer ``U`` with rows of ``U @ basis`` LLL-reduced (delta=0.75).

    Used to make the nearest-image completeness bound tight for skew primitive
    cells: the searched coefficient box is taken in the reduced basis, which
    keeps the dual-vector bound informative even for nearly parallel
    primitive vectors.
    """
    b = np.array(basis, dtype=float)
    U = np.eye(3, dtype=int)

    def gram_schmidt():
        star = np.zeros_like(b)
        mu = np.zeros((3, 3))
        for i in range(3):
            star[i] = b[i]
            for j in range(i):
                mu[i, j] = (b[i] @ star[j]) / (star[j] @ star[j])
                star[i] = star[i] - mu[i, j] * star[j]
        return star, mu

    k = 1
    for _ in range(1000):
        if k >= 3:
            break
        star, mu = gram_schmidt()
        for j in range(k - 1, -1, -1):
            q = int(np.rint(mu[k, j]))
            if q:
                b[k] -= q * b[j]
                U[k] -= q * U[j]
        star, mu = gram_schmidt()
        if star[k] @ star[k] >= 0.75 * (star[k - 1] @ star[k - 1]):
            k += 1
        else:
            b[[k - 1, k]] = b[[k, k - 1]]
            U[[k - 1, k]] = U[[k, k - 1]]
            k = max(k - 1, 1)
    if abs(int(round(np.linalg.det(U)))) != 1:
        raise ValueError("lattice reduction did not produce a unimodular integer transform")
    return U


def _closest_lattice_points(basis: np.ndarray, target: np.ndarray, tol: float) -> np.ndarray:
    """All integer ``k`` minimizing ``|target - k @ basis|`` (within ``tol``).

    Exact sphere enumeration (Fincke-Pohst): the Gram-Schmidt decomposition
    bounds the admissible coefficient range at every level by the remaining
    squared radius, so the enumeration is fast for the LLL-reduced bases used
    here and returns the true minimum and all its near-ties, never a truncated
    shell.  ``tol`` is an absolute squared-distance tolerance (angstrom**2).
    """
    basis = np.asarray(basis, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    n = basis.shape[0]
    star = np.zeros_like(basis)
    mu = np.zeros((n, n))
    for i in range(n):
        star[i] = basis[i]
        for j in range(i):
            mu[i, j] = (basis[i] @ star[j]) / (star[j] @ star[j])
            star[i] -= mu[i, j] * star[j]
    norms2 = np.einsum("ij,ij->i", star, star)
    ystar = np.array([(target @ star[i]) / norms2[i] for i in range(n)])
    k = np.zeros(n, dtype=int)
    best2 = [np.inf]
    visited = []  # (squared distance, coefficients) of every enumerated point

    def visit(level: int, partial2: float) -> None:
        if partial2 > best2[0] + tol:
            return
        if level < 0:
            vec = k @ basis
            d2 = float(((target - vec) ** 2).sum())
            visited.append((d2, k.copy()))
            if d2 < best2[0]:
                best2[0] = d2
            return
        center = ystar[level] - mu[level + 1:, level] @ k[level + 1:]
        radius2 = max((best2[0] + tol - partial2) / norms2[level], 0.0)
        radius = float(np.sqrt(radius2))
        for kk in range(int(np.floor(center - radius)), int(np.ceil(center + radius)) + 1):
            k[level] = kk
            c = center - kk
            visit(level - 1, partial2 + c * c * norms2[level])

    # Seed with the Babai point so the enumeration radius is finite; the
    # enumeration below rediscovers it, so it must not be appended here.
    k[:] = np.rint(ystar).astype(int)
    best2[0] = float(((target - k @ basis) ** 2).sum())
    k[:] = 0
    visit(n - 1, 0.0)
    if not visited:
        return np.empty((0, n), dtype=int)
    # Ties are decided once, after the enumeration: a point that improves the
    # squared distance by less than the tolerance is a tie, not a new minimum
    # (clearing on small improvements would break reciprocal symmetry).
    minimum = min(d2 for d2, _ in visited)
    return np.array([kv for d2, kv in visited if d2 <= minimum + tol], dtype=int)


def _pair_ws_weights(
    cell: np.ndarray,
    positions: np.ndarray,
    reps: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-pair Wigner-Seitz image weights for one supercell periodicity.

    ``weights[i, j, c]`` multiplies the translation-resolved block
    ``kappa[i, :, j, :](T_c)``, where ``T_c = candidates[c]`` is the lattice
    translation of site *j* relative to site *i* -- the same convention as
    ``ifc.wigner_seitz_weights`` (pair offset ``tau_j - tau_i + T``).  The pair
    offset is used unwrapped so the candidate label is the true translation:
    target folding then selects the correct image of a larger cell.

    For every pair and residue class the nearest images are the exact
    closest-vector solutions of the coset ``c0 + (mZ)^3`` (``c0`` the smallest
    representative of the class) with respect to the LLL-reduced superlattice,
    obtained by sphere enumeration: this is the provable closest-image
    decomposition and stays small even for long repetitions such as
    ``(64, 1, 1)``, whose far classes have their nearest image 32 primitive
    vectors away.  ``rho_index[c]`` is the residue class of candidate ``c``;
    ``class_residual`` is the per-class weight-sum deviation from one.
    """
    n = len(positions)
    m = np.array(reps, dtype=int)
    nrho = int(np.prod(m))
    # Residue-class vectors and their smallest representatives (|c0| <= m/2).
    rho_vec = np.stack(
        [
            (np.arange(nrho) // (m[1] * m[2])) % m[0],
            (np.arange(nrho) // m[2]) % m[1],
            np.arange(nrho) % m[2],
        ],
        axis=1,
    )
    class_rep = (rho_vec - m[None, :] * np.rint(rho_vec / m[None, :])).astype(int)
    rep_cart = class_rep @ cell  # (nrho, 3) Cartesian
    # LLL-reduced superlattice basis (rows m_i * a_i); same lattice as the
    # primitive superlattice, so the coefficient labels map back exactly.
    super_cell = np.diag(m.astype(float)) @ cell
    super_reduce = _reduce_lattice(super_cell)
    s_reduced = super_reduce @ super_cell
    x = positions[None, :, :] - positions[:, None, :]  # x[i, j] = tau_j - tau_i
    candidates, weights, rho_index = [], [], []
    class_sums = np.zeros((n, n, nrho))
    for ic in range(nrho):
        # Candidate T must satisfy tau_j - tau_i + T, so minimize |x + coset|.
        offset = x + rep_cart[ic]  # (n, n, 3)
        solutions = {}
        per_pair = []
        for i in range(n):
            for j in range(n):
                # |x + c0 + k.basis| is minimized; enumerate with -offset.
                ks = _closest_lattice_points(s_reduced, -offset[i, j], _WS_TOL)
                per_pair.append((i, j, ks))
                for kv in ks:
                    # k are coefficients of the reduced basis; the primitive
                    # integer translation is (k.reduce) * m elementwise.
                    label = tuple(int(v) for v in (kv @ super_reduce) * m)
                    solutions.setdefault(tuple(class_rep[ic] + label), None)
        labels = np.array(list(solutions), dtype=int)  # (nc, 3) true primitive translations
        if len(labels) > _WS_MAX_CANDIDATES:
            raise ValueError(
                f"Wigner-Seitz image search found {len(labels)} equidistant images for residue "
                f"class {ic} of repetitions {tuple(int(v) for v in m)}; the geometry is too "
                "degenerate for the image decomposition"
            )
        index = {key: pos for pos, key in enumerate(solutions)}
        block = np.zeros((n, n, len(labels)))
        for i, j, ks in per_pair:
            share = 1.0 / len(ks)
            for kv in ks:
                label = tuple(int(v) for v in (kv @ super_reduce) * m)
                block[i, j, index[tuple(class_rep[ic] + label)]] = share
        class_sums[:, :, ic] = block.sum(axis=2)
        candidates.append(labels)
        weights.append(block)
        rho_index.append(np.full(len(labels), ic))
    candidates = np.concatenate(candidates, axis=0)
    weights = np.concatenate(weights, axis=2)
    rho_index = np.concatenate(rho_index)
    class_residual = float(np.max(np.abs(class_sums - 1.0)))
    return candidates, weights, rho_index, class_residual


def _ws_synth(
    kernel: np.ndarray,
    weights: np.ndarray,
    rho_index: np.ndarray,
    candidates: np.ndarray,
    q_frac,
    ft_sign: int,
) -> np.ndarray:
    """WS-weighted Fourier interpolation of an image-resolved kernel at q.

    ``I(q) = sum_c w[i,j,c] kappa[rho(c)] exp(ft_sign*2i pi q.T_c)``.  On the
    matching grid it reproduces the kernel samples; between grid points it is
    the physical Fourier interpolation of the original response (synthesising
    an already target-folded kernel instead would alias and produce spurious
    imaginary parts).  For a reciprocal kernel ``I(q)`` is Hermitian.

    Candidates are grouped by residue class (class-major, matching
    :func:`_pair_ws_weights`), so the per-class weight sums are formed first and
    the block synthesis is a contraction over the classes.
    """
    q_frac = np.atleast_2d(np.asarray(q_frac, dtype=float))
    n = kernel.shape[3]
    blocks = kernel.reshape(-1, n, 3, n, 3)  # (nrho, n, 3, n, 3)
    phase = np.exp(ft_sign * 2j * np.pi * q_frac @ candidates.astype(float).T)  # (nq, ncand)
    out = np.zeros((len(q_frac), n, 3, n, 3), dtype=complex)
    for ic in range(blocks.shape[0]):
        sel = rho_index == ic
        if not np.any(sel):
            continue
        wsum = np.einsum("qc,ijc->qij", phase[:, sel], weights[:, :, sel])  # (nq, n, n)
        out += wsum[..., None, :, None] * blocks[ic][None]
    return out[0] if len(q_frac) == 1 else out


def _fold_to_target(
    weights: np.ndarray,
    source_kernel: np.ndarray,
    rho_index: np.ndarray,
    candidates: np.ndarray,
    target_reps: Tuple[int, int, int],
) -> np.ndarray:
    """Accumulate weighted source kernels into tiled target-cell residues.

    ``kappa[i, j, T] = sum_{candidates c with T_c == T mod target} w[i, j, c] * kernel[rho(c)]``.
    """
    n = weights.shape[0]
    l1, l2, l3 = target_reps
    vals = source_kernel.reshape(-1, n, 3, n, 3)[rho_index]  # (n_cand, n, 3, n, 3)
    t_idx = np.stack([candidates[:, 0] % l1, candidates[:, 1] % l2, candidates[:, 2] % l3], axis=1)
    contributions = weights.transpose(2, 0, 1)[:, :, None, :, None] * vals
    kappa = np.zeros((l1, l2, l3, n, 3, n, 3))
    np.add.at(kappa, (t_idx[:, 0], t_idx[:, 1], t_idx[:, 2]), contributions)
    return kappa


def _nearest_images(
    cell: np.ndarray,
    offsets: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Provably complete nearest-lattice-translation search in the true metric.

    ``offsets`` is (N,3) Cartesian.  Returns ``(distances, images)`` where
    ``images[k]`` is the integer primitive translation ``T`` minimizing
    ``|offset[k] - T|`` and ``distances[k]`` that minimum.  Each offset is
    first shifted by its rounded reduced-basis translation (an exact integer
    lattice vector, restored in the returned labels), so the search window
    stays small even for offsets spanning tens of primitive cells; the window
    is then grown until the dual-basis lower bound
    ``(U_i + 1 - |x.d_i|)/|d_i|`` proves that no unsearched translation is
    nearer than the best found one by more than ``tol`` (``tol`` is the
    caller's acceptance bound, so near-ties are resolved exhaustively).  Raises
    instead of returning an unproved assignment.
    """
    offsets = np.atleast_2d(np.asarray(offsets, dtype=float))
    reduce = _reduce_lattice(cell)
    reduced = reduce @ cell
    red_inv = np.linalg.inv(reduced)
    red_dual_norm = np.linalg.norm(red_inv.T, axis=1)  # |d_i|, d_i . a_j = delta_ij
    # Center each search on its own integer translation so large cell-spanning
    # offsets cost the same as small displacements; the labels stay exact.
    center_k = np.rint(offsets @ red_inv).astype(int)  # reduced-basis coefficients
    center_T = center_k @ reduce  # integer primitive translations
    residual = offsets - center_T.astype(float) @ cell
    x_red = np.abs(residual @ red_inv)  # centered offsets have reduced coords <= 0.5
    U = np.full(3, _WS_SEARCH_INIT, dtype=int)
    for _ in range(_WS_SEARCH_ITERATIONS):
        n_grid = int(np.prod(2 * U + 1))
        if n_grid > _WS_MAX_CANDIDATES or offsets.shape[0] * n_grid > _IMAGE_SEARCH_MAX_WORK:
            raise ValueError(
                f"nearest-image search needs {n_grid} candidate translations for {offsets.shape[0]} "
                f"offsets (reduced window {U.tolist()}); the displacements are too far from the "
                "lattice for a provable assignment"
            )
        grid = np.stack(
            np.meshgrid(*[np.arange(-U[k], U[k] + 1) for k in range(3)], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        candidates = grid @ reduce  # integer translations in the primitive basis
        cart = grid.astype(float) @ reduced
        d2 = ((residual[:, None, :] - cart[None, :, :]) ** 2).sum(-1)  # (N, nc)
        best = np.argmin(d2, axis=1)
        nearest = np.sqrt(d2[np.arange(len(offsets)), best])
        lower = np.maximum(U[None, :] + 1.0 - x_red, 0.0) / red_dual_norm[None, :]
        bound = lower.min(axis=1)  # (N,): distance guaranteed for any unsearched translation
        deficiency = nearest + tol - bound
        if np.all(deficiency <= 0):
            return nearest, center_T + candidates[best]
        for axis in np.unique(np.argmin(lower, axis=1)[deficiency > 0]):
            U[axis] = max(int(U[axis]) + 2, 2 * int(U[axis]) + 1)
    raise ValueError(
        f"nearest-image search did not converge after {_WS_SEARCH_ITERATIONS} window enlargements "
        f"(reduced window {U.tolist()}); the lattice or the displacements are outside the proved domain"
    )


@dataclass
class SiteMapping:
    """Resolved assignment of target atoms to canonical tiled reference sites."""

    canonical_of_target: np.ndarray  # (N_L,) canonical index of each target atom
    target_of_canonical: np.ndarray  # (N_L,) inverse permutation
    positions_canonical: np.ndarray  # (N_L, 3) Cartesian canonical positions
    images: np.ndarray  # (N_L, 3) integer image of each canonical site
    species_canonical: Optional[np.ndarray] = None

    def permute_u(self, u_target: np.ndarray) -> np.ndarray:
        u = np.asarray(u_target, dtype=float)
        if u.shape[0] % 3:
            raise ValueError("displacement vector must carry three components per atom")
        out = np.empty_like(u)
        out.reshape(-1, 3)[self.canonical_of_target] = u.reshape(-1, 3)
        return out

    def unpermute_forces(self, f_canonical: np.ndarray) -> np.ndarray:
        """Return forces in the target-atom order (inverse of permute_u)."""
        f = np.asarray(f_canonical)
        if f.shape[0] % 3:
            raise ValueError("force vector must carry three components per atom")
        return f.reshape(-1, 3)[self.canonical_of_target].reshape(-1)


@dataclass
class MatchedCorrection:
    """Explicit two-branch fixed-cell matched correction for one target cell."""

    primitive_cell: np.ndarray
    positions: np.ndarray
    born: np.ndarray
    dielectric: np.ndarray
    matching_reps: Tuple[int, int, int]
    target_reps: Tuple[int, int, int]
    grid_h: np.ndarray
    correction_matrix: np.ndarray  # (3N_L, 3N_L)
    unsplit_matrix: np.ndarray
    split_matrix: np.ndarray
    diagnostics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    # retained branch data underlying the matrices (explicit construction)
    _ws_matching: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # candidates, weights, rho_index
    _kappa_phi0_match: Optional[np.ndarray] = None  # image-resolved kernel, (m1,m2,m3,n,3,n,3)
    _kappa_dipole_match: Optional[np.ndarray] = None
    _kappa_dipole_target: Optional[np.ndarray] = None
    _kappa_dipole_interp: Optional[np.ndarray] = None
    _d_target_samples: Optional[np.ndarray] = None
    _phi0_samples: Optional[np.ndarray] = None
    _eta: Optional[float] = None
    _nreal: int = 0
    _nrecip: int = 0
    _neutrality_tolerance: float = 1e-4
    _ft_sign: int = 1
    _natom: int = 0
    _response_provenance: Optional[dict] = None

    @property
    def natom_target(self) -> int:
        return self._natom * int(np.prod(self.target_reps))

    @property
    def phi0_samples(self) -> Optional[np.ndarray]:
        """Accepted matching-grid samples used by both branches (tiled grid order)."""
        return self._phi0_samples

    def _check_u(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.shape[0] != 3 * self.natom_target:
            raise ValueError(f"expected {3 * self.natom_target} displacement components, got {u.shape[0]}")
        return u

    def energy(self, u) -> float:
        """Correction energy (eV, whole target cell) in canonical site order."""
        u = self._check_u(u)
        return float(0.5 * u @ (self.correction_matrix @ u))

    def forces(self, u) -> np.ndarray:
        """Correction forces (eV/angstrom) in canonical site order."""
        return -(self.correction_matrix @ self._check_u(u))

    def evaluate(self, u, mapping: Optional[SiteMapping] = None) -> Tuple[float, np.ndarray]:
        """Correction energy and forces in the caller's atom order.

        With ``mapping`` from :meth:`map_sites`, ``u`` and the returned
        forces use the target-atom order the mapping was built from.
        """
        u = self._check_u(u)
        if mapping is None:
            return self.energy(u), self.forces(u)
        u_can = mapping.permute_u(u)
        f_can = self.forces(u_can)
        return float(0.5 * u_can @ (self.correction_matrix @ u_can)), mapping.unpermute_forces(f_can)

    def matching_periodic_tiling(self, pattern) -> np.ndarray:
        """Tile a matching-periodic displacement pattern to canonical target order.

        ``pattern`` is either primitive-cell (``3*n`` components, atom-major)
        or matching-supercell (``3*n*prod(matching_reps)`` components in that
        cell's canonical image-major atom-minor order).  Both describe a
        distortion whose Fourier content lies on the matching grid, so its
        correction vanishes exactly; the primitive form is the special case
        that is independent of the matching image.

        Only defined when the target repetitions are compatible multiples of
        the matching repetitions; otherwise the tiled pattern is not a
        matching-periodic distortion of the target and its correction does
        not carry the preservation guarantee.
        """
        if not self.tiled_compatible():
            raise ValueError(
                f"target repetitions {self.target_reps} are not multiples of the matching repetitions "
                f"{self.matching_reps}; a tiled matching-periodic pattern is not defined (preservation is "
                "asserted on shared wavevectors only)"
            )
        pattern = np.asarray(pattern, dtype=float).reshape(-1)
        m = np.array(self.matching_reps, dtype=int)
        target = np.array(self.target_reps, dtype=int)
        n = self._natom
        n_matching = n * int(np.prod(m))
        if pattern.shape[0] == 3 * n:
            values = np.broadcast_to(pattern.reshape(1, n, 3), (int(np.prod(m)), n, 3))
        elif pattern.shape[0] == 3 * n_matching:
            values = pattern.reshape(int(np.prod(m)), n, 3)
        else:
            raise ValueError(
                f"pattern must carry 3 components per primitive atom ({3 * n}) or per "
                f"matching-supercell atom ({3 * n_matching}, canonical image-major order); "
                f"got {pattern.shape[0]}"
            )
        # Target image t carries the matching-image residue t mod m.
        t = np.array([(t1, t2, t3) for t1 in range(target[0]) for t2 in range(target[1]) for t3 in range(target[2])])
        r = t % m
        flat = (r[:, 0] * m[1] + r[:, 1]) * m[2] + r[:, 2]
        return values[flat].reshape(-1)

    def tiled_compatible(self) -> bool:
        """True when every matching-grid wavevector is also target-periodic."""
        m, target = np.array(self.matching_reps), np.array(self.target_reps)
        return bool(np.all(target % m == 0))

    def _branches(self, q) -> Tuple[np.ndarray, np.ndarray]:
        """``(I(Phi0), I(D))`` from the image-resolved kernels at wavevector(s) q."""
        if self._ws_matching is None:
            raise ValueError("branch data is missing from this correction object")
        cand, weights, rho_index = self._ws_matching
        phi_a = _ws_synth(self._kappa_phi0_match, weights, rho_index, cand, q, self._ft_sign)
        phi_d = _ws_synth(self._kappa_dipole_match, weights, rho_index, cand, q, self._ft_sign)
        return phi_a, phi_d

    def branch_dynamical(self, q, branch: str = "difference"):
        """Branch matrix (n,3,n,3) at a fractional wavevector (primitive basis).

        ``branch`` is ``'unsplit'``, ``'split'`` or ``'difference'``.  Every
        branch is the Wigner-Seitz-weighted Fourier interpolant of the
        image-resolved kernels: on the matching grid it reproduces the
        accepted samples, on the target grid it equals the q-block of the
        assembled matrices, and between grid points it interpolates the
        original response (a folded-kernel synthesis would alias and produce
        spurious imaginary parts).  The analytic dipolar matrix is restored at
        every q through ``D(q) - I(D)(q)``.
        """
        if branch not in ("unsplit", "split", "difference"):
            raise ValueError("branch must be 'unsplit', 'split' or 'difference'")
        q = np.asarray(q, dtype=float).reshape(3)
        phi_a, phi_d = self._branches(q)
        if branch == "unsplit":
            return phi_a
        d_target = self._dipole_at(q)
        if branch == "split":
            return phi_a - phi_d + d_target
        return d_target - phi_d

    def counterterm_matrix(self) -> np.ndarray:
        """Independent algebraic shortcut ``D_L - I_Q[D|_Q]`` as (3N_L, 3N_L).

        Verification quantity only: ticket 02 requires the explicit branches
        for the actual construction.
        """
        return _dense_from_kappa(self._kappa_dipole_target - self._kappa_dipole_interp, self._natom)

    def eta_sensitivity(self, factors: Sequence[float] = (0.5, 2.0)) -> Dict[str, float]:
        """Rebuild the correction at scaled Ewald parameters; report stability.

        The splitting parameter is numerical: converged results must be
        invariant within the reported tolerance, never a fitted physical
        smoothing length.
        """
        cell = self.primitive_cell
        q_match = self.grid_h.astype(float) / np.array(self.matching_reps, dtype=float)
        q_target = _grid_h(self.target_reps) / np.array(self.target_reps, dtype=float)
        report: Dict[str, float] = {}
        cand_m, w_m, rho_m, _ = _pair_ws_weights(cell, self.positions, self.matching_reps)
        cand_l, w_l, rho_l, _ = _pair_ws_weights(cell, self.positions, self.target_reps)
        for f in factors:
            eta = self._eta * float(f)
            d_match = _dipole_grid(cell, self.positions, self.born, self.dielectric, q_match, eta,
                                   self._nreal, self._nrecip, self._neutrality_tolerance)
            d_target = _dipole_grid(cell, self.positions, self.born, self.dielectric, q_target, eta,
                                    self._nreal, self._nrecip, self._neutrality_tolerance)
            if self._ft_sign == -1:
                # Same convention as build_correction: dipole_ifc emits the
                # positive synthesis, so the negative-sign path uses the
                # complex conjugate before the inverse transform.
                d_match = np.conj(d_match)
                d_target = np.conj(d_target)
            d_match, _ = _grid_project(d_match, self.matching_reps)
            d_target, _ = _grid_project(d_target, self.target_reps)
            kd_m = _q_to_r(d_match, self.matching_reps, self._ft_sign)
            kd_l = _q_to_r(d_target, self.target_reps, self._ft_sign)
            if max(np.abs(kd_m.imag).max(), np.abs(kd_l.imag).max()) > _HERM_TOL:
                raise ValueError("dipolar kernel acquired a large imaginary part; Ewald data inconsistent")
            interp = _fold_to_target(w_m, kd_m.real, rho_m, cand_m, self.target_reps)
            dl = _fold_to_target(w_l, kd_l.real, rho_l, cand_l, self.target_reps)
            corr = _dense_from_kappa(dl, self._natom) - _dense_from_kappa(interp, self._natom)
            report[f"max_abs_change_eta_x{f:g}"] = float(np.abs(corr - self.correction_matrix).max())
        return report

    def _dipole_at(self, q: np.ndarray) -> np.ndarray:
        d = dipole_ifc(self.primitive_cell, self.positions, self.born, self.dielectric, q,
                       eta=self._eta, nreal=self._nreal, nrecip=self._nrecip,
                       neutrality_tolerance=self._neutrality_tolerance)
        return np.conj(d) if self._ft_sign == -1 else d

    def _dipole_target_grid(self) -> np.ndarray:
        """Target-grid dipolar samples (stored from construction)."""
        if self._d_target_samples is not None:
            return self._d_target_samples
        q_target = _grid_h(self.target_reps) / np.array(self.target_reps, dtype=float)
        d = _dipole_grid(self.primitive_cell, self.positions, self.born, self.dielectric, q_target,
                         self._eta, self._nreal, self._nrecip, self._neutrality_tolerance)
        return np.conj(d) if self._ft_sign == -1 else d

    def _ws_args(self, reps):
        cand, w, rho, _ = _pair_ws_weights(self.primitive_cell, self.positions, reps)
        return cand, w, rho

    def map_sites(self, cell, positions, species=None, tol: float = 1e-6) -> SiteMapping:
        """Validate the target geometry and map its atoms to canonical sites.

        Accepts arbitrary atom order (permutations), periodic-image choices
        and lattice-vector translations of whole atoms.  Rejects cell strain,
        non-diagonal repetitions, atom-count changes, species mismatches,
        diffusion/topology changes and ambiguous assignments with actionable
        diagnostics.

        ``tol`` is a displacement bound in angstrom, compared against the
        Cartesian distance to the nearest periodic image of a reference site.
        The nearest image is found by a provably complete lattice search
        (reduced basis plus a dual-basis distance bound, raised rather than
        approximated when the bound cannot be proved), so skew cells are
        handled in the true metric; ``tol`` does not scale with the cell size.
        A target atom further than ``tol`` from every reference site, or
        within ``tol`` of two of them, is outside the declared mapping domain
        and is rejected instead of remapped.
        """
        cell = _validate_cell(cell, "target cell")
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("target positions must be finite (N,3) Cartesian coordinates")
        target = self.target_reps
        n = self._natom
        n_l = n * target[0] * target[1] * target[2]
        expected_cell = np.diag(target).astype(float) @ self.primitive_cell
        scale = max(np.abs(expected_cell).max(), 1.0)
        if not np.allclose(cell, expected_cell, rtol=0, atol=1e-8 * scale):
            raise ValueError(
                "target cell is not an unstrained diagonal repetition of the primitive cell "
                "(general supercell transforms and strain are outside the fixed-cell contract); "
                f"expected rows {expected_cell.tolist()}, got {cell.tolist()}"
            )
        if positions.shape[0] != n_l:
            raise ValueError(
                f"target has {positions.shape[0]} atoms but {target} repetitions of {n} sites need {n_l}; "
                "atom-count changes are rejected"
            )
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a positive finite displacement bound in angstrom")
        ref_red = self.positions @ np.linalg.inv(self.primitive_cell)
        frac = positions @ np.linalg.inv(cell)
        frac = frac - np.floor(frac + 1e-12)  # periodic wrap into the target cell
        site_frac = frac * np.array(target, dtype=float)[None, :]  # primitive fractional, images unwrapped
        # Distances are measured in the Cartesian metric against the true
        # nearest lattice image of every reference site; the search is
        # provably complete, so a skew cell can neither select a wrong image
        # nor reject a valid position.
        ref_cart = ref_red @ self.primitive_cell  # (n, 3)
        offsets = (site_frac @ self.primitive_cell)[:, None, :] - ref_cart[None, :, :]  # (N_L, n, 3)
        dist, images = _nearest_images(self.primitive_cell, offsets.reshape(-1, 3), tol)
        dist = dist.reshape(n_l, n)
        images = images.reshape(n_l, n, 3)
        rows = np.arange(n_l)
        best = np.argmin(dist, axis=1)
        best_dist = dist[rows, best]
        if np.any(best_dist > tol):
            bad = np.where(best_dist > tol)[0]
            raise ValueError(
                "target atoms " + np.array2string(bad) +
                " do not coincide with any reference site up to lattice translations "
                f"(max Cartesian residual {best_dist.max():.3e} angstrom > tol={tol:.3e} angstrom); "
                "diffusion or topology changes are rejected"
            )
        if n > 1:
            second = np.partition(dist, 1, axis=1)[:, 1]
            if np.any(second <= tol):
                bad = np.where(second <= tol)[0]
                raise ValueError(
                    "ambiguous site assignment for target atoms " + np.array2string(bad) +
                    f": two reference sites lie within tol={tol:.3e} angstrom of the requested position"
                )
        images = images[rows, best]
        canonical = np.empty(n_l, dtype=int)
        seen = np.zeros(n_l, dtype=bool)
        for k in range(n_l):
            s = int(best[k])
            t = tuple(int(v) for v in (images[k] % np.array(target)))
            idx = ((t[0] * target[1] + t[1]) * target[2] + t[2]) * n + s
            canonical[k] = idx
            if seen[idx]:
                raise ValueError(
                    f"two target atoms map to the same canonical site (site {s}, image {list(t)}); "
                    "duplicate or exchanged assignments are rejected"
                )
            seen[idx] = True
        if not seen.all():
            raise ValueError(
                "canonical sites " + np.array2string(np.where(~seen)[0]) + " are not covered by the target atoms"
            )
        positions_canonical = np.empty((n_l, 3))
        images_canonical = np.empty((n_l, 3), dtype=int)
        for t in np.ndindex(*target):
            for s in range(n):
                idx = ((t[0] * target[1] + t[1]) * target[2] + t[2]) * n + s
                positions_canonical[idx] = (ref_red[s] + np.array(t)) @ self.primitive_cell
                images_canonical[idx] = np.array(t)
        species_canonical = None
        if species is not None:
            species = list(species)
            if len(species) != n_l:
                raise ValueError(f"expected {n_l} species labels, got {len(species)}")
            ref_species = self.metadata.get("species")
            if ref_species is not None:
                for k in range(n_l):
                    if species[k] != ref_species[int(best[k])]:
                        raise ValueError(
                            f"target atom {k} has species {species[k]!r} but mapped reference site "
                            f"{int(best[k])} is {ref_species[int(best[k])]!r}; incompatible species are rejected"
                        )
            species_canonical = np.empty(n_l, dtype=object)
            for k in range(n_l):
                species_canonical[canonical[k]] = species[k]
        return SiteMapping(canonical, np.argsort(canonical), positions_canonical, images_canonical, species_canonical)


def _dense_from_kappa(kappa_tiled: np.ndarray, natom: int) -> np.ndarray:
    """Assemble the real-space (3N_L, 3N_L) matrix from tiled kernels."""
    l1, l2, l3 = kappa_tiled.shape[0], kappa_tiled.shape[1], kappa_tiled.shape[2]
    nimg = l1 * l2 * l3
    mat = np.zeros((3 * natom * nimg, 3 * natom * nimg))
    ta = np.arange(nimg)
    ix, iy, iz = ta // (l2 * l3), (ta // l3) % l2, ta % l3
    for t1 in range(l1):
        for t2 in range(l2):
            for t3 in range(l3):
                kappa = kappa_tiled[t1, t2, t3].reshape(3 * natom, 3 * natom)
                tb = (((ix + t1) % l1) * l2 + ((iy + t2) % l2)) * l3 + ((iz + t3) % l3)
                perm = np.zeros((nimg, nimg))
                perm[ta, tb] = 1.0
                mat += np.kron(perm, kappa)
    return mat


def _grid_project(matrices: np.ndarray, reps: Tuple[int, int, int]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Explicit projection of grid matrices onto a real reciprocal kernel.

    For a real symmetric force-constant kernel the translation-resolved
    q-matrices satisfy ``Phi(q) = conj(Phi(-q))`` and ``Phi(q) = Phi(q)^dag``.
    Finite-difference noise violates both slightly.  The projections are
    applied explicitly and every residual (before and after) is returned for
    recording; nothing is dropped silently.
    """
    m = reps
    neg = [(np.arange(k) * 0 + (k - np.arange(k)) % k) for k in m]
    grid = np.asarray(matrices).reshape(m[0], m[1], m[2], *matrices.shape[1:])
    flip = grid[np.ix_(neg[0], neg[1], neg[2])]
    res_real = float(np.abs(grid - np.conj(flip)).max())
    grid = 0.5 * (grid + np.conj(flip))
    herm = grid.transpose(0, 1, 2, 5, 6, 3, 4).conj()
    res_herm = float(np.abs(grid - herm).max())
    grid = 0.5 * (grid + herm)
    # Residuals of the final data (two alternating projections cannot make a
    # noisy input exactly consistent; report what remains).
    flip2 = grid[np.ix_(neg[0], neg[1], neg[2])]
    herm2 = grid.transpose(0, 1, 2, 5, 6, 3, 4).conj()
    report = {
        "real_kernel_residual_before": res_real,
        "hermiticity_residual_before": res_herm,
        "real_kernel_residual_after": float(np.abs(grid - np.conj(flip2)).max()),
        "hermiticity_residual_after": float(np.abs(grid - herm2).max()),
    }
    return grid.reshape(matrices.shape), report


def build_correction(
    primitive_cell,
    positions,
    born,
    dielectric,
    matching_reps,
    phi0_grid,
    grid_h,
    target_reps,
    *,
    ft_sign: int = 1,
    eta: Optional[float] = None,
    nreal: int = 5,
    nrecip: int = 8,
    neutrality_tolerance: float = 1e-4,
    species: Optional[Sequence] = None,
    phi0_symmetry_tol: Optional[float] = None,
    response_provenance: Optional[dict] = None,
) -> MatchedCorrection:
    """Build the explicit split-minus-unsplit fixed-cell correction.

    Parameters
    ----------
    primitive_cell : (3,3) row-vector lattice (angstrom).
    positions : (n,3) reference Cartesian positions defining the site order.
    born : (n,3,3) Born effective charges (field, displacement), elementary charge.
    dielectric : (3,3) electronic dielectric tensor (symmetric positive definite).
    matching_reps : three positive integers, the matching supercell.
    phi0_grid : (nq, n, 3, n, 3) complex unmassweighted IFC matrices of the
        original potential on the full unshifted matching grid, eV/angstrom**2,
        convention ``Phi0(q) = sum_T kappa(T) exp(+2i pi q.T)`` with no extra
        site phases (``ft_sign`` overrides the synthesis sign).
    grid_h : (nq, 3) integer grid indices of ``phi0_grid`` (any order), each
        in [0, m) and covering every residue class exactly once.
    target_reps : three positive integers, the target supercell (smaller,
        equal or larger than the matching cell; the shared wavevectors carry
        the matching guarantee).
    eta : shared Ewald parameter (angstrom**-2); default from the primitive
        cell using the dipole_ifc formula.  One value is used for every
        dipole evaluation in both branches.
    nreal, nrecip : integer Ewald shell cutoffs shared by all evaluations.
    neutrality_tolerance : absolute charge-neutrality tolerance (elementary
        charge) declared for the Born charges, passed to every ``dipole_ifc``
        evaluation (matching grid, target grid, and later re-evaluations such
        as ``branch_dynamical``, ``_dipole_target_grid`` and
        ``eta_sensitivity``).  The default ``1e-4`` matches the tolerance
        declared by ``ElectricalResponse``; pass
        ``response.neutrality_tolerance`` so the builder accepts exactly the
        response data the caller declared valid.  Born tensors are used as
        supplied: this is a validity bound, never a renormalization.
    species : optional reference species labels (validated during site mapping).
    phi0_symmetry_tol : optional absolute tolerance (eV/angstrom**2) declared
        by the caller for violation of the reciprocal-kernel symmetries by
        ``phi0_grid``.  The real-space construction uses the Hermitian part of
        those samples; this argument is the caller's explicit statement of how
        much symmetry violation the producer's data may carry, and it is the
        only way to permit a projection larger than floating-point roundoff.
        Omitted (default): only roundoff-level violation
        (``_PHI0_ROUNDOFF`` relative) is accepted, and the applied change is
        recorded; any larger violation is rejected rather than repaired,
        because the admissible noise level belongs to the producer, not to
        this construction.  There is deliberately no built-in material guess.
    response_provenance : optional provenance metadata recorded verbatim.  When
        ``phi0_symmetry_tol`` is passed, include the declared value here too
        and state how it was obtained (measured finite-difference noise, an
        empirically estimated in-sample threshold, or a predeclared bound):
        ``metadata["phi0_symmetrization"]["declared_tolerance"]`` records the
        number only, not its status.

    Returns
    -------
    MatchedCorrection with the explicit branch matrices, their difference,
    quadratic energy/forces, site mapping, branch q-matrices, the
    independent counterterm check and Ewald-sensitivity diagnostics.
    """
    ft_sign = int(ft_sign)
    if ft_sign not in (1, -1):
        raise ValueError("ft_sign must be +1 or -1")
    cell = _validate_cell(primitive_cell, "primitive cell")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0 or not np.isfinite(positions).all():
        raise ValueError("reference positions must be a nonempty finite (n,3) array in angstrom")
    born = np.asarray(born, dtype=float)
    dielectric = np.asarray(dielectric, dtype=float)
    n = len(positions)
    if born.shape != (n, 3, 3) or dielectric.shape != (3, 3):
        raise ValueError("born must be (n,3,3) and dielectric (3,3)")
    # Reference sites must be distinct; coincident sites make IFCs and
    # Born charges ambiguous and site mapping ill-defined.
    dr = positions[:, None, :] - positions[None, :, :]
    if n > 1 and np.sqrt((dr**2).sum(-1))[np.triu_indices(n, 1)].min() < 1e-6:
        raise ValueError("reference positions contain coincident sites; the assignment would be ambiguous")
    reps_m = _as_reps(matching_reps, "matching_reps")
    reps_l = _as_reps(target_reps, "target_reps")
    nq = reps_m[0] * reps_m[1] * reps_m[2]
    phi0 = np.asarray(phi0_grid, dtype=complex)
    if phi0.shape != (nq, n, 3, n, 3) or not np.isfinite(phi0).all():
        raise ValueError(f"phi0_grid must be finite ({nq}, {n}, 3, {n}, 3); got {np.asarray(phi0_grid).shape}")
    grid_h = np.asarray(grid_h, dtype=float)
    if grid_h.shape != (nq, 3) or not np.all(np.isfinite(grid_h)) or np.any(grid_h != np.floor(grid_h)):
        raise ValueError("grid_h must be (nq,3) integers")
    grid_h = grid_h.astype(int)
    if np.any(grid_h < 0) or np.any(grid_h >= np.array(reps_m)):
        raise ValueError("grid_h indices must lie in [0, m) per axis")
    if len({(int(a), int(b), int(c)) for a, b, c in grid_h}) != nq:
        raise ValueError("grid_h must cover every matching-grid residue class exactly once (unshifted full grid)")
    if species is not None and len(list(species)) != n:
        raise ValueError(f"expected {n} species labels for the reference sites, got {len(list(species))}")
    if eta is None:
        eta = _default_eta(cell)
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in (nreal, nrecip)):
        raise ValueError("nreal and nrecip must be nonnegative integers")
    neutrality_tolerance = float(neutrality_tolerance)
    if not np.isfinite(neutrality_tolerance) or neutrality_tolerance < 0:
        raise ValueError("neutrality_tolerance must be nonnegative and finite")

    # Reorder phi0 into tiled C order matching _grid_h.
    tiled_h = _grid_h(reps_m)
    lookup = {(int(a), int(b), int(c)): k for k, (a, b, c) in enumerate(grid_h.tolist())}
    order = np.array([lookup[key] for key in map(tuple, tiled_h.tolist())], dtype=int)
    phi0_tiled = phi0[order]
    q_match = tiled_h / np.array(reps_m, dtype=float)

    # Explicit projection onto the real reciprocal-kernel symmetries.  A real
    # conservative kernel cannot represent a non-Hermitian grid, so the
    # Hermitian part is what the branches use.  How much violation is
    # admissible is the producer's property: by default only roundoff-level
    # inconsistency is accepted (the applied change is recorded); anything
    # larger is refused unless the caller declares an absolute tolerance.
    declared_phi0 = phi0_tiled
    phi0_accepted, phi0_sym = _grid_project(declared_phi0, reps_m)
    phi0_violation = max(phi0_sym["real_kernel_residual_before"],
                         phi0_sym["hermiticity_residual_before"])
    phi0_change = float(np.abs(phi0_accepted - declared_phi0).max())
    phi0_scale = max(float(np.abs(declared_phi0).max()), 1e-300)
    roundoff_limit = _PHI0_ROUNDOFF * phi0_scale
    if phi0_symmetry_tol is None:
        limit = roundoff_limit
        mode = "roundoff-only (no tolerance declared; projection bounded by floating-point roundoff)"
    else:
        if not np.isfinite(phi0_symmetry_tol) or phi0_symmetry_tol <= 0:
            raise ValueError(
                "phi0_symmetry_tol must be a positive finite absolute tolerance in eV/angstrom**2"
            )
        limit = float(phi0_symmetry_tol)
        mode = "explicit tolerance (caller-declared)"
    if phi0_violation > limit:
        raise ValueError(
            "phi0_grid violates the reciprocal-kernel symmetries by "
            f"{phi0_violation:.3e} eV/angstrom**2, above the {limit:.3e} accepted here "
            f"({mode}); pass phi0_symmetry_tol=<the producer's measured violation> to declare the "
            "acceptable noise level explicitly, or fix the producer's symmetry treatment. "
            "No built-in material-specific tolerance is applied."
        )
    phi0_tiled = phi0_accepted

    # --- analytic dipolar data: shared eta and shells on both grids ----
    # All dipolar matrices are the primitive-cell analytic response sampled at
    # each grid's physical wavevectors (h/reps in the primitive basis).  The
    # Ewald matrix is periodicity independent at convergence, so this is the
    # "target Ewald response at the target periodicity" evaluated at the
    # target wavevectors rather than a translated matching-cell matrix, and it
    # makes the shared-wavevector samples bitwise identical across grids -
    # the matching identity is then exact instead of truncation limited.
    d_match = _dipole_grid(cell, positions, born, dielectric, q_match, eta, nreal, nrecip,
                           neutrality_tolerance)
    q_prim_target = _grid_h(reps_l) / np.array(reps_l, dtype=float)
    d_target = _dipole_grid(cell, positions, born, dielectric, q_prim_target, eta, nreal, nrecip,
                            neutrality_tolerance)
    if ft_sign == -1:
        # dipole_ifc emits the exp(+2i pi q.T) synthesis; its kernel is real,
        # so the opposite synthesis convention is the complex conjugate.
        d_match = np.conj(d_match)
        d_target = np.conj(d_target)
    d_match, sym_m = _grid_project(d_match, reps_m)
    d_target, sym_l = _grid_project(d_target, reps_l)
    d_herm_res = 0.0
    for sym, dgrid in ((sym_m, d_match), (sym_l, d_target)):
        d_scale = max(float(np.abs(dgrid).max()), 1e-300)
        worst = max(sym["real_kernel_residual_before"], sym["hermiticity_residual_before"])
        if worst > 1e-6 * d_scale:
            raise ValueError(f"dipolar matrices are not reciprocal (residual {worst:.3e}); response input inconsistent")
        d_herm_res = max(d_herm_res, worst)

    # --- kernels on the respective tiled translation grids -------------
    kappa_phi0 = _q_to_r(phi0_tiled, reps_m, ft_sign)
    imag_res = float(np.abs(kappa_phi0.imag).max())
    if imag_res > 1e-8 * max(float(np.abs(kappa_phi0).max()), 1e-300):
        raise ValueError(
            "phi0_grid does not admit a real-space extension after reciprocal projection "
            f"(imaginary kernel residual {imag_res:.3e})"
        )
    kappa_phi0 = kappa_phi0.real
    kappa_res = _q_to_r(phi0_tiled - d_match, reps_m, ft_sign)
    kappa_d_match = _q_to_r(d_match, reps_m, ft_sign)
    kappa_d_target = _q_to_r(d_target, reps_l, ft_sign)
    for k in (kappa_res, kappa_d_match, kappa_d_target):
        if np.abs(k.imag).max() > 1e-8 * max(float(np.abs(k).max()), 1e-300):
            raise ValueError("real-space kernel has a large imaginary residual; inconsistent branch data")
    kappa_res = kappa_res.real
    kappa_d_match = kappa_d_match.real
    kappa_d_target = kappa_d_target.real

    # --- Wigner-Seitz image weights and the common fold ---------------
    cand_m, w_m, rho_m, res_m_w = _pair_ws_weights(cell, positions, reps_m)
    cand_l, w_l, rho_l, res_l_w = _pair_ws_weights(cell, positions, reps_l)
    if max(res_m_w, res_l_w) > 1e-8:
        raise ValueError(
            "Wigner-Seitz class sums deviate from one by "
            f"{max(res_m_w, res_l_w):.3e}; the periodic-image weight construction is inconsistent "
            "(only roundoff-level deviations are expected with the provable search window)"
        )

    kappa_unsplit = _fold_to_target(w_m, kappa_phi0, rho_m, cand_m, reps_l)
    kappa_residual = _fold_to_target(w_m, kappa_res, rho_m, cand_m, reps_l)
    kappa_dipole_interp = _fold_to_target(w_m, kappa_d_match, rho_m, cand_m, reps_l)
    kappa_dipole_target = _fold_to_target(w_l, kappa_d_target, rho_l, cand_l, reps_l)

    unsplit_matrix = _dense_from_kappa(kappa_unsplit, n)
    split_matrix = _dense_from_kappa(kappa_residual, n) + _dense_from_kappa(kappa_dipole_target, n)
    sym_res_a = float(np.abs(unsplit_matrix - unsplit_matrix.T).max())
    sym_res_b = float(np.abs(split_matrix - split_matrix.T).max())
    sym_tol = _HERM_TOL * max(np.abs(unsplit_matrix).max(), np.abs(split_matrix).max(), 1e-300)
    if max(sym_res_a, sym_res_b) > sym_tol:
        raise ValueError(f"branch matrices are not symmetric (residuals {sym_res_a:.3e}, {sym_res_b:.3e})")
    # Explicit roundoff symmetrisation, residuals recorded in diagnostics.
    unsplit_matrix = 0.5 * (unsplit_matrix + unsplit_matrix.T)
    split_matrix = 0.5 * (split_matrix + split_matrix.T)
    correction_matrix = split_matrix - unsplit_matrix

    mc = MatchedCorrection(
        primitive_cell=cell, positions=positions, born=born, dielectric=dielectric,
        matching_reps=reps_m, target_reps=reps_l, grid_h=tiled_h,
        correction_matrix=correction_matrix, unsplit_matrix=unsplit_matrix,
        split_matrix=split_matrix,
        _ws_matching=(cand_m, w_m, rho_m),
        _kappa_phi0_match=kappa_phi0, _kappa_dipole_match=kappa_d_match,
        _kappa_dipole_target=kappa_dipole_target, _kappa_dipole_interp=kappa_dipole_interp,
        _d_target_samples=d_target,
        _phi0_samples=phi0_tiled,
        _eta=eta, _nreal=int(nreal), _nrecip=int(nrecip), _ft_sign=ft_sign, _natom=n,
        _neutrality_tolerance=neutrality_tolerance,
        _response_provenance=response_provenance,
    )
    mc.metadata = {
        "species": list(species) if species is not None else None,
        "units": "angstrom, eV (whole cell), eV/angstrom, eV/angstrom**2",
        "ifc_axes": "(i, mu, j, nu) atom-major Cartesian, unmassweighted",
        "site_order": "image-major atom-minor: site + natom*(iz + nz*(iy + ny*ix)), tiled images",
        "fourier_convention": f"Phi(q) = sum_T kappa(T) exp({'+' if ft_sign == 1 else '-'}2i pi q.T)",
        "boundary": "conducting (macroscopic Gamma zero mode omitted in dipole_ifc, both grids)",
        "interpolation": "matching-grid Fourier kernel with Wigner-Seitz image weights folded into the target cell",
        "weight_class_sum_residual": max(res_m_w, res_l_w),
        "phi0_symmetry_residuals": phi0_sym,
        "phi0_symmetrization": {
            "mode": mode,
            "declared_tolerance": None if phi0_symmetry_tol is None else float(phi0_symmetry_tol),
            "roundoff_limit": float(roundoff_limit),
            "accepted_violation_max_abs": float(phi0_violation),
            "applied_max_abs_change": float(phi0_change),
            "samples_used_for_matching": "hermitian projection of the declared matching-grid samples "
                                         "(see MatchedCorrection.phi0_samples)",
        },
        "dipole_symmetry_residuals": {"matching": sym_m, "target": sym_l},
        "dipole_reciprocity_residual": d_herm_res,
        "symmetrization_residual": max(sym_res_a, sym_res_b),
        "ewald": {"eta": eta, "nreal": int(nreal), "nrecip": int(nrecip),
                  "neutrality_tolerance": neutrality_tolerance},
        "asr": "none applied here; input data used exactly as supplied",
        "response_provenance": response_provenance,
    }
    mc.diagnostics.update(_diagnostics(mc))
    return mc


def _shared_mask(mc: MatchedCorrection) -> np.ndarray:
    m, target = np.array(mc.matching_reps), np.array(mc.target_reps)
    h = mc.grid_h.astype(float)
    return np.all((target[None, :] * h) % m[None, :] == 0, axis=1)


def _diagnostics(mc: MatchedCorrection) -> Dict[str, float]:
    """Explicit matching, cancellation and invariance diagnostics."""
    diag: Dict[str, float] = {}
    m = np.array(mc.matching_reps)
    mask = _shared_mask(mc)
    diag["shared_wavevector_count"] = int(mask.sum())

    # (1) counterterm equivalence on the target matrix
    ct = mc.counterterm_matrix()
    diag["correction_matrix_max_abs"] = float(np.abs(mc.correction_matrix).max())
    diag["counterterm_max_abs_difference"] = float(np.abs(mc.correction_matrix - ct).max())

    # (2) full branch matrices at every shared matching-grid wavevector
    if mask.any():
        q_match = mc.grid_h[mask].astype(float) / m.astype(float)
        phi_a, phi_d = mc._branches(q_match)
        d_target = mc._dipole_target_grid()[_shared_target_indices(mc, mask)]
        diag["shared_grid_max_branch_difference"] = float(np.abs(d_target - phi_d).max())
        # the unsplit branch reproduces the accepted matching-grid samples
        if mc._phi0_samples is not None:
            accepted = mc._phi0_samples[mask]
            diag["phi0_sample_reproduction_max_abs"] = float(np.abs(phi_a - accepted).max())

    # (3) translation invariance of the correction (acoustic behavior)
    row_sums = np.abs(mc.correction_matrix.reshape(mc.natom_target, 3, mc.natom_target, 3).sum(axis=2))
    diag["gamma_uniform_displacement_force_residual"] = float(row_sums.max())

    # (4) acoustic content at Gamma: the dipolar contribution alone is not
    # ASR-invariant (uniform displacement induces site-dependent dipoles), so
    # no projection is applied and the correction's Gamma force residual (3)
    # is the meaningful statement.  The raw dipolar row-sum magnitude is
    # reported for transparency.
    gamma_dd = mc._kappa_dipole_target.sum(axis=(0, 1, 2))
    diag["dipole_target_gamma_row_sum_max"] = float(np.abs(gamma_dd.sum(axis=2)).max())
    return diag


def _shared_target_indices(mc: MatchedCorrection, mask: np.ndarray) -> np.ndarray:
    """Target-grid indices of the shared matching-grid wavevectors."""
    m, target = np.array(mc.matching_reps), np.array(mc.target_reps)
    h_l = np.rint(mc.grid_h[mask].astype(float) * target / m).astype(int) % np.array(mc.target_reps)
    return (h_l[:, 0] * mc.target_reps[1] + h_l[:, 1]) * mc.target_reps[2] + h_l[:, 2]
