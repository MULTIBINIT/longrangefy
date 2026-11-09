"""Screened dipolar IFCs in reciprocal space.

Adapted from pymultibinit.pyeffpot.dipdip.compute_dipdip_dynmat by the
same author. No pymultibinit or libabinit runtime dependency is required.
Cells and positions use angstrom; IFCs use eV/angstrom**2. Born indices are
(electric-field direction, displacement direction). The omitted macroscopic
zero mode implements the conducting boundary at Gamma.
"""

import numpy as np
from scipy.special import erfc

# e**2 / (4 pi epsilon_0), eV angstrom (CODATA 2018).
_COULOMB = 14.3996454784255


def dipole_ifc(cell, positions, born, dielectric, q, *, eta=None, nreal=5, nrecip=8, neutrality_tolerance=1e-8):
    """Return unmassweighted dipole IFCs at a reduced wavevector.

    ``eta`` is the numerical Ewald parameter in inverse angstrom squared;
    ``nreal`` and ``nrecip`` are integer lattice shell cutoffs. Positions and
    cell are Cartesian row vectors. Output axes are atom, displacement, atom,
    displacement in pymultibinit's convention 1; the caller must apply the
    same phase, image and sum-rule conventions in both harmonic branches.
    ``neutrality_tolerance`` bounds each component of the sum of Born charges.
    """
    cell = np.asarray(cell, dtype=float)
    positions = np.asarray(positions, dtype=float)
    born = np.asarray(born, dtype=float)
    dielectric = np.asarray(dielectric, dtype=float)
    q = np.asarray(q, dtype=float)
    if cell.shape != (3, 3) or q.shape != (3,) or positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) == 0:
        raise ValueError('expected a 3x3 cell, nonempty (N,3) positions and a 3-vector q')
    n = len(positions)
    if born.shape != (n, 3, 3) or dielectric.shape != (3, 3):
        raise ValueError('Born charges must have shape (N,3,3) and dielectric shape (3,3)')
    if not all(np.isfinite(v).all() for v in (cell, positions, born, dielectric, q)):
        raise ValueError('cell, positions, response tensors and q must be finite')
    if abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError('cell must be nonsingular')
    if not np.allclose(dielectric, dielectric.T, rtol=1e-12, atol=1e-12) or np.min(np.linalg.eigvalsh(dielectric)) <= 0:
        raise ValueError('dielectric tensor must be symmetric positive definite')
    if not np.isfinite(neutrality_tolerance) or neutrality_tolerance < 0:
        raise ValueError('neutrality tolerance must be nonnegative and finite')
    if not np.allclose(born.sum(axis=0), 0, rtol=0, atol=neutrality_tolerance):
        raise ValueError('Born effective charges must satisfy charge neutrality')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in (nreal, nrecip)):
        raise ValueError('Ewald shell cutoffs must be nonnegative integers')
    invcell = np.linalg.inv(cell)
    if eta is None:
        eta = np.pi * 100.0 / 33.0 * np.sqrt(1.69 * np.sum(invcell @ invcell.T) / np.sum(cell @ cell.T))
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError('eta must be positive and finite')
    rec = 2 * np.pi * invcell.T
    g = np.stack(np.meshgrid(*([np.arange(-nrecip, nrecip + 1)] * 3), indexing='ij'), axis=-1).reshape(-1, 3)
    k = (g + q) @ rec
    kek = np.einsum('gi,ij,gj->g', k, dielectric, k)
    mask = kek > 0
    k, kek = k[mask], kek[mask]
    factor = 4 * np.pi / abs(np.linalg.det(cell)) * np.exp(-kek / (4 * eta)) / kek
    diff = positions[:, None, :] - positions[None, :, :]
    phases = np.exp(1j * np.einsum('gi,abi->gab', k, diff))
    tensor = np.einsum('g,gab,gi,gj->aibj', factor, phases, k, k)
    result = np.einsum('iam,iajb,jbn->imjn', born, tensor, born)

    inv_eps = np.linalg.inv(dielectric)
    inv_sqrt_det = 1 / np.sqrt(np.linalg.det(dielectric))
    reta = np.sqrt(eta)
    self_tensor = 4 / (3 * np.sqrt(np.pi)) * reta**3 * inv_sqrt_det * inv_eps.T
    r_indices = np.stack(np.meshgrid(*([np.arange(-nreal, nreal + 1)] * 3), indexing='ij'), axis=-1).reshape(-1, 3)
    for r in r_indices:
        rij = r @ cell + diff
        mask = np.ones((n, n), dtype=bool)
        if np.all(r == 0):
            for i in range(n):
                result[i, :, i, :] -= born[i].T @ self_tensor @ born[i]
            np.fill_diagonal(mask, False)
        distances = rij[mask]
        if not len(distances):
            continue
        r_eps = np.einsum('mi,ij,mj->m', distances, inv_eps, distances)
        if np.any(r_eps <= 1e-24):
            raise ValueError('reference sites overlap under periodic images')
        y = reta * np.sqrt(r_eps)
        t2 = erfc(y) / y**3
        t3 = (2 / np.sqrt(np.pi)) * np.exp(-y*y) / y**2
        scaled = distances @ inv_eps.T
        t = ((3*t2 + t3*(3 + 2*y*y))[:, None, None] * np.einsum('mi,mj->mij', scaled, scaled) / r_eps[:, None, None]
             - (t2+t3)[:, None, None] * inv_eps.T)
        t *= -(reta**3) * inv_sqrt_det
        phase = np.exp(-2j * np.pi * np.dot(q, r))
        for (i, j), block in zip(zip(*np.where(mask)), t):
            result[i, :, j, :] += phase * (born[i].T @ block @ born[j])
    return result * _COULOMB
