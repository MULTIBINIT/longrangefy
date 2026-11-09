"""Shared deterministic models for the ticket 01 behaviour tests.

Both models are conservative and analytic so the tests can assert recovered
derivatives against exact Hessians rather than against another numerical path.

Cell conventions: rows, Angstrom; positions are Cartesian (n, 3).
"""

from __future__ import annotations

import numpy as np

# Skew triclinic primitive cell (rows) with two sites; deliberately non-cubic.
SKEW_CELL = np.array([
    [4.1, 0.0, 0.0],
    [0.7, 4.6, 0.0],
    [0.4, -1.1, 5.0],
])
SKEW_FRAC = np.array([
    [0.0, 0.0, 0.0],
    [0.31, 0.42, 0.57],
])
SKEW_SPECIES = ("Na", "Cl")


def _spring_hessian(natom: int, rng: np.random.Generator,
                    scale: float = 1.0) -> np.ndarray:
    """Random *symmetric positive definite* dense (3N, 3N) Hessian.

    Built as ``A^T A + shift`` in eV/Angstrom^2; the result is a valid
    conservative harmonic model's Hessian with nontrivial off-diagonal blocks.
    """
    raw = rng.normal(size=(3 * natom, 3 * natom))
    hessian = raw.T @ raw
    hessian *= scale / hessian.max()
    hessian += np.eye(3 * natom) * 0.05
    return 0.5 * (hessian + hessian.T)


class HarmonicModel:
    """Exact quadratic model ``E = 1/2 u^T K u`` with known Hessian."""

    def __init__(self, hessian: np.ndarray, reference_positions: np.ndarray):
        self.hessian = np.asarray(hessian, dtype=float)
        self.reference = np.asarray(reference_positions, dtype=float).copy()
        self.n_calls = 0

    def energy(self, positions: np.ndarray) -> float:
        u = (np.asarray(positions, dtype=float) - self.reference).reshape(-1)
        return float(0.5 * u @ self.hessian @ u)

    def get_forces(self, positions: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        u = (np.asarray(positions, dtype=float) - self.reference).reshape(-1)
        return -(self.hessian @ u).reshape(self.reference.shape)


class AnharmonicModel(HarmonicModel):
    """Quadratic plus quartic on-site term: ``E += k4/4 sum_i |u_i|^4``."""

    def __init__(self, hessian: np.ndarray, reference_positions: np.ndarray,
                 k4: float = 5.0):
        super().__init__(hessian, reference_positions)
        self.k4 = float(k4)

    def energy(self, positions: np.ndarray) -> float:
        u = np.asarray(positions, dtype=float) - self.reference
        return float(0.5 * u.reshape(-1) @ self.hessian @ u.reshape(-1)
                     + 0.25 * self.k4 * np.sum(np.sum(u ** 2, axis=1) ** 2))

    def get_forces(self, positions: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        u = np.asarray(positions, dtype=float) - self.reference
        quadratic = -(self.hessian @ u.reshape(-1)).reshape(u.shape)
        quartic = -self.k4 * np.sum(u ** 2, axis=1)[:, None] * u
        return quadratic + quartic


def random_blocks(rng: np.random.Generator, reps, n_uc: int) -> np.ndarray:
    """Translation-invariant physical blocks: Phi(i,R,j) = Phi(j,-R,i)^T.

    ``returned[irpt, i, a, j, b]`` with the fold ``R mod reps``; the reciprocity
    makes the Fourier matrices Hermitian and the dense matrix symmetric.
    """
    reps = tuple(int(v) for v in reps)
    nrpt = int(np.prod(reps))
    # Random cell-origin blocks for the half of the range; mirror the other half.
    base = rng.normal(size=(nrpt, n_uc, 3, n_uc, 3))
    rpoints = np.array([(r1, r2, r3)
                        for r1 in range(reps[0])
                        for r2 in range(reps[1])
                        for r3 in range(reps[2])], dtype=int)
    index = {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(rpoints)}
    blocks = np.zeros_like(base)
    for irpt, r in enumerate(rpoints):
        mirror = index[(int((-r[0]) % reps[0]), int((-r[1]) % reps[1]),
                        int((-r[2]) % reps[2]))]
        blocks[irpt] = base[irpt]
        if mirror != irpt:
            blocks[mirror] = np.transpose(blocks[irpt], (2, 3, 0, 1))
        else:
            blocks[irpt] = 0.5 * (blocks[irpt]
                                  + np.transpose(blocks[irpt], (2, 3, 0, 1)))
    return blocks
