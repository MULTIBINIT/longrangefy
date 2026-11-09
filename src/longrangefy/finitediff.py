"""NumPy-only force finite differences for harmonic IFCs (R04, R13).

A ``get_forces`` callable maps Cartesian positions ``(n, 3)`` (Angstrom) of a
fixed cell to forces ``(n, 3)`` (eV/Angstrom).  The displaced configurations
differ only in the selected coordinate; the calculator itself is never
modified.

Sign and unit convention:

``K[:, 3*i + mu] = -d F_i,mu / d u_j,nu = d^2 E / du_i,mu du_j,nu`` in
eV/Angstrom^2, atom-major flat indexing (``flat = 3*atom + cartesian``).

Stencil weights are adapted from pymultibinit ``pyeffpot/ifc_targets.py``
(``_STENCILS``, user-authored code reused with permission; provenance recorded
in the module docstring of the ticket 01 implementation notes).  No
projection, drift correction, or symmetrization is applied automatically; the
reported diagnostics make any such treatment an explicit caller decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

__all__ = [
    "FiniteDifferenceError",
    "STENCILS",
    "FiniteDifferenceResult",
    "finite_difference_hessian",
    "hessian_diagnostics",
    "estimate_force_noise",
    "StepStudy",
    "step_convergence",
]

#: Displacement stencils: offset multipliers (units of the step) and
#: derivative weights, adapted from pymultibinit ``pyeffpot/ifc_targets.py``.
STENCILS = {
    "central": ((+1.0, 0.5), (-1.0, -0.5)),
    "central5": ((+2.0, -1.0 / 12.0), (+1.0, 8.0 / 12.0),
                 (-1.0, -8.0 / 12.0), (-2.0, 1.0 / 12.0)),
}


class FiniteDifferenceError(ValueError):
    """Invalid finite-difference setup or non-finite calculator output."""


def _check_forces(forces: np.ndarray, natom: int, where: str) -> np.ndarray:
    forces = np.asarray(forces, dtype=float)
    if forces.shape != (natom, 3):
        raise FiniteDifferenceError(
            f"{where}: get_forces must return shape ({natom}, 3), got {forces.shape}")
    if not np.isfinite(forces).all():
        raise FiniteDifferenceError(f"{where}: get_forces returned non-finite values")
    return forces


def hessian_diagnostics(hessian: np.ndarray) -> dict:
    """Reciprocity and acoustic residuals of a dense (3N, 3N) Hessian.

    ``reciprocity_abs`` is the max |K - K^T| in eV/Angstrom^2;
    ``reciprocity_rel`` scales it by max |K|; ``asr_row_sum_abs`` is the
    max |sum_j K[i, a, j, b]| in eV/Angstrom^2 over the displaced-atom index
    ``j`` only, keeping the Cartesian components ``(i, a, b)`` separate
    (exactly zero for a translation-invariant conservative model up to
    finite-difference noise).  Summing all Cartesian components together
    would let x/y/z violations cancel.
    """
    hessian = np.asarray(hessian, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise FiniteDifferenceError(f"hessian must be square, got {hessian.shape}")
    if hessian.shape[0] % 3:
        raise FiniteDifferenceError(
            "hessian dimension must be 3N for an atom-major Cartesian Hessian, "
            f"got {hessian.shape}")
    natom = hessian.shape[0] // 3
    blocks = hessian.reshape(natom, 3, natom, 3)   # (i, a, j, b)
    asr = blocks.sum(axis=2)                       # sum displaced atoms j only
    scale = float(np.abs(hessian).max())
    reciprocity = float(np.abs(hessian - hessian.T).max())
    return {
        "reciprocity_abs_ev_per_angstrom2": reciprocity,
        "reciprocity_rel": reciprocity / scale if scale > 0 else 0.0,
        "asr_row_sum_abs_ev_per_angstrom2": float(np.abs(asr).max()),
        "max_abs_ev_per_angstrom2": scale,
    }


@dataclass(frozen=True)
class FiniteDifferenceResult:
    """Dense supercell Hessian plus the recorded construction diagnostics."""

    hessian: np.ndarray                 # (3N, 3N) eV/Angstrom^2
    displacement: float                 # Angstrom
    stencil: str
    n_force_calls: int
    diagnostics: dict

    @property
    def natom(self) -> int:
        return int(self.hessian.shape[0] // 3)


def finite_difference_hessian(
    get_forces: Callable[[np.ndarray], np.ndarray],
    reference_positions: np.ndarray,
    displacement: float = 0.01,
    stencil: str = "central",
) -> FiniteDifferenceResult:
    """Dense harmonic Hessian by Cartesian force finite differences.

    Displaces each Cartesian coordinate of each site about
    ``reference_positions`` (Angstrom, exact reference copy) and assembles
    ``K[:, 3*i + mu] = -(sum_j w_j F(offset_j)) / h`` with the selected
    stencil.  Positions differ from the reference in exactly one coordinate.
    """
    if stencil not in STENCILS:
        raise FiniteDifferenceError(
            f"stencil must be one of {sorted(STENCILS)}, got {stencil!r}")
    displacement = float(displacement)
    if not np.isfinite(displacement) or displacement <= 0.0:
        raise FiniteDifferenceError(
            f"displacement must be finite and positive, got {displacement}")

    reference = np.asarray(reference_positions, dtype=float).copy()
    if reference.ndim != 2 or reference.shape[1] != 3 or reference.shape[0] == 0:
        raise FiniteDifferenceError(
            f"reference_positions must have shape (n, 3), got {reference.shape}")
    if not np.isfinite(reference).all():
        raise FiniteDifferenceError("reference_positions contains non-finite values")
    natom = reference.shape[0]

    offsets = STENCILS[stencil]
    hessian = np.zeros((3 * natom, 3 * natom), dtype=float)
    calls = 0
    for i in range(natom):
        for mu in range(3):
            derivative = np.zeros(3 * natom, dtype=float)
            for mult, weight in offsets:
                positions = reference.copy()
                positions[i, mu] += mult * displacement
                forces = _check_forces(
                    get_forces(positions), natom,
                    f"displaced (+{mult}h) site {i} cartesian {mu}")
                calls += 1
                derivative += weight * forces.reshape(-1)
            hessian[:, 3 * i + mu] = -derivative / displacement
    diagnostics = hessian_diagnostics(hessian)
    diagnostics["force_noise_note"] = (
        "row-sum and reciprocity residuals combine finite-difference noise with "
        "calculator noise; measure the calculator noise separately with "
        "estimate_force_noise")
    return FiniteDifferenceResult(
        hessian=hessian,
        displacement=displacement,
        stencil=stencil,
        n_force_calls=calls,
        diagnostics=diagnostics,
    )


def estimate_force_noise(
    get_forces: Callable[[np.ndarray], np.ndarray],
    positions: np.ndarray,
    n_repeats: int = 3,
    fresh_forces: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Runtime (nondeterminism) force noise of the calculator, eV/Angstrom.

    Re-evaluates identical positions ``n_repeats`` times and reports the max
    deviation between repeats.  This bounds the deterministic noise floor that
    finite-difference steps must dominate; it does not measure anharmonicity
    or model error.

    ``fresh_forces`` is the uncached evaluation callback of a caching provider
    (e.g. ``AseForceProvider.get_fresh_forces``).  Without it, repeated calls
    through an ordinary cached getter can return one stored result and report
    zero noise; supply it whenever ``get_forces`` caches identical positions.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise FiniteDifferenceError(
            f"positions must have shape (n, 3) with n > 0, got {positions.shape}")
    if not np.isfinite(positions).all():
        raise FiniteDifferenceError("positions contain non-finite values")
    if n_repeats < 2:
        raise FiniteDifferenceError("n_repeats must be at least 2")
    natom = positions.shape[0]
    evaluate = fresh_forces if fresh_forces is not None else get_forces
    samples = []
    for _ in range(n_repeats):
        # Copy on retention: a provider may reuse one output buffer across calls.
        samples.append(
            _check_forces(evaluate(positions.copy()), natom, "noise probe").copy())
    stacked = np.stack(samples)
    spread = stacked.max(axis=0) - stacked.min(axis=0)
    return {
        "max_repeat_deviation_ev_per_angstrom": float(spread.max()),
        "rms_repeat_deviation_ev_per_angstrom": float(np.sqrt((spread ** 2).mean())),
        "n_repeats": int(n_repeats),
        "fresh_evaluations": fresh_forces is not None,
        "positions_max_abs_ev_per_angstrom": float(np.abs(stacked).max()),
    }


@dataclass(frozen=True)
class StepStudy:
    """One displacement step of a convergence study."""

    displacement: float
    hessian: np.ndarray
    diagnostics: dict
    max_abs_deviation_ev_per_angstrom2: float | None   # vs reference, if supplied
    max_abs_step_change_ev_per_angstrom2: float | None  # vs previous step


def step_convergence(
    get_forces: Callable[[np.ndarray], np.ndarray],
    reference_positions: np.ndarray,
    steps: Sequence[float],
    stencil: str = "central",
    reference_hessian: np.ndarray | None = None,
) -> Tuple[StepStudy, ...]:
    """Hessian versus displacement step size (caller-declared tolerance support).

    Returns one :class:`StepStudy` per requested step, with deviations against
    ``reference_hessian`` (e.g. a known analytic Hessian or the largest-step
    trusted value) and against the previous step.  Interpretation against the
    measured force noise is the caller's declared acceptance decision (R13).
    """
    steps = tuple(float(s) for s in steps)
    if not steps:
        raise FiniteDifferenceError("steps must be non-empty")
    if any(s <= 0.0 for s in steps):
        raise FiniteDifferenceError(f"steps must be positive, got {steps}")
    studies = []
    previous = None
    for step in steps:
        result = finite_difference_hessian(
            get_forces, reference_positions, displacement=step, stencil=stencil)
        deviation = None
        if reference_hessian is not None:
            deviation = float(np.abs(result.hessian - np.asarray(reference_hessian)).max())
        change = None
        if previous is not None:
            change = float(np.abs(result.hessian - previous).max())
        studies.append(StepStudy(
            displacement=step,
            hessian=result.hessian,
            diagnostics=result.diagnostics,
            max_abs_deviation_ev_per_angstrom2=deviation,
            max_abs_step_change_ev_per_angstrom2=change,
        ))
        previous = result.hessian
    return tuple(studies)
