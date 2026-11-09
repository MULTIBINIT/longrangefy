"""Reusable matching-grid harmonics with explicit compatibility checks."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from . import __version__
from .reference import SupercellMapping


@dataclass(frozen=True)
class CachedHarmonics:
    q_matrices: np.ndarray
    grid_h: np.ndarray
    diagnostics: dict


def _manifest(reference, response, reps, identity, displacement, stencil, asr, ws_toldist):
    digest = identity.get("sha256") if isinstance(identity, dict) else None
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("a lowercase SHA-256 model identity is required for harmonic caching")
    mapping = SupercellMapping(reference, reps)
    source_dir = Path(__file__).parent
    source_hash = hashlib.sha256()
    for name in ("ase.py", "deepmd.py", "finitediff.py", "harmonics.py", "ifc.py",
                 "reference.py", "response.py"):
        source_hash.update(name.encode())
        source_hash.update((source_dir / name).read_bytes())
    return {
        "format": 1, "longrangefy_version": __version__, "harmonic_source_sha256": source_hash.hexdigest(),
        "model": identity, "reference": reference.to_dict(),
        "born": response.born_charges.tolist(), "dielectric": response.dielectric.tolist(),
        "response": response.validation_report(), "matching_reps": list(mapping.reps),
        "displacement": float(displacement), "stencil": stencil, "asr": asr,
        "ws_toldist": float(ws_toldist),
    }


def save_harmonics(path, harmonics):
    """Create a cache; never overwrite an existing file or infer model identity."""
    c = harmonics.conventions
    manifest = _manifest(harmonics.reference, harmonics.response, harmonics.matching.reps,
                         dict(harmonics.calculator_identity), c["finite_difference_displacement_angstrom"],
                         c["finite_difference_stencil"], c["asr"], c["ws_toldist_angstrom2"])
    metadata = json.dumps({"manifest": manifest, "diagnostics": harmonics.diagnostics},
                          sort_keys=True, allow_nan=False)
    path = Path(path)
    with path.open("xb") as handle:
        try:
            np.savez_compressed(handle, metadata=np.array(metadata),
                                q_matrices=harmonics.q_matrices, grid_h=harmonics.grid_h)
        except Exception:
            handle.close()
            path.unlink(missing_ok=True)
            raise


def load_harmonics(path, reference, response, reps, identity, *, displacement=0.01,
                   stencil="central", asr="none", ws_toldist=1e-8):
    """Load only when the full model, reference, response and numerical contract matches."""
    expected = json.loads(json.dumps(
        _manifest(reference, response, reps, identity, displacement, stencil, asr, ws_toldist),
        sort_keys=True, allow_nan=False))
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata["manifest"] != expected:
            raise ValueError("harmonic cache is incompatible with the requested inputs or settings")
        matrices = np.array(data["q_matrices"], copy=True)
        grid = np.array(data["grid_h"], copy=True)
    mapping = SupercellMapping(reference, reps)
    nq = int(np.prod(mapping.reps))
    expected_grid = np.array(list(np.ndindex(*mapping.reps)))
    if (matrices.shape != (nq, reference.natom, 3, reference.natom, 3)
            or not np.isfinite(matrices).all() or not np.array_equal(grid, expected_grid)):
        raise ValueError("harmonic cache has invalid grid matrices or ordering")
    return CachedHarmonics(matrices, grid, metadata["diagnostics"])
