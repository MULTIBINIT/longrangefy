"""Optional DeepMD original-potential adapter (R12/R17).

Importing this module requires the ``deepmd`` extra (``deepmd-kit`` + ``ase``);
the array core never imports it.  The model file is a user-supplied external
asset: this adapter records its absolute path, SHA-256, size, runtime versions,
and the model-parameter metadata extracted from the serialized graph.

Eligibility (method specification section 2) is a **declared** property: the
caller must state, in ``eligibility_declaration``, why the model contains no
explicit asymptotic electrostatic tail.  This adapter records the declaration
and the extracted model type/descriptor next to it; it does not infer
eligibility from force samples.  DeepMD ``ener`` models with short-range
descriptors (e.g. ``se_e2_a``) contain no explicit Ewald/dipole module, but
the declaration remains the user's recorded statement about provenance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "DeepMDError",
    "ModelIdentity",
    "load_model_identity",
    "DeepMDForceProvider",
]


class DeepMDError(RuntimeError):
    """Missing optional dependency, unreadable model, or unsupported use."""


def _require_deepmd():
    try:
        import deepmd  # noqa: F401
        from deepmd.calculator import DP  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DeepMDError(
            "DeepMD support needs the 'deepmd' extra: install deepmd-kit and ase "
            "(the array core stays usable without them)") from exc
    return DP


def _require_ase():
    try:
        import ase  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DeepMDError(
            "DeepMD support needs the 'ase' extra: install ase") from exc
    import ase

    return ase


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_model_params_json(raw: bytes) -> dict | None:
    """Extract the embedded model-parameter JSON from a serialized DeepMD model.

    DeepMD stores its model parameters (``type_map``, ``descriptor``,
    ``fitting_net``, ...) as one JSON document inside the graph (TensorFlow
    constant ``train_attr/training_script``) or archive (PyTorch).  The
    document is located by its keys and decoded from the enclosing object;
    trailing binary data is ignored by the decoder.  Returns ``None`` when no
    such document is present.
    """
    decoder = json.JSONDecoder()
    markers = []
    for key in (b'"type_map"', b'"fitting_net"', b'"descriptor"'):
        start = 0
        while True:
            index = raw.find(key, start)
            if index < 0:
                break
            markers.append(index)
            start = index + 1
            if len(markers) > 64:
                break
    seen_starts = set()
    for marker_index in sorted(markers):
        begin = raw.rfind(b"{", 0, marker_index)
        if begin < 0 or begin in seen_starts:
            continue
        seen_starts.add(begin)
        window = raw[begin: begin + (1 << 20)].decode("utf-8", errors="ignore")
        try:
            obj, _ = decoder.raw_decode(window)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if len(obj) == 1 and isinstance(obj.get("model"), dict):
            obj = obj["model"]
        if any(key in obj for key in ("type_map", "descriptor", "fitting_net")):
            return obj
    return None


def _runtime_versions() -> dict:
    versions = {}
    try:
        from importlib.metadata import version

        for package in ("deepmd-kit", "ase", "numpy"):
            try:
                versions[package] = version(package)
            except Exception:  # pragma: no cover - packaging metadata absence
                versions[package] = None
    except Exception:  # pragma: no cover
        pass
    return versions


@dataclass(frozen=True)
class ModelIdentity:
    """Recorded identity of the real original model (R17, ticket criterion 1)."""

    path: str
    sha256: str
    size_bytes: int
    units: str
    type_map: tuple
    model_type: str | None
    descriptor: str | None
    backend: str | None
    model_version: str | None
    deepmd_version: str | None
    runtime_versions: Mapping[str, Any]
    extraction_method: str
    eligibility_declaration: str
    declared_by: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
            "units": self.units,
            "type_map": list(self.type_map),
            "model_type": self.model_type,
            "descriptor": self.descriptor,
            "backend": self.backend,
            "model_version": self.model_version,
            "deepmd_version": self.deepmd_version,
            "runtime_versions": dict(self.runtime_versions),
            "extraction_method": self.extraction_method,
            "eligibility_declaration": self.eligibility_declaration,
            "declared_by": self.declared_by,
        }


def load_model_identity(
    model_path: str | Path,
    *,
    eligibility_declaration: str,
    declared_by: str,
    model_type: str | None = None,
    descriptor: str | None = None,
) -> ModelIdentity:
    """Record model identity, units, and declared eligibility.

    ``model_type``/``descriptor`` may be supplied explicitly when the metadata
    cannot be extracted from the file; the extraction method then says so.
    An empty eligibility declaration is rejected: the no-explicit-tail
    contract must be a stated fact about the model, not an assumption.
    """
    _require_deepmd()   # existence check; the model file is read directly below
    if not isinstance(eligibility_declaration, str) or not eligibility_declaration.strip():
        raise DeepMDError(
            "eligibility_declaration is required: state why this model contains "
            "no explicit asymptotic electrostatic tail (method specification "
            "section 2); eligibility is never inferred automatically")
    if not isinstance(declared_by, str) or not declared_by.strip():
        raise DeepMDError("declared_by is required: name the eligibility declarer")
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise DeepMDError(f"DeepMD model file not found: {path}")
    raw = path.read_bytes()

    embedded = _extract_model_params_json(raw)
    type_map: tuple = ()
    inferred_type = None
    inferred_descriptor = None
    backend = None
    model_version = None
    extraction = "unavailable (no embedded model-parameter JSON found)"
    if embedded is not None:
        type_map = tuple(embedded.get("type_map", ()))
        descriptor_block = embedded.get("descriptor") or {}
        fitting_block = embedded.get("fitting_net") or {}
        inferred_type = fitting_block.get("type") or embedded.get("model_type")
        inferred_descriptor = descriptor_block.get("type") or embedded.get("descriptor_type")
        backend = embedded.get("backend")
        model_version = embedded.get("model_version")
        if model_version is not None:
            model_version = str(model_version)
        extraction = "embedded model-parameter JSON in the serialized graph"
    if model_type is not None:
        inferred_type = str(model_type)
        extraction += "; model_type supplied by caller"
    if descriptor is not None:
        inferred_descriptor = str(descriptor)
        extraction += "; descriptor supplied by caller"

    versions = _runtime_versions()
    return ModelIdentity(
        path=str(path),
        sha256=_file_sha256(path),
        size_bytes=int(path.stat().st_size),
        units="angstrom, eV, eV/angstrom (DeepMD native)",
        type_map=type_map,
        model_type=inferred_type,
        descriptor=inferred_descriptor,
        backend=backend or ("tensorflow" if path.suffix == ".pb"
                            else "pytorch" if path.suffix == ".pth" else None),
        model_version=model_version,
        deepmd_version=versions.get("deepmd-kit"),
        runtime_versions=versions,
        extraction_method=extraction,
        eligibility_declaration=str(eligibility_declaration),
        declared_by=str(declared_by),
    )


class DeepMDForceProvider:
    """Forces/energies of a fixed-cell DeepMD model in a fixed site order.

    The cell, species, and site order are frozen at construction (the initial
    contract excludes cell strain and topology changes).  ``get_forces`` only
    moves the atoms; ASE's calculator cache is reset on each position change,
    so repeated identical positions reuse the cached evaluation.
    ``get_fresh_forces`` clears that cache first, for force-noise probes that
    must measure runtime variability at one geometry.
    """

    def __init__(
        self,
        model_path: str | Path,
        cell: np.ndarray,
        species: Sequence[str],
        *,
        eligibility_declaration: str,
        declared_by: str,
        model_type: str | None = None,
        descriptor: str | None = None,
        initial_positions: np.ndarray,
    ) -> None:
        ase = _require_ase()
        DP = _require_deepmd()
        identity = load_model_identity(
            model_path, eligibility_declaration=eligibility_declaration,
            declared_by=declared_by, model_type=model_type, descriptor=descriptor)
        cell = np.asarray(cell, dtype=float)
        if cell.shape != (3, 3) or not np.isfinite(cell).all():
            raise DeepMDError(f"cell must be finite (3, 3) Angstrom, got {cell.shape}")
        species = tuple(str(s) for s in species)
        if not species:
            raise DeepMDError("species must be non-empty")
        positions = np.asarray(initial_positions, dtype=float)
        if positions.shape != (len(species), 3):
            raise DeepMDError(
                f"initial_positions must have shape ({len(species)}, 3), "
                f"got {positions.shape}")
        # The runtime type map is authoritative; the embedded training JSON is
        # optional provenance and may be absent from a loadable model.
        calculator = DP(model=str(identity.path))
        runtime_type_map = tuple(str(t) for t in calculator.dp.get_type_map())
        if not runtime_type_map:
            raise DeepMDError(
                f"DeepMD runtime reported an empty native type map for "
                f"{identity.path}; species support cannot be verified")
        missing = sorted(set(species) - set(runtime_type_map))
        if missing:
            raise DeepMDError(
                f"reference species {missing} are not supported by the model "
                f"type_map {list(runtime_type_map)}")
        metadata_type_map = tuple(identity.type_map)
        note = "; type_map from DeepMD runtime type_map"
        if metadata_type_map and metadata_type_map != runtime_type_map:
            note += f" (training metadata declared {list(metadata_type_map)})"
        identity = replace(
            identity,
            type_map=runtime_type_map,
            extraction_method=identity.extraction_method + note)
        atoms = ase.Atoms(symbols=list(species), positions=positions,
                          cell=cell, pbc=True)
        atoms.calc = calculator
        self._ase = ase
        self._atoms = atoms
        self._identity = identity
        self._natom = len(species)

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def natom(self) -> int:
        return self._natom

    def _set_positions(self, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self._natom, 3):
            raise DeepMDError(
                f"positions must have shape ({self._natom}, 3), got {positions.shape}")
        if not np.isfinite(positions).all():
            raise DeepMDError("positions contain non-finite values")
        self._atoms.set_positions(positions)   # invalidates the ASE calculator cache

    def get_forces(self, positions: np.ndarray) -> np.ndarray:
        self._set_positions(positions)
        forces = np.asarray(self._atoms.get_forces(), dtype=float)
        if not np.isfinite(forces).all():
            raise DeepMDError("DeepMD returned non-finite forces")
        return forces

    def get_fresh_forces(self, positions: np.ndarray) -> np.ndarray:
        """Same positions, but from an uncached calculator evaluation."""
        self._set_positions(positions)
        self._atoms.calc.reset()
        return self.get_forces(positions)

    def get_energy(self, positions: np.ndarray) -> float:
        """Whole-cell energy (eV) at ``positions``; reuses the calculator cache."""
        self._set_positions(positions)
        return float(self._atoms.get_potential_energy())
