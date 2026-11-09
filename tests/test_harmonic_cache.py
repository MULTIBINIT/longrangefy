import numpy as np
import pytest

from longrangefy import ElectricalResponse, ReferenceStructure, ResponseProvenance, build_reference_harmonics
from longrangefy.cache import load_harmonics, save_harmonics


def test_cache_reuses_only_matching_contract(tmp_path):
    ref = ReferenceStructure(np.eye(3) * 4, [[0, 0, 0]], ("Si",), source="synthetic")
    response = ElectricalResponse(np.zeros((1, 3, 3)), np.eye(3),
                                  ResponseProvenance("synthetic", "test", "Si"))
    identity = {"sha256": "1" * 64, "backend": "test"}
    h = build_reference_harmonics(ref, response, (1, 1, 1), lambda p: -2 * p,
                                  calculator_identity=identity)
    path = tmp_path / "harmonics.npz"
    save_harmonics(path, h)
    cached = load_harmonics(path, ref, response, (1, 1, 1), identity)
    np.testing.assert_array_equal(cached.q_matrices, h.q_matrices)
    np.testing.assert_array_equal(cached.grid_h, h.grid_h)
    assert cached.diagnostics == h.diagnostics
    with pytest.raises(FileExistsError):
        save_harmonics(path, h)
    with pytest.raises(ValueError, match="incompatible"):
        load_harmonics(path, ref, response, (1, 1, 1), {"sha256": "2" * 64, "backend": "test"})
    with pytest.raises(ValueError, match="incompatible"):
        load_harmonics(path, ref, response, (1, 1, 1), identity, displacement=0.02)
    other = ElectricalResponse(np.zeros((1, 3, 3)), np.eye(3) * 2,
                               ResponseProvenance("synthetic", "test", "Si"))
    with pytest.raises(ValueError, match="incompatible"):
        load_harmonics(path, ref, other, (1, 1, 1), identity)
    with pytest.raises(ValueError, match="SHA-256"):
        load_harmonics(path, ref, response, (1, 1, 1), {})
