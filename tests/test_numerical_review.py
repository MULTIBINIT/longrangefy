"""Consumer-visible regressions for reviewer findings C3-C8.

Every test drives the public workflow rather than a private helper: dense
finite differences, the force-noise API, frozen reference/response inputs, and
the optional DeepMD adapter.
"""

import numpy as np
import pytest

import longrangefy
from longrangefy.deepmd import DeepMDError
from longrangefy.finitediff import (
    FiniteDifferenceError,
    estimate_force_noise,
    finite_difference_hessian,
    hessian_diagnostics,
)
from longrangefy.reference import ReferenceStructure, SymmetryOperation
from longrangefy.response import ElectricalResponse, ResponseProvenance


def make_response(natom=2):
    born = (np.array([np.eye(3) * 2.0, -np.eye(3) * 2.0]) if natom == 2
            else np.zeros((natom, 3, 3)))
    return ElectricalResponse(
        born_charges=born, dielectric=9.0 * np.eye(3),
        provenance=ResponseProvenance(
            source="review fixture", method="fixture",
            reference_description="review fixture"))


# C3: the acoustic residual keeps Cartesian components separate.
def test_acoustic_diagnostic_does_not_cancel_cartesian_components():
    hessian = np.array([[1.0, -1.0, 0.0],
                        [-1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0]])
    # The old scalar row sum is zero; a rigid x translation still responds.
    assert float(np.abs(hessian.sum(axis=1)).max()) == 0.0
    assert not np.allclose(hessian @ np.array([1.0, 0.0, 0.0]), 0.0)

    diagnostics = hessian_diagnostics(hessian)
    assert diagnostics["asr_row_sum_abs_ev_per_angstrom2"] == pytest.approx(1.0)

    # The same diagnostic is what every FiniteDifferenceResult carries.
    result = finite_difference_hessian(
        lambda positions: -(positions @ hessian), np.zeros((1, 3)),
        displacement=0.01)
    assert result.diagnostics["asr_row_sum_abs_ev_per_angstrom2"] == pytest.approx(1.0)


def test_hessian_diagnostics_rejects_non_cartesian_dimensions():
    with pytest.raises(FiniteDifferenceError, match="3N"):
        hessian_diagnostics(np.eye(4))


# C4: noise probes request fresh calculator evaluations without losing caching.
class _CachingProvider:
    """Minimal cache-backed provider with the two-callback provider contract."""

    def __init__(self):
        self.calls = 0
        self._key = None
        self._value = None

    def get_forces(self, positions):
        key = positions.tobytes()
        if key != self._key:
            self.calls += 1
            self._key = key
            self._value = np.full(positions.shape, float(self.calls))
        return self._value

    def get_fresh_forces(self, positions):
        self.calls += 1
        self._key = positions.tobytes()
        self._value = np.full(positions.shape, float(self.calls))
        return self._value


def test_force_noise_bypasses_cache_only_when_fresh_callback_supplied():
    provider = _CachingProvider()
    reference = np.zeros((2, 3))

    cached = estimate_force_noise(provider.get_forces, reference, n_repeats=3)
    assert provider.calls == 1
    assert cached["fresh_evaluations"] is False
    assert cached["max_repeat_deviation_ev_per_angstrom"] == 0.0

    fresh = estimate_force_noise(
        provider.get_forces, reference, n_repeats=3,
        fresh_forces=provider.get_fresh_forces)
    assert provider.calls == 4
    assert fresh["fresh_evaluations"] is True
    assert fresh["max_repeat_deviation_ev_per_angstrom"] == pytest.approx(2.0)

    # Ordinary evaluations still reuse the cache after the fresh probes.
    provider.get_forces(reference)
    assert provider.calls == 4


def test_ase_provider_fresh_forces_bypass_cache_but_get_forces_does_not():
    ase = pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator
    from longrangefy.ase import AseForceProvider

    class CountingCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.calls = 0

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=()):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            self.results["energy"] = 0.0
            self.results["forces"] = np.full((len(atoms), 3), float(self.calls))

    atoms = ase.Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                      cell=[4.0, 4.0, 4.0], pbc=True)
    calculator = CountingCalculator()
    provider = AseForceProvider(atoms, calculator=calculator)
    reference = atoms.get_positions()

    cached = estimate_force_noise(provider.get_forces, reference, n_repeats=3)
    assert calculator.calls == 1
    assert cached["max_repeat_deviation_ev_per_angstrom"] == 0.0

    fresh = estimate_force_noise(
        provider.get_forces, reference, n_repeats=3,
        fresh_forces=provider.get_fresh_forces)
    assert calculator.calls == 4
    assert fresh["max_repeat_deviation_ev_per_angstrom"] == pytest.approx(2.0)


# C5: frozen reference and response inputs own their arrays.
def test_reference_and_response_do_not_alias_caller_arrays():
    cell = np.eye(3) * 5.0
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    rot = -np.eye(3)
    trans = np.zeros(3)
    reference = ReferenceStructure(
        cell=cell, frac_positions=frac, species=("Na", "Cl"),
        symmetry_operations=(SymmetryOperation(rot, trans),))

    cell[0, 0] = 99.0
    frac[0, 0] = 0.25
    rot[0, 0] = 7.0
    trans[0] = 3.0
    assert reference.cell[0, 0] == 5.0
    assert reference.frac_positions[0, 0] == 0.0
    assert reference.symmetry_operations[0].rot[0, 0] == -1.0
    assert reference.symmetry_operations[0].trans[0] == 0.0

    for array in (reference.cell, reference.frac_positions,
                  reference.symmetry_operations[0].rot,
                  reference.symmetry_operations[0].trans):
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 1.0

    born = np.array([np.eye(3) * 2.0, -np.eye(3) * 2.0])
    dielectric = np.eye(3) * 9.0
    response = ElectricalResponse(
        born_charges=born, dielectric=dielectric,
        provenance=ResponseProvenance(
            source="review fixture", method="fixture",
            reference_description="review fixture"))
    born[0, 0, 0] = 42.0
    dielectric[0, 0] = 3.0
    assert response.born_charges[0, 0, 0] == 2.0
    assert response.dielectric[0, 0] == 9.0
    for array in (response.born_charges, response.dielectric):
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 1.0


# C6: a missing optional training JSON does not imply unsupported species.
def test_deepmd_uses_native_runtime_type_map_without_training_metadata(
        tmp_path, monkeypatch):
    pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator
    import longrangefy.deepmd as deepmd_module

    class FakeDeepPot:
        def __init__(self, type_map):
            self.type_map = list(type_map)

        def get_type_map(self):
            return list(self.type_map)

    class FakeDP(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, model):
            super().__init__()
            self.dp = FakeDeepPot(("Na", "Cl"))
            self.calls = 0

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=()):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            self.results["energy"] = 0.0
            self.results["forces"] = np.full((len(atoms), 3), float(self.calls))

    monkeypatch.setattr(deepmd_module, "_require_deepmd", lambda: FakeDP)

    model = tmp_path / "model.pth"
    model.write_bytes(b"loadable-model-without-training-json")
    with pytest.raises(deepmd_module.DeepMDError, match="eligibility_declaration"):
        deepmd_module.load_model_identity(str(model), eligibility_declaration=None, declared_by="user")
    with pytest.raises(deepmd_module.DeepMDError, match="declared_by"):
        deepmd_module.load_model_identity(str(model), eligibility_declaration="no tail", declared_by=None)
    identity = deepmd_module.load_model_identity(
        str(model), eligibility_declaration="declared no tail",
        declared_by="review fixture")
    assert identity.type_map == ()

    provider = deepmd_module.DeepMDForceProvider(
        str(model), np.eye(3) * 5.0, ("Na", "Cl"),
        eligibility_declaration="declared no tail", declared_by="review fixture",
        initial_positions=np.zeros((2, 3)))
    assert provider.identity.type_map == ("Na", "Cl")
    assert "runtime type_map" in provider.identity.extraction_method

    calculator = provider._atoms.calc
    reference = np.zeros((2, 3))
    cached = estimate_force_noise(provider.get_forces, reference, n_repeats=3)
    assert calculator.calls == 1
    assert cached["max_repeat_deviation_ev_per_angstrom"] == 0.0
    fresh = estimate_force_noise(
        provider.get_forces, reference, n_repeats=3,
        fresh_forces=provider.get_fresh_forces)
    assert calculator.calls == 4
    assert fresh["max_repeat_deviation_ev_per_angstrom"] == pytest.approx(2.0)

    with pytest.raises(DeepMDError, match="not supported by the model type_map"):
        deepmd_module.DeepMDForceProvider(
            str(model), np.eye(3) * 5.0, ("Na", "H"),
            eligibility_declaration="declared no tail",
            declared_by="review fixture", initial_positions=np.zeros((2, 3)))


# C7: malformed reference geometries are rejected before any evaluation.
def test_nonfinite_reference_positions_rejected_before_callback():
    calls = []

    def get_forces(positions):
        calls.append(positions.copy())
        return np.zeros_like(positions)

    with pytest.raises(FiniteDifferenceError, match="non-finite"):
        finite_difference_hessian(get_forces, np.array([[np.nan, 0.0, 0.0]]))
    assert calls == []

    with pytest.raises(FiniteDifferenceError, match="non-finite"):
        estimate_force_noise(get_forces, np.array([[0.0, np.inf, 0.0]]))
    with pytest.raises(FiniteDifferenceError, match="positions"):
        estimate_force_noise(get_forces, np.zeros((0, 3)))


# C8: retained noise samples must not alias a reused provider buffer.
def test_force_noise_copies_reused_output_buffer():
    buffer = np.zeros((2, 3))
    counter = {"calls": 0}

    def shared_buffer(positions):
        counter["calls"] += 1
        buffer[:] = counter["calls"]
        return buffer

    noise = estimate_force_noise(shared_buffer, np.zeros((2, 3)), n_repeats=3)
    assert noise["max_repeat_deviation_ev_per_angstrom"] == pytest.approx(2.0)
    assert noise["positions_max_abs_ev_per_angstrom"] == pytest.approx(3.0)


# The harmonics workflow forwards the fresh callback and owns its residual forces.
def test_harmonics_workflow_forwards_fresh_callback_and_copies_residuals():
    reference = ReferenceStructure(
        cell=np.eye(3) * 5.0, frac_positions=np.zeros((1, 3)), species=("Na",))
    response = make_response(natom=1)
    buffer = np.zeros((1, 3))
    fresh_calls = {"count": 0}

    def cached_forces(positions):
        return buffer

    def fresh_forces(positions):
        fresh_calls["count"] += 1
        buffer[:] = fresh_calls["count"]
        return buffer

    harmonics = longrangefy.build_reference_harmonics(
        reference, response, (1, 1, 1), cached_forces,
        force_noise_repeats=2, fresh_get_forces=fresh_forces)

    assert harmonics.diagnostics["force_noise_fresh_evaluations"] is True
    assert harmonics.diagnostics["force_noise"][
        "max_repeat_deviation_ev_per_angstrom"] == pytest.approx(1.0)
    # The residual-force record was taken before the fresh probes overwrote the buffer.
    assert harmonics.diagnostics["residual_forces_max_ev_per_angstrom"] == 0.0
