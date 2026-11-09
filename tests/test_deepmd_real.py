"""Real DeepMD model through ASE: identity, finite differences, composition (R12/R17).

Enabled only when a real-model configuration exists (default path from the
ticket 01 evidence directory, or ``LONGRANGEFY_REAL_CONFIG``).  The model bytes
and the acquired reference/tensor data live outside the repository; this test
never fabricates a model path or reference values.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

CONFIG_DEFAULT = Path(
    "/STORAGE1/hexu/projects_bak/longrangefy/ticket01/real_config.json")
CONFIG_ENV = "LONGRANGEFY_REAL_CONFIG"


def load_config():
    path = Path(os.environ.get(CONFIG_ENV, CONFIG_DEFAULT))
    if not path.exists():
        pytest.skip(f"real DeepMD configuration not available at {path}")
    config = json.loads(path.read_text())
    if not Path(config["model_path"]).exists():
        pytest.skip(f"configured DeepMD model missing: {config['model_path']}")
    for package in ("deepmd", "ase"):
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"real-model test needs the optional '{package}' package")
    return config


@pytest.fixture(scope="module")
def real():
    from longrangefy import ElectricalResponse, ReferenceStructure, ResponseProvenance

    config = load_config()
    base = Path(config["data_dir"])
    reference = ReferenceStructure.from_dict(
        json.loads((base / config["reference_file"]).read_text()))
    tensors = np.load(base / config["response_file"])
    provenance_data = json.loads((base / config["response_provenance_file"]).read_text())
    response = ElectricalResponse(
        born_charges=tensors["born_charges"],
        dielectric=tensors["dielectric"],
        provenance=ResponseProvenance(**provenance_data))
    return config, base, reference, response


def test_identity_units_and_declared_eligibility(real):
    from longrangefy.deepmd import load_model_identity

    config, _, _, _ = real
    identity = load_model_identity(
        config["model_path"],
        eligibility_declaration=config["eligibility_declaration"],
        declared_by=config["declared_by"])
    assert len(identity.sha256) == 64
    assert identity.size_bytes > 0
    assert identity.type_map == tuple(config["expected_type_map"])
    assert identity.model_type == config["expected_model_type"]
    assert identity.descriptor == config["expected_descriptor"]
    assert "eV" in identity.units and "angstrom" in identity.units.lower()
    assert identity.eligibility_declaration == config["eligibility_declaration"]
    assert identity.deepmd_version


def test_reference_forces_and_cache_semantics(real):
    from longrangefy import SupercellMapping
    from longrangefy.deepmd import DeepMDForceProvider

    config, _, reference, _ = real
    matching = SupercellMapping(reference, tuple(config["matching_reps"]))
    positions = matching.cart_positions_supercell()
    provider = DeepMDForceProvider(
        config["model_path"], matching.supercell_cell,
        matching.species_supercell(),
        eligibility_declaration=config["eligibility_declaration"],
        declared_by=config["declared_by"],
        initial_positions=positions)

    forces = provider.get_forces(positions)
    assert forces.shape == positions.shape
    assert np.isfinite(forces).all()
    residual = float(np.abs(forces).max())
    assert residual <= config["residual_force_tolerance_ev_per_angstrom"]

    # Position changes invalidate the ASE calculator cache; identical positions reuse it.
    # The counter is attached after the residual-force call above, so the first
    # assertion relative to the attached state must not assume a fresh cache.
    calculator = provider._atoms.calc
    counter = {"calls": 0}
    original_calculate = calculator.calculate

    def counting_calculate(*args, **kwargs):
        counter["calls"] += 1
        return original_calculate(*args, **kwargs)

    calculator.calculate = counting_calculate
    provider.get_forces(positions)
    after_first = counter["calls"]
    provider.get_forces(positions.copy())
    assert counter["calls"] == after_first    # identical positions: cache hit, no recompute
    moved = positions.copy()
    moved[0, 0] += 1e-3
    provider.get_forces(moved)
    assert counter["calls"] == after_first + 1    # position change invalidated the cache


def test_primitive_cell_hessian_step_convergence_and_diagnostics(real):
    from longrangefy import finite_difference_hessian, step_convergence
    from longrangefy.deepmd import DeepMDForceProvider

    config, _, reference, _ = real
    positions = reference.cart_positions
    provider = DeepMDForceProvider(
        config["model_path"], reference.cell, reference.species,
        eligibility_declaration=config["eligibility_declaration"],
        declared_by=config["declared_by"],
        initial_positions=positions)

    result = finite_difference_hessian(provider.get_forces, positions,
                                      displacement=0.01, stencil="central")
    diagnostics = result.diagnostics
    assert abs(diagnostics["reciprocity_rel"]) <= config["reciprocity_rel_tolerance"]
    assert diagnostics["asr_row_sum_abs_ev_per_angstrom2"] <= \
        config["asr_row_sum_tolerance_ev_per_angstrom2"]
    assert np.isfinite(result.hessian).all()

    studies = step_convergence(provider.get_forces, positions,
                               steps=config["convergence_steps"], stencil="central")
    changes = [s.max_abs_step_change_ev_per_angstrom2 for s in studies[1:]]
    assert changes, "convergence study needs at least two steps"
    # Successive changes shrink and end below the empirically estimated
    # step-change smoke threshold (10x the measured residual from this same
    # run; not a predeclared independent acceptance bound).
    assert changes[-1] <= config["step_change_tolerance_ev_per_angstrom2"]
    assert changes[-1] < changes[0]


def test_matching_grid_harmonics_from_real_model(real):
    from longrangefy import SupercellMapping, build_reference_harmonics
    from longrangefy.deepmd import DeepMDForceProvider

    config, _, reference, response = real
    reps = tuple(config["matching_reps"])
    matching = SupercellMapping(reference, reps)
    positions = matching.cart_positions_supercell()
    provider = DeepMDForceProvider(
        config["model_path"], matching.supercell_cell,
        matching.species_supercell(),
        eligibility_declaration=config["eligibility_declaration"],
        declared_by=config["declared_by"],
        initial_positions=positions)

    harmonics = build_reference_harmonics(
        reference, response, reps, provider.get_forces,
        displacement=config["displacement"], asr="none",
        force_noise_repeats=2,
        fresh_get_forces=provider.get_fresh_forces,
        calculator_identity=provider.identity.to_dict())

    assert harmonics.q_matrices.shape[0] == int(np.prod(reps))
    assert harmonics.grid_h.shape == (int(np.prod(reps)), 3)
    assert harmonics.folded_ws_weights.shape == (reference.natom, reference.natom,
                                                int(np.prod(reps)))
    assert harmonics.calculator_identity["sha256"] == provider.identity.sha256
    diagnostics = harmonics.diagnostics
    assert diagnostics["iq_on_grid_identity_max_abs_ev_per_angstrom2"] < 1e-8
    assert diagnostics["reciprocity_rel"] <= config["reciprocity_rel_tolerance"]
    assert diagnostics["residual_forces_max_ev_per_angstrom"] <= \
        config["residual_force_tolerance_ev_per_angstrom"]
    # Unmassweighted matrices: symmetric-bond consistency is checked through the
    # dense Hessian, and no mass scaling is applied anywhere in the core.
    for qmatrix in harmonics.q_matrices:
        np.testing.assert_allclose(qmatrix, qmatrix.conj().transpose(2, 3, 0, 1),
                                   atol=config["hermiticity_tolerance"])
