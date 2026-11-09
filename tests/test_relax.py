"""Reference preparation: no automatic relaxation, symmetry preservation (R02)."""

import numpy as np
import pytest

from longrangefy import (
    ReferencePreparationError,
    ReferenceStructure,
    SymmetryOperation,
    prepare_reference,
)


def cell():
    return np.diag([4.0, 4.0, 4.0])


def two_site_reference(displacement=0.0):
    """Two sites related by inversion; the model's minimum is the symmetric point."""
    frac = np.array([[0.25 + displacement, 0.25, 0.25],
                     [0.75 - displacement, 0.75, 0.75]])
    operations = (
        SymmetryOperation(-np.eye(3), np.zeros(3)),
        SymmetryOperation(np.eye(3), np.zeros(3)),
    )
    return ReferenceStructure(cell=cell(), frac_positions=frac,
                              species=("Na", "Cl"),
                              symmetry_label="P-1 (inversion)",
                              symmetry_operations=operations,
                              source="synthetic inversion fixture")


def spring_forces(rest_frac):
    """Polar spring pair: forces pull each site toward its rest position."""
    rest = rest_frac @ cell()

    def get_forces(positions):
        positions = np.asarray(positions, dtype=float)
        return -0.5 * (positions - rest)      # eV/Angstrom per (eV/Angstrom) displacement

    return get_forces


def test_none_mode_never_moves_and_reports_residuals():
    reference = two_site_reference(displacement=0.02)
    # The model minimum is the symmetric point, so the declared reference is
    # off stationarity and its residual forces are nonzero.
    get_forces = spring_forces(np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]))
    prepared, record = prepare_reference(reference, get_forces, mode="none")
    np.testing.assert_allclose(prepared.frac_positions, reference.frac_positions, atol=0)
    assert record.mode == "none"
    assert record.n_steps == 0
    assert record.max_residual_force_ev_per_angstrom > 0.0
    assert record.symmetry_preserved is False
    forces = get_forces(reference.frac_positions @ reference.cell)
    assert record.max_residual_force_ev_per_angstrom == pytest.approx(
        np.abs(forces).max())


def test_symmetry_preserving_relaxation_moves_sites_together():
    reference = two_site_reference(displacement=0.02)
    rest = np.array([[0.26, 0.25, 0.25], [0.74, 0.75, 0.75]])
    get_forces = spring_forces(rest)
    prepared, record = prepare_reference(
        reference, get_forces, mode="symmetry_preserving",
        fmax=1e-6, max_steps=400, step_size=2e-3)

    assert record.converged
    assert record.symmetry_preserved
    assert record.n_steps > 0
    delta = prepared.frac_positions - reference.frac_positions
    # Symmetry projection keeps the inversion pair moving together.
    np.testing.assert_allclose(delta[0], -delta[1], atol=1e-12)
    # Residual forces were reported at the final positions.
    forces = get_forces(prepared.cart_positions)
    assert record.max_residual_force_ev_per_angstrom == pytest.approx(
        np.abs(forces).max())
    # And the residual is below the requested fmax.
    assert record.max_residual_force_ev_per_angstrom <= 1e-6
    # The declared symmetry label and operations survive the preparation.
    assert prepared.symmetry_label == reference.symmetry_label
    assert prepared.symmetry_operations == reference.symmetry_operations


def test_unrestricted_mode_requires_explicit_intent():
    reference = two_site_reference(displacement=0.02)
    get_forces = spring_forces(reference.frac_positions)
    with pytest.raises(ReferencePreparationError, match="allow_symmetry_breaking"):
        prepare_reference(reference, get_forces, mode="unrestricted")

    prepared, record = prepare_reference(
        reference, get_forces, mode="unrestricted",
        allow_symmetry_breaking=True, fmax=1e-6, max_steps=20, step_size=2e-3)
    assert record.mode == "unrestricted"
    assert record.symmetry_preserved is False
    assert any("unrestricted" in warning for warning in record.warnings)
    # Unrestricted relaxation may move the two sites unequally; the reference
    # object reports its own residual forces for the record.
    forces = get_forces(prepared.cart_positions)
    assert record.max_residual_force_ev_per_angstrom == pytest.approx(
        np.abs(forces).max())


def test_relaxation_never_changes_the_lattice_or_species():
    reference = two_site_reference(displacement=0.02)
    get_forces = spring_forces(reference.frac_positions)
    prepared, _ = prepare_reference(reference, get_forces,
                                    mode="symmetry_preserving",
                                    fmax=1e-6, max_steps=20, step_size=2e-3)
    np.testing.assert_allclose(prepared.cell, reference.cell, atol=0)
    assert prepared.species == reference.species
    assert prepared.natom == reference.natom


def test_inconsistent_declared_symmetry_is_rejected():
    reference = ReferenceStructure(
        cell=cell(),
        frac_positions=np.array([[0.11, 0.23, 0.37], [0.74, 0.75, 0.75]]),
        species=("Na", "Cl"),
        symmetry_label="bogus inversion",
        symmetry_operations=(
            SymmetryOperation(-np.eye(3), np.zeros(3)),
            SymmetryOperation(np.eye(3), np.zeros(3)),),
        source="synthetic inconsistent fixture")
    get_forces = spring_forces(reference.frac_positions)
    with pytest.raises(ReferencePreparationError, match="does not map"):
        prepare_reference(reference, get_forces, mode="symmetry_preserving")


def test_invalid_mode_and_force_shape_rejected():
    reference = two_site_reference()
    get_forces = spring_forces(reference.frac_positions)
    with pytest.raises(ReferencePreparationError, match="mode must be"):
        prepare_reference(reference, get_forces, mode="relax_everything")
    with pytest.raises(ReferencePreparationError, match="shape"):
        prepare_reference(reference, lambda p: np.zeros((3, 3)), mode="none")
    with pytest.raises(ReferencePreparationError, match="non-finite"):
        prepare_reference(reference, lambda p: np.full(p.shape, np.inf), mode="none")
