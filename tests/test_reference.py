"""Reference structure, supercell mapping, and index-order behaviour (R01/R09)."""

import numpy as np
import pytest

from conftest import SKEW_CELL, SKEW_FRAC, SKEW_SPECIES
from longrangefy import (
    ReferenceStructure,
    ReferenceStructureError,
    SupercellMapping,
    SymmetryOperation,
)


def make_reference(**overrides):
    data = dict(cell=SKEW_CELL, frac_positions=SKEW_FRAC, species=SKEW_SPECIES,
                symmetry_label="test", source="synthetic skew fixture")
    data.update(overrides)
    return ReferenceStructure(**data)


def test_reference_validation_rejects_bad_inputs():
    with pytest.raises(ReferenceStructureError, match="singular"):
        make_reference(cell=np.zeros((3, 3)))
    with pytest.raises(ReferenceStructureError, match="shape"):
        make_reference(cell=np.eye(2))
    with pytest.raises(ReferenceStructureError, match="non-finite"):
        make_reference(cell=np.full((3, 3), np.nan))
    with pytest.raises(ReferenceStructureError, match="species labels"):
        make_reference(species=("Na",))
    with pytest.raises(ReferenceStructureError, match="3D-periodic"):
        make_reference(pbc=(True, True, False))
    with pytest.raises(ReferenceStructureError, match="n > 0"):
        make_reference(frac_positions=np.zeros((0, 3)), species=())


def test_cartesian_convention_is_rows():
    reference = make_reference()
    cart = reference.cart_positions
    # row-vector convention: cart = frac @ cell
    np.testing.assert_allclose(cart, reference.frac_positions @ SKEW_CELL, atol=0)


def test_supercell_index_order_is_cell_major_and_invertible():
    reference = make_reference()
    mapping = SupercellMapping(reference, (2, 1, 3))
    assert mapping.natom_supercell == 12
    assert mapping.n_cells == 6
    assert mapping.reps == (2, 1, 3)
    # The documented formula ((c1*n2+c2)*n3+c3)*n_uc + site.
    assert mapping.atom_index((0, 0, 0), 0) == 0
    assert mapping.atom_index((0, 0, 0), 1) == 1
    assert mapping.atom_index((0, 0, 1), 0) == 2
    assert mapping.atom_index((1, 0, 2), 1) == 11
    for atom in range(mapping.natom_supercell):
        cell, site = mapping.cell_and_site(atom)
        assert mapping.atom_index(cell, site) == atom
    # Supercell-basis fractional positions: (offset + primitive frac) / reps.
    frac = mapping.frac_positions_supercell()
    reps = np.asarray(mapping.reps, dtype=float)[None, :]
    np.testing.assert_allclose(frac[0], reference.frac_positions[0] / reps[0])
    np.testing.assert_allclose(frac[2], (reference.frac_positions[0] + [0, 0, 1]) / reps[0])
    np.testing.assert_allclose(frac[11], (reference.frac_positions[1] + [1, 0, 2]) / reps[0])
    # Cartesian positions use the *primitive* cell for offsets and agree with
    # mapping the supercell-basis fractional positions through the supercell.
    cart = mapping.cart_positions_supercell()
    np.testing.assert_allclose(cart, frac @ mapping.supercell_cell, atol=1e-12)
    np.testing.assert_allclose(cart[1], reference.frac_positions[1] @ reference.cell,
                               atol=1e-12)
    np.testing.assert_allclose(
        cart[11],
        (reference.frac_positions[1] + np.array([1, 0, 2])) @ reference.cell, atol=1e-12)
    # Distinct supercell sites have distinct Cartesian positions (no aliasing).
    pairwise = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=2)
    assert pairwise[np.triu_indices(mapping.natom_supercell, 1)].min() > 0.5
    species = mapping.species_supercell()
    assert species[0] == "Na" and species[1] == "Cl" and species[2] == "Na"
    assert len(species) == mapping.natom_supercell


def test_supercell_cell_rows_are_diagonal_repetitions():
    reference = make_reference()
    mapping = SupercellMapping(reference, (2, 3, 1))
    np.testing.assert_allclose(mapping.supercell_cell,
                               np.diag([2.0, 3.0, 1.0]) @ SKEW_CELL, atol=0)


def test_matching_q_grid_is_full_unshifted_commensurate_set():
    reference = make_reference()
    mapping = SupercellMapping(reference, (2, 1, 3))
    q = mapping.q_grid()
    assert q.shape == (6, 3)
    # one representative per commensurate point: q = integer / reps, in [0, 1)
    expected = {(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 1 / 3),
                (0.5, 0.0, 1 / 3), (0.0, 0.0, 2 / 3), (0.5, 0.0, 2 / 3)}
    observed = np.array(sorted(map(tuple, q)))
    expected = np.array(sorted(expected))
    assert observed.shape == expected.shape
    assert np.allclose(observed, expected, atol=1e-12)


def test_invalid_reps_rejected():
    reference = make_reference()
    with pytest.raises(ReferenceStructureError):
        SupercellMapping(reference, (2, 0, 2))
    with pytest.raises(ReferenceStructureError):
        SupercellMapping(reference, (2, -1, 2))
    with pytest.raises(ReferenceStructureError):
        SupercellMapping(reference, (2.0, 2, 2))


def test_atom_index_rejects_out_of_range():
    mapping = SupercellMapping(make_reference(), (2, 1, 1))
    with pytest.raises(ReferenceStructureError):
        mapping.atom_index((2, 0, 0), 0)
    with pytest.raises(ReferenceStructureError):
        mapping.atom_index((0, 0, 0), 2)
    with pytest.raises(ReferenceStructureError):
        mapping.cell_and_site(4)


def test_symmetry_operation_roundtrip_and_invalid():
    op = SymmetryOperation(rot=-np.eye(3), trans=np.array([0.0, 0.5, 0.5]))
    cell = SKEW_CELL
    cart = op.cart_matrix(cell)
    # reciprocal consistency: rotating twice returns the identity mapping
    np.testing.assert_allclose(cart @ cart, np.eye(3), atol=1e-12)
    with pytest.raises(ReferenceStructureError, match="singular"):
        SymmetryOperation(rot=np.zeros((3, 3)), trans=np.zeros(3))
    with pytest.raises(ReferenceStructureError, match="non-finite"):
        SymmetryOperation(rot=np.eye(3), trans=np.array([0.0, np.inf, 0.0]))


def test_reference_dict_roundtrip_records_everything():
    reference = make_reference(
        symmetry_operations=(SymmetryOperation(-np.eye(3), np.zeros(3)),))
    restored = ReferenceStructure.from_dict(reference.to_dict())
    np.testing.assert_allclose(restored.cell, reference.cell)
    np.testing.assert_allclose(restored.frac_positions, reference.frac_positions)
    assert restored.species == reference.species
    assert restored.symmetry_label == reference.symmetry_label
    assert restored.source == reference.source
    assert len(restored.symmetry_operations) == 1
    np.testing.assert_allclose(restored.symmetry_operations[0].rot,
                               reference.symmetry_operations[0].rot)
