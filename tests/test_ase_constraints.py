"""Constraints stay at the outer ASE boundary; the model surface stays raw.

The composed calculator and the finite-difference force provider must evaluate
the unconstrained original potential exactly at the requested geometry.  The
consumer's ordinary ``Atoms`` getters then apply each restraint once.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms, Hookean

from longrangefy.ase import AseForceProvider, AseIntegrationError
from longrangefy.ase_correction import CorrectedCalculator
from longrangefy.finitediff import finite_difference_hessian
from longrangefy.matched import build_correction


def _correction():
    cell = np.eye(3) * 4
    positions = np.array([[0., 0., 0.], [2., 2., 2.]])
    born = np.array([np.eye(3), -np.eye(3)])
    return build_correction(
        cell, positions, born, np.eye(3) * 3, (1, 1, 1),
        np.zeros((1, 2, 3, 2, 3)), np.array([[0, 0, 0]]), (2, 1, 1),
        species=['Ar', 'Ar'], eta=.2, nreal=3, nrecip=4,
    )


def _composed(correction, constraint=None):
    positions = np.array([[0., 0., 0.], [2., 2., 2.]])
    atoms = Atoms('Ar4', positions=np.vstack((positions, positions + [4, 0, 0])),
                  cell=np.diag([8, 4, 4]), pbc=True)
    atoms.calc = CorrectedCalculator(LennardJones(), correction, atoms, max_displacement=.4)
    if constraint is not None:
        atoms.set_constraint(constraint)
    return atoms


def test_hookean_restraint_is_applied_exactly_once():
    correction = _correction()
    free = _composed(correction)
    raw_energy = free.get_potential_energy()
    raw_forces = free.get_forces()
    distance = np.linalg.norm(free.positions[1] - free.positions[0])
    direction = (free.positions[1] - free.positions[0]) / distance

    restrained = _composed(correction, Hookean(0, 1, k=2.0, rt=distance - 0.5))
    energy = restrained.get_potential_energy()
    forces = restrained.get_forces()
    assert np.isclose(energy - raw_energy, 0.25, rtol=0, atol=1e-10)
    assert np.allclose(forces[0] - raw_forces[0], direction, rtol=0, atol=1e-10)
    assert np.allclose(forces[1] - raw_forces[1], -direction, rtol=0, atol=1e-10)
    assert np.allclose(forces[2:], raw_forces[2:], rtol=0, atol=1e-12)

    assert np.isclose(restrained.get_potential_energy(apply_constraint=False),
                      raw_energy, rtol=0, atol=1e-10)
    assert np.allclose(restrained.get_forces(apply_constraint=False),
                       raw_forces, rtol=0, atol=1e-12)


def test_fixatoms_keeps_raw_calculator_forces_behind_the_boundary():
    correction = _correction()
    reference = _composed(correction)
    displaced = reference.positions.copy()
    displaced[0, 0] += 0.1
    reference.set_positions(displaced)
    raw_forces = reference.get_forces()
    assert np.linalg.norm(raw_forces[0]) > 1e-3

    fixed = _composed(correction, FixAtoms(indices=[0]))
    fixed.set_positions(displaced, apply_constraint=False)
    assert np.all(fixed.get_forces()[0] == 0.0)
    assert np.allclose(fixed.get_forces(apply_constraint=False), raw_forces,
                       rtol=0, atol=1e-12)


def test_composed_calculator_still_rejects_changed_cell_under_restraint():
    atoms = _composed(_correction(), Hookean(0, 1, k=2.0, rt=np.sqrt(12) - 0.5))
    atoms.set_cell([8.1, 4, 4])
    with pytest.raises(ValueError, match='changed cell'):
        atoms.get_potential_energy()


def _pair(constraint=None, calculator=None):
    atoms = Atoms('Ar2', positions=[[0., 0., 0.], [1.2, 0., 0.]],
                  cell=np.diag([5., 5., 5.]), pbc=True)
    atoms.calc = LennardJones() if calculator is None else calculator
    if constraint is not None:
        atoms.set_constraint(constraint)
    return atoms


def test_provider_finite_differences_ignore_fixatoms_projection():
    reference = np.array([[0., 0., 0.], [1.2, 0., 0.]])
    free = AseForceProvider(_pair())
    fixed_atoms = _pair(FixAtoms(indices=[0, 1]))
    fixed = AseForceProvider(fixed_atoms)
    free_hessian = finite_difference_hessian(free.get_forces, reference).hessian
    fixed_hessian = finite_difference_hessian(fixed.get_forces, reference).hessian
    assert np.abs(free_hessian).max() > 1.0
    assert np.allclose(fixed_hessian, free_hessian, rtol=0, atol=1e-12)

    displaced = reference.copy()
    displaced[0, 0] += 0.1
    fixed.get_forces(displaced)
    assert np.array_equal(fixed_atoms.positions, displaced)
    assert np.all(fixed_atoms.get_forces() == 0.0)


def test_provider_energy_and_forces_exclude_restraint_potential():
    reference = np.array([[0., 0., 0.], [1.2, 0., 0.]])
    free = AseForceProvider(_pair())
    raw_energy = free.get_energy(reference)
    raw_forces = free.get_forces(reference)

    restrained_atoms = _pair()
    restrained = AseForceProvider(restrained_atoms)
    restrained_atoms.set_constraint(Hookean(0, 1, k=2.0, rt=0.7))
    assert np.isclose(restrained.get_energy(reference), raw_energy, rtol=0, atol=1e-12)
    assert np.allclose(restrained.get_forces(reference), raw_forces, rtol=0, atol=1e-12)
    assert np.isclose(restrained_atoms.get_potential_energy(), raw_energy + 0.25,
                      rtol=0, atol=1e-12)


def test_provider_validation_still_rejects_changed_cell_with_constraint():
    atoms = _pair(FixAtoms(indices=[0, 1]))
    provider = AseForceProvider(atoms)
    atoms.set_cell([6., 5., 5.])
    with pytest.raises(AseIntegrationError, match='changed'):
        provider.get_forces(atoms.positions)


class _CountingLennardJones(LennardJones):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self.calls += 1
        super().calculate(atoms, properties, system_changes)


def test_fresh_forces_bypass_the_ase_cache():
    atoms = _pair(calculator=_CountingLennardJones())
    provider = AseForceProvider(atoms)
    reference = atoms.positions.copy()
    cached = provider.get_forces(reference)
    calls = atoms.calc.calls
    assert np.array_equal(provider.get_forces(reference), cached)
    assert atoms.calc.calls == calls
    assert np.array_equal(provider.get_fresh_forces(reference), cached)
    assert atoms.calc.calls == calls + 1
