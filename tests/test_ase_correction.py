"""Consumer-visible energy/force and fixed-cell behavior of ASE composition."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.lj import LennardJones

from longrangefy.ase_correction import CorrectedCalculator
from longrangefy.matched import build_correction


def make_corrected():
    cell = np.eye(3) * 4
    positions = np.array([[0., 0., 0.], [2., 2., 2.]])
    born = np.array([np.eye(3), -np.eye(3)])
    correction = build_correction(
        cell, positions, born, np.eye(3)*3, (1, 1, 1),
        np.zeros((1, 2, 3, 2, 3)), np.array([[0, 0, 0]]), (2, 1, 1),
        species=['Ar', 'Ar'], eta=.2, nreal=3, nrecip=4,
    )
    atoms = Atoms('Ar4', positions=np.vstack((positions, positions + [4, 0, 0])),
                  cell=np.diag([8, 4, 4]), pbc=True)
    atoms.calc = CorrectedCalculator(LennardJones(), correction, atoms, max_displacement=.4)
    return atoms


def test_composed_energy_force_derivative_and_position_cache():
    atoms = make_corrected()
    reference_energy = atoms.get_potential_energy()
    atoms.positions[0, 0] += .1
    energy = atoms.get_potential_energy()
    force = atoms.get_forces()[0, 0]
    assert abs(energy - reference_energy) > 1e-5
    step = 1e-5
    atoms.positions[0, 0] += step
    plus = atoms.get_potential_energy()
    atoms.positions[0, 0] -= 2*step
    minus = atoms.get_potential_energy()
    assert np.isclose(force, -(plus-minus)/(2*step), atol=1e-8, rtol=1e-6)
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()


def test_composed_calculator_rejects_cell_pbc_and_unbounded_displacements():
    atoms = make_corrected()
    atoms.set_cell([8.1, 4, 4])
    with pytest.raises(ValueError, match='changed cell'):
        atoms.get_potential_energy()
    atoms = make_corrected()
    atoms.pbc = [True, True, False]
    with pytest.raises(ValueError, match='periodicity'):
        atoms.get_forces()
    atoms = make_corrected()
    atoms.positions[0, 0] += .5
    with pytest.raises(ValueError, match='mapping bound'):
        atoms.get_forces()
