import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from longrangefy.ase import AseForceProvider, AseIntegrationError


def test_energy_uses_requested_positions_and_invalidates_cache():
    atoms = Atoms('Ar2', positions=[[0, 0, 0], [1.2, 0, 0]])
    atoms.calc = LennardJones()
    provider = AseForceProvider(atoms)
    initial = provider.get_energy(atoms.positions.copy())
    moved = atoms.positions.copy()
    moved[1, 0] = 1.4
    actual = provider.get_energy(moved)
    atoms.set_positions(moved)
    assert np.isclose(actual, atoms.get_potential_energy())
    assert not np.isclose(actual, initial)


@pytest.mark.parametrize('change', ['cell', 'pbc', 'species'])
def test_original_atoms_mutations_are_rejected(change):
    atoms = Atoms('Ar2', positions=[[0, 0, 0], [1.2, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.calc = LennardJones()
    provider = AseForceProvider(atoms)
    if change == 'cell':
        atoms.set_cell([6, 5, 5])
    elif change == 'pbc':
        atoms.pbc = [True, True, False]
    else:
        atoms.numbers[0] = 2
    with pytest.raises(AseIntegrationError, match='changed'):
        provider.get_energy(atoms.positions)
