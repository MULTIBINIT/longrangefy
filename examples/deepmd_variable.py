"""Evaluate a frozen matched correction under a declared uniaxial strain."""

import argparse
import json
from pathlib import Path

import numpy as np
from deepmd.calculator import DP

from longrangefy import ElectricalResponse, ReferenceStructure, ResponseProvenance, SupercellMapping
from longrangefy.ase import supercell_to_atoms
from longrangefy.ase_variable import VariableCellCorrectedCalculator
from longrangefy.cache import load_harmonics, save_harmonics
from longrangefy.deepmd import DeepMDForceProvider
from longrangefy.harmonics import build_reference_harmonics
from longrangefy.matched import build_correction
from longrangefy.variable_cell import build_variable_correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--target', type=int, nargs=3, required=True)
    parser.add_argument('--phi0-symmetry-tolerance', type=float, required=True)
    parser.add_argument('--max-displacement', type=float, required=True)
    parser.add_argument('--max-strain', type=float, required=True)
    parser.add_argument('--strain-xx', type=float, required=True)
    parser.add_argument('--displace', type=float, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--nreal', type=int, default=5)
    parser.add_argument('--nrecip', type=int, default=8)
    args = parser.parse_args()
    path = args.config.expanduser().resolve()
    config = json.loads(path.read_text())
    reference = ReferenceStructure.from_dict(json.loads((path.parent / config['reference_file']).read_text()))
    tensors = np.load(path.parent / config['response_file'])
    provenance = json.loads((path.parent / config['response_provenance_file']).read_text())
    response = ElectricalResponse(tensors['born_charges'], tensors['dielectric'], ResponseProvenance(**provenance))
    matching = SupercellMapping(reference, tuple(config['matching_reps']))
    provider = DeepMDForceProvider(config['model_path'], matching.supercell_cell, matching.species_supercell(),
                                   eligibility_declaration=config['eligibility_declaration'],
                                   declared_by=config['declared_by'], initial_positions=matching.cart_positions_supercell())
    identity = provider.identity.to_dict()
    if args.cache and args.cache.exists():
        harmonics = load_harmonics(args.cache, reference, response, matching.reps, identity,
                                   displacement=config['displacement'])
    else:
        harmonics = build_reference_harmonics(reference, response, matching.reps, provider.get_forces,
                                              displacement=config['displacement'], calculator_identity=identity,
                                              fresh_get_forces=provider.get_fresh_forces)
        if args.cache:
            save_harmonics(args.cache, harmonics)
    matched = build_correction(reference.cell, reference.cart_positions, response.born_charges,
                               response.dielectric, matching.reps, harmonics.q_matrices, harmonics.grid_h,
                               tuple(args.target), species=reference.species, nreal=args.nreal,
                               nrecip=args.nrecip, neutrality_tolerance=response.neutrality_tolerance,
                               phi0_symmetry_tol=args.phi0_symmetry_tolerance, response_provenance={'source': provenance})
    correction = build_variable_correction(matched, max_displacement=args.max_displacement,
                                           max_abs_strain=args.max_strain)
    atoms = supercell_to_atoms(SupercellMapping(reference, tuple(args.target)))
    atoms.calc = VariableCellCorrectedCalculator(DP(model=config['model_path']), correction, atoms,
                                                  max_displacement=args.max_displacement,
                                                  max_abs_strain=args.max_strain)
    deformation = np.eye(3)
    deformation[0, 0] += args.strain_xx
    atoms.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
    atoms.positions[0, 0] += args.displace
    print(json.dumps({'energy_eV': atoms.get_potential_energy(),
                      'forces_eV_per_A': atoms.get_forces().tolist(),
                      'stress_eV_per_A3': atoms.get_stress().tolist(),
                      'strain_xx': args.strain_xx}, indent=2))


if __name__ == '__main__':
    main()
