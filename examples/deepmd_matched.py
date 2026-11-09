"""Build a fixed-cell DeepMD matched correction from explicit external inputs.

The JSON configuration references a reference.json, a response.npz containing
born_charges/dielectric, and a response provenance JSON, relative to the config
file. Model and evidence files are user-provided and remain outside this repo.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from deepmd.calculator import DP

from longrangefy import ElectricalResponse, ReferenceStructure, ResponseProvenance, SupercellMapping
from longrangefy.ase import supercell_to_atoms
from longrangefy.ase_correction import CorrectedCalculator
from longrangefy.cache import load_harmonics, save_harmonics
from longrangefy.deepmd import DeepMDForceProvider
from longrangefy.harmonics import build_reference_harmonics
from longrangefy.matched import build_correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='JSON model, reference and response configuration')
    parser.add_argument('--target', nargs=3, type=int, required=True, metavar=('N1', 'N2', 'N3'))
    parser.add_argument('--phi0-symmetry-tolerance', type=float, required=True,
                        help='caller-selected IFC symmetry bound, eV/angstrom²')
    parser.add_argument('--max-displacement', type=float, required=True, help='mapping bound, angstrom')
    parser.add_argument('--displace', type=float, required=True, help='site-0 x displacement, angstrom')
    parser.add_argument('--cache', type=Path, help='reuse compatible harmonic data at this path')
    parser.add_argument('--nreal', type=int, default=5)
    parser.add_argument('--nrecip', type=int, default=8)
    args = parser.parse_args()
    path = args.config.expanduser().resolve()
    config = json.loads(path.read_text())
    data = path.parent
    reference = ReferenceStructure.from_dict(json.loads((data / config['reference_file']).read_text()))
    tensors = np.load(data / config['response_file'])
    provenance = json.loads((data / config['response_provenance_file']).read_text())
    response = ElectricalResponse(tensors['born_charges'], tensors['dielectric'],
                                  ResponseProvenance(**provenance))
    matching = SupercellMapping(reference, tuple(config['matching_reps']))
    provider = DeepMDForceProvider(
        config['model_path'], matching.supercell_cell, matching.species_supercell(),
        eligibility_declaration=config['eligibility_declaration'], declared_by=config['declared_by'],
        initial_positions=matching.cart_positions_supercell(),
    )
    identity = provider.identity.to_dict()
    if args.cache and args.cache.exists():
        harmonics = load_harmonics(args.cache, reference, response, matching.reps, identity,
                                   displacement=config['displacement'])
    else:
        harmonics = build_reference_harmonics(
            reference, response, matching.reps, provider.get_forces,
            displacement=config['displacement'], calculator_identity=identity,
            fresh_get_forces=provider.get_fresh_forces,
        )
        if args.cache:
            save_harmonics(args.cache, harmonics)
    correction = build_correction(
        reference.cell, reference.cart_positions, response.born_charges, response.dielectric,
        matching.reps, harmonics.q_matrices, harmonics.grid_h, tuple(args.target),
        species=reference.species, nreal=args.nreal, nrecip=args.nrecip,
        neutrality_tolerance=response.neutrality_tolerance,
        phi0_symmetry_tol=args.phi0_symmetry_tolerance,
        response_provenance={'source': provenance, 'phi0_symmetry_tolerance': args.phi0_symmetry_tolerance,
                             'tolerance_status': 'user-selected; verify convergence independently'},
    )
    target = SupercellMapping(reference, tuple(args.target))
    atoms = supercell_to_atoms(target)
    atoms.calc = CorrectedCalculator(DP(model=config['model_path']), correction, atoms,
                                     max_displacement=args.max_displacement)
    atoms.positions[0, 0] += args.displace
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(json.dumps({'energy_eV': energy, 'forces_eV_per_A': forces.tolist(),
                      'matching_residual_eV_per_A2': correction.diagnostics['shared_grid_max_branch_difference'],
                      'phi0_projection': correction.metadata['phi0_symmetrization'],
                      'force_noise': harmonics.diagnostics.get('force_noise'),
                      'ewald_shells': [args.nreal, args.nrecip]}, indent=2))


if __name__ == '__main__':
    main()
