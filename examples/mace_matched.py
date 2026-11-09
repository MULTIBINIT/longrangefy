"""Evaluate a fixed-cell matched correction with a user-declared MACE checkpoint."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from longrangefy import ElectricalResponse, ReferenceStructure, ResponseProvenance, SupercellMapping
from longrangefy.ase import AseForceProvider, supercell_to_atoms
from longrangefy.ase_correction import CorrectedCalculator
from longrangefy.harmonics import build_reference_harmonics
from longrangefy.matched import build_correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--target', type=int, nargs=3, required=True)
    parser.add_argument('--phi0-symmetry-tolerance', type=float, required=True)
    parser.add_argument('--max-displacement', type=float, required=True)
    parser.add_argument('--displace', type=float, required=True)
    parser.add_argument('--nreal', type=int, default=5)
    parser.add_argument('--nrecip', type=int, default=8)
    args = parser.parse_args()
    path = args.config.expanduser().resolve()
    config = json.loads(path.read_text())
    if not all(isinstance(config.get(k), str) and config[k].strip()
               for k in ('eligibility_declaration', 'declared_by')):
        raise ValueError('explicit model eligibility and declarant are required')
    ref = ReferenceStructure.from_dict(json.loads((path.parent / config['reference_file']).read_text()))
    tensors = np.load(path.parent / config['response_file'])
    provenance = json.loads((path.parent / config['response_provenance_file']).read_text())
    response = ElectricalResponse(tensors['born_charges'], tensors['dielectric'], ResponseProvenance(**provenance))
    matching = SupercellMapping(ref, tuple(config['matching_reps']))
    model = Path(config['model_path']).expanduser().resolve()
    digest = hashlib.sha256()
    with model.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    digest = digest.hexdigest()
    identity = {'path': str(model), 'sha256': digest, 'declared_by': config['declared_by'],
                'eligibility_declaration': config['eligibility_declaration']}
    with redirect_stdout(sys.stderr):
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=str(model), device='cpu', default_dtype='float64')
    provider = AseForceProvider(supercell_to_atoms(matching), calc, identity)
    h = build_reference_harmonics(ref, response, matching.reps, provider.get_forces,
                                  displacement=config['displacement'], calculator_identity=identity,
                                  fresh_get_forces=provider.get_fresh_forces)
    correction = build_correction(ref.cell, ref.cart_positions, response.born_charges, response.dielectric,
                                  matching.reps, h.q_matrices, h.grid_h, tuple(args.target),
                                  species=ref.species, nreal=args.nreal, nrecip=args.nrecip,
                                  neutrality_tolerance=response.neutrality_tolerance,
                                  phi0_symmetry_tol=args.phi0_symmetry_tolerance,
                                  response_provenance={'source': provenance})
    atoms = supercell_to_atoms(SupercellMapping(ref, tuple(args.target)))
    atoms.calc = CorrectedCalculator(calc, correction, atoms, max_displacement=args.max_displacement)
    atoms.positions[0, 0] += args.displace
    print(json.dumps({'model': identity, 'energy_eV': atoms.get_potential_energy(),
                      'forces_eV_per_A': atoms.get_forces().tolist(),
                      'matching_residual_eV_per_A2': correction.diagnostics['shared_grid_max_branch_difference'],
                      'force_noise': h.diagnostics['force_noise']}, indent=2))


if __name__ == '__main__':
    main()
