# longrangefy

A fixed-cell matched harmonic correction for an existing interatomic potential. From force-finite-difference IFCs of the original potential on a declared reference supercell, the package constructs an unsplit extension and a split extension. The split extension subtracts the analytic dipole IFCs on the same q-grid, extends the residual using the same image convention, and restores the analytic dipolar response. The difference, **split minus unsplit**, is added to the original calculator's energy and forces without retraining.

See [the method specification](docs/method-specification.md) for conventions, applicability and acceptance criteria. This implementation is under scientific validation; an algebraic matching check is not evidence of material accuracy.

## Install

Use the shared Python environment and optional DeepMD/ASE dependencies:

```bash
source /home/hexu/projects/myenvs/mydev/bin/activate
rtk uv pip install -e '.[ase,deepmd]'
rtk python -m pytest tests -q
```

The NumPy/SciPy array core does not require ASE, DeepMD, pymultibinit or libabinit. Downstream pymultibinit integration is optional and has the opposite dependency direction.

## DeepMD example

Supply an eligible 3D-periodic insulating model, an explicit reference structure and matching repetitions, and separately sourced Born charges and electronic dielectric response. The example reads a JSON configuration whose `reference_file`, `response_file` and `response_provenance_file` are relative to the JSON file. The response NPZ contains `born_charges` and `dielectric`; the provenance JSON records the physical source, units and conducting Gamma boundary. The model path and a statement of its no-explicit-asymptotic-tail eligibility are required inputs. Force data alone cannot establish the electrical response or tail eligibility.

The owner-selected SrTiO₃ input set is at `/STORAGE1/hexu/projects_bak/longrangefy/sto/workflow_config.json`. It uses the owner's 2×2×2-trained DeepMD checkpoint and selected DDB structure and response; the owner declares that this checkpoint has no explicit electrostatic/Ewald tail. This does not establish shared model–DDB training provenance. From the project root:

```bash
rtk python examples/deepmd_matched.py \
  /STORAGE1/hexu/projects_bak/longrangefy/sto/workflow_config.json \
  --target 2 2 4 --phi0-symmetry-tolerance 0.0001 \
  --max-displacement 0.1 --displace 0.02 --nreal 7 --nrecip 10 \
  --cache /STORAGE1/hexu/projects_bak/longrangefy/sto/workflow_harmonics.npz
```

This runs original-model finite differences when the compatible cache is absent, constructs both branches, and evaluates corrected whole-cell energy and forces through ASE. The symmetry tolerance is a caller-selected numerical bound supported by the recorded SrTiO₃ finite-difference study, not a transferable acceptance bound. Check Ewald convergence for each system and geometry before drawing conclusions. The explicitly recorded Hermitian projection of accepted finite-difference samples is visible in the output.
Pass `--cache /path/outside/the/repo/harmonics.npz` to create reusable matching-grid data. A cache is reused only when the model SHA-256 and runtime identity, harmonic source, reference, electrical response, matching repetitions and finite-difference settings match; an incompatible cache is rejected rather than overwritten. Ewald settings apply when the correction is built and are not stored as harmonic data.


## MACE example

The ASE adapter also has an experimental MACE workflow. Install `.[mace]`, then use the same reference, response and eligibility JSON contract:

```bash
rtk python examples/mace_matched.py \
  /STORAGE1/hexu/projects_bak/longrangefy/mace-example/config.json \
  --target 2 2 2 --phi0-symmetry-tolerance 0.00001 \
  --max-displacement 0.1 --displace 0.01 --nreal 3 --nrecip 4
```

This local smoke uses `/home/hexu/Downloads/2024-01-07-mace-128-L2_epoch-199.model` (SHA-256 `514bd2f1938653c37929f7c275a489378f03a7e3ffb27aa2ae0ddce8d0c859c8`), MACE 0.3.16 on CPU and **synthetic** NaCl response tensors; it is not a material-accuracy benchmark. The inspected `ScaleShiftMACE` checkpoint uses a 6 Å neighbor cutoff and its energy consists of atomic and finite-range interaction contributions. This assessment applies only to the inspected checkpoint, not arbitrary MACE models. Load only trusted checkpoints: the installed PyTorch environment permits pickle-based loading.

## Variable-cell experiment

The approved frozen correction preserves the original model's clamped-ion elastic and bilinear internal-strain response; it does **not** correct missing long-range elastic or piezoelectric physics. `VariableCellCorrectedCalculator` uses the affine ASE stress convention. One BiFeO₃ DeepMD smoke at 0.5% uniaxial strain gave a stress/energy finite-difference discrepancy of 2.5e-10 eV/Å³; this is not a supported strain range or an accuracy or NPT claim.

```bash
rtk python examples/deepmd_variable.py \
  /STORAGE1/hexu/projects_bak/longrangefy/ticket01/real_config.json \
  --target 2 2 4 --phi0-symmetry-tolerance 0.02375 \
  --max-displacement 0.3 --max-strain 0.02 --strain-xx 0.005 \
  --displace 0.02 --nreal 3 --nrecip 4
```

`--max-strain` is a caller-selected safety bound, not a validated material limit. The example uses unconverged Ewald settings and an in-sample IFC tolerance. Finite-strain convergence, variable-cell relaxation and NPT remain unqualified.


## Existing explicit tails

`longrangefy.existing_tail` implements the approved matching-preserving operator for explicitly declared DPLR response, including both first- and second-order Wannier-centroid response. Synthetic energy/derivative and matching checks do not qualify a real model: a compatible DPLR energy/Wannier checkpoint, response extractor, runtime-tail verification and real-model benchmark remain outstanding. Do not apply the ordinary no-tail workflow to a model with an explicit asymptotic tail.


## Boundaries

The fixed-cell geometry contract covers a fixed reference, bounded displacements, 3D PBC and diagonal repetitions of an orthogonal or skew primitive cell. Synthetic skew checks cover matching, anisotropic response, rotation covariance and Ewald convergence; they do not establish material accuracy. The ASE wrapper fixes site assignment and rejects changed cells, PBC, species and out-of-bound displacements. Corrected stress/NPT, diffusion and models with existing explicit asymptotic electrostatics are not yet supported. Initial calculator qualification is DeepMD; MACE is selected for later qualification but is not yet claimed as supported.

Public lengths are Å, whole-cell energies eV, forces eV/Å, and unmassweighted IFCs eV/Å². Born indices are electric-field direction then atomic-displacement direction. Gamma uses the conducting/tin-foil boundary; directional $q\to0$ is distinct. Reuse of upstream Python routines is documented at their source; the full scientific validation and variable-cell/tail designs remain separate gates.
