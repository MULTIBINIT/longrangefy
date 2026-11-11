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

Reference harmonics default to `asr="none"`. Choosing `asr="simple"` or
`asr="weighted"` explicitly symmetrizes the sampled IFCs and enforces the
corresponding full Cartesian acoustic sum rule before either harmonic branch is
built. This changes the sampled IFCs; the input finite-difference reciprocity
residual remains available in the diagnostics. The weighted convention requires
reciprocal site/image weights with nonzero support for each site pair; it is not
interchangeable with the unweighted convention.

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

The ASE adapter also supports a fixed-cell workflow for the user-selected, inspected MACE checkpoint below. Install `.[mace]`, then use the same reference, response and eligibility JSON contract:

```bash
rtk python examples/mace_matched.py \
  /STORAGE1/hexu/projects_bak/longrangefy/mace-real-material/inputs/config.json \
  --target 2 2 2 --phi0-symmetry-tolerance 0.0012 \
  --max-displacement 0.2 --displace 0.02 --nreal 5 --nrecip 8
```

This example uses `/home/hexu/Downloads/2024-01-07-mace-128-L2_epoch-199.model` (SHA-256 `514bd2f1938653c37929f7c275a489378f03a7e3ffb27aa2ae0ddce8d0c859c8`), MACE 0.3.16 on CPU and a SrTiO₃ DFPT response; it is not a material-accuracy benchmark. The inspected `ScaleShiftMACE` checkpoint uses a 6 Å neighbor cutoff and its energy consists of atomic and finite-range interaction contributions. The user accepted this investigator assessment of its eligibility; it is not a provider declaration or a claim about arbitrary MACE models. Load only trusted checkpoints: the installed PyTorch environment permits pickle-based loading. Evidence and convergence checks are in `/STORAGE1/hexu/projects_bak/longrangefy/mace-real-material/`.

## Variable-cell experiment

The approved frozen correction preserves the original model's clamped-ion elastic and bilinear internal-strain response; it does **not** correct missing long-range elastic or piezoelectric physics. `VariableCellCorrectedCalculator` uses the affine ASE stress convention. For the owner-supplied SrTiO₃ DeepMD model, finite-difference checks at sampled normal and shear strains up to 2% gave maximum stress and force discrepancies of 8.75e-10 eV/Å³ and 4.71e-7 eV/Å. Bounded FIRE relaxations starting at 0.5–1% uniaxial strain and a separate 0.5% engineering-shear relaxation converged. These are numerical workflow checks at the recorded configurations, not a general material strain limit, accuracy claim, or NPT qualification. Evidence: `/STORAGE1/hexu/projects_bak/longrangefy/ticket09_variable_cell_20261111/` (uniaxial runs using wrapped calculator inputs) and `ticket09_unwrapped_20261111/` (current adapter).

```bash
rtk python examples/deepmd_variable.py \
  /STORAGE1/hexu/projects_bak/longrangefy/sto/workflow_config.json \
  --target 2 2 4 --phi0-symmetry-tolerance 0.0001 \
  --max-displacement 0.1 --max-strain 0.0205 --strain-xx 0.005 \
  --displace 0.02 --nreal 7 --nrecip 10
```

`--max-strain` is a caller-selected safety bound, not a validated material limit. In the tested CPU DeepMD 3.1.3 shear trajectory, `loc idx out of ... bound` messages arose from floating-point roundoff at cell boundaries: equivalent wrapped states and an independent ASE neighbor-list path agreed within 1.1e-13 eV/Å in forces. This diagnosis is specific to the recorded states and does not dismiss warnings in other models or geometries. See `/STORAGE1/hexu/projects_bak/longrangefy/ticket09_locidx_rootcause_20261111/`.


## Existing explicit tails

`longrangefy.existing_tail` implements the approved matching-preserving operator for explicitly declared DPLR response, including both first- and second-order Wannier-centroid response. Its declared existing tail reproduces the DeePMD-kit 3.1.3 **TensorFlow** `EwaldRecp` rectangular wavevector grid determined by `ewald_h`, including float32 op-attribute rounding and the runtime Coulomb constant. `CorrectedCalculator` adds the correction exactly once. Matching and target cells have independent per-axis runtime grids, so the builder compares the full existing-tail q-blocks at every shared wavevector and rejects—rather than retunes or projects—a pair whose target grid or declared Wannier response cannot preserve them. A published NaCl DPLR pair has been exercised against this runtime, but model selection, response precision and physical material accuracy remain qualification gates; do not generalize the runtime match to other DeePMD backends or versions. Do not apply the ordinary no-tail workflow to a model with an explicit asymptotic tail.


## Boundaries

The fixed-cell geometry contract covers a fixed reference, bounded displacements, 3D PBC and diagonal repetitions of an orthogonal or skew primitive cell. Synthetic skew checks cover matching, anisotropic response, rotation covariance and Ewald convergence; they do not establish material accuracy. The fixed-cell ASE wrapper fixes site assignment and rejects changed cells, PBC, species and out-of-bound displacements. Variable-cell stress is experimental under the frozen-response restrictions above; NPT, diffusion and models with existing explicit asymptotic electrostatics are not yet qualified. Calculator qualification covers the documented DeepMD model and, for fixed-cell energy and forces only, the inspected MACE checkpoint; it does not generalize to arbitrary models.

Public lengths are Å, whole-cell energies eV, forces eV/Å, and unmassweighted IFCs eV/Å². Born indices are electric-field direction then atomic-displacement direction. Gamma uses the conducting/tin-foil boundary; directional $q\to0$ is distinct. Reuse of upstream Python routines is documented at their source; the full scientific validation and variable-cell/tail designs remain separate gates.
