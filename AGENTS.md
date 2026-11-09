# Repository Guidelines

## Project Overview

`longrangefy` is a Python project for **matched harmonic long-range corrections to existing interatomic potentials**, machine-learned or conventional. The `longrangefy` source package is under `src/`; implementation and scientific validation are in progress.

Before implementing or revising the method, read [the agreed method specification](docs/method-specification.md) and [domain glossary](CONTEXT.md). The specification records the confirmed scope and acceptance criteria; it takes precedence over the earlier handoff. The first implementation must obtain finite-difference IFCs and explicitly construct both harmonic branches before considering the direct-counterterm optimization.

The dipole–dipole implementation is Python/NumPy/SciPy owned by this package: no libabinit or pymultibinit runtime dependency, including optional adapters. DeepMD through ASE is the initial calculator workflow; an experimental MACE workflow exercises one inspected checkpoint. Keep both optional for array-core users and do not generalize their eligibility to arbitrary models.

The correction is the difference between two harmonic representations built from the **same reference-supercell IFC data**:
- **(a) Unsplit:** treat the total finite-cell IFCs as finite-range and extend them by lattice translations.
- **(b) Split:** subtract the analytic dipole contribution on the reference q-grid, extend the residual short-range IFCs with the same interpolation/translation convention, and restore target-cell Ewald dipole IFCs.

Add **(b) minus (a)** to the original potential, retaining its nonlinear behavior and fitted parameters. The two harmonic representations agree on the reference q-grid; the correction changes the extension to larger cells/new wavevectors. The initial construction obtains the existing model's reference IFCs by force finite differences. Born effective charges and electronic dielectric response are separate physical inputs, not inferable from a generic force calculator alone.

The earlier [`../pymultibinit/standalone_dipole_handoff.md`](../pymultibinit/standalone_dipole_handoff.md) provides kernel/unit/provenance background, but its emphasis on extracting `nonlinear_dipdip.py` and composing a full dipole term with a residual MLIP is **not the current correction design**. The user clarification and harmonic implementation take precedence.

Initial physical scope: fixed-reference insulating polar crystals, fixed Born/dielectric response, 3D PBC, matched electrical/Gamma boundary conventions. This does not establish support for diffusion, reactive systems, heterogeneous dielectric interfaces, or strained cells without verification. The user approved a frozen variable-cell design that preserves the base elastic/internal-strain response and a matching-preserving existing-tail design; neither is a support claim until implemented and validated.

### Companion Paper

A paper and literature research are explicit project goals. Establish prior art before claiming novelty. Keep the introduction, method, and validation claims aligned with the implementation.

- Overleaf Git: `https://git@git.overleaf.com/6aeef35284dcc0b2d494c612`.
- Canonical paper checkout: `/home/hexu/projects/mypapers/long_range_correction_atomic_potential`.
- Local access: [`papers/long_range_correction_atomic_potential`](papers/long_range_correction_atomic_potential), a machine-local symlink; maintain its Git history separately.
- Before method/novelty discussions, read the paper's `literature_notes.md`. Compare finite-grid harmonic matching against Monacelli--Marzari's Gaussian-smoothed correction, not against an unmodified full-Ewald addition. Matched IFC interpolation is established physics; novelty of a retrofit or application remains to be researched.

## Architecture & Data Flow

1. Reference structure, supercell/q-grid, and force finite differences → total harmonic IFCs with explicit atom/image ordering.
2. Fixed Born/dielectric inputs → analytic dipole matrices on the **same** reference grid.
3. Build the unsplit and split extensions using identical Fourier phases, Wigner--Seitz weights, translation mapping, and compatible acoustic sum rules.
4. Target cell plus displacements → harmonic difference → correction energy/forces; add it to the unchanged base calculator. Use ASE's composition/cache protocol rather than a custom framework.
5. Define and finite-strain-check the cell dependence before advertising stress. Fixed-reference displacement matching alone does not prove preservation of strained-cell data.

Keep the array core independent of ASE/DeepMD and DDB/XML parsing; keep all package runtime paths independent of `pymultibinit` and libabinit. `pymultibinit` may optionally consume this package downstream for consistency checks, never the reverse. **Copy and adapt existing Python code from pymultibinit and atomchain**, including their strain–phonon, elastic, internal-strain, finite-difference, and phonon helpers; do not recreate these implementations from Fortran or import the source packages at runtime. The user confirms authorship and authorizes reuse; retain provenance for traceability. Do not conflate this correction with `dipdip_model="geometry"`.

## Key Directories

Current layout:

| Path | Responsibility |
| --- | --- |
| `src/longrangefy/` | Reference harmonics, IFC extension, matched correction and Python Ewald arithmetic. |
| `src/longrangefy/cache.py` | Compatibility-checked matching-grid harmonic cache; model hash, source, response and numerical settings are part of its key. |
| `src/longrangefy/ase_correction.py` | Optional fixed-cell ASE correction composition. |
| `tests/` | Self-contained IFC matching, target-cell extension, derivative and optional ASE checks. |
| `docs/`, `examples/` | Correction contract and worked existing-potential example. |

`../pymultibinit/` and read-only `../Refs/abinit_pymb/` are implementation/source references, not runtime backends. Optional downstream checks do not establish parity until exercised with matched conventions.

## Development Commands

Use the shared environment; install the local source before running tests:

```bash
source /home/hexu/projects/myenvs/mydev/bin/activate
rtk uv pip install -e ".[ase]"
rtk python -m pytest tests/ -q
rtk uvx ruff check src/ tests/
rtk uvx ruff format --check src/ tests/
rtk uv build
rtk python -c "import longrangefy"
```

No CLI or test extra is defined. Install test tools in the shared environment. Exercise a correction evaluation, not only imports, for numerical changes.

## Code Conventions & Common Patterns

- Synchronous NumPy/SciPy arrays; `snake_case` functions/modules, `PascalCase` public classes, explicit reference/settings data. Keep immutable reference IFCs separate from per-target geometry; ASE owns calculator cache invalidation.
- Public units: Å, cell-total eV, eV/Å, eV/Å³; lattice vectors as rows. Full `(3,3)` stress in the core, ASE `[xx, yy, zz, yz, xz, xy]` at the adapter. Pin upstream Hartree/Bohr conversions, force/IFC axis order, Fourier phases, and mass weighting explicitly.
- Born tensors use `Z[i,alpha,beta] = dP_alpha/du_(i,beta)`. Validate finite shapes, cell nonsingularity, symmetric positive-definite dielectric, Born neutrality, supported PBC, and reference/site matching; preserve supplied tensors.
- Preserve reference-grid metadata and image weights. Use the same extension for both branches: mismatched ASR, boundary terms, or interpolation destroys cancellation. Test skew cells; cubic examples hide transposes.
- Ewald splitting is a numerical convergence parameter, not Monacelli's physical smoothing length. Do not substitute one for the other.
- Python version and tooling configuration are in `pyproject.toml`.

## Important Files

Upstream paths below are relative to `../pymultibinit/`:

- `src/pymultibinit/pyeffpot/supercell_builder.py`: `_build_supercell_ifcs_fourier` subtracts reference-grid dipole matrices, Fourier-interpolates/weights the residual, restores target-cell Ewald, and replicates IFCs. `build_supercell(..., dipdip=False|True)` provides the two harmonic construction routes.
- `src/pymultibinit/pyeffpot/dipdip.py`: `compute_dipdip_dynmat(s)` supplies analytic harmonic dipole matrices; distinct from finite-geometry energy arithmetic.
- `src/pymultibinit/pyeffpot/potential.py`: `EffectivePotential` assembles local plus Ewald harmonic matrices and evaluates quadratic energies/linear forces.
- `src/pymultibinit/pyeffpot/ifc_targets.py`: `generate_ifc_target(spec, get_forces, ...)` already generates dense force-finite-difference IFCs in phonopy order and eV/Å². Mapping them into the cell-major, R-point-indexed extension is separate glue; do not pass the dense array directly to `_replicate_ifcs`.
- `src/pymultibinit/pyeffpot/nonlinear_dipdip.py`: separate geometry-dependent option; not the definition of this correction.
- Reuse authorization: the user confirms authorship and permits copying/adapting their pymultibinit and atomchain code. Do not keep a separate licensing-clearance blocker for that code; retain source provenance and existing third-party notices.
- Existing Python strain implementation: `potential.py` has `_evaluate_elastic` (including bilinear internal strain), `_evaluate_strain_coupling` (higher-order phonon–strain), and strain/Voigt/stress-chain helpers.
- Atomchain reuse sources, relative to `../atomchain/`: `atomchain/ddb/finite_difference.py` supplies stress/internal-strain finite differences; `atomchain/phonon/frozenphonon.py` supplies finite-displacement phonon helpers. Copy the needed parts and adapt imports; do not depend on the whole source application.

`pyproject.toml` owns tooling/dependencies, and the package `__init__.py` owns public exports. Check current test and benchmark evidence before claiming a production workflow or numerical accuracy.

## Runtime/Tooling Preferences

Use `/home/hexu/projects/myenvs/mydev` via `uv`; no project-local `.venv`. Prefix shell tool commands with `rtk`. Keep ASE optional; do not inherit upstream CFFI/libabinit/JAX or publishing infrastructure by default. Use CPU execution for small investigations when the GPU is occupied. Store run evidence outside checkouts under `/STORAGE1/hexu/projects_bak/longrangefy/`.

The user-supplied SrTiO₃ DeepMD model is `STO_full_dpmd.pb` at the project root, a symlink to `/STORAGE1/hexu/projects_bak/longrangefy/models/STO_full_dpmd.pb` (SHA-256 `a50ab468bb871a32d7fd948cbc2d9320308d8d09a39eae22d0ff7b15ffe0db2e`). The user reports training on a 2×2×2 supercell of five-atom SrTiO₃, selected the local DDB reference and response, and declares no explicit electrostatic/Ewald tail in this model. Raw and charge-neutralized tensors and reference geometry are in `/STORAGE1/hexu/projects_bak/longrangefy/sto/response.json`. The DDB selection does not establish shared model-training provenance; do not claim it does.

## Testing & QA

Use pytest, deterministic fixtures, `numpy.testing.assert_allclose`, and justified convergence tolerances. No coverage-percentage gate exists yet.

- Reference-cell matrix, energy, and force differences must vanish to numerical tolerance; also check tiled reference-periodic patterns in larger cells and all original q-grid points.
- Larger-cell/new-wavevector corrections must reproduce the analytic long-range response minus its finite-grid interpolant. Verify symmetry, translations, atom permutations, acoustic modes, and exactly-once composition with the original potential.
- Check finite-difference IFC recovery and displacement-step convergence; match axes, units, masses, image convention, and sum rules.
- Check energy/force derivatives and Ewald convergence. Establish a cell-dependent contract and finite-strain stress before claiming NPT support or strained-training-data preservation.
- Verify optional ASE cache/property behavior and import independence. Physical accuracy needs independent data and observable-specific comparisons, not only interpolation identities.
- The q-grid builder and IFC-only fallback are different routes; audit each before claiming parity. Keep DDB/libabinit fixtures out of mandatory standalone tests.
- The first production workflow must exercise real DeepMD IFC generation and corrected energies/forces; synthetic fixtures alone do not prove calculator support. Both core and ASE/DeepMD use must work without pymultibinit/libabinit; downstream pymultibinit use is a separate optional integration.
