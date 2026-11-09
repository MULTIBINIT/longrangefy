# Matched Harmonic Long-Range Correction: Method Specification

Status: agreed design following the completed design interview. This specifies the method and acceptance criteria; it does not claim that a standalone correction implementation or physical benchmark already exists. Domain terminology is defined in [CONTEXT.md](../CONTEXT.md).

## 1. Purpose and staged scope

Correct the missing long-wavelength dipolar response of an **unchanged existing interatomic potential**, machine-learned or conventional, by adding the difference between two matched harmonic representations.

The first stage is fixed-cell. It explicitly obtains the original potential's harmonic IFCs by force finite differences and constructs both representations. A direct-counterterm simplification is permitted only as a subsequent, equivalence-checked optimization.

The guarantees are:

1. Preserve the response on an explicitly chosen matching grid, to the numerical accuracy of the matched construction.
2. Restore the specified long-wavelength dipolar contribution when it is absent from the original potential.

Accuracy throughout the Brillouin zone, nonlinear polarization, finite-temperature observables, and agreement with first-principles data are **validation outcomes, not algebraic guarantees**. Matching one supercell is not a claim about the original potential's entire training history.

Required later stages, not discarded scope:

- Variable-cell behavior: harmonic elasticity, bilinear atomic-displacement–strain coupling, and consistent stress.
- Original potentials that already contain explicit long-range electrostatic response: identify the existing contribution and define a replacement rather than blindly stacking terms.

### Implementation ownership and calculator rollout

The dipole–dipole/Ewald arithmetic is implemented here in Python/NumPy/SciPy. There is no libabinit dependency and no pymultibinit runtime dependency, including in optional adapters. DeepMD through ASE is the first supported original calculator; other calculators are later extensions. ASE/DeepMD dependencies are isolated from the array core. DeepMD models with existing explicit asymptotic electrostatics still fall outside the initial applicability contract.

Dependency direction is one-way: pymultibinit may optionally consume this package to investigate consistency with existing results. This package must never use pymultibinit or MULTIBINIT/libabinit as a backend. Copy and adapt the existing Python implementations in pymultibinit and atomchain—including relevant harmonic, elastic, internal-strain, strain–phonon, and finite-difference helpers—rather than recreating them from Fortran. The user confirms authorship and authorizes reuse of their code; retain provenance for traceability without treating separate clearance as a blocker. Downstream consistency is not yet demonstrated.

## 2. Physical applicability

Initially support fully 3D-periodic insulating crystals with fixed reference Born effective charges and a fixed electronic/high-frequency dielectric tensor. Use the conducting/tin-foil electrical boundary convention, consistently in the matching and target calculations.

The original potential must not already contain the explicit asymptotic electrostatic response being restored. Establish this from the model's declared physics; finite force samples cannot certify absence of such a tail. A finite but large interaction cutoff is not, by itself, an explicit asymptotic tail.

“Short-range” and “long-range” identify a decomposition. The residual after dipole subtraction need not have strictly finite support, and this correction is not a correction to every omitted interaction. Metallic screening, slabs, wires, molecules, dielectric interfaces, diffusion, changing site topology, and reactive chemistry are outside the initial physical contract.

This is not the separate `pymultibinit` current-minus-reference geometry correction, and it is not addition of a full dipole term to a residual-trained potential.

## 3. Reference, cells, and response inputs

### 3.1 Reference structure and symmetry

The user explicitly supplies the reference lattice, atomic sites, species, and intended symmetry. A high-symmetry reference is preferred when it represents the intended parent phase, but a stable minimum is not required. High symmetry prevents unintended reference symmetry breaking; it does not guarantee model accuracy or lack of bias.

Do not relax the reference automatically. If relaxation is requested, the initial workflow keeps the lattice fixed and preserves the declared symmetry. Unrestricted relaxation requires an explicit choice because it may change the reference phase. Record residual forces and reference preparation; a harmonic correction does not remove the original model's residual force at zero displacement.

Born/dielectric response must correspond to the declared reference. A nonstationary supplied reference is not silently moved; report that normal-mode stability interpretations then require care.

### 3.2 Matching and target supercells

Choose one explicit matching supercell. Initially, both matching and target supercells use positive diagonal integer repetitions of the same primitive lattice; that primitive lattice may be skew. General integer supercell transformations are a later geometric extension.

The matching grid is the full, unshifted set of wavevectors commensurate with the matching supercell. The target may be smaller, equal, or larger; preservation is asserted on shared matching-grid wavevectors, not on arbitrary new wavevectors. A tiled matching-cell displacement pattern is available when the target repetitions are compatible multiples.

Each target has the corresponding unstrained repeated reference lattice. Changing target size is supported; straining that lattice is not part of the fixed-cell contract.

### 3.3 Electrical response and provenance

Accept externally supplied material response or response supplied by the original model. Record its source, reference structure, conventions, and units. The imposed asymptotic response is the one specified by these tensors; do not infer it from a generic mechanical Hessian.

Require finite Born tensors `(n,3,3)` in units of elementary charge and a symmetric positive-definite electronic dielectric tensor `(3,3)`. The Born index order maps displacement component beta to dipole component alpha. Require charge neutrality within a documented tolerance; report incompatibility rather than silently altering tensors.

The boundary and Gamma convention are part of the response input. Omission of the macroscopic zero mode at Gamma is not the same as a direction-dependent limit approaching Gamma. Validate LO behavior through directional small-wavevector limits, not by assigning a directional LO term to every Gamma calculation.

## 4. Numerical and data conventions

- Public distances, displacements, and row-vector lattice cells: Å.
- Energies: eV per whole periodic target cell, not per atom or primitive cell.
- Forces: eV/Å; unmassweighted IFCs: eV/Å².
- The finite-difference supercell Hessian is dense `(3N_m,3N_m)` with atom-major Cartesian indices. Reciprocal primitive-cell IFC matrices are `(3n,3n)` at each matching wavevector.
- The target-cell real correction matrix is `(3N_L,3N_L)`, in the same site order as its displacement vector and returned forces.
- Phonon mass weighting belongs to spectral analysis, not to dipole subtraction. Convert frequency dynamical matrices back to the common unmassweighted convention before combining them.
- Record site/image ordering, Fourier phases and normalization, reciprocal-coordinate units, Wigner–Seitz image weights, acoustic-sum-rule handling, and symmetry treatment. A common convention is mandatory, not optional metadata.

Reject malformed/non-finite inputs, singular cells, incompatible site order or species, unsupported PBC/cell deformation, invalid dielectric/Born response, and ambiguous displacement mapping with actionable diagnostics. Do not silently switch cells, branches, electrical boundaries, or tensor conventions.

## 5. Explicit finite-difference construction

The original calculator supplies conservative energies and forces without modifying its parameters. In the matching supercell, displace each independent Cartesian coordinate about the reference and recover the total harmonic Hessian. For a central force stencil,

\[
 K_{ij}=-\frac{F_i(+h e_j)-F_i(-h e_j)}{2h}.
\]

The displaced configurations differ only in the selected coordinate. Establish displacement-step convergence against force noise and anharmonic contamination; the specification does not assign one universal step size.

Check reciprocity, translation symmetry, point-group consistency, and acoustic behavior. Any numerical symmetrization or sum-rule treatment must be explicit, recorded, and compatible between the two branches. Do not use inconsistent projections to manufacture cancellation or silently repair a symmetry-incompatible supplied response.

Map the dense supercell Hessian to translation-resolved IFCs and primitive-cell matrices on the matching grid. Preserve periodic-image degeneracies and normalized image weights. Dense phonopy order cannot be assumed identical to the cell-major/R-point representation used by the upstream builder.

This process characterizes the existing model. It neither refits that model nor asserts that its harmonic response is already physically accurate.

## 6. The two harmonic representations

Let `Q` be the matching grid, `Phi_0(q)` the original model's unmassweighted IFC matrices sampled there, and `D(q)` the analytic dipolar IFC matrices for the specified reference response and electrical boundary.

Let `I_Q` denote the **same linear, sample-reproducing interpolation/translation operation** in both branches. The notation includes the declared Fourier/image convention; compatible sum-rule handling must preserve the cancellation and input samples.

### 6.1 Representation (a): unsplit

Treat the total finite-cell harmonic response as finite-range for its extension:

\[
 \Phi_a(q)=\mathcal I_Q[\Phi_0|_Q](q).
\]

This representation already includes whatever electrical effects the original potential has encoded on the matching grid. “Unsplit” does not mean “without electrostatics.” Nor is its extension automatically identical to the original calculator's actual harmonic response at every unsampled wavevector; characterize that relationship rather than assuming it.

### 6.2 Representation (b): split

Subtract the analytic dipolar matrices on the matching grid, extend the residual with the same operation, and restore the analytic target response using Ewald summation:

\[
 \Phi_b(q)=\mathcal I_Q[(\Phi_0-D)|_Q](q)+D(q).
\]

Use the same reference response, normalization, phase, and boundary conventions for subtraction and restoration. Evaluate the target Ewald response for the target periodicity; do not merely translate the matching-cell Ewald matrix as though it were finite-range.

The Ewald splitting parameter is numerical: converged results should be invariant to it within the declared accuracy. It is not a physical smoothing parameter chosen to hide changes on the matching grid.

### 6.3 Difference and matching identity

Define

\[
 \Delta\Phi(q)=\Phi_b(q)-\Phi_a(q)
              =D(q)-\mathcal I_Q[D|_Q](q).
\]

The second equality follows from using the same linear extension. If that extension reproduces its input samples, the difference vanishes at each `q` in `Q`. This also preserves matching-periodic patterns embedded in compatible larger supercells, subject to the same conventions and numerical accuracy.

The first implementation must construct the two branches from the finite-difference data and verify their difference. The simplified expression is an independent equivalence check and a possible later optimization; it does not replace the agreed first construction.

## 7. Fixed-cell energy, forces, and site mapping

Assemble the real, symmetric target-cell matrix `DeltaPhi_L` from the two representations. It remains fixed for that reference and unstrained target lattice. Let `u` contain displacements assigned to its fixed reference sites. Then

\[
 E_{\rm corrected}=E_{\rm original}
                   +\tfrac12 u^{\mathsf T}\Delta\Phi_Lu,
 \qquad
 F_{\rm corrected}=F_{\rm original}-\Delta\Phi_Lu.
\]

Evaluate the original potential normally at the current atomic positions. Add the correction exactly once; retain the original reference energy, linear forces, and anharmonic terms. No residual retraining is required.

For the matching cell, a vanishing correction matrix leaves the original energy and displacement forces unchanged throughout the supported displacement branch—not because the original potential is harmonic, but because the added quadratic form is zero. Away from the matching grid the correction is harmonic; it does not update Born charges, dielectric response, or nonlinear polarization.

Site/species correspondence and a continuous displacement branch must be explicit. Initial use assumes bounded crystal displacements without site exchanges or diffusion. The core accepts assigned displacements; an adapter must document any periodic-image convention and reject unsupported/ambiguous mapping rather than silently rematching atoms.

The optional ASE boundary should use the existing calculator state/cache and composition mechanisms. Advertise only genuinely implemented properties. The fixed-cell stage provides correction energy and forces; it must not fabricate zero stress, advertise corrected stress, or enable variable-cell relaxation/NPT as though stress were already defined.

The first supported end-to-end workflow uses an actual compatible DeepMD calculator for reference forces, finite-difference IFCs, original energies/forces, and corrected evaluation. A generic array seam and synthetic tests do not establish support for other production calculators.

## 8. Acceptance and first scientific milestone

| Area | Required observable acceptance |
| --- | --- |
| Finite differences | Recover a known harmonic Hessian; demonstrate step convergence for an anharmonic conservative model; record force noise, ordering, and units. |
| Matching | Compare the full branch matrices on every matching-grid point, not just selected eigenvalues. Reference-cell correction energies and forces vanish to stated tolerance. |
| Embedded matching patterns | Tiled matching-periodic distortions in compatible larger cells retain vanishing correction. |
| New wavevectors | Show a nonzero correction where appropriate and agreement between explicit branch subtraction and the analytic response minus its interpolant. |
| Asymptotic response | Recover the specified directional long-wavelength dipolar behavior while handling Gamma with the declared boundary convention. |
| Derivatives and invariances | Energy finite differences reproduce correction forces; check symmetry, translations/acoustic modes, site permutations, and skew primitive cells. |
| Ewald accuracy | Tighten summation accuracy and vary the numerical splitting; results remain stable within reported tolerances. |
| Original-potential composition | Verify direct component sums and exactly-once addition; no original parameter changes or residual retraining. |
| Applicability/errors | Reject unsupported PBC, cell strain, invalid response, and incompatible mapping; document whether the original model contains an explicit tail. |
| Independence | Python dipole arithmetic/core evaluation work without ASE, DeepMD, pymultibinit, DDB parsing, or libabinit. The ASE/DeepMD path also works without pymultibinit/libabinit. |
| DeepMD integration | Exercise a real compatible model, record model/runtime identity, generate its finite-difference IFCs, and compare direct versus composed energies/forces with cache updates. |
| Optional downstream use | pymultibinit may consume this package in a separate integration environment; compare matched components before claiming consistency. No reverse runtime dependency is introduced. |

Use absolute tolerances with units and relative tolerances appropriate to each scale. Separate finite-difference noise, Ewald truncation, interpolation roundoff, and physical model error; do not choose tolerances merely to make a comparison pass.

The first scientific demonstration uses DeepMD and includes phonons and larger-cell frozen-distortion energy/force comparisons. Use both a controlled reference with known response and an independently justified physical comparison. Include the original potential and an existing correction where applicable. A phonon-plot-only adjustment does not prove a corrected energy/force calculator; synthetic matching does not prove material accuracy or DeepMD integration.

## 9. Required extensions and their boundaries

### 9.1 Variable-cell correction

Extend matching to the joint reference harmonic response: displacement IFCs, clamped-ion harmonic elasticity, and **bilinear internal-strain coupling** between atomic displacement and homogeneous strain. Relaxed-ion elasticity must be consistent with internal relaxation, not substituted silently for the clamped-ion block.

Define how both representations, reference sites, tensors, and counterterms depend on cell deformation, then derive and finite-strain-check the stress. Establish what is preserved on the matching cell under strain. Strain-dependent IFCs are a higher-order coupling and are not synonymous with the agreed bilinear internal-strain block.

This extension is required later, but fixed-cell displacement matching alone does not settle its construction or establish strain-data preservation.

The relevant numerical logic already exists in pymultibinit Python code. Copy and adapt its elastic, bilinear internal-strain, strain–phonon, and stress-convention helpers; reuse complementary atomchain finite-difference/phonon helpers where appropriate. Keep the adapted code independent of both source packages at runtime. Verify units, tensor order, coupling order, and derivative behavior rather than rederiving the implementation from Fortran. Reuse does not itself complete variable-cell matching or make every higher-order term an initial requirement.

### 9.2 Potentials with existing asymptotic response

Identify the specific electrostatic response already present and its boundary/partition convention. Define the desired response relative to that existing contribution before constructing a matched replacement. Do not classify eligibility by an arbitrary distance threshold or assume that matching-grid cancellation prevents off-grid double counting.

No automatic inference of a model's existing asymptotic response from an arbitrary force calculator is promised. The exact replacement contract and its validation require a subsequent design decision.

## 10. Prior art, implementation references, and evidence status

- [Monacelli–Marzari](https://doi.org/10.1103/7ygl-8db2), [reviewed full-text preprint](https://arxiv.org/html/2412.01642v1): closely related fixed-response correction and existing ASE software. The comparison is matched-grid counterterms versus Gaussian suppression, not a claim that no-retraining correction itself is new.
- [Yu et al.](https://doi.org/10.1103/74l8-yfn7), [ABINIT dipdip documentation](https://docs.abinit.org/variables/anaddb/#dipdip), and [Gonze–Lee](https://doi.org/10.1103/PhysRevB.55.10355): harmonic dipole subtraction/interpolation/readdition is established. Novelty of the generic retrofit and its applications remains to be researched.
- [Paper literature notes](../papers/long_range_correction_atomic_potential/literature_notes.md): source comparisons and limits of the targeted survey.
- Upstream `pymultibinit/src/pymultibinit/pyeffpot/supercell_builder.py`, `_build_supercell_ifcs_fourier`: reference-grid subtraction, weighted residual interpolation, target Ewald, and translation mapping.
- Upstream `pyeffpot/ifc_targets.py`, `generate_ifc_target`: force finite differences in dense phonopy order; conversion to the matched extension remains separate integration work.
- Upstream `pyeffpot/potential.py`, `EffectivePotential`: local/Ewald harmonic matrix assembly and quadratic evaluation. The geometry-dependent option is a different correction.
- `pymultibinit/src/pymultibinit/pyeffpot/potential.py`: `_evaluate_elastic` already includes bilinear internal strain; `_evaluate_strain_coupling` handles higher-order phonon–strain terms. Strain/Voigt and stress-chain helpers are also implemented in Python.
- `atomchain/atomchain/ddb/finite_difference.py`: `strain_atoms`, `calculate_stress_response`, and `calculate_internal_strain_response`; copy/adapt the numerical helpers without pulling in DDB writing.
- `atomchain/atomchain/phonon/frozenphonon.py`: `calculate_phonon` and supporting finite-displacement/force-collection logic; reuse only what the independent workflow needs.

Existing investigation evidence is outside both repositories under `/STORAGE1/hexu/projects_bak/longrangefy/`. `specification_checks.py` verifies symbolic cancellation/force signs and exercises the upstream synthetic builder, including a tiled matching-periodic pattern and force finite differences. These checks do not constitute the complete acceptance matrix or an end-to-end DeepMD correction. The user authorizes reuse of their pymultibinit/atomchain code; preserve source provenance and existing third-party notices. No new distribution-license decision is made here.
