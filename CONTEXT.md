# Matched Harmonic Long-Range Correction

A correction to an existing interatomic potential based on the difference between matched unsplit and split harmonic representations. The original potential may be machine-learned or conventional.

## Language

**Original potential**:
The existing interatomic potential before the correction is applied, including its harmonic and anharmonic response.
_Avoid_: Residual potential, short-range residual model

**Reference structure**:
The explicitly declared crystal structure about which the harmonic response is defined, including its reference lattice, sites, and symmetry. It need not be a stable minimum.

**Matching supercell**:
The explicitly chosen periodic supercell on which the split and unsplit harmonic representations are required to agree. It may coincide with a training supercell, but does not stand for the original potential's entire training history.
_Avoid_: Training supercell when no training provenance is established

**Matching grid**:
The set of wavevectors commensurate with the matching supercell, at which the two harmonic representations agree under the chosen conventions.

**Target supercell**:
The periodic supercell in which the corrected potential is evaluated; it need not be the matching supercell.

**Unsplit harmonic representation**:
The harmonic representation that treats the total finite-supercell force constants as finite-range interactions and extends them by lattice translations.
_Avoid_: Potential without electrostatics

**Split harmonic representation**:
The harmonic representation obtained by separating the analytic dipolar contribution from the same finite-supercell force constants and restoring its long-range response in the target cell.
_Avoid_: Independently fitted long-range potential

**Matched harmonic correction**:
The split harmonic representation minus the unsplit harmonic representation, added to the original potential. The representations agree on their matching grid under consistent conventions.
_Avoid_: Full dipole addition, geometry correction

**Corrected potential**:
The original potential plus the matched harmonic correction; the original fitted parameters and anharmonic terms are retained.
_Avoid_: Retrained potential

**Internal-strain coupling**:
The bilinear coupling between atomic displacement and homogeneous strain about the reference structure; it is the mixed block of the joint harmonic response.
_Avoid_: Strain-dependent IFCs, strain–lattice coupling without qualification

**Specified dipolar response**:
The long-wavelength dipolar response defined by the supplied Born effective charges, electronic dielectric tensor, and electrical boundary convention.

**Diagonal supercell transformation**:
A supercell formed by independently repeating the primitive lattice vectors an integer number of times. The primitive lattice vectors may be skew; diagonal repetition does not imply an orthogonal crystal.

**Frozen distortion**:
A prescribed atomic displacement pattern evaluated without relaxing it away, used to compare energies and forces of the original and corrected potentials.

**Dipolar long-range contribution**:
The analytic dipolar interaction selected by the specified electrical response and boundary convention. It is not defined merely as every interaction beyond a fixed distance.
_Avoid_: Everything missing from a short-range model

**Short-range residual**:
The harmonic response remaining after the selected dipolar contribution is subtracted on the matching grid. The name identifies a decomposition; it does not assert that every residual interaction has exactly zero range beyond a cutoff.

**Reference symmetry**:
The explicitly declared symmetry of the reference structure used to define the harmonic and electrical response. It may describe a stationary high-symmetry parent phase rather than the stable low-symmetry phase.
