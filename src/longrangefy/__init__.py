"""Matched harmonic long-range correction: reference harmonics core (ticket 01).

The array core (this import) depends only on NumPy.  Optional calculator
integration lives in :mod:`longrangefy.deepmd` and :mod:`longrangefy.ase` and
is imported explicitly by callers that installed the corresponding extras.

Public surface for the correction stages: :class:`ReferenceStructure`,
:class:`SupercellMapping`, :class:`ElectricalResponse`, the finite-difference
helpers, the IFC mapping/extension operators, and
:func:`build_reference_harmonics` returning ``Phi_0(q)`` on the matching grid
with recorded conventions and diagnostics.
"""

from .finitediff import (
    FiniteDifferenceResult,
    StepStudy,
    estimate_force_noise,
    finite_difference_hessian,
    hessian_diagnostics,
    step_convergence,
)
from .harmonics import HarmonicsError, ReferenceHarmonics, build_reference_harmonics
from .ifc import (
    IfcMappingError,
    SupercellIFC,
    apply_asr_simple,
    apply_asr_weighted,
    blocks_asr_row_sums,
    blocks_to_dense,
    blocks_to_qmatrices,
    dense_to_blocks,
    extend_blocks,
    fold_ws_weights,
    interpolation_rpoints,
    qmatrices_to_blocks,
    wigner_seitz_weights,
)
from .reference import (
    ReferenceStructure,
    ReferenceStructureError,
    SupercellMapping,
    SymmetryOperation,
)
from .relax import (
    ReferencePreparation,
    ReferencePreparationError,
    prepare_reference,
)
from .response import (
    ElectricalResponse,
    ResponseProvenance,
    ResponseValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # reference
    "ReferenceStructure",
    "ReferenceStructureError",
    "SupercellMapping",
    "SymmetryOperation",
    # response
    "ElectricalResponse",
    "ResponseProvenance",
    "ResponseValidationError",
    # finite differences
    "FiniteDifferenceResult",
    "StepStudy",
    "estimate_force_noise",
    "finite_difference_hessian",
    "hessian_diagnostics",
    "step_convergence",
    # IFC mapping
    "IfcMappingError",
    "SupercellIFC",
    "apply_asr_simple",
    "apply_asr_weighted",
    "blocks_asr_row_sums",
    "blocks_to_dense",
    "blocks_to_qmatrices",
    "dense_to_blocks",
    "extend_blocks",
    "fold_ws_weights",
    "interpolation_rpoints",
    "qmatrices_to_blocks",
    "wigner_seitz_weights",
    # harmonics workflow
    "HarmonicsError",
    "ReferenceHarmonics",
    "build_reference_harmonics",
    # reference preparation
    "ReferencePreparation",
    "ReferencePreparationError",
    "prepare_reference",
]
