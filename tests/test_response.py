"""Electrical response validation and provenance behaviour (R03)."""

import numpy as np
import pytest

from longrangefy import ElectricalResponse, ResponseProvenance, ResponseValidationError


def provenance(**overrides):
    data = dict(source="unit-test tensor fixture", method="analytic test values",
                reference_description="2-site skew test cell",
                extracted_by="tests/test_response.py")
    data.update(overrides)
    return ResponseProvenance(**data)


def born_ok():
    return np.array([
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        [[-2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -2.0]],
    ])


def test_valid_response_records_provenance_and_report():
    response = ElectricalResponse(born_charges=born_ok(),
                                  dielectric=9.0 * np.eye(3),
                                  provenance=provenance())
    report = response.validation_report()
    assert report["natom"] == 2
    assert report["max_neutrality_residual_e"] < 1e-12
    assert report["provenance"]["boundary_convention"].startswith("tin-foil")
    assert report["provenance"]["gamma_handling"]
    assert report["dielectric_eigenvalues"] == pytest.approx([9.0, 9.0, 9.0])
    assert "elementary charge" in report["born_units"]


def test_non_neutral_born_charges_rejected_with_measured_residual():
    born = born_ok()
    born[0, 0, 0] = 2.5
    with pytest.raises(ResponseValidationError, match="neutrality"):
        ElectricalResponse(born_charges=born, dielectric=9.0 * np.eye(3),
                           provenance=provenance())
    # A documented looser tolerance accepts the same tensors; the residual is reported.
    response = ElectricalResponse(born_charges=born, dielectric=9.0 * np.eye(3),
                                  provenance=provenance(), neutrality_tolerance=0.6)
    assert response.validation_report()["max_neutrality_residual_e"] == pytest.approx(0.5)


def test_non_finite_and_bad_shapes_rejected():
    born = born_ok()
    born[1, 2, 2] = np.nan
    with pytest.raises(ResponseValidationError, match="non-finite"):
        ElectricalResponse(born_charges=born, dielectric=np.eye(3) * 9,
                           provenance=provenance())
    with pytest.raises(ResponseValidationError, match="shape"):
        ElectricalResponse(born_charges=np.zeros((2, 3)), dielectric=np.eye(3) * 9,
                           provenance=provenance())
    with pytest.raises(ResponseValidationError, match="shape"):
        ElectricalResponse(born_charges=born_ok(), dielectric=np.eye(2) * 9,
                           provenance=provenance())
    with pytest.raises(ResponseValidationError, match="non-finite"):
        ElectricalResponse(born_charges=born_ok(),
                           dielectric=np.array([[9.0, 0, 0], [0, np.inf, 0], [0, 0, 9]]),
                           provenance=provenance())


def test_asymmetric_and_indefinite_dielectric_rejected():
    asymmetric = np.array([[9.0, 0.5, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
    with pytest.raises(ResponseValidationError, match="not symmetric"):
        ElectricalResponse(born_charges=born_ok(), dielectric=asymmetric,
                           provenance=provenance())
    indefinite = np.diag([9.0, 9.0, -1.0])
    with pytest.raises(ResponseValidationError, match="positive definite"):
        ElectricalResponse(born_charges=born_ok(), dielectric=indefinite,
                           provenance=provenance())


def test_provenance_must_be_stated():
    with pytest.raises(ResponseValidationError, match="source"):
        ResponseProvenance(source="", method="m", reference_description="r")
    with pytest.raises(ResponseValidationError, match="boundary"):
        ResponseProvenance(source="s", method="m", reference_description="r",
                           boundary_convention="  ")
    with pytest.raises(ResponseValidationError, match="boundary"):
        ResponseProvenance(source="s", method="m", reference_description="r",
                           boundary_convention="vacuum")
    with pytest.raises(ResponseValidationError, match="source"):
        ResponseProvenance(source=None, method="m", reference_description="r")


def test_natom_mismatch_is_explicit():
    response = ElectricalResponse(born_charges=born_ok(),
                                  dielectric=9.0 * np.eye(3),
                                  provenance=provenance())
    response.check_natom(2)
    with pytest.raises(ResponseValidationError, match="reference structure has 5"):
        response.check_natom(5)


def test_tensors_are_never_inferred_from_forces():
    # The public constructor requires the tensors themselves; there is no API
    # deriving them from a force callable, so this test documents the contract.
    import longrangefy
    assert not hasattr(longrangefy, "infer_born_charges")
    assert not hasattr(longrangefy.ElectricalResponse, "from_forces")
