"""Array-core independence: no ASE/DeepMD/pymultibinit/libabinit at runtime (R11).

Runs in a subprocess with an import blocker that makes the optional packages
unimportable, then exercises actual evaluation (finite differences plus the
matching-grid harmonics build), not only an import statement.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

BLOCKED = ("ase", "deepmd", "pymultibinit", "abipy", "libabinit")


def test_core_imports_and_evaluates_without_optional_packages():
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import json
        import sys

        BLOCKED = {BLOCKED!r}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".")[0]
                if root in BLOCKED:
                    raise ImportError(f"blocked optional package: {{fullname}}")
                return None

        sys.meta_path.insert(0, Blocker())

        import numpy as np
        sys.path.insert(0, {str(SRC)!r})
        import longrangefy

        for name in BLOCKED:
            assert name not in sys.modules, name

        # Deterministic two-site model with an exact harmonic Hessian on a
        # 2x1x1 matching supercell, built with the package's own block mapping.
        cell = np.diag([4.0, 4.0, 4.0])
        frac = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.5]])
        reference = longrangefy.ReferenceStructure(
            cell=cell, frac_positions=frac, species=("Na", "Cl"),
            symmetry_label="test", source="core independence check")
        response = longrangefy.ElectricalResponse(
            born_charges=np.array([np.eye(3) * 2.0, -np.eye(3) * 2.0]),
            dielectric=np.eye(3) * 9.0,
            provenance=longrangefy.ResponseProvenance(
                source="core independence check", method="analytic",
                reference_description="two-site cell"))
        matching = longrangefy.SupercellMapping(reference, (2, 1, 1))
        rpoints = np.array([[0, 0, 0], [1, 0, 0]])
        blocks = np.zeros((2, 2, 3, 2, 3))
        blocks[0, :, :, :, :] = 0.0
        blocks[1, 0, :, 1, :] = -0.4 * np.eye(3)
        blocks[1, 1, :, 0, :] = -0.4 * np.eye(3)
        diagonal = np.zeros((2, 2, 3, 2, 3))
        diagonal[0, 0, :, 0, :] = 1.0 * np.eye(3)
        diagonal[0, 1, :, 1, :] = 1.0 * np.eye(3)
        block_set = longrangefy.SupercellIFC(reps=(2, 1, 1), rpoints=rpoints,
                                             blocks=blocks + diagonal)
        hessian = longrangefy.blocks_to_dense(block_set, (2, 1, 1))
        positions = matching.cart_positions_supercell()

        def get_forces(points):
            u = (np.asarray(points) - positions).reshape(-1)
            return -(hessian @ u).reshape(points.shape)

        harmonics = longrangefy.build_reference_harmonics(
            reference, response, (2, 1, 1), get_forces, force_noise_repeats=2)
        report = {{
            "version": longrangefy.__version__,
            "loaded_blocked": sorted(m for m in sys.modules
                                     if m.split(".")[0] in BLOCKED),
            "nq": int(harmonics.q_matrices.shape[0]),
            "iq_identity": harmonics.diagnostics[
                "iq_on_grid_identity_max_abs_ev_per_angstrom2"],
            "hessian_max": float(np.abs(harmonics.dense_hessian).max()),
        }}
        print(json.dumps(report))
        """
    )
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["loaded_blocked"] == []
    assert report["nq"] == 2
    assert report["iq_identity"] < 1e-8
    assert report["hessian_max"] > 0.0


def test_no_optional_package_imports_in_core_sources():
    """Top-level imports of calculator/diagonalization packages are forbidden.

    Lazy in-function imports (spglib symmetry detection in ``relax``) are the
    documented exception: they never execute on core evaluation paths.
    """
    import re

    core = ("reference", "response", "finitediff", "ifc", "harmonics",
            "relax", "__init__")
    forbidden = BLOCKED + ("phonopy", "spglib")
    pattern = re.compile(
        r"^(?:from|import)\s+(?:" + "|".join(re.escape(p) for p in forbidden) + r")\b",
        re.MULTILINE)
    for module in core:
        source = (SRC / "longrangefy" / f"{module}.py").read_text()
        matches = pattern.findall(source)
        assert not matches, (module, matches)
