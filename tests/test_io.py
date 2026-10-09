import numpy as np
import pytest

from mdsim.constants import MOLECULE_DIR
from mdsim.io import read_itp, read_xyz, write_itp, write_xyz
from mdsim.system import load_molecule


@pytest.mark.parametrize("name", ["Water", "Ethanol", "Oxygen", "Hydrogen"])
def test_itp_round_trip(tmp_path, name):
    _, topology = load_molecule(name, MOLECULE_DIR)
    write_itp(tmp_path / "out.itp", topology)
    original = read_itp(MOLECULE_DIR / f"{name}.itp")
    for key, value in read_itp(tmp_path / "out.itp").items():
        np.testing.assert_allclose(value, original[key], rtol=1e-12)


def test_xyz_round_trip(tmp_path):
    atoms, x = read_xyz(MOLECULE_DIR / "Ethanol.xyz")
    with open(tmp_path / "out.xyz", "w") as f:
        write_xyz(f, x, atoms, 0)
    atoms2, x2 = read_xyz(tmp_path / "out.xyz")
    assert list(atoms2) == list(atoms)
    np.testing.assert_allclose(x2, x, atol=1e-6)


def test_xyz_writer_keeps_small_values(tmp_path):
    x = np.array([[1.23456789e-6, -0.000123456789, 1.23456789]])
    with open(tmp_path / "out.xyz", "w") as f:
        write_xyz(f, x, np.array(["H"]), 0)
    _, x2 = read_xyz(tmp_path / "out.xyz")
    np.testing.assert_allclose(x2, x, atol=1e-6)


def test_xyz_writer_does_not_modify_input(tmp_path):
    x = np.ones((1, 3))
    with open(tmp_path / "out.xyz", "w") as f:
        write_xyz(f, x, np.array(["H"]), 0)
    np.testing.assert_array_equal(x, 1)


def test_itp_reports_malformed_header(tmp_path):
    (tmp_path / "bad.itp").write_text("bond 1\n0 1 1 1\n")
    with pytest.raises(ValueError, match="bonds"):
        read_itp(tmp_path / "bad.itp")
