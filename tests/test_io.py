import numpy as np
import pytest

from parse_functions import array_to_xyz, data_to_itp, itp_to_data, xyz_to_array


@pytest.mark.parametrize("name", ["Water", "Ethanol", "Oxygen", "Hydrogen"])
def test_itp_round_trip(tmp_path, name):
    data = itp_to_data(f"molecule_data/{name}.itp")
    data_to_itp(tmp_path / "out.itp", *data)
    for original, reread in zip(data, itp_to_data(tmp_path / "out.itp")):
        np.testing.assert_allclose(reread, original, rtol=1e-12)


def test_xyz_round_trip(tmp_path):
    atoms, x = xyz_to_array("molecule_data/Ethanol.xyz")
    array_to_xyz(x.copy(), atoms, tmp_path / "out.xyz", 0, "write")
    atoms2, x2 = xyz_to_array(tmp_path / "out.xyz")
    assert list(atoms2) == list(atoms)
    np.testing.assert_allclose(x2, x, atol=1e-6)


@pytest.mark.xfail(strict=True, reason="writer truncates str(float) to 10 "
                                       "characters")
def test_xyz_writer_keeps_small_values(tmp_path):
    x = np.array([[1.23456789e-6, -0.000123456789, 1.23456789]])
    array_to_xyz(x.copy(), np.array(["H"]), tmp_path / "out.xyz", 0, "write")
    _, x2 = xyz_to_array(tmp_path / "out.xyz")
    np.testing.assert_allclose(x2, x, atol=1e-6)
