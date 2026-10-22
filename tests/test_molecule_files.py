import numpy as np
import pytest

from jostle.constants import BUILTIN_MOLECULE_DIR
from jostle.molecule_files import read_xyz, write_xyz
from jostle.system import load_molecule


def test_xyz_round_trip(tmp_path):
    atoms, x = read_xyz(BUILTIN_MOLECULE_DIR / "Ethanol.xyz")
    with open(tmp_path / "out.xyz", "w") as f:
        write_xyz(f, atoms, x, 0)
    atoms2, x2 = read_xyz(tmp_path / "out.xyz")
    assert list(atoms2) == list(atoms)
    np.testing.assert_allclose(x2, x, atol=1e-6)


@pytest.mark.parametrize("name", ["Water", "Ethanol", "Oxygen", "Hydrogen"])
def test_shipped_molecules_pass_consistency_check(name):
    load_molecule(name, BUILTIN_MOLECULE_DIR)
