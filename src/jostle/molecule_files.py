"""Reading and writing .xyz coordinates, and reading .itp force field files.

Unit conversions happen here: files use Angstrom for positions and bond
lengths, and kJ/mol/A^2 for bond constants (x 100 gives kJ/mol/nm^2); the
code uses nm.
"""
import numpy as np


# ---------------------- xyz read/write file functions ------------------------
def read_xyz(path):
    """Returns the element symbols and positions (nm) of the first frame."""
    with open(path) as file:
        nr_atoms = int(file.readline())
        file.readline()  # Skip comment line
        atoms = np.empty(nr_atoms, dtype=object)
        # float32, as always: changing it would change every initial box
        positions = np.empty((nr_atoms, 3), dtype='f')
        for k in range(nr_atoms):
            symbol, x, y, z = file.readline().split()[:4]
            atoms[k] = symbol
            positions[k] = float(x), float(y), float(z)
    # The code works in nm, not Angstrom. So scale units.
    return atoms.astype(str), 0.1 * positions


def write_xyz(file, atoms, positions, time):
    """Writes one frame (positions in nm, time in ps) to an open .xyz
    file."""
    # VMD works with Angstrom, not nm. So scale the positions
    positions = 10 * positions
    lines = [f"{len(atoms)}\n", f"t={time}\n"]
    lines += [f"{atom:<2}{x:16.6f}{y:16.6f}{z:16.6f}\n"
              for atom, (x, y, z) in zip(atoms, positions, strict=True)]
    file.write("".join(lines))


# ---------------------------- itp file reading -------------------------------
def _read_section(file, name, nr_columns):
    """Reads a '<name> N' header followed by N rows of numbers."""
    header = file.readline().split()
    if len(header) != 2 or header[0] != name:
        raise ValueError(f"Expected '{name} <count>' in {file.name}, "
                         f"got {' '.join(header)!r}.")
    rows = [file.readline().split() for _ in range(int(header[1]))]
    return np.array(rows, dtype=float).reshape(-1, nr_columns)


def read_itp(path):
    """Parse an ITP-like file into a dict of force field arrays (nm units)."""
    with open(path) as file:
        bonds = _read_section(file, "bonds", 4)
        angles = _read_section(file, "angles", 5)
        dihedrals = _read_section(file, "dihedrals", 8)
        atom_rows = _read_section(file, "atoms", 3)

    return dict(
        # Store bonds in sorted order
        bonds=np.sort(bonds[:, :2].astype(int), axis=1),
        k_bond=100 * bonds[:, 2],
        bond_r0=0.1 * bonds[:, 3],
        angles=angles[:, :3].astype(int),
        k_angle=angles[:, 3],
        theta0=angles[:, 4],
        dihedrals=dihedrals[:, :4].astype(int),
        rb_coefficients=dihedrals[:, 4:],
        lj_epsilon=atom_rows[:, 0],
        lj_sigma=atom_rows[:, 1],
        masses=atom_rows[:, 2],
    )

