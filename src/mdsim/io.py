"""Readers and writers for .xyz coordinates and .itp force field files.

Unit conversions happen here: files use Angstrom for positions and bond
lengths and kJ/mol/A^2 / 100 for bond constants; the code uses nm.
"""
import numpy as np


# ---------------------- xyz read/write file functions ------------------------
def read_xyz(path):
    """Returns the element symbols and positions (nm) of the first frame."""
    with open(path) as file:
        nr_atoms = int(file.readline())
        file.readline()  # Skip comment line
        atoms = np.empty(nr_atoms, dtype=object)
        positions = np.empty((nr_atoms, 3), dtype='f')
        for k in range(nr_atoms):
            symbol, x, y, z = file.readline().split()[:4]
            atoms[k] = symbol
            positions[k] = float(x), float(y), float(z)
    # The code works in nm, not Angstrom. So scale units.
    return atoms.astype(str), 0.1 * positions


def write_xyz(file, positions, atoms, t):
    """Writes one frame (positions in nm) to an open .xyz file."""
    # VMD works with Angstrom, not nm. So scale the positions
    positions = 10 * positions
    lines = [f"{len(atoms)}\n", f"Some text: t={t}\n"]
    lines += [f"{atom:<2}{x:16.6f}{y:16.6f}{z:16.6f}\n"
              for atom, (x, y, z) in zip(atoms, positions)]
    file.write("".join(lines))


# --------------------- itp read/write file functions -------------------------
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
        atoms = _read_section(file, "atoms", 3)

    return dict(
        # Store bonds in sorted order
        bonds=np.sort(bonds[:, :2].astype(int), axis=1),
        kb=100 * bonds[:, 2],
        r0=0.1 * bonds[:, 3],
        angles=angles[:, :3].astype(int),
        ktheta=angles[:, 3],
        theta0=angles[:, 4],
        dihedrals=dihedrals[:, :4].astype(int),
        C=dihedrals[:, 4:],
        epsilon=atoms[:, 0],
        sigma=atoms[:, 1],
        masses=atoms[:, 2],
    )


def write_itp(path, topology):
    """Writes the force field of a topology to an ITP-like file."""
    t = topology
    with open(path, 'w') as f:
        f.write(f"bonds {len(t.bonds)}\n")
        for (i, j), kb, r0 in zip(t.bonds, t.kb, t.r0):
            f.write(f"{i} {j} {0.01 * kb} {10 * r0}\n")
        f.write(f"angles {len(t.angles)}\n")
        for (i, j, k), ktheta, theta0 in zip(t.angles, t.ktheta, t.theta0):
            f.write(f"{i} {j} {k} {ktheta} {theta0}\n")
        f.write(f"dihedrals {len(t.dihedrals)}\n")
        for (i, j, k, l), (c1, c2, c3, c4) in zip(t.dihedrals, t.C):
            f.write(f"{i} {j} {k} {l} {c1} {c2} {c3} {c4}\n")
        f.write(f"atoms {len(t.masses)}\n")
        for eps, sig, mass in zip(t.epsilon, t.sigma, t.masses):
            f.write(f"{eps} {sig} {mass}\n")
