"""Shared fixtures for the tests: random molecules and small real systems."""
import numpy as np

from auxiliary import compute_LJ_indices, find_bond_indices
from parse_functions import itp_to_data, xyz_to_array

MASSES = {"H": 1.00784, "C": 12.011, "O": 15.999}


# ----------------------------- Random molecules ------------------------------
def random_connected_graph(n, rng):
    """Random connected graph on n vertices as a set of sorted edges."""
    # A random spanning tree guarantees connectivity, extra edges add cycles
    edges = {tuple(sorted((k, int(rng.integers(k))))) for k in range(1, n)}
    nr_edges = int(rng.integers(n - 1, n * (n - 1) // 2))
    while len(edges) < nr_edges:
        a, b = rng.choice(n, 2, replace=False)
        edges.add((min(a, b), max(a, b)))
    return edges


def paths_of_length(edges, n, length):
    """All simple paths with `length` vertices, each undirected path once."""
    neighbours = [[] for _ in range(n)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)

    paths = []

    def extend(path):
        if len(path) == length:
            if path[0] < path[-1]:
                paths.append(path)
            return
        for nb in neighbours[path[-1]]:
            if nb not in path:
                extend(path + [nb])

    for start in range(n):
        extend([start])
    return np.array(paths, dtype=int).reshape(-1, length)


def random_molecule(n, rng):
    """A random nonsense molecule: random topology, random parameters and no
    LJ interactions. Does not respect valency."""
    edges = random_connected_graph(n, rng)
    bonds = paths_of_length(edges, n, 2)
    angles = paths_of_length(edges, n, 3)
    dihedrals = paths_of_length(edges, n, 4)

    atoms = rng.choice(list(MASSES), size=n)
    return dict(
        x=rng.random((n, 3)),
        atoms=atoms,
        masses=np.array([MASSES[a] for a in atoms]),
        bonds=bonds,
        kb=rng.random(len(bonds)),
        r0=rng.random(len(bonds)),
        angles=angles,
        ktheta=rng.random(len(angles)),
        theta0=rng.random(len(angles)),
        dihedrals=dihedrals,
        C1=np.zeros(len(dihedrals)),
        C2=np.zeros(len(dihedrals)),
        C3=np.zeros(len(dihedrals)),
        C4=np.zeros(len(dihedrals)),
    )


# ------------------------------ Real molecules -------------------------------
def load_molecule(name):
    """Positions and itp data of one molecule from molecule_data/."""
    _, x = xyz_to_array(f"molecule_data/{name}.xyz")
    (bonds, kb, r0, angles, ktheta, theta0, epsilon, sigma, dihedrals,
     C1, C2, C3, C4, masses) = itp_to_data(f"molecule_data/{name}.itp")
    return dict(x=x.astype(float), bonds=bonds, kb=kb, r0=r0, angles=angles,
                ktheta=ktheta, theta0=theta0, epsilon=epsilon, sigma=sigma,
                dihedrals=dihedrals, C1=C1, C2=C2, C3=C3, C4=C4,
                masses=masses)


def combine(molecules, shifts):
    """Concatenate molecules into one system, shifting each by a vector."""
    out = {}
    offset = 0
    for mol, shift in zip(molecules, shifts):
        n = len(mol["masses"])
        for key, val in mol.items():
            if key == "x":
                val = val + np.asarray(shift)
            elif key in ("bonds", "angles", "dihedrals"):
                val = val + offset
            out.setdefault(key, []).append(val)
        out.setdefault("molec_index", []).append(np.full(n, len(out["x"]) - 1))
        offset += n
    return {key: np.concatenate(val) for key, val in out.items()}


def water_ethanol_pair():
    """One water and one ethanol molecule, close enough to interact via LJ."""
    system = combine([load_molecule("Water"), load_molecule("Ethanol")],
                     [(0.0, 0.0, 0.0), (0.45, 0.1, -0.05)])
    n = len(system["masses"])
    i_LJ, j_LJ = compute_LJ_indices(n, system["dihedrals"], system["angles"],
                                    system["epsilon"], system["sigma"],
                                    system["molec_index"])
    system["i_LJ"], system["j_LJ"] = i_LJ, j_LJ
    system["sigma_LJ"] = 0.5 * (system["sigma"][i_LJ] + system["sigma"][j_LJ])
    system["epsilon_LJ"] = np.sqrt(system["epsilon"][i_LJ]
                                   * system["epsilon"][j_LJ])
    system["angle_mapping"] = find_bond_indices(system["angles"],
                                                system["bonds"])
    system["dihedral_mapping"] = find_bond_indices(system["dihedrals"],
                                                   system["bonds"])
    return system
