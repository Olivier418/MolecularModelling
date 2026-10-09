"""Shared fixtures for the tests: small topologies and helpers."""
import numpy as np

from mdsim.constants import MOLECULE_DIR
from mdsim.forcefield import ForceField
from mdsim.system import Topology, load_molecule

MASSES = {"H": 1.00784, "C": 12.011, "O": 15.999}


def make_topology(n, **arrays):
    """A single-molecule topology of n atoms; missing data is empty/zero."""
    defaults = dict(
        atoms=np.full(n, "H"), masses=np.ones(n), epsilon=np.zeros(n),
        sigma=np.zeros(n), bonds=np.zeros((0, 2), int), kb=np.zeros(0),
        r0=np.zeros(0), angles=np.zeros((0, 3), int), ktheta=np.zeros(0),
        theta0=np.zeros(0), dihedrals=np.zeros((0, 4), int),
        C=np.zeros((0, 4)), molec_index=np.zeros(n, int),
        molec_names=np.full(n, "Test"))
    defaults.update(arrays)
    return Topology(**defaults)


def forcefield(topology, boxsize=0., r_cutoff_scalar=np.inf):
    """ForceField without periodic boundaries (boxsize=0) by default."""
    return ForceField(topology, boxsize, r_cutoff_scalar=r_cutoff_scalar)


def forces_and_energy(ff, x):
    """Total force and potential energy at positions x."""
    result = ff.compute(x, compute_energy=True)
    return result.forces, sum(result.energies.values())


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
    """Positions and topology of a random nonsense molecule: random topology,
    random parameters and no LJ interactions. Does not respect valency."""
    edges = random_connected_graph(n, rng)
    bonds = paths_of_length(edges, n, 2)
    angles = paths_of_length(edges, n, 3)
    dihedrals = paths_of_length(edges, n, 4)
    atoms = rng.choice(list(MASSES), size=n)
    topology = make_topology(
        n, atoms=atoms, masses=np.array([MASSES[a] for a in atoms]),
        bonds=bonds, kb=rng.random(len(bonds)), r0=rng.random(len(bonds)),
        angles=angles, ktheta=rng.random(len(angles)),
        theta0=rng.random(len(angles)), dihedrals=dihedrals,
        C=np.zeros((len(dihedrals), 4)))
    return rng.random((n, 3)), topology


# ------------------------------ Real molecules -------------------------------
def water_ethanol_pair():
    """One water and one ethanol molecule, close enough to interact via LJ."""
    x_w, water = load_molecule("Water", MOLECULE_DIR)
    x_e, ethanol = load_molecule("Ethanol", MOLECULE_DIR)
    x = np.vstack([x_w, x_e + np.array([0.45, 0.1, -0.05])]).astype(float)
    return x, Topology.concatenate([water, ethanol])
