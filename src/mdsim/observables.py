"""Measurements recorded on save steps: phase space data and RDF histograms."""
import numpy as np

from .geometry import pair_vectors


def _parse_species_pair(entry, separator):
    """'O~H (Water~Ethanol)' -> sorted [('H', 'Ethanol'), ('O', 'Water')].
    A single molecule ('O-H (Water)') applies to both atoms."""
    atom_part, molecule_part = entry.split(' ', 1)
    atoms = atom_part.split(separator)
    molecules = molecule_part.strip('()').split(separator)
    if len(molecules) == 1:
        molecules = molecules * 2
    if len(atoms) != 2 or len(molecules) != 2:
        raise ValueError(f"Cannot parse {entry!r}; expected e.g. "
                         f"'O{separator}H (Water{separator}Ethanol)'.")
    return sorted(zip(atoms, molecules))


# ------------------------------ Phase space ----------------------------------
class PhaseSpace:
    """Tracks the length of every bond of the requested types."""

    def __init__(self, topology, bonds_of_interest, nr_saves):
        t = topology
        a, b = t.bonds[:, 0], t.bonds[:, 1]
        bond_atoms = np.sort(np.column_stack((t.atoms[a], t.atoms[b])), axis=1)
        bond_molecs = t.molec_names[a]

        self.names, self.masks = [], []
        for entry in dict.fromkeys(bonds_of_interest):
            (atom1, molecule), (atom2, _) = _parse_species_pair(entry, '-')
            mask = ((bond_atoms[:, 0] == atom1) & (bond_atoms[:, 1] == atom2)
                    & (bond_molecs == molecule))
            # Only keep bonds of interest that are present in the system.
            if mask.any():
                self.names.append(f"{atom1}~{atom2} ({molecule})")
                self.masks.append(mask)
        self.distances = [np.empty((nr_saves, m.sum())) for m in self.masks]

    def record(self, save_idx, result):
        for data, mask in zip(self.distances, self.masks):
            data[save_idx] = result.D_bonds[mask]


# ---------------------------------- RDF --------------------------------------
class RDF:
    """Histograms of the distances between pairs of interest, e.g.
    'O~H (Water~Ethanol)': an O atom of a Water molecule and an H atom of an
    Ethanol molecule. Intramolecular pairs are included."""

    def __init__(self, topology, pairs_of_interest, boxsize, thin, dr,
                 nr_saves, rng):
        t = topology
        self.boxsize = boxsize
        # Calculate the bins, up to the largest minimum-image distance
        domain = boxsize / 2
        nr_bins = int(domain / dr)
        self.bin_edges = np.linspace(0, domain, nr_bins + 1)
        self.bin_volumes = 4 / 3 * np.pi * (self.bin_edges[1:]**3
                                            - self.bin_edges[:-1]**3)

        self.names, self.pairs = [], []
        system_species = set(zip(t.atoms, t.molec_names))
        for entry in pairs_of_interest:
            species = _parse_species_pair(entry, '~')
            # Only keep pairs of interest that are present in the system.
            if not all(sp in system_species for sp in species):
                continue
            (atom1, mol1), (atom2, mol2) = species
            first = np.nonzero((t.atoms == atom1) & (t.molec_names == mol1))[0]
            second = np.nonzero((t.atoms == atom2) & (t.molec_names == mol2))[0]
            if (atom1, mol1) == (atom2, mol2):
                i, j = np.triu_indices(len(first), k=1)
                i, j = first[i], first[j]
            else:
                i = np.repeat(first, len(second))
                j = np.tile(second, len(first))
            # Select an RDF_thin fraction of all pairs to reduce time spent
            # on computing distances.
            keep = rng.choice(len(i), int(np.ceil(len(i) * thin)),
                              replace=False)
            self.names.append(entry)
            self.pairs.append((i[keep], j[keep]))

        self.histograms = np.empty((nr_saves, len(self.names), nr_bins))

    def record(self, save_idx, x):
        for k, (i, j) in enumerate(self.pairs):
            _, D = pair_vectors(x, i, j, self.boxsize)
            self.histograms[save_idx, k] = np.histogram(D, self.bin_edges)[0]
