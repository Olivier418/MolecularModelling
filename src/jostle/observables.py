"""Measurements recorded during a run: phase space data and RDF histograms.
Each has its own times (ps): the frames it was recorded on."""
import numpy as np
from numba import njit

# At most this many bonds of each type are tracked in the phase space, evenly
# spaced through the bond list (no random draws, so the trajectory does not
# depend on it). More only overlap in the diagram, and the data would grow
# with the box (the default box has about 6,300 tracked bonds).
PHASE_SPACE_MAX_BONDS = 20


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
    return sorted(zip(atoms, molecules, strict=True))


def _unique_species_pairs(entries, separator):
    """(entry, parsed species) of each entry, leaving out entries that name
    the same species as an earlier one (e.g. 'H-O (Water)' after 'O-H
    (Water)')."""
    unique = {}
    for entry in entries:
        species = _parse_species_pair(entry, separator)
        unique.setdefault(tuple(species), entry)
    return [(entry, list(species)) for species, entry in unique.items()]


# ------------------------------ Phase space ----------------------------------
class PhaseSpace:
    """Tracks the length of the bonds of the requested types (at most
    PHASE_SPACE_MAX_BONDS of each), and its rate of change
    dD/dt = (v_i - v_j) . (x_i - x_j) / D, at the given times (ps).
    bond_types holds the requested types present in the system, as given
    (e.g. 'O-H (Water)'); lengths and length_rates hold one array of shape
    (frames, bonds) per bond type."""

    def __init__(self, topology, requested_bond_types, times):
        top = topology
        a, b = top.bonds[:, 0], top.bonds[:, 1]
        bond_atoms = np.sort(np.column_stack((top.elements[a],
                                              top.elements[b])), axis=1)
        bond_molecules = top.molecule_names[a]

        self.all_bonds = top.bonds
        self.times = times
        self.bond_types, self.bond_indices = [], []
        for entry, species in _unique_species_pairs(requested_bond_types,
                                                    '-'):
            (atom1, molecule), (atom2, _) = species
            mask = ((bond_atoms[:, 0] == atom1) & (bond_atoms[:, 1] == atom2)
                    & (bond_molecules == molecule))
            # Only keep bond types that are present in the system.
            if mask.any():
                index = np.nonzero(mask)[0]
                if len(index) > PHASE_SPACE_MAX_BONDS:
                    index = index[np.linspace(0, len(index) - 1,
                                              PHASE_SPACE_MAX_BONDS,
                                              dtype=int)]
                self.bond_types.append(entry)
                self.bond_indices.append(index)
        shapes = [(len(times), len(index)) for index in self.bond_indices]
        self.lengths = [np.empty(shape) for shape in shapes]
        self.length_rates = [np.empty(shape) for shape in shapes]

    @classmethod
    def from_arrays(cls, bond_types, times, lengths, length_rates):
        """Phase space data loaded from a file: it can be plotted, but no
        longer records."""
        phase_space = cls.__new__(cls)
        phase_space.all_bonds, phase_space.bond_indices = None, None
        phase_space.bond_types = list(bond_types)
        phase_space.times = times
        phase_space.lengths = list(lengths)
        phase_space.length_rates = list(length_rates)
        return phase_space

    def record(self, frame, evaluation, velocities):
        for k, index in enumerate(self.bond_indices):
            a, b = self.all_bonds[index, 0], self.all_bonds[index, 1]
            D = evaluation.bond_lengths[index]
            self.lengths[k][frame] = D
            self.length_rates[k][frame] = np.sum(
                (velocities[a] - velocities[b])
                * evaluation.bond_vectors[index], axis=1) / D


# ---------------------------------- RDF --------------------------------------
class RDF:
    """Histograms of the distances between pairs of interest, e.g.
    'O~H (Water~Ethanol)': an O atom of a Water molecule and an H atom of an
    Ethanol molecule. Only intermolecular pairs are counted. pair_labels
    holds the requested pairs present in the system, as given; nr_pairs holds
    the number of sampled pairs of each, which normalises the histograms to
    g(r) (see g_of_r). times (ps) are those of the frames."""

    def __init__(self, topology, requested_pairs, box_length, pair_fraction,
                 bin_width, times, rng):
        top = topology
        self.box_length = box_length
        self.times = times
        # Calculate the bins, up to the largest minimum-image distance
        r_max = box_length / 2
        nr_bins = int(r_max / bin_width)
        self.bin_edges = np.linspace(0, r_max, nr_bins + 1)

        self.pair_labels, self.atom_pairs = [], []
        system_species = set(zip(top.elements, top.molecule_names,
                                 strict=True))
        for entry, species in _unique_species_pairs(requested_pairs, '~'):
            # Only keep pairs that are present in the system.
            if not all(sp in system_species for sp in species):
                continue
            (atom1, mol1), (atom2, mol2) = species
            atoms_a = np.nonzero((top.elements == atom1)
                                 & (top.molecule_names == mol1))[0]
            atoms_b = np.nonzero((top.elements == atom2)
                                 & (top.molecule_names == mol2))[0]
            if (atom1, mol1) == (atom2, mol2):
                i, j = np.triu_indices(len(atoms_a), k=1)
                i, j = atoms_a[i], atoms_a[j]
            else:
                i = np.repeat(atoms_a, len(atoms_b))
                j = np.tile(atoms_b, len(atoms_a))
            # Intermolecular pairs only: within a molecule, the distances are
            # set by the bonds and angles, not by the liquid structure.
            is_intermolecular = top.molecule_index[i] != top.molecule_index[j]
            i, j = i[is_intermolecular], j[is_intermolecular]
            # Select an rdf_pair_fraction of all pairs to reduce time spent
            # on computing distances.
            keep = rng.choice(len(i), int(np.ceil(len(i) * pair_fraction)),
                              replace=False)
            self.pair_labels.append(entry)
            self.atom_pairs.append((i[keep].astype(np.int32),
                                    j[keep].astype(np.int32)))

        self.nr_pairs = np.array([len(i) for i, _ in self.atom_pairs])
        self.histograms = np.zeros((len(times), len(self.pair_labels),
                                    nr_bins))

    @classmethod
    def from_arrays(cls, pair_labels, times, bin_edges, histograms, nr_pairs):
        """RDF data loaded from a file: it can be plotted, but no longer
        records."""
        rdf = cls.__new__(cls)
        rdf.times = times
        rdf.box_length = 2 * bin_edges[-1]
        rdf.bin_edges = bin_edges
        rdf.pair_labels, rdf.atom_pairs = list(pair_labels), None
        rdf.histograms, rdf.nr_pairs = histograms, nr_pairs
        return rdf

    def g_of_r(self):
        """The histograms as g(r), shape (frames, pairs, bins): the counts
        divided by those of the same number of pairs spread uniformly over
        the box, so g -> 1 at large r. An entry without pairs is NaN."""
        edges = self.bin_edges
        shell_volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
        ideal = (self.nr_pairs[None, :, None] * shell_volumes[None, None]
                 / self.box_length**3)
        with np.errstate(invalid='ignore', divide='ignore'):
            return self.histograms / ideal

    def record(self, frame, positions):
        for k, (i, j) in enumerate(self.atom_pairs):
            accumulate_pair_histogram(positions, i, j, self.box_length,
                                      self.bin_edges, self.histograms[frame, k])


@njit(cache=True)
def accumulate_pair_histogram(positions, i, j, box_length, bin_edges, counts):
    """Adds the minimum-image distances of the pairs (i, j) to the histogram
    counts. Same binning as np.histogram on uniform bin_edges (the last bin
    includes its right edge), without storing the distances."""
    inv_box_length = 1. / box_length
    nr_bins = len(bin_edges) - 1
    first, last = bin_edges[0], bin_edges[-1]
    for k in range(len(i)):
        dx = positions[i[k], 0] - positions[j[k], 0]
        dy = positions[i[k], 1] - positions[j[k], 1]
        dz = positions[i[k], 2] - positions[j[k], 2]
        # Minimum image (floor is much faster than % in compiled code)
        dx -= box_length * np.floor(dx * inv_box_length + 0.5)
        dy -= box_length * np.floor(dy * inv_box_length + 0.5)
        dz -= box_length * np.floor(dz * inv_box_length + 0.5)
        D = np.sqrt(dx * dx + dy * dy + dz * dz)
        if D < first or D > last:
            continue
        b = min(int((D - first) / (last - first) * nr_bins), nr_bins - 1)
        # Correct for rounding, as np.histogram does
        if D < bin_edges[b]:
            b -= 1
        elif b != nr_bins - 1 and D >= bin_edges[b + 1]:
            b += 1
        counts[b] += 1
