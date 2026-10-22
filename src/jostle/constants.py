"""Physical constants and molecule data.

Internal units: nm, ps, amu, kJ/mol and K.
"""
from pathlib import Path

BOLTZMANN_CONSTANT = 0.0083144626   # Boltzmann constant (kJ / (mol K))
AVOGADRO = 6.022e23     # Avogadro's constant (1 / mol)

# Molar volumes (cm^3 / mol), used to fill the box at the right density.
# Water and ethanol at 298.15K; oxygen and hydrogen as liquids at their
# normal boiling points (90 K, 1.141 g/cm^3 and 20 K, 0.0708 g/cm^3), so the
# gases are packed at liquid density. Every molecule in the data directory
# needs an entry here.
MOLAR_VOLUMES = {
    "Water": 18.07,
    "Ethanol": 58.4,
    "Oxygen": 28.0,
    "Hydrogen": 28.5,
}

# Directory with the .xyz and .itp files of the built-in molecules
BUILTIN_MOLECULE_DIR = Path(__file__).parent / "data" / "molecules"

# Name of the trajectory file in a run folder
TRAJECTORY_FILE = "trajectory.xyz"
