"""Physical constants and molecule data.

Internal units: nm, ps, amu, kJ/mol and K.
"""
from pathlib import Path

KBOLTZ = 0.0083144626   # Boltzmann constant (kJ / (mol K))
AVOGADRO = 6.022e23     # Avogadro's constant (1 / mol)

# Molar volumes (cm^3 / mol) at 298.15K, used to fill the box at the right
# density. Every molecule in the data directory needs an entry here.
MOLAR_VOLUMES = {
    "Water": 18.07,
    "Ethanol": 58.4,
    "Oxygen": 22.430,
    "Hydrogen": 22.390,
}

# Directory with the .xyz and .itp files of the built-in molecules
MOLECULE_DIR = Path(__file__).parent / "data" / "molecules"
