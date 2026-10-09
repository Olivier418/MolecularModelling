"""Opening a trajectory in VMD (https://www.ks.uiuc.edu/Research/vmd/)."""
import os
import shutil
import subprocess
import tempfile
from itertools import product

import numpy as np


def visualize_xyz_with_vmd(xyz_filepath, boxsize, d_max=0.15,
                           vmd_path="vmd"):
    """Launch VMD showing the trajectory with dynamic bonds up to d_max and
    the edges of the box (both in Angstrom)."""
    if not os.path.isfile(xyz_filepath):
        raise FileNotFoundError(f"The file {xyz_filepath} does not exist.")
    executable = shutil.which(vmd_path) or vmd_path
    if not os.path.isfile(executable):
        raise FileNotFoundError(f"VMD not found at {vmd_path!r}; set "
                                f"vmd_path in the [output] config section.")

    xyz_filepath = os.path.abspath(xyz_filepath).replace("\\", "/")
    tcl_script = f"""
    # Load the .xyz file
    mol new {{{xyz_filepath}}} type xyz

    # Add a representation for the molecule
    mol representation DynamicBonds {d_max}
    mol selection all
    mol material Opaque
    mol color Name
    mol modstyle 0 0 DynamicBonds

    # Draw a box around the simulation
    graphics top color red
    """
    # Draw the 12 edges of the box: corner pairs that differ in one coordinate
    corners = [np.array(c) for c in product([0, boxsize], repeat=3)]
    for p1, p2 in product(corners, repeat=2):
        if np.sum(p1 != p2) == 1 and tuple(p1) < tuple(p2):
            tcl_script += (f"graphics top line {{{p1[0]} {p1[1]} {p1[2]}}} "
                           f"{{{p2[0]} {p2[1]} {p2[2]}}}\n")

    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as f:
        f.write(tcl_script)
    try:
        subprocess.run([executable, "-e", f.name], check=True)
    finally:
        os.remove(f.name)
