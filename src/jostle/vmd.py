"""Opening a trajectory in VMD (https://www.ks.uiuc.edu/Research/vmd/)."""
import os
import shutil
import subprocess
import sys
from glob import glob
from itertools import product
from pathlib import Path

import numpy as np

# Environment variable with the VMD executable, so every notebook on a machine
# finds it without passing vmd_path to view_in_vmd
VMD_ENV_VAR = "JOSTLE_VMD"
# Depth cueing (fog towards the background colour): exp(-(density * d)^2) of
# an atom's colour remains at a distance d from the camera (the scene, with
# its periodic images, spans about 1 to 3)
CUE_MODE = "Exp2"
CUE_DENSITY = 0.4


def _standard_locations():
    """Executables in the folders the VMD installers use; on Windows,
    newest-looking first (e.g. VMD2 before VMD 1.9.4)."""
    if sys.platform == "win32":
        roots = [os.environ.get(name) for name in
                 ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
        folders = [("University of Illinois", "VMD*"), ("VMD*",),
                   ("Programs", "VMD*")]
        patterns = [os.path.join(root, *parts, "vmd.exe")
                    for root in filter(None, roots) for parts in folders]
    elif sys.platform == "darwin":
        patterns = ["/Applications/VMD*.app/Contents/Resources/VMD.app/"
                    "Contents/MacOS/VMD"]
    else:
        patterns = ["/usr/local/bin/vmd", "/opt/vmd*/bin/vmd"]
    found = [path for pattern in patterns for path in glob(pattern)]
    return sorted(found, key=lambda p: os.path.basename(os.path.dirname(p)),
                  reverse=True)


def find_vmd(vmd_path=None):
    """Path of the VMD executable, or None if it cannot be found.

    Looks, in this order, at: vmd_path (a file, or a command on the PATH;
    when given, nothing else is tried), the JOSTLE_VMD environment variable,
    a `vmd` command on the PATH, and the standard install folders."""
    if vmd_path is None:
        vmd_path = os.environ.get(VMD_ENV_VAR) or None
    candidates = ([vmd_path] if vmd_path is not None
                  else ["vmd", *_standard_locations()])
    for candidate in candidates:
        executable = shutil.which(str(candidate))
        if executable is not None:
            return Path(executable)
        # shutil.which skips files that are not marked executable (e.g. a
        # path given without its .exe), so accept existing files as well
        if os.path.isfile(candidate):
            return Path(candidate)
    return None


def _box_line_commands(lower, upper, grid, style):
    """Tcl commands drawing the lines parallel to the axes from lower to upper
    that pass through grid points (in the other two coordinates), with the
    given Tcl line style options."""
    commands = []
    for axis in range(3):
        for p, q in product(grid, repeat=2):
            start, end = (np.insert([p, q], axis, lower),
                          np.insert([p, q], axis, upper))
            commands.append("graphics top line {%g %g %g} {%g %g %g} %s"
                            % (*start, *end, style))
    return commands


def write_vmd_script(script_path, trajectory_path, box_length_angstrom,
                     bonds, show_periodic_images=True, show_box=True,
                     show_image_outlines=True):
    """Writes a Tcl script that loads the trajectory in licorice style with
    the given bonds (B x 2 atom indices), on a white background, with depth
    cueing (atoms fade with distance from the camera). The box length is in
    Angstrom. With show_periodic_images, it also shows the 26 periodic images
    of the box, outlined in dashed grey if show_image_outlines; with
    show_box, the box itself is outlined in red. Run it with
    `vmd -e <script>`, or in VMD with File > Load Visualization State."""
    trajectory_path = os.path.abspath(trajectory_path).replace("\\", "/")
    # ---- Bonds: VMD guesses them from distances; set the real ones ----
    # Molecules are written whole, so the bonds hold in every frame
    nr_atoms = int(np.max(bonds)) + 1 if len(bonds) else 0
    bonded_atoms = [[] for _ in range(nr_atoms)]
    for i, j in bonds:
        bonded_atoms[i].append(int(j))
        bonded_atoms[j].append(int(i))
    bond_list = " ".join("{%s}" % " ".join(map(str, n))
                         for n in bonded_atoms)

    L = box_length_angstrom
    tcl = [f"""\
# Load the .xyz file
mol new {{{trajectory_path}}} type xyz waitfor all
color Display Background white

# Bonds from the topology
set all_atoms [atomselect top "index < {nr_atoms}"]
$all_atoms setbonds {{{bond_list}}}
$all_atoms delete

# Licorice, coloured by element
mol modstyle 0 top Licorice 0.2 12 12
mol modcolor 0 top Name
mol modmaterial 0 top Opaque

# The periodic box (the same in every frame)
for {{set i 0}} {{$i < [molinfo top get numframes]}} {{incr i}} {{
    molinfo top set frame $i
    molinfo top set {{a b c alpha beta gamma}} \\
        {{{L:g} {L:g} {L:g} 90 90 90}}
}}
animate goto 0
"""]
    if show_periodic_images:
        tcl += ["# Periodic images: the 26 neighbouring boxes",
                "mol showperiodic top 0 xyzXYZ",
                "mol numperiodic top 0 1"]
    if show_periodic_images and show_image_outlines:
        tcl += ["# Outlines of the periodic images",
                "graphics top color gray"]
        tcl += _box_line_commands(-L, 2 * L, [-L, 0, L, 2 * L], "style dashed")
    if show_box:
        tcl += ["# The simulation box",
                "graphics top color red"]
        tcl += _box_line_commands(0, L, [0, L], "width 3")
    # Centre on the box; zoom out to fit its images (3 boxes wide)
    tcl += ["display resetview"] + (["scale by 0.35"]
                                       if show_periodic_images else [])
    # Depth cueing: atoms further from the camera fade into the background
    tcl += ["display depthcue on",
            f"display cuemode {CUE_MODE}",
            f"display cuedensity {CUE_DENSITY}"]
    Path(script_path).write_text("\n".join(tcl) + "\n")
    return Path(script_path)


def launch_vmd(executable, script_path):
    """Starts VMD running the script, without waiting for it to close."""
    script_path = Path(script_path).resolve()
    return subprocess.Popen([str(executable), "-e", str(script_path)],
                            cwd=script_path.parent)
