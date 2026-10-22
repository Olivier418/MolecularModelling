"""Figures of the measurements of a run. The *_figure functions build and
return a matplotlib Figure without showing it: in a notebook it is displayed
as the cell result, and Results saves it on request."""
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.stats import t as student_t
from tqdm import tqdm

# Confidence level of the intervals shaded around the running mean RDFs
RDF_CONFIDENCE = 0.9
# Default block length (ps) for the RDF confidence intervals. Consecutive
# frames are correlated, so they are averaged in blocks of this length and
# the block means are treated as independent. It should be longer than the
# correlation time of the RDF; plot_rdf and animate_rdf take it as
# block_time.
RDF_BLOCK_TIME = 1.0


def _figure(**kwargs):
    """A new figure with one axes. It is closed in pyplot right away, so a
    notebook shows it once (as the returned value), not twice."""
    fig, ax = plt.subplots(**kwargs)
    plt.close(fig)
    return fig, ax


def timings_figure(timings, wall_time, nr_steps):
    """Pie chart of where the computation time went. timings maps a label to
    seconds; the remainder of wall_time is shown as 'Other'."""
    labels = list(timings) + ["Other"]
    seconds = np.array(list(timings.values()) + [0.])
    seconds[-1] = max(wall_time - seconds[:-1].sum(), 0.)

    # Filter small values
    mask = seconds > 0.01 * wall_time
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)

    fig, ax = _figure(figsize=(8, 6))
    ax.pie(seconds[mask], labels=np.array(labels)[mask], colors=colors[mask],
           autopct=lambda p: f'{p * seconds[mask].sum() / 100:.2f} s',
           startangle=140)
    ax.set_title(f"Total computation time for {nr_steps} iterations: "
                 f"{wall_time:.2f} s")
    return fig


def temperature_figure(times, temperatures):
    """Plots the temperature over time."""
    fig, ax = _figure()
    ax.plot(times, temperatures, label="Temperature")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Plot of system temperature")
    ax.legend()
    ax.grid()
    return fig


def energy_figure(times, energies):
    """Plots each type of energy in the system (a dict of arrays), as well as
    the total energy."""
    fig, ax = _figure()
    for name, values in energies.items():
        ax.plot(times, values, label=f"{name} energy")
    ax.plot(times, np.sum(list(energies.values()), axis=0),
            label="Total energy")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Energy (kJ/mol)")
    ax.set_title("Plot of energy in the system")
    ax.legend()
    ax.grid()
    return fig


# ------------------------- phase space diagram functions ---------------------
def phase_space_figure(bond_type, lengths, length_rates):
    """Phase space diagram of one bond type: bond length against its rate of
    change, one line per bond. lengths and length_rates have shape
    (frames, bonds)."""
    nr_bonds = lengths.shape[1]

    fig, ax = _figure(figsize=(8, 6))
    segments = [np.column_stack((lengths[:, j], length_rates[:, j]))
                for j in range(nr_bonds)]
    line_collection = LineCollection(segments, cmap='viridis', alpha=0.7)
    line_collection.set_array(np.linspace(0, 1, nr_bonds))
    ax.add_collection(line_collection)

    ax.set_xlabel('Bond length D (nm)')
    ax.set_ylabel('Bond velocity dD/dt (nm/ps)')
    # Set axis limits to properly view the trajectory.
    for set_lim, data in ((ax.set_xlim, lengths), (ax.set_ylim, length_rates)):
        margin = 0.05 * (data.max() - data.min())
        set_lim(data.min() - margin, data.max() + margin)
    ax.set_title(f'Phase Space Diagram for {bond_type} Bonds')
    return fig


# ----------------------------------- RDF functions --------------------------
class RDFStatistics(NamedTuple):
    """Running means of g(r) with confidence bounds, each of shape
    (frames, pairs, bins), and the largest final upper bound (max_g)."""
    means: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    max_g: float


def rdf_statistics(rdf, frame_interval, block_time=RDF_BLOCK_TIME):
    """Running means of g(r) of each pair over the saved frames and their
    confidence intervals, each of shape (frames, pairs, bins), plus the
    largest final upper bound (used as the plot range).

    The intervals come from block averaging: the frames are grouped into
    consecutive blocks of block_time (ps; at least one frame), and the
    spread of the block means gives the standard error of the mean. Up to a
    frame, only the complete blocks count; with fewer than two, the bounds
    are NaN (no interval is drawn)."""
    g = rdf.g_of_r()
    nr_frames, nr_pairs, nr_bins = g.shape

    # Running means over the saved frames
    frame_counts = np.arange(1, nr_frames + 1)[:, None, None]
    means = np.cumsum(g, axis=0) / frame_counts

    # Means of the complete blocks, and their running (sample) variance
    # from running sums of the values and their squares
    frames_per_block = max(1, round(block_time / frame_interval))
    nr_blocks = nr_frames // frames_per_block
    blocks = g[:nr_blocks * frames_per_block].reshape(
        nr_blocks, frames_per_block, nr_pairs, nr_bins).mean(axis=1)
    block_counts = np.arange(1, nr_blocks + 1)[:, None, None]
    sums = np.cumsum(blocks, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        variances = ((np.cumsum(blocks**2, axis=0) - sums**2 / block_counts)
                     / (block_counts - 1))
        # Student's t, as there may be only a few blocks
        quantile = student_t.ppf(1 - 0.5 * (1 - RDF_CONFIDENCE),
                                 block_counts - 1)
        # NaN for a single block (0/0, and t has no degrees of freedom)
        block_margins = quantile * np.sqrt(np.maximum(variances, 0)
                                           / block_counts)

    # Margin at each frame, from the blocks completed by then
    completed = np.arange(1, nr_frames + 1) // frames_per_block
    margin_of_error = np.full_like(means, np.nan)
    has_blocks = completed >= 1
    margin_of_error[has_blocks] = block_margins[completed[has_blocks] - 1]
    lower, upper = means - margin_of_error, means + margin_of_error

    # The largest final value sets the ylim
    final = np.where(np.isfinite(upper[-1]), upper[-1], means[-1])
    final = final[np.isfinite(final)]
    max_g = final.max() if final.size else 1.
    return RDFStatistics(means, lower, upper, max_g)


def _rdf_colors(nr_pairs):
    return plt.cm.tab10(np.linspace(0, 1, nr_pairs))


def _style_rdf_axes(ax, bin_edges, max_g):
    """Limits and labels of an RDF plot, with g = 1 (no structure) marked."""
    ax.axhline(1, color="gray", linewidth=0.75, linestyle="--")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(0, 1.25 * max_g)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("g(r)")


def rdf_figure(rdf, stats, frame, frame_time, pairs=None):
    """Running mean RDF of the given pairs (default: all) up to save index
    frame, with shaded confidence intervals."""
    means, lower, upper, max_g = stats
    colors = _rdf_colors(len(rdf.pair_labels))
    r = rdf.bin_edges[:-1]

    fig, ax = _figure(figsize=(9, 6))
    for k, pair in enumerate(rdf.pair_labels):
        if pairs is not None and pair not in pairs:
            continue
        ax.plot(r, means[frame, k], color=colors[k], label=pair)
        ax.fill_between(r, lower[frame, k], upper[frame, k],
                        color=colors[k], alpha=0.25, linewidth=0)
    _style_rdf_axes(ax, rdf.bin_edges, max_g)
    ax.set_title(f"RDF at t={frame_time:.4f} ps (shaded: "
                 f"{int(100 * RDF_CONFIDENCE)}% confidence interval)")
    ax.legend(loc='upper right')
    return fig


def save_rdf_gif(rdf, stats, frame_interval, folder, pair=None,
                 max_frames=200):
    """Creates a gif of the RDF over time in folder, plus pngs of its first
    and last frames: of one pair, with its running mean and confidence
    interval (in folder/<pair>/), or of the running means of all pairs if
    pair is None. The statistics use every saved frame; the gif shows at
    most max_frames of them. Returns the path of the gif."""
    means, lower, upper, max_g = stats
    colors = _rdf_colors(len(rdf.pair_labels))
    r = rdf.bin_edges[:-1]
    # Evenly spaced frames to animate, always including the first and last
    nr_frames = len(means)
    frames = np.unique(np.linspace(0, nr_frames - 1,
                                   min(nr_frames, max_frames)).astype(int))

    # Each line is drawn from (values, pair index) at every frame
    fig, ax = _figure(figsize=(9, 6))
    if pair is None:
        folder, suffix, title = Path(folder), "", "RDF"
        lines = [(ax.plot(r, means[0, k], color=colors[k], label=label)[0],
                  means, k) for k, label in enumerate(rdf.pair_labels)]
    else:
        k = rdf.pair_labels.index(pair)
        folder, suffix = Path(folder) / pair, f"_{pair}"
        title = f"RDF of {pair}"
        lines = [
            (ax.plot(r, means[0, k], color=colors[k], label="mean")[0],
             means, k),
            (ax.plot(r, lower[0, k], color="lightgray", linewidth=0.75)[0],
             lower, k),
            (ax.plot(r, upper[0, k], color="lightgray", linewidth=0.75,
                     label=f"{int(100 * RDF_CONFIDENCE)}% confidence "
                           f"interval")[0], upper, k)]
    folder.mkdir(parents=True, exist_ok=True)
    _style_rdf_axes(ax, rdf.bin_edges, max_g)
    ax.set_title(f"{title} at t=0")
    ax.legend(loc='upper right')
    fig.savefig(folder / f"first_frame{suffix}.png")

    progress = tqdm(total=len(frames))

    def update(n):
        frame = frames[n]
        for line, values, k in lines:
            line.set_ydata(values[frame, k])
        ax.set_title(f"{title} at t={frame * frame_interval:.4f}")
        # Absolute count: FuncAnimation may draw a frame more than once
        progress.update(n + 1 - progress.n)

    path = folder / f"RDF{suffix}.gif"
    FuncAnimation(fig, update, frames=range(len(frames)),
                  interval=1).save(path, writer='pillow')
    progress.close()
    fig.savefig(folder / f"last_frame{suffix}.png")
    return path
