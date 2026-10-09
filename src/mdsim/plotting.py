"""Figures made at the end of a run. Every figure is saved to the run folder;
show=True also opens it in a (blocking) window."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.stats import norm

from .constants import KBOLTZ
from .progress import ProgressBar


def _finish(fig, path, show, **savefig_kwargs):
    fig.savefig(path, **savefig_kwargs)
    if show:
        plt.show()
    plt.close(fig)


def computation_piechart(timings, total_time, nr_steps, folder_name,
                         show=False):
    """Pie chart of where the computation time went. timings maps a label to
    seconds; the remainder of total_time is shown as 'Other'."""
    labels = list(timings) + ["Other"]
    times = np.array(list(timings.values()) + [0.])
    times[-1] = max(total_time - times[:-1].sum(), 0.)

    # Filter small values
    mask = times > 0.01 * total_time
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)

    fig = plt.figure(figsize=(8, 6))
    plt.pie(times[mask], labels=np.array(labels)[mask], colors=colors[mask],
            autopct=lambda p: f'{p * times[mask].sum() / 100:.2f} s',
            startangle=140)
    plt.title(f"Total computation time for {nr_steps} iterations: "
              f"{total_time:.2f} s")
    _finish(fig, os.path.join(folder_name, 'comptimes_piechart.png'), show,
            dpi=300, bbox_inches='tight')


def temperature_plot(KE, Nf, save_interval, folder_name, show=False):
    """Plots the temperature computed from the kinetic energy."""
    time_points = save_interval * np.arange(len(KE))
    T = 2 * KE / (KBOLTZ * Nf)

    fig = plt.figure()
    plt.plot(time_points, T, label="Temperature")
    plt.xlabel("Time (ps)")
    plt.ylabel("Temperature (K)")
    plt.title("Plot of system temperature")
    plt.legend()
    plt.grid()
    _finish(fig, os.path.join(folder_name, 'temperature.png'), show)


def energy_plot(energies, save_interval, folder_name, show=False):
    """Plots each type of energy in the system (a dict of arrays), as well as
    the total energy."""
    time_points = save_interval * np.arange(len(next(iter(energies.values()))))

    fig = plt.figure()
    for name, values in energies.items():
        plt.plot(time_points, values, label=f"{name} energy")
    plt.plot(time_points, np.sum(list(energies.values()), axis=0),
             label="Total energy")
    plt.xlabel("Time (ps)")
    plt.ylabel("Energy (kJ/mol)")
    plt.title("Plot of energy in the system")
    plt.legend()
    plt.grid()
    _finish(fig, os.path.join(folder_name, 'energy.png'), show)


# ------------------------- phase space diagram functions ---------------------
def phase_space_plots(phase_space, folder_name, show=False):
    """Creates a phase space diagram (bond length against its rate of change)
    for each tracked bond type."""
    for name, D, V in zip(phase_space.names, phase_space.distances,
                          phase_space.velocities):
        nr_trajectories = D.shape[1]

        fig, ax = plt.subplots(figsize=(8, 6))
        segments = [np.column_stack((D[:, j], V[:, j]))
                    for j in range(nr_trajectories)]
        lc = LineCollection(segments, cmap='viridis', alpha=0.7)
        lc.set_array(np.linspace(0, 1, nr_trajectories))
        ax.add_collection(lc)

        ax.set_xlabel('Bond length D (nm)')
        ax.set_ylabel('Bond velocity dD/dt (nm/ps)')
        # Set axis limits to properly view the trajectory.
        for set_lim, data in ((ax.set_xlim, D), (ax.set_ylim, V)):
            margin = 0.05 * (data.max() - data.min())
            set_lim(data.min() - margin, data.max() + margin)
        ax.set_title(f'Phase Space Diagram for {name} Bonds')
        _finish(fig, os.path.join(folder_name, f"ps_diag_{name}.png"), show,
                bbox_inches='tight')


# ----------------------------------- RDF functions --------------------------
def _animate(fig, update, nr_frames, path, progress, offset):
    def frame(k):
        update(k)
        progress.update(offset + k)

    ani = FuncAnimation(fig, frame, frames=range(nr_frames), interval=1)
    ani.save(path, writer='pillow')


def rdf_plots(rdf, save_interval, folder_name, r0, RDF_conf=0.9):
    """Creates gifs (and first/last frame pngs) of the RDF of each pair of
    interest over time, with running means and confidence intervals, plus one
    gif comparing the means of all pairs."""
    T, nr_pairs, nr_bins = rdf.histograms.shape
    bin_edges = rdf.bin_edges
    colors = plt.cm.tab10(np.linspace(0, 1, nr_pairs))
    os.makedirs(folder_name, exist_ok=True)

    # Normalize each frame of each pair to sum to 1 over the bins
    data = rdf.histograms / rdf.bin_volumes[None, None, :]
    data /= np.sum(data, axis=2, keepdims=True)

    # Running means and standard deviations over the saved frames
    n = np.arange(1, T + 1)[:, None, None]
    means = np.cumsum(data, axis=0) / n
    std_devs = np.sqrt(np.cumsum((data - means)**2, axis=0) / n)
    margin_of_error = norm.ppf(1 - 0.5 * (1 - RDF_conf)) * std_devs / np.sqrt(n)
    lower, upper = means - margin_of_error, means + margin_of_error

    # Identify the largest frequency beyond the bond lengths to set ylim
    start = int(1.2 * np.max(r0) / bin_edges[-1] * nr_bins)
    final_upper = upper[-1, :, start:]
    max_freq = np.max(final_upper[np.isfinite(final_upper)])

    progress = ProgressBar((nr_pairs + 1) * T)
    r = bin_edges[:-1]

    # One gif per pair, with confidence interval
    for k, pair in enumerate(rdf.names):
        pair_folder = os.path.join(folder_name, pair)
        os.makedirs(pair_folder, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        line_mean, = ax.plot(r, means[0, k], color=colors[k], label="mean")
        line_lower, = ax.plot(r, lower[0, k], color="lightgray",
                              linewidth=0.75)
        line_upper, = ax.plot(r, upper[0, k], color="lightgray",
                              linewidth=0.75,
                              label=f"{int(100 * RDF_conf)}% confidence "
                                    f"interval")
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_ylim(0, 1.25 * max_freq)
        ax.set_title(f"RDF of {pair} at t=0")
        ax.legend(loc='upper right')
        fig.savefig(os.path.join(pair_folder, f'first_frame_{pair}.png'))

        def update(frame, k=k, pair=pair, ax=ax, lines=(line_mean, line_lower,
                                                        line_upper)):
            for line, values in zip(lines, (means, lower, upper)):
                line.set_ydata(values[frame, k])
            ax.set_title(f"RDF of {pair} at t={frame * save_interval:.4f}")

        _animate(fig, update, T, os.path.join(pair_folder, f'RDF_{pair}.gif'),
                 progress, k * T)
        fig.savefig(os.path.join(pair_folder, f'last_frame_{pair}.png'))
        plt.close(fig)

    # All means in one figure
    fig, ax = plt.subplots(figsize=(9, 6))
    lines = [ax.plot(r, means[0, k], color=colors[k], label=pair)[0]
             for k, pair in enumerate(rdf.names)]
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(0, 1.25 * max_freq)
    ax.set_title("RDF at t=0")
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(folder_name, 'first_frame.png'))

    def update(frame):
        for k, line in enumerate(lines):
            line.set_ydata(means[frame, k])
        ax.set_title(f"RDF at t={frame * save_interval:.4f}")

    _animate(fig, update, T, os.path.join(folder_name, 'RDF.gif'), progress,
             nr_pairs * T)
    fig.savefig(os.path.join(folder_name, 'last_frame.png'))
    plt.close(fig)
    progress.finish()
