"""Command line interface: python -m mdsim [--config FILE] [options]."""
import argparse

from .config import load_config
from .simulation import run


def _weight(text):
    name, _, value = text.partition("=")
    try:
        return name, float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected NAME=WEIGHT, e.g. Water=1, got {text!r}") from None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mdsim",
        description="Molecular dynamics of small-molecule mixtures in a "
                    "periodic box.")
    parser.add_argument("-c", "--config",
                        help="TOML settings file (default: built-in "
                             "defaults, see configs/default.toml)")
    parser.add_argument("-w", "--weight", type=_weight, action="append",
                        metavar="NAME=WEIGHT",
                        help="relative amount of a molecule; replaces all "
                             "weights from the config when given. Repeat "
                             "for each molecule, e.g. -w Water=3 -w "
                             "Ethanol=1")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--no-show", dest="show_plots", action="store_false",
                        default=None,
                        help="save figures without opening windows")
    args = parser.parse_args(argv)

    overrides = {}
    if args.weight:
        overrides["weights"] = dict(args.weight)
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.show_plots is not None:
        overrides["show_plots"] = args.show_plots

    config = load_config(args.config, **overrides)
    folder = run(config)
    print(f"Results written to {folder}")
