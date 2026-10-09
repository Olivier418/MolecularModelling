from pathlib import Path

import pytest

from mdsim.cli import main
from mdsim.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_file_matches_built_in_defaults():
    config = load_config(ROOT / "configs" / "default.toml")
    defaults = Config()
    assert config.output_dir.resolve() == (ROOT / "trajectories").resolve()
    for name in ("weights", "boxsize", "dt", "end_time", "save_steps",
                 "method", "r_cutoff_scalar", "f_threshold",
                 "r_threshold_scalar", "bonds_of_interest",
                 "pairs_of_interest", "RDF_dr", "seed"):
        assert getattr(config, name) == getattr(defaults, name), name


def test_f_threshold_replaces_default_cutoff(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[lennard_jones]\nf_threshold = 0.05\n")
    config = load_config(path)
    assert config.f_threshold == 0.05 and config.r_cutoff_scalar is None


def test_unknown_setting_is_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[integration]\ndtt = 0.001\n")
    with pytest.raises(ValueError, match="dtt"):
        load_config(path)


@pytest.mark.parametrize("changes, message", [
    (dict(weights={"Water": -1}), "weights"),
    (dict(weights={"Unobtainium": 1}), "does not exist"),
    (dict(method="leapfrog"), "method"),
    (dict(save_steps=0), "save_steps"),
    (dict(dt=0), "dt"),
    (dict(f_threshold=0.1), "Exactly one"),
    (dict(r_cutoff_scalar=None), "Exactly one"),
    (dict(RDF_thin=0), "RDF_thin"),
    (dict(temp_control=1), "boolean"),
])
def test_validation(changes, message):
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        Config(**changes).validate()


def test_cli_runs_with_overrides(tmp_path, capsys):
    path = tmp_path / "c.toml"
    path.write_text(
        "[system]\nboxsize = 1.5\n"
        "[integration]\nend_time = 0.004\nsave_steps = 2\n"
        "[output]\ndirectory = 'out'\n"
        "[plots]\nenergy = false\ntemperature = false\nphase_space = false\n"
        "computation_time = false\nrdf = false\n")
    main(["-c", str(path), "-w", "Water=1", "--seed", "1", "--no-show"])
    assert "Results written to" in capsys.readouterr().out
    (folder,) = (tmp_path / "out").iterdir()
    assert folder.name.endswith("__1.0Water")
