import pytest

from jostle.config import Config
from jostle.simulation import run_folder


def test_settings_by_name_go_to_their_group():
    config = Config(dt=0.001, box_length=3.0, rdf_pairs=[], seed=2)
    assert config.numerics.dt == 0.001 and config.system.box_length == 3.0
    assert config.recording.rdf_pairs == [] and config.seed == 2
    with pytest.raises(TypeError, match="Unknown setting"):
        Config(nonexistent=0.1)


@pytest.mark.parametrize("changes, message", [
    (dict(weights={"Water": -1}), "weights"),
    (dict(weights={"Unobtainium": 1}), "does not exist"),
    (dict(integrator="leapfrog"), "integrator"),
    (dict(dt=0), "dt"),
    (dict(end_time=0), "end_time"),
    (dict(box_length=1.5), "half the box length"),
    (dict(rdf_pairs=["O-O Water"]), "Cannot parse"),
])
def test_validation(changes, message):
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        Config(**changes).validate()


def test_run_folder_is_unique_per_config():
    # Equal settings share a folder (a rerun overwrites); any other setting,
    # including the seed or a recording setting, gets its own
    base = Config(box_length=2, seed=1, output_dir="a")
    assert run_folder(base).name == run_folder(
        Config(box_length=2.0, seed=1, output_dir="a")).name
    assert run_folder(base).parent.name == "a"
    others = [base.replace(seed=2), base.replace(rdf_bin_width=0.01),
              base.replace(dt=0.0001)]
    names = {run_folder(c).name for c in [base, *others]}
    assert len(names) == 4
