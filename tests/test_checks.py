"""The pre-run checks: the default settings give no warnings."""
import warnings

from jostle.checks import JostleWarning, check_config
from jostle.config import Config


def test_default_settings_give_no_warnings(capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("error", JostleWarning)
        assert check_config(Config()) == []
    assert "Expected output: trajectory.xyz" in capsys.readouterr().out
