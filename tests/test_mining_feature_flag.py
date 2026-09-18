"""The feature gate: default off, three activation paths, no cost when disabled.

NO network, NO LLM draw, NO cache write. Everything here is in-process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.mining import flags  # noqa: E402
from t2pw.mining.flags import (  # noqa: E402
    FEATURE_ENV_VAR,
    PathwayMiningDisabled,
    mining_enabled,
    require_mining_enabled,
)


def test_default_is_off(monkeypatch):
    monkeypatch.delenv(FEATURE_ENV_VAR, raising=False)
    monkeypatch.setattr("t2pw.config._ENV_LOADED", True, raising=False)
    assert mining_enabled() is False


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", "enabled"])
def test_environment_enables(monkeypatch, value):
    monkeypatch.setenv(FEATURE_ENV_VAR, value)
    assert mining_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "nonsense"])
def test_environment_does_not_enable(monkeypatch, value):
    monkeypatch.setenv(FEATURE_ENV_VAR, value)
    assert mining_enabled() is False


def test_override_wins_both_ways(monkeypatch):
    """The CLI flag / UI toggle is a per-invocation override in both directions."""
    monkeypatch.delenv(FEATURE_ENV_VAR, raising=False)
    monkeypatch.setattr("t2pw.config._ENV_LOADED", True, raising=False)
    assert mining_enabled(True) is True
    monkeypatch.setenv(FEATURE_ENV_VAR, "true")
    assert mining_enabled(False) is False


def test_require_raises_with_activation_instructions(monkeypatch):
    monkeypatch.delenv(FEATURE_ENV_VAR, raising=False)
    monkeypatch.setattr("t2pw.config._ENV_LOADED", True, raising=False)
    with pytest.raises(PathwayMiningDisabled) as excinfo:
        require_mining_enabled()
    message = str(excinfo.value)
    assert FEATURE_ENV_VAR in message
    assert "--enable-pathway-mining" in message
    assert "does not start a run" in message


def test_controller_refuses_to_construct_when_disabled(monkeypatch, tmp_path):
    from helpers_mining import make_config
    from t2pw.mining.controller import MiningController

    monkeypatch.delenv(FEATURE_ENV_VAR, raising=False)
    monkeypatch.setattr("t2pw.config._ENV_LOADED", True, raising=False)
    config = make_config(tmp_path)
    with pytest.raises(PathwayMiningDisabled):
        MiningController(config)


def test_importing_the_package_is_cheap_and_touches_nothing():
    """Importing t2pw.mining must not build an LLM client or a vector store.

    A disabled feature that still constructs a provider client, reads a 39 MB
    cache or imports chromadb is not disabled in the way that matters.
    """
    script = (
        "import sys\n"
        "import t2pw.mining\n"
        "bad = [m for m in ('chromadb', 'openai', 'streamlit') if m in sys.modules]\n"
        "client = 't2pw.llm.client' in sys.modules\n"
        "print('LOADED', bad, client)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "", "SYSTEMROOT": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "LOADED [] False" in result.stdout, result.stdout


def test_disabled_run_writes_nothing(monkeypatch, tmp_path):
    """With the flag off, no run directory is created by any entry point."""
    from t2pw.mining.cli import EXIT_DISABLED, main

    monkeypatch.delenv(FEATURE_ENV_VAR, raising=False)
    monkeypatch.setattr("t2pw.config._ENV_LOADED", True, raising=False)
    runs = tmp_path / "runs_mining"
    code = main(
        ["run", "--species", "Arabidopsis thaliana", "--topic", "x", "--runs-dir", str(runs)]
    )
    assert code == EXIT_DISABLED
    assert not runs.exists()


def test_as_bool_helper_is_conservative():
    assert flags._as_bool("maybe", default=False) is False
    assert flags._as_bool("maybe", default=True) is True
    assert flags._as_bool(None) is False
