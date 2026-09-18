"""The CLI: gating, exit codes, and the shared controller.

The CLI and the Streamlit page must not be two implementations. These tests pin
the contract the CLI exposes and check that it routes into the same controller.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.cli import (  # noqa: E402
    EXIT_DISABLED,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_PREFLIGHT,
    main,
)
from t2pw.mining.flags import FEATURE_ENV_VAR  # noqa: E402


@pytest.fixture
def disabled_env(monkeypatch):
    monkeypatch.delenv(FEATURE_ENV_VAR, raising=False)
    monkeypatch.setattr("t2pw.config._ENV_LOADED", True, raising=False)


def test_run_is_refused_when_the_feature_is_off(disabled_env, capsys, tmp_path):
    code = main(["run", "--species", "Homo sapiens", "--runs-dir", str(tmp_path / "r")])
    assert code == EXIT_DISABLED
    err = capsys.readouterr().err
    assert FEATURE_ENV_VAR in err
    assert "--enable-pathway-mining" in err


def test_the_cli_flag_enables_the_feature_for_one_invocation(disabled_env, tmp_path, capsys):
    """`--enable-pathway-mining` is the per-invocation override."""
    code = main(
        [
            "--enable-pathway-mining",
            "preflight",
            "--species", "Arabidopsis thaliana",
            "--topic", "jasmonate biosynthesis",
            "--runs-dir", str(tmp_path / "runs"),
            "--no-probe",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    names = {c["name"] for c in payload["checks"]}
    assert "feature.flag" in names
    flag = next(c for c in payload["checks"] if c["name"] == "feature.flag")
    assert flag["status"] == "ok"
    assert flag["data"]["enabled"] is True
    # Without a provider configured the preflight legitimately fails; what is
    # under test is that the gate opened at all.
    assert code in (EXIT_OK, EXIT_PREFLIGHT)


def test_preflight_fails_loudly_rather_than_starting_an_expensive_run(
    disabled_env, tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    code = main(
        [
            "--enable-pathway-mining",
            "preflight",
            "--species", "Arabidopsis thaliana",
            "--runs-dir", str(tmp_path / "runs"),
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_PREFLIGHT
    assert "PREFLIGHT FAILED" in out


def test_an_unparseable_paper_reference_is_a_usage_error(disabled_env, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--enable-pathway-mining",
                "run",
                "--species", "Homo sapiens",
                "--paper", "not-a-reference",
                "--runs-dir", str(tmp_path / "runs"),
                "--no-probe",
            ]
        )
    assert "unrecognised paper reference" in str(excinfo.value)


def test_a_missing_starting_graph_is_refused(disabled_env, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--enable-pathway-mining",
                "run",
                "--species", "Homo sapiens",
                "--starting-graph", str(tmp_path / "nope.json"),
                "--runs-dir", str(tmp_path / "runs"),
                "--no-probe",
            ]
        )
    assert "starting graph not found" in str(excinfo.value)


def test_status_report_and_list_read_a_finished_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(FEATURE_ENV_VAR, "true")
    mined = H.run_fixture(tmp_path)
    run_path = str(mined.outcome.run_dir.root)
    runs_dir = str(mined.outcome.run_dir.root.parent)

    code = main(["status", run_path])
    out = capsys.readouterr().out
    assert code == EXIT_INCOMPLETE, "a limit-stop is reported as incomplete, not success"
    assert "PWML              none" in out
    assert "database check    deferred" in out

    assert main(["report", run_path]) == EXIT_OK
    written = capsys.readouterr().out.strip().splitlines()
    assert any("report.md" in line for line in written)

    assert main(["list", "--runs-dir", runs_dir]) == EXIT_OK
    listing = capsys.readouterr().out
    assert mined.outcome.run_dir.run_id in listing
    assert H.TARGET_SPECIES in listing


def test_status_json_is_machine_readable(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(FEATURE_ENV_VAR, "true")
    mined = H.run_fixture(tmp_path)
    main(["status", str(mined.outcome.run_dir.root), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["produces_pwml"] is False
    assert payload["external_database_validation"] == "deferred_not_evaluated"
    assert payload["graph"]["primary_reactions"] >= 1


def test_budget_flags_reach_the_config():
    import argparse

    from t2pw.mining.cli import _budget_overrides, _make_config

    args = argparse.Namespace(
        species="Homo sapiens",
        topic="",
        seed_metabolite=[], seed_enzyme=[], seed_reaction=[], paper=[],
        starting_graph="", config="", run_id="", runs_dir="", model="vendor/m",
        offline=True, no_screen=True, vector_backend="memory",
        max_rounds=3, max_papers_discovered=None, max_papers_processed=None,
        max_llm_calls=None, max_input_tokens=None, max_output_tokens=None,
        max_cost_usd=0.5, max_wall_clock_seconds=None, max_frontier_size=None,
    )
    assert _budget_overrides(args) == {"max_rounds": 3, "max_cost_usd": 0.5}
    config = _make_config(args, enabled_via="cli_flag")
    assert config.budgets.max_rounds == 3
    assert config.budgets.max_cost_usd == 0.5
    assert config.model.model == "vendor/m"
    assert config.offline is True
    assert config.screen_papers is False
    assert config.enabled_via == "cli_flag"


def test_a_config_file_supplies_overrides(tmp_path):
    import argparse

    from t2pw.mining.cli import _make_config

    path = tmp_path / "mining.json"
    path.write_text(
        json.dumps({"budgets": {"max_rounds": 2}, "model": {"model": "vendor/from-file"}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        species="Homo sapiens", topic="", seed_metabolite=[], seed_enzyme=[],
        seed_reaction=[], paper=[], starting_graph="", config=str(path), run_id="",
        runs_dir=str(tmp_path), model="", offline=True, no_screen=False,
        vector_backend="", max_rounds=None, max_papers_discovered=None,
        max_papers_processed=None, max_llm_calls=None, max_input_tokens=None,
        max_output_tokens=None, max_cost_usd=None, max_wall_clock_seconds=None,
        max_frontier_size=None,
    )
    config = _make_config(args, enabled_via="cli_flag")
    assert config.budgets.max_rounds == 2
    assert config.model.model == "vendor/from-file"


def test_the_cli_and_the_ui_build_the_same_controller():
    """One controller, two front ends -- not two implementations."""
    import inspect

    from t2pw.mining import cli
    from t2pw.mining.ui import page

    cli_source = inspect.getsource(cli)
    page_source = inspect.getsource(page)
    assert "MiningController" in cli_source
    assert "MiningController" in page_source
    # Neither front end reimplements a round loop.
    for source in (cli_source, page_source):
        assert "_run_round" not in source
        assert "extract_claims" not in source


def test_exit_codes_are_distinct():
    assert len({EXIT_OK, 2, EXIT_DISABLED, EXIT_PREFLIGHT, EXIT_INCOMPLETE}) == 5
