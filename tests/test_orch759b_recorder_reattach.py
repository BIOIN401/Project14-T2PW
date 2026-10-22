"""ORCH-759b (D-107): the Stage-1 diagnostics recorder is re-attached in the post-pipeline run.

WHY. ``ORCH-758-RESULT.md`` § 4 D4. D-104 rule 3, the Stage-6 database-identity merge, reads
the run's diagnostics recorder: ``_pooled_draws()`` needs the Stage-1 ``stage1_ensemble``
boundary and ``_write_identity_side_file`` needs its artifact directory. The recorder is a
ContextVar, set in the "Run pipeline" script run. "Run audit and DB mapping" is another script
run on another thread, so it starts with an empty context: rule 3 never ran in the app, and
the audit and gap round summaries the same run records were never written (0 of 312 run
directories hold them).

HOW. The REAL app, through ``streamlit.testing.v1.AppTest``, exactly as ``t2pw.batch.driver``
drives it: paste a paper, click "Run pipeline", click "Run audit and DB mapping". Nothing in
the pipeline is stubbed except what would leave the machine:

* the provider transport (``t2pw.llm.client._client``) is a fake that records every request
  and answers Stage 0 with a context and every Stage-1 draw with the same two reactions,
  which differ only in a synonym (``protoheme`` / ``heme b``) that rule 3 exists to merge;
* compound lookup answers from a table (both synonyms -> CHEBI:17627), the PathBank DB is
  absent, and every HTTP request raises (the app's other lookups, ChEBI's web service among
  them, still try and are refused, at base and at the tip alike);
* Stage 7 enrichment (UniProt over the network) copies its input;
* ``t2pw.paths.PROJECT_ROOT`` is this test's ``tmp_path``, so the app's ``tmp/``,
  ``outputs/``, ``out/`` and ``data/`` writes never touch the checkout.

G9, BEHAVIOURAL (``test_g9_*``): fail on base ``4f750030`` -- the post-pipeline run sees a
detached recorder, rule 3 returns ``{}``, no ``ensemble_identity_merge_report.json`` is
written and no audit/gap summary reaches the Stage-1 artifact directory -- and pass at the tip.

PRESERVATION (``test_still_*``): pass on base and at the tip. One draw: rule 3 stays off
(D-104 § 2). No Stage-1 run in the session: the post-pipeline run is exactly today's.

D-104 AMENDMENT 2 (``test_llm_inputs_*``): every request the app sends a model in both script
runs is byte-identical to base for the same paper (sha256 per request, measured on base
``4f750030``), and no request carries the identity-merge record.

NEW ACCEPTANCE TEST (``test_new_*``): the ``reenter`` helper across threads; fails at base on
the missing name, which is why it is not offered as a G9 proof.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT / "src", ROOT / "tests"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

# ``real_streamlit`` is autouse at its definition, so importing it arms it here too: without
# it ``AppTest`` can be a MagicMock left in ``sys.modules`` by an earlier test module.
from test_streamlit_quarantine_boundary import _app_exceptions, real_streamlit  # noqa: E402,F401

APP_PATH = ROOT / "src" / "t2pw" / "app" / "streamlit_app.py"
APP_TIMEOUT = 300.0
DRAWS_ENV = "T2PW_EXTRACTION_DRAWS"
KEY_POST_PIPELINE = "pwml_generate_btn"
KEY_DIAGNOSTICS = "extraction_diagnostics"
KEY_DIAGNOSTICS_DIR = "extraction_diagnostics_dir"
IDENTITY_REPORT = "ensemble_identity_merge_report.json"
AUDIT_SUMMARY = "audit_iteration_summary.json"
GAP_SUMMARY = "gap_iteration_summary.json"
STAGE1_SIDE_FILE = "stage1_ensemble_report.json"
#: Strings only the identity-merge record carries (``_write_identity_side_file``); none may
#: reach a model (D-104 AMENDMENT 2).
RECORD_STRINGS = ("ensemble_identity_merge", "DATABASE-IDENTITY MERGE", "bridged_by", "database-id")

SENTENCE_1 = "Ferrochelatase inserts Fe2+ into protoporphyrin IX to form protoheme."
SENTENCE_2 = "The same enzyme converts protoporphyrin IX and Fe2+ into heme b."
PAPER = f"{SENTENCE_1} {SENTENCE_2}"
CONTEXT = {"pathway_name": "Heme biosynthesis", "likely_organism": "Escherichia coli",
           "scope_clarity_score": 0.9, "key_compounds": ["protoporphyrin IX", "protoheme"]}
DRAW = {
    "entities": {"compounds": [
        {"name": "protoporphyrin IX", "evidence": SENTENCE_1},
        {"name": "Fe2+", "evidence": SENTENCE_1},
        {"name": "protoheme", "evidence": SENTENCE_1},
        {"name": "heme b", "evidence": SENTENCE_2},
    ]},
    "processes": {"reactions": [
        {"name": "protoheme formation", "inputs": ["protoporphyrin IX", "Fe2+"],
         "outputs": ["protoheme"], "evidence": SENTENCE_1},
        {"name": "heme b formation", "inputs": ["protoporphyrin IX", "Fe2+"],
         "outputs": ["heme b"], "evidence": SENTENCE_2},
    ]},
}
CHEBI = {"protoheme": "CHEBI:17627", "heme b": "CHEBI:17627",
         "protoporphyrin IX": "CHEBI:15430", "Fe2+": "CHEBI:29033"}


class _Wire:
    """``t2pw.llm.client._client`` stand-in: records every request and answers it."""

    def __init__(self, prompts: Dict[str, str]) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.prompts = prompts
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        messages = kwargs.get("messages") or [{}]
        system = str(messages[0].get("content") or "")
        content = {self.prompts["preprocess"]: json.dumps(CONTEXT),
                   self.prompts["extraction"]: json.dumps(DRAW)}.get(system, "{}")
        message = types.SimpleNamespace(content=content, tool_calls=None)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], usage=None)


def _resolve_compound(*, name: str, **_: Any) -> Dict[str, Any]:
    """``map_ids._map_compound_with_strategy`` from a table: a confident, admitted ChEBI id."""

    chebi = CHEBI.get(name)
    if chebi is None:
        return {"status": "unmapped", "reason": "no_match", "candidates": []}
    return {"status": "mapped", "mapped_ids": {"chebi": chebi}, "confidence": 1.0, "provider": "fake",
            "source": "api", "chosen_rule": "exact_name",
            "candidates": [{"name": name, "mapped_ids": {"chebi": chebi}}], "resolution": {"status": "matched"}}


def _copy_enrichment(input_path: Any, output_path: Any, report_path: Any, **_: Any) -> Dict[str, Any]:
    """Stage 7, offline: it reaches UniProt over the network and changes no gate outcome."""

    report = {"summary": {"enrichment_skipped_for_test": True}}
    Path(output_path).write_text(Path(input_path).read_text(encoding="utf-8"), encoding="utf-8")
    Path(report_path).write_text(json.dumps(report), encoding="utf-8")
    return report


@pytest.fixture()
def offline_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, Any]:
    """Everything that would leave the machine or touch the checkout, redirected."""

    import requests

    import t2pw.config  # noqa: F401 - bind the real PROJECT_ROOT before it is redirected
    import t2pw.llm.client as client
    import t2pw.mapping.enrich_entities as enrich_entities
    import t2pw.mapping.map_ids as map_ids
    import t2pw.paths as paths
    import t2pw.pipeline.pipeline as pipeline_module
    import t2pw.pwml.writer  # noqa: F401 - reads reference data under the real root
    import t2pw.sbml.render_pathwhiz_like as render_module
    from t2pw.paths import PROMPTS_DIR

    def _offline(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network call during an offline test")

    for name in ("RAG_ENABLED", "RAG_ELIGIBILITY_ENABLED"):
        monkeypatch.setenv(name, "0")
    for name in ("T2PW_SPECIES_LLM", "T2PW_SPECIES_NCBI"):
        monkeypatch.setenv(name, "0")
    monkeypatch.delenv("T2PW_OFFLINE_CURATOR", raising=False)  # the curator's request is captured too
    monkeypatch.setenv("LLM_CALL_SPACING", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "0")
    wire = _Wire({"preprocess": (PROMPTS_DIR / "preprocess_system.txt").read_text(encoding="utf-8"),
                  "extraction": (PROMPTS_DIR / "pwml_system.txt").read_text(encoding="utf-8")})
    monkeypatch.setattr(client, "_client", wire)
    monkeypatch.setattr(client, "_model", "orch759b-test-model")
    monkeypatch.setattr(client, "PROVIDER", "local")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_module, "TMP_DIR", tmp_path / "tmp")
    monkeypatch.setattr(render_module, "TMP_DIR", tmp_path / "tmp")
    monkeypatch.setattr(map_ids, "_map_compound_with_strategy", _resolve_compound)
    monkeypatch.setattr(map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None))
    monkeypatch.setattr(map_ids, "_ai_protein_synonym_lookup", lambda *_a, **_k: [])
    monkeypatch.setattr(map_ids.HttpClient, "get", _offline)
    monkeypatch.setattr(requests.Session, "request", _offline)
    monkeypatch.setattr(enrich_entities, "run_enrichment", _copy_enrichment)
    return {"wire": wire, "root": tmp_path}


def _app():
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - after real_streamlit

    at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    at.run()
    return at


def _run_pipeline(at: Any, monkeypatch: pytest.MonkeyPatch, draws: str) -> Path:
    """Paste the paper and click "Run pipeline" at ``draws`` draws. Returns Stage 1's artifact dir."""

    monkeypatch.setenv(DRAWS_ENV, draws)
    at.text_area(key="pathway_text_0").set_value(PAPER)
    next(box for box in at.checkbox if box.label == "Run inference/enrichment stage").uncheck()
    next(button for button in at.button if button.label == "Run pipeline").click().run()
    assert _app_exceptions(at) == []
    assert at.session_state["pipeline_ready"], [str(e.value) for e in at.error]
    return Path(at.session_state[KEY_DIAGNOSTICS_DIR])


def _run_post_pipeline(at: Any, monkeypatch: pytest.MonkeyPatch, draws: str = "3") -> Dict[str, Any]:
    """Click "Run audit and DB mapping" in a new script run, with the operator's ``.env`` count."""

    monkeypatch.setenv(DRAWS_ENV, draws)
    at.button(key=KEY_POST_PIPELINE).click().run()
    assert _app_exceptions(at) == []
    assert not [e for e in at.error if "Post-pipeline conversion failed" in str(e.value)], \
        [str(e.value) for e in at.error]
    return dict(at.session_state["post_pipeline_artifacts"])


def _reactions(payload: Any) -> List[str]:
    return sorted(str(r.get("name")) for r in ((payload or {}).get("processes") or {}).get("reactions") or [])


def _written(root: Path, name: str) -> List[Path]:
    return sorted((root / "tmp" / "extraction_diagnostics").glob(f"**/{name}"))


# ---------------------------------------------------------------- G9


def test_g9_rule3_runs_at_stage6_and_files_its_record_beside_stage1(
    offline_app: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app()
    stage1_dir = _run_pipeline(at, monkeypatch, "3")
    assert (stage1_dir / STAGE1_SIDE_FILE).is_file(), "Stage 1 did not pool: the premise is wrong"
    assert not (stage1_dir / IDENTITY_REPORT).exists()

    artifacts = _run_post_pipeline(at, monkeypatch)

    # What came into Stage 6 still has both rows; what came out has one (rule 3, D-104 § 3(c)).
    assert _reactions(artifacts["final_audited"]) == ["heme b formation", "protoheme formation"]
    assert _reactions(artifacts["final_mapped_db"]) == ["protoheme formation"]
    [record] = json.loads((stage1_dir / IDENTITY_REPORT).read_text(encoding="utf-8"))
    assert [(m["survivor"], m["merged"], m["bridged_by"]) for m in record["merges"]] == [
        ("protoheme formation", "heme b formation", "database-id")]
    assert "DATABASE-IDENTITY MERGE" in (stage1_dir / "stage1_ensemble_report.txt").read_text(encoding="utf-8")
    # Filed in THIS run's directory and nowhere else.
    assert _written(offline_app["root"], IDENTITY_REPORT) == [stage1_dir / IDENTITY_REPORT]


def test_g9_audit_and_gap_rounds_reach_the_stage1_directory_and_the_driver(
    offline_app: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app()
    stage1_dir = _run_pipeline(at, monkeypatch, "3")
    artifacts = _run_post_pipeline(at, monkeypatch)

    rounds = len(artifacts["audit_iterations"])
    assert rounds >= 1 and len(artifacts["gap_resolution_iterations"]) == rounds
    for name in (AUDIT_SUMMARY, GAP_SUMMARY):
        on_disk = json.loads((stage1_dir / name).read_text(encoding="utf-8"))
        assert on_disk["round_count"] == rounds, name
    # The hand-off ``t2pw.batch.driver`` copies into the leg's run directory keeps Stage 1's
    # boundaries AND gains the rounds, instead of being replaced by a detached recorder's {}.
    published = at.session_state[KEY_DIAGNOSTICS]
    boundaries = published["extraction_boundary_report.json"]["boundaries"]
    assert any(b.get("boundary") == "stage1_ensemble" for b in boundaries)
    run_id = published["extraction_boundary_report.json"]["run_id"]
    assert run_id and published[AUDIT_SUMMARY]["run_id"] == published[GAP_SUMMARY]["run_id"] == run_id
    assert published[AUDIT_SUMMARY]["round_count"] == rounds
    assert published[GAP_SUMMARY]["round_count"] == rounds


def test_g9_a_new_pipeline_run_replaces_the_stored_recorder(
    offline_app: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run 1 pools 3 draws, run 2 in the same session draws once. The post-pipeline run must
    use run 2's recorder: its rounds land in run 2's directory, and rule 3 stays off, because
    run 2 did not pool -- even with the operator's ``.env`` saying 3."""

    at = _app()
    first = _run_pipeline(at, monkeypatch, "3")
    second = _run_pipeline(at, monkeypatch, "1")
    assert first != second
    artifacts = _run_post_pipeline(at, monkeypatch, "3")

    assert _reactions(artifacts["final_mapped_db"]) == ["heme b formation", "protoheme formation"]
    assert _written(offline_app["root"], IDENTITY_REPORT) == []
    assert (second / AUDIT_SUMMARY).is_file()
    assert not (first / AUDIT_SUMMARY).exists()


# ---------------------------------------------------------------- preservation


def test_still_one_draw_never_runs_rule3(offline_app: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """D-104 § 2: at one draw nothing pools, so Stage 6 merges nothing, even under an ``.env``
    that says 3 when the post-pipeline button is clicked."""

    at = _app()
    stage1_dir = _run_pipeline(at, monkeypatch, "1")
    assert not (stage1_dir / STAGE1_SIDE_FILE).exists()
    artifacts = _run_post_pipeline(at, monkeypatch, "3")

    assert _reactions(artifacts["final_mapped_db"]) == ["heme b formation", "protoheme formation"]
    assert _written(offline_app["root"], IDENTITY_REPORT) == []


def test_still_no_stage1_run_in_the_session_means_todays_post_pipeline(
    offline_app: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session that never ran Stage 1 (a seeded payload, as the boundary suite seeds one):
    no recorder is stored, so the post-pipeline run records into a detached recorder exactly as
    at base -- nothing merges, nothing is written under ``tmp/extraction_diagnostics``, and what
    is published is that detached recorder's rounds, with no Stage-1 boundary and no run id."""

    at = _app()
    payload = json.loads(json.dumps(DRAW))
    payload["metadata"] = {"pathway_name": CONTEXT["pathway_name"], "organism": CONTEXT["likely_organism"]}
    for key, value in {"pipeline_ready": True, "stage_one": payload, "stage_two": None, "stage_two_chunks": [],
                       "stage_two_rounds": [], "chunk_details": [], "qa_hints": None,
                       "final_payload": payload, "final_payload_snapshot": payload,
                       "pipeline_source_text": PAPER, "pathway_context": dict(CONTEXT)}.items():
        at.session_state[key] = value
    at.run()
    artifacts = _run_post_pipeline(at, monkeypatch, "3")

    assert _reactions(artifacts["final_mapped_db"]) == ["heme b formation", "protoheme formation"]
    assert not (offline_app["root"] / "tmp" / "extraction_diagnostics").exists()
    published = at.session_state[KEY_DIAGNOSTICS]
    assert sorted(published) == [AUDIT_SUMMARY, GAP_SUMMARY]
    assert [published[name]["run_id"] for name in sorted(published)] == ["", ""]


# ---------------------------------------------------------------- D-104 AMENDMENT 2


def _request_digest(call: Dict[str, Any]) -> str:
    kept = {k: v for k, v in call.items() if k not in ("max_tokens", "timeout")}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, default=repr).encode("utf-8")).hexdigest()


#: sha256 of every request, in order, minus ``max_tokens``/``timeout``, measured on base
#: ``4f750030`` with this file's fixture: Stage 0, three Stage-1 draws, then the post-pipeline.
GOLDEN: Dict[str, List[str]] = {
    "pipeline_run": [
        "98824bc06dbc3f79b9b6ac30671f71eb473ccab83e653556de8380958e0ba05f",
        "16d8bc5eb5c243934c6d5d3254b26d30801cceacc84d42ee205f9f3204bf05ae",
        "16d8bc5eb5c243934c6d5d3254b26d30801cceacc84d42ee205f9f3204bf05ae",
        "16d8bc5eb5c243934c6d5d3254b26d30801cceacc84d42ee205f9f3204bf05ae",
    ],
    "post_pipeline": [
        "99ed836fd78cb0a7abc579a8ec0b9dca0b1ea65b469ea0caa9759cbcd911d6d8",
        "eb95ccbc7e9f191967f1c3b5ac9bf88910fe69b0d9e1411e7a393bc4e532d61e",
        "05c6e15f8bac5aeb5b74ceabe2549ec947cc903aa3bd23f8e57c462d588de313",
        "718aba4d196b99f6e9026b97686736e9d23f613e4d05df9098665283e5d692e0",
        "f6a3746772c56898e1219acf80471b0121fa5ff1e17ae7b02c2ba856e0a2b805",
        "a18b7dddd242a7566feabf6622649ae0ca9b75e8498fb7d44531dea55e7c509a",
        "980588f3721636bd7439a096788338edb286b56e12e8816dd5ce97adc0e37708",
        "83885a7cd8bebd48973bfabb4e7b9687c5631b9a4a128a1dcc780fd6404ea101",
        "d771b2a8777929ad6f70962b86e240cad353cb374b94eb134f110d1fa571a503",
        "427c214fdcb685a747a71dfa927631bb85c3a224bbe69bb418e8294d9d3877b3",
        "9ce4caeb7c955113f28a0bcdae68c4a4177744e62d98bcb7bfd99d2d0b8d158a",
    ],
}


def test_llm_inputs_are_the_bases_bytes_and_never_carry_the_merge_record(
    offline_app: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app()
    wire = offline_app["wire"]
    _run_pipeline(at, monkeypatch, "3")
    stage1 = [_request_digest(c) for c in wire.calls]
    artifacts = _run_post_pipeline(at, monkeypatch)
    post = [_request_digest(c) for c in wire.calls[len(stage1):]]

    measured = {"pipeline_run": stage1, "post_pipeline": post}
    assert measured == GOLDEN, json.dumps(measured, indent=1)
    # Nor does it reach the texts later model calls read: the mapping report
    # (``compact_mapping_misses`` -> the interactive curator) and the reaction summary
    # (the curator and the enrichment agent).
    for text in (json.dumps(wire.calls, default=repr), json.dumps(artifacts["mapping_report"], default=repr),
                 str(at.session_state["reaction_summary"])):
        assert [s for s in RECORD_STRINGS if s in text] == []


# ---------------------------------------------------------------- the helper itself


def test_new_reenter_crosses_threads_and_drops_anything_else() -> None:
    """NEW ACCEPTANCE TEST, labelled as such: ``extraction_diagnostics.reenter`` is new in this
    lane, so this is not a G9 proof (at base it fails on the missing name, which proves nothing).

    Each Streamlit script run, live or under AppTest, is its own ``ScriptRunner`` thread. A new
    thread sees no recorder; ``reenter`` hands it the earlier run's; and anything that is not a
    recorder clears what the thread holds, so a reused thread cannot record into a recorder
    another session left behind."""

    import threading

    import t2pw.pipeline.extraction_diagnostics as xd

    stage1 = xd.activate(run_id="orch759b-stage1")
    seen: Dict[str, bool] = {}

    def post_pipeline_run() -> None:
        seen["starts_detached"] = xd.current() is not stage1
        xd.reenter(stage1)
        seen["sees_stage1"] = xd.current() is stage1
        for stale in (None, {}, "orch759b-stage1"):
            xd.reenter(stage1)
            xd.reenter(stale)
            seen[f"drops_{stale!r}"] = xd.current() is not stage1 and xd.current().artifact_dir is None

    try:
        worker = threading.Thread(target=post_pipeline_run)
        worker.start()
        worker.join()
    finally:
        xd.deactivate()
    assert seen == {"starts_detached": True, "sees_stage1": True, "drops_None": True, "drops_{}": True,
                    "drops_'orch759b-stage1'": True}
