"""App-vs-service equivalence harness (R2.5, new capability). Test-only; no ``src`` change.

* :func:`run_app_arm` drives the real Streamlit app through ``t2pw.batch.driver.run_one`` -- the batch
  driver's own widget path (load, strict mode, paste, "Run pipeline", "Run audit and DB mapping",
  "Generate PWML") -- and captures the ``AppTest`` it built, the canonical payload hash and the PWML bytes.
  Two spies on ``t2pw.service.postpipeline`` record the exact arguments the app handed to
  ``run_post_pipeline_sbml_artifacts`` and ``run_pwml_export``; they call straight through.
* :func:`run_service_arm` starts from the case's text and calls the SERVICE functions directly, as the app
  does: ``run_extraction`` (R3d), then the post-pipeline and the export. ``# TODO(R5)`` marks where this
  becomes ``run_pathway``.
* Every arm runs in its own project root (:func:`isolate`): ``t2pw.paths.PROJECT_ROOT``/``TMP_DIR`` and
  every loaded ``t2pw`` module's binding of them point at it, and the cwd is it. The fixed
  ``outputs/pathway.pwml``, ``tmp/``, ``out/`` and the ``data/`` mapping and enrichment caches (which start
  EMPTY, so every lookup is a recorded call) therefore never touch the checkout.

Record (live LLM, live HTTP, live PathBank DB; through the bounded wrapper, heavy lock). Repeat the same
command until it writes the fixture; each call is one segment (see :data:`MAX_SEGMENTS`):

    python tests/equivalence/harness.py record --env-file <primary .env> --work <scratch dir>
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from equivalence import llm_replay  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "equivalence" / "sour_taste_v2_strict.json.gz"
#: Copied into every isolated root: the app's PWML exporter reads its reference model from
#: ``PROJECT_ROOT/reference`` (``_pwml_reference_path``).
REFERENCE_FILES = ("PW000001.pwml", "PW012926.pwml")
#: The case, verbatim from docs/evaluation_v2/paper_runs_orch755/topics/sour_taste_v2.txt (a pinned id with
#: scope). Committed evidence that strict mode exported a PWML from it: docs/pwml_recovery_sprint/
#: ORCH-767-RESULT.md (sour taste: PWML yes, review_required) and tests/fixtures/orch768 (the
#: ORCH-767/sour_taste_v2-strict leg wrote pathway.review_required.pwml; test_orch768_replay_corpus.py pins
#: that its replay at this tree still writes it). Its full text is fetched ONCE at record time, exactly as
#: the batch runner fetches it (``t2pw.batch.fetch.fetch_papers``), and stored in the fixture.
TOPIC_LINE = "PMC7059786 | Taste transduction: sour | Mus musculus"
MODE = "strict"
#: The equivalence test runs at a NON-DEFAULT "Audit repair timeout": the sidebar's maximum, 1800 s, instead
#: of 240 s, identically when recording and replaying. ``run_post_pipeline_sbml_artifacts`` branches on wall
#: clock time against this budget (postpipeline.py:920/940/976/1131), so at 240 s the audit round a run
#: stops at depends on live LLM latency and a replay (which is fast) takes a different path (R2.5 sec. 8).
#: At 1800 s the loop ends by its own data-driven conditions for any audit that finishes in under ~28 min.
AUDIT_TIMEOUT_SECONDS = 1800
AUDIT_TIMEOUT_KEY = "post_audit_timeout_seconds"
#: The one instant every ``retrieved_at`` stamp reads during a replay (PO ruling 2026-11-05, option (a)).
#: ``t2pw.mapping.enrich_entities._utc_now_iso`` stamps wall-clock UTC seconds into enrichment provenance
#: (lines 502-1173), and the canonical payload hash covers them, so without this no two runs -- the live one
#: included -- share a canonical hash. It is the only clock that reaches the canonical payload (R2.5 sec. 12.4).
FROZEN_UTC = "1970-01-01T00:00:00Z"
#: The second recorded case (R26, additive; the first case above is untouched): an open-access paper in
#: RESEARCH mode, so the equivalence check sees Stage 2 additions, the out-of-scope filter and a non-default
#: export mode (R3d review F1). The requested pathway carries the scope sentence that asks the model to label
#: the paper's dolichol-cycle reaction ``out_of_scope`` (a natural hit was 1 in 182 labelled past runs,
#: docs/webserver/results/R26.md sec. 2). It goes through the app's real extraction-focus box.
SECOND_CASE = {
    "fixture": ROOT / "tests" / "fixtures" / "equivalence" / "farnesol_research.json.gz",
    "topic_line": ("PMC12505894 | farnesol biosynthesis (the dolichol cycle is a different pathway: "
                   "extract its dolichyl pyrophosphate dephosphorylation but label it out_of_scope) "
                   "| Candida albicans"),
    "mode": "research",
    "source": "docs/webserver/results/R26.md",
}
CASES = {
    "sour_taste_v2": {"fixture": FIXTURE, "topic_line": TOPIC_LINE, "mode": MODE,
                      "source": "docs/evaluation_v2/paper_runs_orch755/topics/sour_taste_v2.txt"},
    "farnesol_research": SECOND_CASE,
}
#: Keys the extraction publishes that the app and the service must agree on (R26): what the mutations of R3d
#: review F1 change, compared directly, not only through the canonical payload they feed.
PUBLISHED_KEYS = ("export_mode", "stage_one", "stage_two", "final_payload", "out_of_scope_removed_reactions")


def freeze_clock(mp: pytest.MonkeyPatch) -> int:
    """Point every binding of ``enrich_entities._utc_now_iso`` at :data:`FROZEN_UTC`. Returns how many.

    Every stamping site resolves the module global at call time; the scan also catches any module that
    bound the function by name (none does today: ``src/enrich_entities.py``'s star import skips ``_`` names).
    """

    import t2pw.mapping.enrich_entities as enrich_entities

    real = enrich_entities._utc_now_iso
    bound = 0
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "_utc_now_iso", None) is real:
            mp.setattr(module, "_utc_now_iso", lambda: FROZEN_UTC)
            bound += 1
    return bound


#: The service functions the app arm observes. One spy per function per PROCESS, recording into whichever
#: arm is current (:data:`_ARM`). ``install_spies`` points every ``sys.modules`` binding of the real function
#: (or of the spy, left behind by an earlier arm) at the spy, the way :func:`freeze_clock` does, so a binding
#: made by ``from t2pw.service.postpipeline import ...`` in a module that stays loaded is reached too, not
#: only the attribute the app script re-imports on every AppTest run (review F1).
SPIED = ("run_post_pipeline_sbml_artifacts", "run_pwml_export")
_SPIES: Dict[Any, Any] = {}  # real function -> its spy
_ARM: Dict[str, Any] = {"current": None}  # the arm's capture dict while an arm runs, else None


def _on_call(name: str, args: Any, kwargs: Any) -> None:
    arm = _ARM["current"]
    if arm is None:
        return
    if name == "run_post_pipeline_sbml_artifacts" and "post_call" not in arm:
        arm["post_call"] = deepcopy((args, kwargs))
        snapshot = arm["root"].with_name(arm["root"].name + ".at_post_pipeline")
        shutil.copytree(arm["root"], snapshot)  # the on-disk state the app's post-pipeline starts from
        arm["snapshot"] = snapshot
    elif name == "run_pwml_export" and "export_call" not in arm:
        arm["export_call"] = deepcopy((args, kwargs))


def _spy(name: str, real: Any) -> Any:
    if real not in _SPIES:
        def spy(*args: Any, **kwargs: Any) -> Any:
            _on_call(name, args, kwargs)
            return real(*args, **kwargs)

        spy.__wrapped__ = real  # type: ignore[attr-defined]
        _SPIES[real] = spy
    return _SPIES[real]


def real_function(fn: Any) -> Any:
    """``fn`` itself, or the real function behind one of our spies."""

    return getattr(fn, "__wrapped__", fn) if fn in _SPIES.values() else fn


def install_spies(mp: pytest.MonkeyPatch, arm: Dict[str, Any], source: Any = None,
                  names: tuple = SPIED) -> list:
    """Make ``arm`` current and bind the spies everywhere. Returns the ``module.name`` bindings patched."""

    if source is None:
        from t2pw.service import postpipeline as source
    mp.setitem(_ARM, "current", arm)
    patched = []
    for name in names:
        real = real_function(getattr(source, name))
        spy = _spy(name, real)
        for mod_name, module in list(sys.modules.items()):
            if module is not None and getattr(module, name, None) in (real, spy):
                mp.setattr(module, name, spy)
                patched.append(f"{mod_name}.{name}")
    return sorted(patched)


def leaves(value: Any, path: str = "") -> Dict[str, Any]:
    """Every leaf of a JSON value by its pointer path, for a field-by-field comparison."""

    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in leaves(item, f"{path}/{key}").items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value) for k, v in leaves(item, f"{path}/{i}").items()}
    return {path: value}


#: The driver fields of a batch paper the app path reads (``_paper_text``, ``_extraction_focus``,
#: ``_requested_scope``, ``_paper_id``).
CASE_FIELDS = ("paper_id", "title", "full_text", "requested_pathway", "requested_organism", "topic")


def fetch_case(cache_dir: Path, spec: Dict[str, Any] = CASES["sour_taste_v2"]) -> Dict[str, Any]:
    from t2pw.batch.fetch import fetch_papers

    topic_line, mode = spec["topic_line"], spec["mode"]
    papers, skipped = fetch_papers(topic_line, limit=1, cache_dir=cache_dir, use_cache=True, screen=False)
    assert papers and papers[0].paper_id == topic_line.split("|")[0].strip(), skipped
    case = {name: str(getattr(papers[0], name, "") or "") for name in CASE_FIELDS}
    case.update(mode=mode, topic_line=topic_line, full_text_sha256=sha256(case["full_text"].encode("utf-8")),
                source=spec["source"])
    return case


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    (path / "reference").mkdir()
    for name in REFERENCE_FILES:
        shutil.copyfile(ROOT / "reference" / name, path / "reference" / name)
    return path


def isolate(mp: pytest.MonkeyPatch, root: Path) -> None:
    import t2pw.paths as paths

    mp.setattr(paths, "PROJECT_ROOT", root)
    mp.setattr(paths, "TMP_DIR", root / "tmp")
    for name, module in list(sys.modules.items()):
        if name.startswith("t2pw") and module is not None and module is not paths:
            for attr, value in (("PROJECT_ROOT", root), ("TMP_DIR", root / "tmp")):
                if hasattr(module, attr):
                    mp.setattr(module, attr, value)
    mp.chdir(root)


def checkout_state() -> Dict[str, Any]:
    """(size, mtime_ns) of the checkout files a leaky run would write. Must not move during any arm."""

    out = {}
    for rel in ("data/id_mapping_cache.json", "data/enrichment_cache.json", "outputs/pathway.pwml",
                "out/enrichment_dump.json", "tmp/draft_graph.json"):
        p = ROOT / rel
        out[rel] = (p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None
    return out


def export_verdict(result: Dict[str, Any]) -> Dict[str, Any]:
    """What ``run_pwml_export`` decided, path-free: ok, error, and the required-field gate's findings."""

    gate = result.get("required_gate_report") if isinstance(result.get("required_gate_report"), dict) else {}
    return {"ok": result.get("ok"), "error": result.get("error"), "issues": result.get("issues"),
            "required_gate_errors": gate.get("errors"), "required_gate_warnings": gate.get("warnings")}


def _pwml_files(root: Path) -> Dict[str, str]:
    return {p.name: sha256(p.read_bytes()) for p in sorted((root / "outputs").glob("*.pwml"))}


def run_app_arm(root: Path, cassette: llm_replay.Cassette, timeout: float, case: Dict[str, Any],
                freeze: bool = True) -> Dict[str, Any]:
    from streamlit.testing.v1 import AppTest

    from t2pw.batch import driver
    from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY

    new_root(root)
    captured: Dict[str, Any] = {}
    with pytest.MonkeyPatch.context() as mp:
        cassette.install(mp)
        isolate(mp, root)
        # Replay only: a recording keeps the real clock (its stamps are the live times).
        captured["frozen_bindings"] = freeze_clock(mp) if freeze and cassette.mode == "replay" else 0
        real_from_file = AppTest.from_file

        def from_file(*a: Any, **k: Any) -> Any:
            captured["at"] = real_from_file(*a, **k)
            captured["at"].session_state[AUDIT_TIMEOUT_KEY] = AUDIT_TIMEOUT_SECONDS  # see AUDIT_TIMEOUT_SECONDS
            return captured["at"]

        captured["root"] = root
        captured["spied"] = install_spies(mp, captured)
        mp.setattr(AppTest, "from_file", from_file)
        started = time.monotonic()
        paper = {name: case[name] for name in CASE_FIELDS}
        outcome = driver.run_one(paper, case["mode"], timeout=timeout)
        seconds = time.monotonic() - started
    at = captured.get("at")
    artifacts = driver._safe_dict(driver._ss(at, "post_pipeline_artifacts")) if at is not None else {}
    pwml_result = driver._safe_dict(driver._ss(at, "pwml_export_result")) if at is not None else {}
    return {
        "outcome": {"status": outcome.status, "failure_kind": outcome.failure_kind, "stage": outcome.stage,
                    "message": outcome.message, "detail": (outcome.detail or "")[:4000]},
        "seconds": round(seconds, 1),
        "canonical_payload_sha256": artifacts.get("canonical_payload_sha256"),
        "canonical_payload": deepcopy(artifacts.get(CANONICAL_PAYLOAD_KEY)),
        "xml": driver._xml_bytes(pwml_result),
        "pwml_files": _pwml_files(root),
        "export_verdict": export_verdict(pwml_result),
        "stage2_payload": deepcopy(driver._ss(at, "final_payload")) if at is not None else None,
        "published": {key: deepcopy(driver._ss(at, key)) for key in PUBLISHED_KEYS} if at is not None else {},
        "pathway_context": deepcopy(driver._ss(at, "pathway_context")) if at is not None else None,
        "post_call": captured.get("post_call"),
        "export_call": captured.get("export_call"),
        "snapshot": captured.get("snapshot"),
        "misses": list(cassette.misses),
        "refused": list(cassette.refused),
        "unused": cassette.unused() if cassette.mode == "replay" else [],
        "frozen_bindings": captured["frozen_bindings"],
        "spied": captured["spied"],
    }


def run_service_arm(app: Dict[str, Any], root: Path, cassette: llm_replay.Cassette,
                    case: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The whole case on the service, from the case's text (R3d): ``run_extraction``, then the post-pipeline
    and the export, called directly on the service modules, as the app does.

    ``run_extraction`` gets the request the app's form makes for this case: the batch driver's text, focus,
    export mode and input mode, and every other setting read off the real form's widgets (the app is loaded,
    never submitted). The post-pipeline and the export get the settings the app handed them
    (``app["post_call"]`` / ``app["export_call"]``), except what the run itself produced, which comes from
    the service's own ``state``, where the app's handlers read it from session state.

    TODO(R5): call ``run_pathway(PathwayRequest(...))`` and compare its ``PathwayResult`` (PWML path bytes,
    canonical hash) to the app arm the same way.
    """

    from streamlit.testing.v1 import AppTest

    from t2pw.batch import driver
    from t2pw.config import rag_config
    from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY
    from t2pw.pipeline.pipeline import propagate_context_organism
    from t2pw.service import postpipeline
    from t2pw.service.extraction import ExtractionRequest, run_extraction
    from t2pw.service.reporter import LoggingReporter

    def failed(error: str, started: Optional[float] = None) -> Dict[str, Any]:
        return {"seconds": round(time.monotonic() - started, 1) if started else 0.0, "ok": False, "error": error,
                "canonical_payload_sha256": None, "canonical_payload": None, "xml": b"", "pwml_files": {},
                "export_verdict": None, "misses": list(cassette.misses), "refused": list(cassette.refused),
                "unused": cassette.unused(), "published": {}}

    if app["post_call"] is None:  # no settings to run with; the test reports the app arm's own failure
        return failed("the app arm never reached the post-pipeline")
    new_root(root)
    case = load_fixture()["case"] if case is None else case
    paper = {name: case[name] for name in CASE_FIELDS}
    (post_args, post_kwargs) = deepcopy(app["post_call"])
    (export_args, export_kwargs) = deepcopy(app["export_call"]) if app["export_call"] else ((), {})
    with pytest.MonkeyPatch.context() as mp:
        cassette.install(mp)
        isolate(mp, root)
        frozen_bindings = freeze_clock(mp)
        # The form's settings, as the driver leaves them: load, choose the export and input modes, re-render.
        form = AppTest.from_file(str(driver.DEFAULT_APP_PATH), default_timeout=120).run()
        form.radio(key=driver.KEY_EXPORT_MODE).set_value(driver._resolve_mode(case["mode"])[1])
        form.radio(key=driver.KEY_INPUT_MODE).set_value(driver.INPUT_MODE_PASTE)
        form.run()
        widget = {w.label: w.value for kind in (form.checkbox, form.toggle, form.number_input, form.slider)
                  for w in kind}
        request = ExtractionRequest(
            text_entries=[driver._paper_text(paper)], pdfs=[], uses_pdf_input=False,
            input_mode=driver.INPUT_MODE_PASTE, pdf_page_range="", pdf_skip_refs=True, pdf_ocr=False,
            user_task_context=driver._extraction_focus(paper), export_mode=form.session_state["export_mode"],
            temperature=widget["LLM temperature"], run_inference=widget["Run inference/enrichment stage"],
            enable_chunking=widget["Enable automatic chunking for long inputs"],
            chunk_size=widget["Chunk size (approx. words)"], chunk_overlap=widget["Chunk overlap (words)"],
            extract_attempts=widget["Stage 1 auto-repair attempts"],
            infer_attempts=widget["Stage 2 auto-repair attempts"], infer_rounds=widget["Stage 2 QA rounds"],
            extract_tokens=widget["Stage 1 max tokens"], infer_tokens=widget["Stage 2 max tokens"],
            rag_incomplete_flag=widget.get("Multi-paper RAG (novel pathway from multiple papers)", False),
        )
        state: Dict[str, Any] = {}
        started = time.monotonic()
        extraction = run_extraction(request, reporter=LoggingReporter(), state=state)
        if extraction.failure is not None:
            return failed(f"run_extraction failed: {extraction.failure}", started)
        # As the "Run audit and DB mapping" handler: the Stage-2 payload after propagate_context_organism, the
        # sidebar's settings, the run's own reports and context, and this root's outputs directory.
        post_kwargs.update(
            qa_report=state.get("qa_report"),
            reaction_summary=state.get("reaction_summary"),
            rag_evidence_context=(state.get("rag_evidence_context") or "") if rag_config()["enabled"] else "",
            pathway_context=postpipeline._safe_dict(state.get("pathway_context")),
            outputs_dir=root / "outputs",
            state=state,
        )
        post_payload = propagate_context_organism(extraction.final_payload, state.get("pathway_context"))
        try:
            artifacts = real_function(postpipeline.run_post_pipeline_sbml_artifacts)(
                post_payload, *post_args[1:], **post_kwargs)
        except Exception as exc:  # noqa: BLE001 -- reported by test (c); the app's handler catches it too
            return failed(f"run_post_pipeline_sbml_artifacts raised {type(exc).__name__}: {str(exc)[:300]}",
                          started)
        # As "Generate PWML": refinement review starts from deepcopy(canonical payload)
        # (initialize_refinement_review_state); the boundary's decision rides along
        # (quarantine_session_state); the reference model comes from this root.
        result: Dict[str, Any] = {"error": "the app arm never reached the PWML export"}
        if app["export_call"]:
            export_kwargs.update(
                project_root=root,
                ref_path=root / "reference" / REFERENCE_FILES[0],
                pathway_context=postpipeline._safe_dict(state.get("pathway_context")),
                quarantine_report=artifacts.get("quarantine_report") or {},
                quarantine_artifacts=artifacts.get("quarantine_artifacts") or {},
            )
            result = real_function(postpipeline.run_pwml_export)(deepcopy(artifacts.get(CANONICAL_PAYLOAD_KEY)),
                                                  *export_args[1:], **export_kwargs)
        seconds = time.monotonic() - started

    return {
        "seconds": round(seconds, 1),
        "ok": bool(result.get("ok")),
        "error": result.get("error"),
        "canonical_payload_sha256": artifacts.get("canonical_payload_sha256"),
        "canonical_payload": deepcopy(artifacts.get(CANONICAL_PAYLOAD_KEY)),
        "xml": driver._xml_bytes(result),
        "pwml_files": _pwml_files(root),
        "export_verdict": export_verdict(result),
        "misses": list(cassette.misses),
        "refused": list(cassette.refused),
        "unused": cassette.unused(),
        "frozen_bindings": frozen_bindings,
        "published": {key: deepcopy(state.get(key)) for key in PUBLISHED_KEYS},
    }


def load_fixture(path: Path = FIXTURE) -> Dict[str, Any]:
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def replay_env(mp: pytest.MonkeyPatch, fixture: Dict[str, Any]) -> None:
    """Pin the fixture's environment for replay. Must precede the first ``t2pw.llm.client`` import."""

    if "t2pw.llm.client" in sys.modules:
        raise RuntimeError("t2pw.llm.client is already imported: run the equivalence test in its own process")
    llm_replay.apply_env(mp, {**fixture["env"], **llm_replay.REPLAY_ONLY_ENV}, llm_replay.REPLAY_PLACEHOLDERS)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


#: A live run of this case outlasts one bounded foreground window (585 s): Stage 2 alone took longer
#: (R2.5 attempt 1). So a recording is made in SEGMENTS: each one is a fresh process and a fresh root that
#: replays everything the previous segment got (seconds) and goes live only beyond it. The fixture is
#: written only by the segment whose run completes, and holds exactly that run's answers.
MAX_SEGMENTS = 20


def record(env_file: Path, work: Path, timeout: float, spec: Dict[str, Any] = CASES["sour_taste_v2"]) -> int:
    """One segment of the live recording; writes the case's fixture if this segment's run reaches its end
    (a strict case: a PWML; a research case: a passing report, because research mode writes no PWML)."""

    from dotenv import dotenv_values

    segments = work / "segments"
    segments.mkdir(parents=True, exist_ok=True)
    done = sorted(segments.glob("seg-*.jsonl"))
    if len(done) >= MAX_SEGMENTS:
        print(f"segment cap ({MAX_SEGMENTS}) reached; not recording")
        return 3
    n = len(done) + 1
    prior = llm_replay.Cassette("replay", llm_replay.load_transcript(done[-1])) if done else None
    values = dotenv_values(env_file)
    secrets = {k: str(v) for k, v in values.items() if k in llm_replay.SECRET_ENV and v}
    pinned = llm_replay.pinned_env_from(values)
    mp = pytest.MonkeyPatch()
    try:
        llm_replay.apply_env(mp, pinned, secrets)
        cassette = llm_replay.Cassette("record", redact=secrets.values(), prior=prior,
                                       sink=segments / f"seg-{n:02d}.jsonl")
        before = checkout_state()
        print(f"segment {n}: prior answers {prior.counts() if prior else {}}", flush=True)
        case_file = work / "case.json"
        if not case_file.exists():  # fetched once, then reused verbatim by every later segment
            case_file.write_text(json.dumps(fetch_case(work / "fetch_cache", spec), ensure_ascii=False), encoding="utf-8")
        case = json.loads(case_file.read_text(encoding="utf-8"))
        print(f"case {case['paper_id']}: {len(case['full_text'])} chars, sha256 {case['full_text_sha256'][:16]}", flush=True)
        app = run_app_arm(work / f"record-{n:02d}", cassette, timeout, case)
        assert checkout_state() == before, "the recording run wrote into the checkout"
    finally:
        mp.undo()
    summary = {k: app[k] for k in ("outcome", "seconds", "canonical_payload_sha256", "pwml_files", "export_verdict")}
    summary.update(pwml_sha256=sha256(app["xml"]) if app["xml"] else None, pwml_bytes=len(app["xml"]),
                   counts=cassette.counts(), live_llm_errors=cassette.live_llm_errors, segment=n,
                   served_from_prior=len(prior._served) if prior else 0)
    print(json.dumps(summary, indent=1))
    if app["outcome"]["status"] != "pass" or (case["mode"] == MODE and not app["xml"]):
        print("NOT WRITING THE FIXTURE: the run did not reach its export step")
        return 2
    if case["mode"] != MODE:  # R26: the live payload the replay is compared against
        summary["canonical_payload"] = app["canonical_payload"]
    write_fixture(cassette.entries, pinned, secrets, summary, case,
                  {"tree_head": _git("rev-parse", "HEAD"),
                   "tree_dirty": bool(_git("status", "--porcelain", "--", "src", "tests/equivalence"))},
                  path=spec["fixture"])
    return 0


def write_fixture(entries: Dict[str, Any], pinned: Dict[str, str], secrets: Dict[str, str],
                  summary: Dict[str, Any], case: Dict[str, Any], provenance: Dict[str, Any],
                  path: Path = FIXTURE) -> None:
    models = sorted({row["response"].get("model") for row in entries["llm"]})
    fixture = {
        "schema": 1,
        "case": case,
        "audit_timeout_seconds": AUDIT_TIMEOUT_SECONDS,
        "env": pinned,
        "keying": llm_replay.KEYING,
        "recorded": {**summary, **provenance, "models": models, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "entries": entries,
    }
    text = json.dumps(fixture, indent=1, sort_keys=True, ensure_ascii=False)
    llm_replay.assert_no_secret_values(text, secrets)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    print(f"wrote {path} ({path.stat().st_size} bytes gz, {len(text)} bytes json)")


def assemble(env_file: Path, transcript: Path, live_log: Path, recorded_head: str) -> int:
    """Write the fixture from a COMPLETED segment's transcript, without any live call: for a completed run
    that did not reach a PWML (R2.5 attempt 2), so the harness can still pin what that case does.

    The summary is the one the live segment printed to ``live_log``; the replay test then checks the fixture
    reproduces it. ``recorded_head`` is the commit the wrapper reported for that segment."""

    from dotenv import dotenv_values

    text = live_log.read_text(encoding="utf-8", errors="replace")
    start = text.index('{\n "outcome"')
    summary = json.loads(text[start:text.index("\nNOT WRITING", start)])
    values = dotenv_values(env_file)
    secrets = {k: str(v) for k, v in values.items() if k in llm_replay.SECRET_ENV and v}
    entries = llm_replay.load_transcript(transcript)
    assert {k: len(v) for k, v in entries.items()} == summary["counts"], "transcript and live summary disagree"
    case = json.loads((transcript.parent.parent / "case.json").read_text(encoding="utf-8"))
    write_fixture(entries, llm_replay.pinned_env_from(values), secrets, summary, case,
                  {"tree_head": recorded_head, "tree_dirty": False,
                   "assembled_from": f"{transcript.parent.parent.name}/{transcript.parent.name}/{transcript.name}"})
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--env-file", type=Path, required=True)
    rec.add_argument("--work", type=Path, required=True)
    rec.add_argument("--timeout", type=float, default=3600.0, help="driver budget; the wrapper bounds wall time")
    rec.add_argument("--case", choices=sorted(CASES), default="sour_taste_v2")
    asm = sub.add_parser("assemble")
    asm.add_argument("--env-file", type=Path, required=True)
    asm.add_argument("--transcript", type=Path, required=True)
    asm.add_argument("--live-log", type=Path, required=True)
    asm.add_argument("--recorded-head", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "assemble":
        return assemble(args.env_file, args.transcript, args.live_log, args.recorded_head)
    return record(args.env_file, args.work, args.timeout, CASES[args.case])


if __name__ == "__main__":
    sys.exit(main())
