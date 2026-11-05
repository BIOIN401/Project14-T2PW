"""App-vs-service equivalence harness (R2.5, new capability). Test-only; no ``src`` change.

* :func:`run_app_arm` drives the real Streamlit app through ``t2pw.batch.driver.run_one`` -- the batch
  driver's own widget path (load, strict mode, paste, "Run pipeline", "Run audit and DB mapping",
  "Generate PWML") -- and captures the ``AppTest`` it built, the canonical payload hash and the PWML bytes.
  Two spies on ``t2pw.service.postpipeline`` record the exact arguments the app handed to
  ``run_post_pipeline_sbml_artifacts`` and ``run_pwml_export``; they call straight through.
* :func:`run_service_arm` starts from the Stage-2 payload the app reached and calls the SERVICE functions
  directly, as the app does. ``# TODO(R3)`` marks where this becomes ``run_extraction``/``run_pathway``.
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

FIXTURE = ROOT / "tests" / "fixtures" / "equivalence" / "heme_ferrochelatase_strict.json.gz"
#: Copied into every isolated root: the app's PWML exporter reads its reference model from
#: ``PROJECT_ROOT/reference`` (``_pwml_reference_path``).
REFERENCE_FILES = ("PW000001.pwml", "PW012926.pwml")
#: The case: tests/test_orch759b_recorder_reattach.py ``PAPER`` (SENTENCE_1 + SENTENCE_2), verbatim.
CASE = {
    "paper_id": "equivalence-heme-ferrochelatase",
    "text": "Ferrochelatase inserts Fe2+ into protoporphyrin IX to form protoheme. "
            "The same enzyme converts protoporphyrin IX and Fe2+ into heme b.",
    "mode": "strict",
    "source": "tests/test_orch759b_recorder_reattach.py::PAPER",
}


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


def _pwml_files(root: Path) -> Dict[str, str]:
    return {p.name: sha256(p.read_bytes()) for p in sorted((root / "outputs").glob("*.pwml"))}


def run_app_arm(root: Path, cassette: llm_replay.Cassette, timeout: float) -> Dict[str, Any]:
    from streamlit.testing.v1 import AppTest

    from t2pw.batch import driver
    from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY
    from t2pw.service import postpipeline

    new_root(root)
    captured: Dict[str, Any] = {}
    with pytest.MonkeyPatch.context() as mp:
        cassette.install(mp)
        isolate(mp, root)
        real_from_file = AppTest.from_file
        real_post = postpipeline.run_post_pipeline_sbml_artifacts
        real_export = postpipeline.run_pwml_export

        def from_file(*a: Any, **k: Any) -> Any:
            captured["at"] = real_from_file(*a, **k)
            return captured["at"]

        def post_spy(*a: Any, **k: Any) -> Any:
            captured["post_call"] = deepcopy((a, k))
            snapshot = root.with_name(root.name + ".at_post_pipeline")
            shutil.copytree(root, snapshot)  # the on-disk state the app's post-pipeline starts from
            captured["snapshot"] = snapshot
            return real_post(*a, **k)

        def export_spy(*a: Any, **k: Any) -> Any:
            captured["export_call"] = deepcopy((a, k))
            return real_export(*a, **k)

        mp.setattr(AppTest, "from_file", from_file)
        mp.setattr(postpipeline, "run_post_pipeline_sbml_artifacts", post_spy)
        mp.setattr(postpipeline, "run_pwml_export", export_spy)
        started = time.monotonic()
        outcome = driver.run_one({"paper_id": CASE["paper_id"], "text": CASE["text"]}, CASE["mode"], timeout=timeout)
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
        "stage2_payload": deepcopy(driver._ss(at, "final_payload")) if at is not None else None,
        "pathway_context": deepcopy(driver._ss(at, "pathway_context")) if at is not None else None,
        "post_call": captured.get("post_call"),
        "export_call": captured.get("export_call"),
        "snapshot": captured.get("snapshot"),
        "misses": list(cassette.misses),
        "refused": list(cassette.refused),
        "unused": cassette.unused() if cassette.mode == "replay" else [],
    }


def run_service_arm(app: Dict[str, Any], root: Path, cassette: llm_replay.Cassette) -> Dict[str, Any]:
    """The post-pipeline half, called directly on the service module, from the app's Stage-2 payload.

    TODO(R3): once ``run_extraction(request, reporter, state)`` exists, start this arm from ``CASE["text"]``
    instead of ``app["post_call"]``'s payload; from R5 call ``run_pathway(PathwayRequest(...))`` and compare
    its ``PathwayResult`` (PWML path bytes, canonical hash) to the app arm the same way.
    """

    from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY
    from t2pw.service import postpipeline

    shutil.copytree(app["snapshot"], root)
    (post_args, post_kwargs) = deepcopy(app["post_call"])
    (export_args, export_kwargs) = deepcopy(app["export_call"])
    with pytest.MonkeyPatch.context() as mp:
        cassette.install(mp)
        isolate(mp, root)
        started = time.monotonic()
        # As the "Run audit and DB mapping" handler: the Stage-2 payload (after propagate_context_organism),
        # the sidebar's settings, and this root's outputs directory.
        post_kwargs["outputs_dir"] = root / "outputs"
        artifacts = postpipeline.run_post_pipeline_sbml_artifacts(*post_args, **post_kwargs)
        # As "Generate PWML": refinement review starts from deepcopy(canonical payload)
        # (initialize_refinement_review_state); the boundary's decision rides along
        # (quarantine_session_state); the reference model comes from this root.
        export_kwargs.update(
            project_root=root,
            ref_path=root / "reference" / REFERENCE_FILES[0],
            quarantine_report=artifacts.get("quarantine_report") or {},
            quarantine_artifacts=artifacts.get("quarantine_artifacts") or {},
        )
        result = postpipeline.run_pwml_export(deepcopy(artifacts.get(CANONICAL_PAYLOAD_KEY)),
                                              *export_args[1:], **export_kwargs)
        seconds = time.monotonic() - started
    from t2pw.batch import driver

    return {
        "seconds": round(seconds, 1),
        "ok": bool(result.get("ok")),
        "error": result.get("error"),
        "canonical_payload_sha256": artifacts.get("canonical_payload_sha256"),
        "canonical_payload": deepcopy(artifacts.get(CANONICAL_PAYLOAD_KEY)),
        "xml": driver._xml_bytes(result),
        "pwml_files": _pwml_files(root),
        "misses": list(cassette.misses),
        "refused": list(cassette.refused),
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
MAX_SEGMENTS = 8


def record(env_file: Path, work: Path, timeout: float) -> int:
    """One segment of the live recording; writes :data:`FIXTURE` if this segment's run reaches a PWML."""

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
        app = run_app_arm(work / f"record-{n:02d}", cassette, timeout)
        assert checkout_state() == before, "the recording run wrote into the checkout"
    finally:
        mp.undo()
    summary = {k: app[k] for k in ("outcome", "seconds", "canonical_payload_sha256", "pwml_files")}
    summary.update(pwml_sha256=sha256(app["xml"]) if app["xml"] else None, pwml_bytes=len(app["xml"]),
                   counts=cassette.counts(), live_llm_errors=cassette.live_llm_errors, segment=n,
                   served_from_prior=len(prior._served) if prior else 0)
    print(json.dumps(summary, indent=1))
    if not app["xml"] or app["outcome"]["status"] != "pass":
        print("NOT WRITING THE FIXTURE: the run did not reach a strict PWML export")
        return 2
    models = sorted({row["response"].get("model") for row in cassette.entries["llm"]})
    fixture = {
        "schema": 1,
        "case": CASE,
        "env": pinned,
        "keying": llm_replay.KEYING,
        "recorded": {**summary, "models": models, "tree_head": _git("rev-parse", "HEAD"),
                     "tree_dirty": bool(_git("status", "--porcelain", "--", "src", "tests/equivalence")),
                     "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "entries": cassette.entries,
    }
    text = json.dumps(fixture, indent=1, sort_keys=True, ensure_ascii=False)
    llm_replay.assert_no_secret_values(text, secrets)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes gz, {len(text)} bytes json)")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--env-file", type=Path, required=True)
    rec.add_argument("--work", type=Path, required=True)
    rec.add_argument("--timeout", type=float, default=3600.0, help="driver budget; the wrapper bounds wall time")
    args = parser.parse_args(argv)
    return record(args.env_file, args.work, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
