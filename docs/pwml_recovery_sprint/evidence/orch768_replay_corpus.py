"""ORCH-768 (D-112 § 1): the frozen replay corpus. Every saved leg, replayed offline from its
latest checkpoint that needs no LLM call, through the production functions of a given source tree.

WHAT A CHECKPOINT IS. Every strict leg that reached Stage 6 has ``final_mapped.json``, written by
``batch.driver._add_identity_artifacts``: the canonical payload when the quarantine boundary passed
(post-enrichment, post-quarantine, the object the final gate report is bound to by
``payload_sha256``), otherwise ``final_mapped_db`` (post-Stage-6 remap, PRE-enrichment). The last LLM
calls of the app's post-pipeline (``streamlit_app.run_post_pipeline_sbml_artifacts``) are the audit
loop, the gap resolver and the curator, all BEFORE the Stage-6 remap; after it only the SBML
overwatch calls an LLM, and nothing it returns reaches the PWML. So:

* ``canonical``  -- ``final_mapped.json`` hashes to the recorded final gate report's
  ``payload_sha256``. Replayed: ``freeze_canonical_payload`` (quarantine again, final pre-export
  gate), the driver's gate verdict, ``initialize_refinement_review_state`` (draft graph + PNG), then
  ``run_pwml_export`` (Stage-8 revalidation, required-field gate, pre-freeze seam, IR, writer).
* ``pre_freeze`` -- it does not (the boundary refused, or no final report exists). Replayed:
  ``run_enrichment`` on a pinned copy of the enrichment cache, ``run_prefreeze_resolution``, then
  everything above.

Stage 6 itself is NOT replayed: it runs ``use_cache=False`` against live UniProt and the PathBank
DB, and no leg recorded those responses. A fix upstream of the checkpoint (Stage 3 normalizer,
Stage 6 mapping) is therefore not measurable on this corpus.

Upstream facts the tail cannot re-derive are carried as recorded: the leg's top-level
``*_contract_report`` objects (``contract_reports.json``), combined with the REPLAYED final gate
report and canonical payload into the artifact set the batch driver judges, by the driver's own
``gate_verdict`` / ``_blocking_reports`` / ``_superseded_contract_reports``.

OFFLINE. Every socket connect and every DNS lookup is refused and counted from before ``t2pw`` is
imported (``Offline``); the count is part of each leg's record. ``probe_offline`` proves the block.
The PathBank DB is therefore unreachable: the pre-freeze seam falls back to the tree's offline name
index (``data/pathwhiz_id_db.json``). Measured (ORCH-768-RESULT): canonical legs still reproduce their
PWML byte for byte; pre_freeze legs reproduce every verdict but not the bytes of the payload the leg
quarantined.

Usage (always through ``bounded_run.py``; ``--src`` names the tree under measurement)::

    python orch768_replay_corpus.py inventory              # rebuild orch768_corpus_manifest.json
    python orch768_replay_corpus.py run --src <tree>/src --out <dir> [--json <file>] [--legs <id> ...]
                                       [--src-sha <sha>]   # only the legs that ran on that tree
    python orch768_replay_corpus.py compare <base.json> <tip.json> [--target <leg id>]   # D-112 § 2
    python orch768_replay_corpus.py fixture --legs <id> ... --out tests/fixtures/orch768

``run`` prints the per-leg table, the PWML pass rate, the replayable count and gold, and writes
``--json`` (byte-identical across runs of the same tree: no clock, no absolute output path).

ORCH-772 (D-114 § 1): each replayed leg also carries ``pwml_coverage``, computed by the MEASURED tree's
``t2pw.app.reaction_evidence`` on the replay's own artifacts (n/a for a tree without it), and ``run``
prints the mean, median and share of legs at or above 75% on both denominators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
import urllib.request
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIMARY = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
MANIFEST = HERE / "orch768_corpus_manifest.json"
FULLTEXT = Path(r"C:/t/eval-fulltext")
GOLD_PAPERS = {"PMC12096016": "enterobactin", "PMC12421875": "menaquinone"}

# The app's PWML-export inputs, exactly as ``_generate_pwml_from_refinement_working_json`` reads them
# from a session the batch driver never edits.
EXPORT_ARGS = dict(pathway_name="Generated Pathway", pathway_description="", pathway_subject="Metabolic",
                   vis_width=3200, vis_height=1400, background_color="#FFFFFF", grounding_dict=None, strict_db=True)
MODE = "pathwhiz"

# ── the corpus: runs whose code SHA is recorded (night.log ``head=``) ─────────────────────
RUNS = [
    # (run id, legs root relative to PRIMARY, night.log relative to PRIMARY)
    ("ORCH-755", "runs_papers/ORCH-755/legs", "runs_papers/ORCH-755/night.log"),
    ("ORCH-756", ".claude/worktrees/orch756/runs_papers/ORCH-756/legs", "runs_papers/ORCH-756/night.log"),
    ("ORCH-759", "runs_papers/ORCH-759/legs", "runs_papers/ORCH-759/night.log"),
    ("ORCH-759p", "runs_papers/ORCH-759p/legs", "runs_papers/ORCH-759p/night.log"),
    ("ORCH-767", ".claude/worktrees/orch767/runs_papers/ORCH-767/legs", "runs_papers/ORCH-767/night.log"),
]
LEG_FILES = ("final_mapped.json", "quarantine_report.json", "final_stage3_gate_report.json", "contract_reports.json",
             "pwml_required_field_gate_report.json", "RESULT.txt", "LEG_TERMINAL.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str, repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          encoding="utf-8", errors="replace").stdout.strip()


def stdlib_payload_sha256(payload: Any) -> str:
    """``gate_reports.payload_sha256``, restated so the inventory imports no t2pw (the replay re-checks
    the kind with the measured tree's own function)."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── inventory (stdlib only; imports no t2pw) ─────────────────────────────────────────────


def night_heads(night_log: Path) -> List[tuple]:
    """``[(start 'YYYY-MM-DD_HHMM', head sha), ...]`` from the runner's start lines."""

    out = []
    for line in night_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):\d{2} .*\bstart\b.* head=([0-9a-f]{7,40})", line)
        if m:
            out.append((f"{m.group(1)}_{m.group(2)}{m.group(3)}", m.group(4)))
    return out


def recorded_verdict(row: Dict[str, Any], leg: Path) -> Dict[str, Any]:
    q = load(leg / "quarantine_report.json") if (leg / "quarantine_report.json").is_file() else None
    g = load(leg / "final_stage3_gate_report.json") if (leg / "final_stage3_gate_report.json").is_file() else None
    rf = leg / "pwml_required_field_gate_report.json"
    rel = row.get("release_status")
    return {
        "status": row.get("status"), "stage": row.get("stage"), "failure_kind": row.get("failure_kind"),
        "message": (row.get("message") or "")[:300], "issue_codes": sorted(set(row.get("issue_codes") or [])),
        "pwml": sorted(p.name for p in leg.glob("*.pwml")),
        "release_status": rel.get("status") if isinstance(rel, dict) else rel,
        "quarantine_ok": None if q is None else q.get("ok"),
        "quarantine_refusal": [] if q is None else sorted(q.get("refusal_reasons") or []),
        "quarantine_release": None if q is None else (q.get("release") or {}).get("status"),
        # The itemised gate and required-field errors are compared at replay time, read from the pinned
        # report files by the measured tree's own functions (``recorded_reports``), never restated here.
        "final_gate_ok": None if g is None else g.get("ok"),
        "required_gate_ok": load(rf).get("ok") if rf.is_file() else None,
    }


def checkpoint_of(leg: Path) -> tuple:
    if not (leg / "final_mapped.json").is_file():
        return None, "no final_mapped.json: the leg ended before Stage 6, in or before an LLM stage"
    q = leg / "quarantine_report.json"
    g = leg / "final_stage3_gate_report.json"
    if not q.is_file():
        return None, "final_mapped.json but no quarantine_report.json: pre-boundary artifact set, no requested context"
    if g.is_file() and load(g).get("payload_sha256") == stdlib_payload_sha256(load(leg / "final_mapped.json")):
        return "canonical", ("final_mapped.json hashes to the final gate report's payload_sha256: it is the "
                             "canonical (enriched, quarantined) payload")
    if g.is_file():
        return "pre_freeze", ("a final gate report exists but final_mapped.json does not hash to it: "
                              "final_mapped.json is final_mapped_db (post-remap, pre-enrichment)")
    return "pre_freeze", "no final gate report: final_mapped.json is final_mapped_db (post-remap, pre-enrichment)"


def reflog_heads(repo: Path) -> List[tuple]:
    """``[(local 'YYYY-MM-DD HH:MM:SS', sha), ...]`` of a checkout's HEAD, oldest first."""

    out = []
    for line in git("reflog", "--date=iso", "--format=%h %gd", repo=repo).splitlines():
        m = re.match(r"(\w+) HEAD@\{(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m:
            out.append((m.group(2), m.group(1)))
    return sorted(out)


def leg_record(base: Dict[str, Any], row: Dict[str, Any], leg: Path, caches: List[tuple]) -> Dict[str, Any]:
    kind, why = checkpoint_of(leg)
    if row["mode"] != "strict":
        kind, why = None, "research mode: produces no PWML by design (not in the PWML corpus)"
    rec: Dict[str, Any] = {**base, "mode": row["mode"], "paper": row["paper_id"],
                           "leg_dir": leg.relative_to(PRIMARY).as_posix(), "checkpoint": kind,
                           "checkpoint_reason": why, "recorded": recorded_verdict(row, leg), "inputs": {}}
    for name in LEG_FILES + tuple(p.name for p in leg.glob("*.pwml")):
        p = leg / name
        if p.is_file():
            rec["inputs"][name] = {"path": p.relative_to(PRIMARY).as_posix(), "sha256": sha256_file(p),
                                   "bytes": p.stat().st_size}
    if kind == "pre_freeze":
        cache, why_c = next((c for c in caches if c[0].is_file()), (None, ""))
        rec["inputs"]["enrichment_cache.json"] = {"path": cache.relative_to(PRIMARY).as_posix(),
                                                  "sha256": sha256_file(cache), "bytes": cache.stat().st_size,
                                                  "why": why_c}
    return rec


def inventory() -> Dict[str, Any]:
    legs: List[Dict[str, Any]] = []
    trees: Dict[str, str] = {}
    for run_id, legs_root, night in RUNS:
        heads = night_heads(PRIMARY / night)
        stamps = sorted((d for d in (PRIMARY / legs_root).glob("*/*") if d.is_dir()), key=lambda d: d.name)
        for idx, stamp_dir in enumerate(stamps):
            prior = [h for start, h in heads if start <= stamp_dir.name]
            head = prior[-1] if prior else None
            src_tree = git("rev-parse", f"{head}:src") if head else None
            if src_tree and src_tree not in trees:
                trees[src_tree] = head
            man = stamp_dir / "manifest.jsonl"
            rows = [json.loads(x) for x in man.read_text(encoding="utf-8").splitlines()] if man.is_file() else []
            base = {"tier": "A", "run": run_id, "condition": stamp_dir.parent.name, "stamp": stamp_dir.name,
                    "id": f"{run_id}/{stamp_dir.parent.name}/{stamp_dir.name}", "head": head,
                    "head_source": f"{night} start line <= {stamp_dir.name}", "src_tree": src_tree,
                    "src_sha": trees.get(src_tree)}
            if not rows:
                legs.append({**base, "mode": None, "paper": None, "checkpoint": None,
                             "checkpoint_reason": "no manifest row: the leg was stopped before it reported (aborted)",
                             "recorded": None, "inputs": {}})
                continue
            # The enrichment cache as it stood when the leg ENDED: the next leg's start snapshot (the
            # runner snapshots data/ at each leg start), else this leg's own.
            nxt = stamps[idx + 1] if idx + 1 < len(stamps) else None
            caches = ([(nxt / "cache_snapshot/enrichment_cache.json", f"next leg's start snapshot ({nxt.parent.name}/{nxt.name})")]
                      if nxt else []) + [(stamp_dir / "cache_snapshot/enrichment_cache.json", "this leg's own start snapshot")]
            for row in rows:
                legs.append(leg_record(base, row, stamp_dir / "papers" / row["paper_id"] / row["mode"], caches))
    # Tier B: runs_verify. No SHA is recorded in those runs; the primary's HEAD at plan.json created_at is
    # read from its reflog. The working tree's state at that moment is NOT recorded.
    heads_b = reflog_heads(PRIMARY)
    runs_b = sorted(d for d in (PRIMARY / "runs_verify").glob("2026-*") if (d / "manifest.jsonl").is_file())
    for idx, run in enumerate(runs_b):
        created = load(run / "plan.json")["created_at"].replace("T", " ")
        prior = [h for t, h in heads_b if t <= created]
        head = git("rev-parse", "--short=8", prior[-1]) if prior else None
        src_tree = git("rev-parse", f"{head}:src") if head else None
        if src_tree and src_tree not in trees:
            trees[src_tree] = head
        base = {"tier": "B", "run": f"runs_verify/{run.name}", "stamp": run.name, "head": head,
                "head_source": f"primary reflog: HEAD at plan.json created_at {created} (working tree not recorded)",
                "src_tree": src_tree, "src_sha": trees.get(src_tree)}
        nxt = runs_b[idx + 1] if idx + 1 < len(runs_b) else None
        caches = ([(nxt / "cache_snapshot/enrichment_cache.json", f"next run's start snapshot ({nxt.name})")] if nxt else []) \
            + [(run / "cache_snapshot/enrichment_cache.json", "this run's own start snapshot")]
        for line in (run / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            leg = run / "papers" / row["paper_id"] / row["mode"]
            if row["mode"] != "strict" or not (leg / "final_mapped.json").is_file() or not (leg / "quarantine_report.json").is_file():
                continue  # tier B keeps only legs with a replayable checkpoint ("where usable")
            legs.append(leg_record({**base, "condition": row["paper_id"], "id": f"runs_verify/{run.name}/{row['paper_id']}"},
                                   row, leg, caches))
    fulltext = {pid: {"path": (FULLTEXT / f"{pid}.txt").as_posix(), "sha256": sha256_file(FULLTEXT / f"{pid}.txt")}
                for pid in GOLD_PAPERS if (FULLTEXT / f"{pid}.txt").is_file()}
    return {"schema": 1, "task": "ORCH-768", "decision": "D-112 § 1", "primary": PRIMARY.as_posix(),
            "src_trees": {tree: sha for tree, sha in trees.items()}, "fulltext": fulltext, "legs": legs}


# ── offline: every connect and every lookup is refused and counted ──────────────────────


class Offline:
    refused: List[str] = []
    _saved: Dict[str, Any] = {}

    @classmethod
    def _refuse(cls, *args: Any, **_kw: Any) -> Any:
        target = args[1] if len(args) > 1 and isinstance(args[0], socket.socket) else (args[0] if args else None)
        cls.refused.append(repr(target)[:120])
        raise OSError(f"ORCH-768 replay is offline: refused {target!r}")

    @classmethod
    def install(cls) -> None:
        if cls._saved:
            return
        cls._saved = {"connect": socket.socket.connect, "connect_ex": socket.socket.connect_ex,
                      "create_connection": socket.create_connection, "getaddrinfo": socket.getaddrinfo}
        socket.socket.connect = cls._refuse  # type: ignore[assignment]
        socket.socket.connect_ex = cls._refuse  # type: ignore[assignment]
        socket.create_connection = cls._refuse  # type: ignore[assignment]
        socket.getaddrinfo = cls._refuse  # type: ignore[assignment]

    @classmethod
    def uninstall(cls) -> None:
        if not cls._saved:
            return
        socket.socket.connect = cls._saved["connect"]  # type: ignore[assignment]
        socket.socket.connect_ex = cls._saved["connect_ex"]  # type: ignore[assignment]
        socket.create_connection = cls._saved["create_connection"]  # type: ignore[assignment]
        socket.getaddrinfo = cls._saved["getaddrinfo"]  # type: ignore[assignment]
        cls._saved = {}


def probe_offline() -> Dict[str, Any]:
    """Three real attempts that must each be refused by the block, and counted."""

    before = len(Offline.refused)
    results = {}
    for name, attempt in (
        ("urlopen_uniprot", lambda: urllib.request.urlopen("https://rest.uniprot.org/uniprotkb/P11454.json", timeout=5)),
        ("create_connection_loopback", lambda: socket.create_connection(("127.0.0.1", 9), timeout=1)),
        ("raw_socket_connect", lambda: socket.socket().connect(("1.1.1.1", 443))),
    ):
        try:
            attempt()
            results[name] = "CONNECTED"
        except OSError as exc:
            results[name] = "refused" if "ORCH-768 replay is offline" in str(exc) else f"other:{type(exc).__name__}"
    return {"attempts": results, "counted": len(Offline.refused) - before,
            "ok": all(v == "refused" for v in results.values()) and len(Offline.refused) - before == 3}


# ── the tree under measurement ───────────────────────────────────────────────────────────


def _stub_streamlit() -> None:
    """The established stub (``tests/test_prefreeze_third_export_seam.py``): ``streamlit_app`` runs its
    script at import; the stub neutralises the script and nothing the replayed functions call."""

    if "streamlit" in sys.modules:
        return
    from unittest.mock import MagicMock

    st = MagicMock(name="streamlit")
    st.columns.side_effect = lambda spec=1, **__: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs.side_effect = lambda labels, **__: [MagicMock() for _ in labels]
    st.form_submit_button.return_value = False
    st.button.return_value = False
    st.checkbox.return_value = False
    st.session_state = MagicMock()
    st.session_state.get.return_value = None
    st.session_state.__contains__ = lambda _s, _k: False
    sys.modules["streamlit"] = st


def load_tree(src: Optional[Path]) -> SimpleNamespace:
    """Import the production modules from ``src`` (or from ``sys.path`` as it stands) and prove where
    ``t2pw`` resolved: the venv's editable ``.pth`` names the primary checkout, so an unpinned import
    would silently measure the wrong tree."""

    Offline.install()
    if src is not None:
        sys.path.insert(0, str(Path(src).resolve()))
    _stub_streamlit()
    import t2pw  # noqa: F401

    resolved = Path(t2pw.__file__).resolve().parent
    if src is not None and resolved != (Path(src).resolve() / "t2pw"):
        raise SystemExit(f"ORCH768_WRONG_TREE: t2pw resolved to {resolved}, expected {Path(src).resolve() / 't2pw'}")
    from t2pw.app import streamlit_app as app
    from t2pw.batch import driver
    from t2pw.pipeline import gate_reports
    from t2pw.pipeline import strict_quarantine
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    try:  # ORCH-772: PWML coverage is the measured tree's own Reaction Evidence model, or n/a
        from t2pw.app import reaction_evidence
    except ImportError:
        reaction_evidence = None
    return SimpleNamespace(app=app, driver=driver, gr=gate_reports, sq=strict_quarantine, prefreeze=run_prefreeze_resolution,
                           t2pw=resolved, root=resolved.parents[1], evidence=reaction_evidence)


# ── one leg ──────────────────────────────────────────────────────────────────────────────


def _codes(T: SimpleNamespace, report: Any) -> List[str]:
    return sorted({T.driver._issue_code(e) for e in T.driver._report_errors(report)})


def _gate_errors(report: Any) -> List[List[str]]:
    """``[path, reason]`` per gate error: the itemised verdict, compared exactly."""

    return sorted([str(e.get("path")), str(e.get("reason"))] for e in (report or {}).get("errors") or [] if isinstance(e, dict))


def _tree_digest(work: Path) -> Dict[str, str]:
    """sha256 (16) of every file the replay wrote, except the scratch copy of the enrichment cache.

    The one producer that embeds an output path (``run_enrichment`` records its ``dump_path``) would make
    two runs into two ``--out`` directories differ for that reason alone, so the work directory's own
    spellings are replaced by ``<work>`` before hashing; nothing else is normalised."""

    forms = sorted({str(work), work.as_posix(), json.dumps(str(work))[1:-1]}, key=len, reverse=True)
    out = {}
    for p in sorted(work.rglob("*")):
        if p.is_file() and p.name != "enrichment_cache.json":
            data = p.read_bytes()
            for form in forms:
                data = data.replace(form.encode("utf-8"), b"<work>")
            out[p.relative_to(work).as_posix()] = hashlib.sha256(data).hexdigest()[:16]
    return out


def replay_leg(T: SimpleNamespace, leg: Dict[str, Any], work: Path, primary: Path = PRIMARY) -> Dict[str, Any]:
    """Replay one manifest leg. ``work`` is scratch (created, never read back). Returns the record.

    ``streamlit_app.PROJECT_ROOT`` points at ``work`` for the duration: the only tail writers keyed on it
    (``_write_reaction_preservation_report``, diagnostic, read by no gate) then neither write into the
    measured tree nor read a stale ``tmp/locked_reaction_manifest.json`` from it."""

    saved = T.app.PROJECT_ROOT
    T.app.PROJECT_ROOT = work
    try:
        return _replay_leg(T, leg, work, primary)
    finally:
        T.app.PROJECT_ROOT = saved


def _replay_leg(T: SimpleNamespace, leg: Dict[str, Any], work: Path, primary: Path) -> Dict[str, Any]:
    refused_before = len(Offline.refused)
    inputs = leg["inputs"]
    path = {name: primary / meta["path"] for name, meta in inputs.items() if meta.get("path")}
    for name in path:
        meta = inputs[name]
        if sha256_file(path[name]) != meta["sha256"]:
            raise RuntimeError(f"input drifted: {meta['path']} no longer has sha256 {meta['sha256'][:12]}")
    fm = load(path["final_mapped.json"])
    qrec = load(path["quarantine_report.json"])
    grec = load(path["final_stage3_gate_report.json"]) if "final_stage3_gate_report.json" in path else {}
    contracts = load(path["contract_reports.json"]) if "contract_reports.json" in path else {}
    ctx = (qrec.get("coverage") or {}).get("requested_context")
    work.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {"id": leg["id"], "paper": leg["paper"]}
    rf = path.get("pwml_required_field_gate_report.json")
    out["recorded_reports"] = {  # the leg's own reports, read by THIS tree's functions, for the fidelity check
        "final_gate_errors": _gate_errors(grec),
        "required_gate_codes": _codes(T, load(rf)) if rf else None,
    }

    fm_hash = T.gr.payload_sha256(fm)
    kind = "canonical" if grec.get("payload_sha256") == fm_hash else "pre_freeze"
    out["checkpoint"] = kind
    out["checkpoint_sha256"] = inputs["final_mapped.json"]["sha256"]

    payload = deepcopy(fm)
    if kind == "pre_freeze":
        cache = work / "enrichment_cache.json"
        shutil.copyfile(path["enrichment_cache.json"], cache)
        (work / "final.mapped.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            rep = T.app.run_enrichment(work / "final.mapped.json", work / "final.enriched.json",
                                       work / "enrichment_report.json", cache_path=cache,
                                       dump_path=work / "enrichment_dump.json", qa_report=None)
            payload = load(work / "final.enriched.json")
            out["enrichment"] = {"ok": True, "summary_digest": digest((rep or {}).get("summary"))}
        except Exception as exc:  # noqa: BLE001 -- the app falls back to the mapped payload, so does the replay
            out["enrichment"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        pf = T.prefreeze(payload, strict_db=True)
        out["prefreeze"] = {"ok": bool(pf.get("ok")), "failures": pf.get("failures"),
                            "db_resolution": payload.get("prefreeze_db_resolution")}
        # The leg's PathBank DB was reachable (every canonical payload of these runs carries
        # {"available": true}); the replay's is not. Is that marker the ONLY byte the replay changes?
        probe = deepcopy(payload)
        probe["prefreeze_db_resolution"] = {"available": True}
        out["prefreeze"]["admitted_hash_matches_leg_with_db_available"] = (
            T.sq.admitted_payload_hash(probe) == qrec.get("admitted_payload_hash"))

    freeze = T.app.freeze_canonical_payload(payload, strict_db=True, outputs_dir=work / "outputs", pathway_context=ctx,
                                            export_mode=MODE, research_mode=False, tmp=work)
    qres = freeze["quarantine_result"]
    qrep = qres.get("quarantine_report") or {}
    canonical = freeze["payload"]
    final_gate = freeze["final_stage3_gate_report"]
    out["quarantine"] = {
        "ok": bool(freeze["quarantine_ok"]), "refusal": sorted(qres.get("refusal_reasons") or []),
        "release": (qrep.get("release") or {}).get("status"),
        "decision_input_hash_matches_leg": qrep.get("decision_input_hash") == qrec.get("decision_input_hash"),
        "admitted_hash_matches_leg": qrep.get("admitted_payload_hash") == qrec.get("admitted_payload_hash"),
        "counts": qrep.get("counts"),
    }
    out["canonical_sha256"] = freeze["payload_hash"] or None
    out["canonical_matches_leg"] = bool(freeze["payload_hash"]) and freeze["payload_hash"] == grec.get("payload_sha256")
    out["final_gate"] = None if not final_gate else {"ok": final_gate.get("ok") is True, "codes": _codes(T, final_gate),
                                                     "errors": _gate_errors(final_gate)}

    # The batch driver's verdict on this artifact set: upstream contract reports as recorded, the final
    # gate and canonical payload as replayed.
    arts = {k: v for k, v in contracts.items() if "." not in k and isinstance(v, dict)}
    if final_gate:
        arts[T.gr.FINAL_GATE_REPORT_KEY] = final_gate
    arts[T.gr.CANONICAL_PAYLOAD_KEY] = canonical
    arts = T.gr.stamp_artifact_set(arts)
    verdict = T.gr.gate_verdict(arts)
    # C-119 (2026-09) added the superseded-snapshot rule; a tree from before it has none, which is the
    # old driver's behaviour exactly.
    superseded = getattr(T.driver, "_superseded_contract_reports", lambda _a: [])(arts)
    codes, _lines, contract_errors = T.driver._collect_issue_codes(T.driver._blocking_reports(arts))
    for issue in verdict.errors:
        code = T.driver._issue_code(issue)
        if code and code not in codes:
            codes.append(code)
    out["driver"] = {"gate_failed": bool(verdict.failed), "gate_stage": verdict.stage, "gate_reason": verdict.reason,
                     "contract_errors": contract_errors, "codes": sorted(codes)}

    stage, pwml, ir = None, None, None
    if verdict.failed or contract_errors:
        # The driver names the gate stage in both cases; a passing final gate blocked only by an upstream
        # *_contract_report is a different failure, and one the replay carries as recorded.
        stage = str(verdict.stage) if verdict.failed else "upstream_contract_report"
    else:
        try:  # the app opens refinement review; an exception here means no "Generate PWML" button
            T.app.initialize_refinement_review_state(deepcopy(canonical), {})
        except Exception as exc:  # noqa: BLE001
            stage = "refinement_review_crash"
            out["refinement_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            # INFORMATION ONLY, never a verdict: would the exporter have written a PWML had the review
            # step opened? Tells a lane fixing the crash whether anything else stands behind it.
            cf_root = work / "if_refinement_opened"
            cf = T.app.run_pwml_export(deepcopy(canonical), project_root=cf_root,
                                       ref_path=T.app._pwml_reference_path(T.root), quarantine_report=qrep,
                                       quarantine_artifacts=qres.get("quarantine_artifacts") or {},
                                       pathway_context=ctx, export_mode=MODE, **EXPORT_ARGS)
            out["export_if_refinement_opened"] = {"ok": bool(cf.get("ok")), "error": str(cf.get("error") or "")[:240],
                                                  "required_gate_codes": _codes(T, cf.get("required_gate_report"))}
    if stage is None:
        res = T.app.run_pwml_export(deepcopy(canonical), project_root=work, ref_path=T.app._pwml_reference_path(T.root),
                                    quarantine_report=qrep, quarantine_artifacts=qres.get("quarantine_artifacts") or {},
                                    pathway_context=ctx, export_mode=MODE, **EXPORT_ARGS)
        xml = T.driver._xml_bytes(res)
        out["export"] = {
            "ok": bool(res.get("ok")), "error": str(res.get("error") or "")[:240],
            "quarantine_reused": bool(res.get("quarantine_reused")),
            "stage3_contract_codes": _codes(T, res.get("stage3_contract_report")),
            "required_gate_ok": (res.get("required_gate_report") or {}).get("ok"),
            "required_gate_codes": _codes(T, res.get("required_gate_report")),
            "ir_blocking": sorted({str(i.get("code")) for i in T.app.blocking_pwml_ir_errors(res.get("pwml_ir_report") or {})})
            if res.get("pwml_ir_report") else [],
            "ir_validation": sorted({str(i.get("code")) for i in (res.get("pwml_ir_validation") or {}).get("errors") or []}),
        }
        if res.get("ok") and xml:
            release = (T.driver._frozen_release_record(res, superseded) if superseded
                       else T.driver._frozen_release_record(res))  # one-argument form before C-119
            name = T.driver._pwml_artifact_name(release)
            out["export"]["release"] = release.get("status")
            out["export"]["pwml_name"] = name or "(withheld: diagnostic_only)"
            out["pwml_sha256"] = hashlib.sha256(xml).hexdigest()
            leg_pwml = [n for n in inputs if n.endswith(".pwml")]
            out["pwml_matches_leg"] = bool(leg_pwml) and out["pwml_sha256"] == inputs[leg_pwml[0]]["sha256"]
            if name:
                pwml = work / name
                pwml.write_bytes(xml)
                ir = res.get("pwml_ir")
                (work / "pwml_ir.json").write_text(json.dumps(res.get("pwml_ir"), indent=1, default=str), encoding="utf-8")
            else:
                stage = "pwml_withheld_diagnostic_only"
        else:
            stage = "pwml_export"
    out["pwml_written"] = pwml is not None
    out["failure_stage"] = stage
    (work / "canonical.json").write_text(json.dumps(canonical, sort_keys=True, indent=1, ensure_ascii=False), encoding="utf-8")
    out["network_refused"] = Offline.refused[refused_before:]
    out["gold"] = gold_scores(leg, canonical, work) if leg["paper"] in GOLD_PAPERS else None
    # A refused quarantine leaves no canonical payload; the leg's final_mapped.json is then the payload the
    # quarantine was given (batch.driver._add_identity_artifacts), and so is this.
    out["pwml_coverage"] = pwml_coverage(T, qres, canonical or payload, ir, pwml.name if pwml else None,
                                         recorded_removals=qrec.get("quarantined") if kind == "canonical" else None)
    out["files"] = _tree_digest(work)
    return out


def pwml_coverage(T: SimpleNamespace, qres: Dict[str, Any], payload: Dict[str, Any], ir: Any,
                  pwml_name: Optional[str], recorded_removals: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
    """ORCH-772 (D-114 § 1): the MEASURED tree's ``reaction_evidence`` on this replay's artifacts, the same
    set a leg directory holds. ``None`` (n/a) for a tree whose model carries no ``pwml_coverage``.

    ``recorded_removals``: on a CANONICAL checkpoint the replay re-quarantines a payload the leg already
    quarantined, so its own ``quarantined[]`` holds only what the measured tree removes on top; the leg's
    pre-freeze removals are the recorded report's rows, carried as recorded (as the upstream contract reports
    are) and listed before the replay's own. A canonical replay whose quarantine refused would leave those rows
    in no known place, so its coverage is not recorded (0 such legs at ORCH-772)."""

    E = T.evidence
    if E is None:
        return None
    arts: Dict[str, Any] = {}
    for name, path in sorted((qres.get("quarantine_artifacts") or {}).items()):
        if name in (E.COVERAGE_REPORT_FILENAME, E.QUARANTINE_REPORT_FILENAME, E.REMOVED_ENTITY_REPORT_FILENAME):
            arts[name] = load(Path(path))
    if recorded_removals:
        quarantine = arts.get(E.QUARANTINE_REPORT_FILENAME)
        if not isinstance(quarantine, dict) or quarantine.get("ok") is not True:
            return {"drawn": None, "total": None, "by_kind": {}, "by_design_withheld": 0, "merged_duplicates": 0,
                    "not_counted": 0, "pwml_written": bool(pwml_name), "omitted_by_reason": [],
                    "not_recorded": "canonical checkpoint whose replayed quarantine did not pass: the recorded "
                                    "removals cannot be placed"}
        quarantine["quarantined"] = [*recorded_removals, *(quarantine.get("quarantined") or [])]
    if payload:
        arts[E.FINAL_MAPPED_FILENAME] = payload
    if pwml_name:  # the IR is read only when a PWML was written (the module's own rule)
        arts[E.PWML_IR_FILENAME] = ir
        arts[pwml_name] = True
    report = E.build_report(arts)
    cov = report.get("pwml_coverage")
    if cov is None:
        return None
    reasons: Dict[tuple, int] = {}
    for e in report["omitted"]:
        if e["counted"]:
            key = (e["ki"], e["group"], e["reason"] or e["message"][:120])
            reasons[key] = reasons.get(key, 0) + 1
    return {**{k: cov[k] for k in ("drawn", "total", "by_kind", "by_design_withheld", "merged_duplicates", "not_counted",
                                     "not_recorded", "pwml_written")},
            "omitted_by_reason": [[n, *key] for key, n in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))]}


def gold_scores(leg: Dict[str, Any], canonical: Dict[str, Any], work: Path) -> Dict[str, Any]:
    """eval-v2 ``score_leg`` on the replayed canonical payload and on the IR rows drawn in the replayed
    PWML, through the ORCH-756 scorer's own loaders (imported, not rebuilt; its ``main`` writes into
    review folders and is never called)."""

    import importlib
    import importlib.util

    try:
        importlib.import_module("t2pw.bench.eval_v2.reactions")
    except ModuleNotFoundError:
        return None  # a tree from before eval-v2 (fidelity runs only): no gold, and not an error
    try:
        spec = importlib.util.spec_from_file_location("orch756_gold_accuracy", HERE / "orch756_gold_accuracy.py")
        G = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(G)
        case = G.load_supplement()[leg["paper"]]
        text = G.load_fulltext(case, FULLTEXT)
        if text is None:
            return {"error": f"fulltext missing: {FULLTEXT}/{leg['paper']}.txt"}
        levels = {"final": canonical} if canonical else {}
        pw = next(iter(sorted(work.glob("*.pwml"))), None)
        if pw is not None:
            ir_pl, _check, _left = G.ir_payload(load(work / "pwml_ir.json"), canonical, G.pwml_equations(pw))
            levels["pwml"] = ir_pl
        out = {}
        for lvl, payload in levels.items():
            t1 = G.R.score_leg(case, payload, [], text, expected_species=case.species)["table1_paper_extraction"]
            out[lvl] = {"P": t1["precision"], "R": t1["recall"], "F1": t1["f1"],
                        "matched": len(t1["gold_reactions_matched"]), "gold": t1["gold_reactions"],
                        "matched_ids": sorted(t1["gold_reactions_matched"])}
        return out
    except Exception as exc:  # noqa: BLE001 -- a scorer failure is reported, never hidden
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


# ── fidelity, classification, the table ──────────────────────────────────────────────────


def recorded_class(rec: Dict[str, Any]) -> str:
    """The recorded failure in the replay's own vocabulary."""

    if rec["pwml"]:
        return "pwml"
    msg = rec["message"]
    if rec["stage"] == "post_pipeline" and "final_pre_export_gate_lifecycle" in msg:
        return "final_pre_export_gate_lifecycle"
    if rec["stage"] == "post_pipeline" and "final_pre_export_stage3_gates" in msg and rec["final_gate_ok"] is True:
        return "upstream_contract_report"
    if rec["stage"] == "post_pipeline" and "final_pre_export_stage3_gates" in msg:
        return "final_pre_export_stage3_gates"
    if "refinement_generate_pwml" in msg:
        return "refinement_review_crash"
    if rec["stage"] == "pwml_export":
        return "pwml_export"
    return f"other:{rec['stage']}"


def fidelity(leg: Dict[str, Any], r: Dict[str, Any]) -> List[str]:
    """Each recorded verdict the replay did NOT reproduce; ``[]`` means reproduced."""

    rec = leg["recorded"]
    diffs = []
    got = "pwml" if r["pwml_written"] else r["failure_stage"]
    if recorded_class(rec) != got:
        diffs.append(f"outcome recorded={recorded_class(rec)} replay={got}")
    if rec["quarantine_ok"] is not None and rec["quarantine_ok"] != r["quarantine"]["ok"]:
        diffs.append(f"quarantine_ok recorded={rec['quarantine_ok']} replay={r['quarantine']['ok']}")
    if rec["quarantine_refusal"] != r["quarantine"]["refusal"]:
        diffs.append(f"refusal recorded={rec['quarantine_refusal']} replay={r['quarantine']['refusal']}")
    if rec["quarantine_release"] != r["quarantine"]["release"]:
        diffs.append(f"release recorded={rec['quarantine_release']} replay={r['quarantine']['release']}")
    fg = r["final_gate"] or {}
    ex = r.get("export") or {}
    if rec["final_gate_ok"] is not None and rec["final_gate_ok"] != fg.get("ok"):
        diffs.append(f"final_gate_ok recorded={rec['final_gate_ok']} replay={fg.get('ok')}")
    if r["recorded_reports"]["final_gate_errors"] != fg.get("errors", []):
        diffs.append(f"final_gate_errors recorded={r['recorded_reports']['final_gate_errors']} replay={fg.get('errors')}")
    if rec["required_gate_ok"] is not None and rec["required_gate_ok"] != ex.get("required_gate_ok"):
        diffs.append(f"required_gate_ok recorded={rec['required_gate_ok']} replay={ex.get('required_gate_ok')}")
    if r["recorded_reports"]["required_gate_codes"] not in (None, ex.get("required_gate_codes")):
        diffs.append(f"required_gate_codes recorded={r['recorded_reports']['required_gate_codes']} replay={ex.get('required_gate_codes')}")
    if recorded_class(rec) == "pwml_export" and got == "pwml_export" and ex.get("error", "")[:60] not in rec["message"]:
        diffs.append(f"export_error recorded={rec['message'][:120]!r} replay={ex.get('error', '')[:120]!r}")
    if rec["pwml"] and r["pwml_written"] and rec["pwml"] != [r["export"]["pwml_name"]]:
        diffs.append(f"pwml_name recorded={rec['pwml']} replay={r['export']['pwml_name']}")
    return diffs


def classify(leg: Dict[str, Any], r: Optional[Dict[str, Any]]) -> str:
    if r is None:
        rec = leg.get("recorded")
        if not rec:
            return "aborted (stopped before it reported; no outcome)"
        return f"llm_originating ({rec.get('status')} at {rec.get('stage')}: no checkpoint)"
    if r.get("error"):
        return "replay_error"
    if r["pwml_written"]:
        return "pass"
    return f"replayable ({r['failure_stage']})"


def fmt_gold(g: Optional[Dict[str, Any]]) -> str:
    if not g:
        return ""
    if "error" in g:
        return f"gold error: {g['error'][:60]}"
    return "  ".join(f"{lvl} P{v['P']:.3f} R{v['R']:.3f} F1 {v['F1']:.3f} ({v['matched']}/{v['gold']})"
                     for lvl, v in g.items() if v.get("P") is not None)


def run(args: argparse.Namespace) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    manifest = load(Path(args.manifest))
    T = load_tree(Path(args.src) if args.src else None)
    probe = probe_offline()
    lines: List[str] = []

    def emit(text: str) -> None:
        print(text, flush=True)
        lines.append(text)

    emit(f"t2pw resolved: {T.t2pw}")
    emit(f"offline probe: {probe}")
    if not probe["ok"]:
        print("ORCH768_NETWORK_NOT_BLOCKED")
        return 3
    out_dir = Path(args.out)
    wanted = set(args.legs or [])
    records = []
    for leg in manifest["legs"]:
        if wanted and leg["id"] not in wanted:
            continue
        if args.src_sha and leg["src_sha"] != args.src_sha:
            continue
        if leg["mode"] not in (None, "strict"):
            continue  # research legs are inventoried, never in the PWML corpus
        r = None
        if leg["checkpoint"]:
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", leg["id"])
            try:
                r = replay_leg(T, leg, out_dir / slug)
            except Exception as exc:  # noqa: BLE001 -- a leg that cannot be replayed is reported, not hidden
                r = {"id": leg["id"], "error": f"{type(exc).__name__}: {str(exc)[:300]}", "pwml_written": False}
        row = {"id": leg["id"], "tier": leg.get("tier"), "paper": leg["paper"], "src_sha": leg["src_sha"],
               "checkpoint": leg["checkpoint"],
               "recorded": recorded_class(leg["recorded"]) if leg.get("recorded") else "aborted",
               "class": classify(leg, r), "replay": r}
        if r and not r.get("error"):
            row["fidelity_diffs"] = fidelity(leg, r)
        records.append(row)
        emit(line_for(row))
    summary = {"all": summarize(records)}
    for tier in sorted({r["tier"] for r in records if r.get("tier")}):
        summary[f"tier_{tier}"] = summarize([r for r in records if r.get("tier") == tier])
    print_summary(summary, records, emit)
    coverage = coverage_summary(records)
    print_coverage(coverage, records, emit)
    doc = {"task": "ORCH-768", "src": "t2pw from --src" if args.src else "t2pw from sys.path", "offline_probe": probe,
           "summary": summary, "pwml_coverage": coverage, "legs": records}
    if args.json:
        Path(args.json).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.txt:
        Path(args.txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def line_for(row: Dict[str, Any]) -> str:
    r = row["replay"] or {}
    fid = row.get("fidelity_diffs")
    # At the leg's own tree "AS RECORDED" is the fidelity verdict; at any other tree it says whether the
    # code moved the leg's outcome since it ran.
    fid_s = "" if fid is None else ("AS RECORDED" if not fid else "CHANGED vs recorded: " + "; ".join(fid))
    q = r.get("quarantine") or {}
    fg = r.get("final_gate") or {}
    ex = r.get("export") or {}
    bits = [f"{row['id']:58s}", f"{row['class']:52s}"]
    if r and not r.get("error"):
        bits.append(f"q={q.get('ok')}{q.get('refusal') or ''}/{q.get('release')} gate={fg.get('ok')}{fg.get('codes') or ''}"
                    f" export={ex.get('ok')} req={ex.get('required_gate_ok')} net={len(r.get('network_refused') or [])}")
    if r.get("error"):
        bits.append(r["error"])
    if r.get("export_if_refinement_opened"):
        cf = r["export_if_refinement_opened"]
        bits.append(f"[info: export had review opened: ok={cf['ok']} {cf['error'][:60]}]")
    bits.append(fid_s)
    bits.append(f"cov={fmt_cov(row)}")
    g = fmt_gold(r.get("gold"))
    return "  ".join(b for b in bits if b) + (f"\n{'':60s}gold: {g}" if g else "")


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    strict = [r for r in records if r["recorded"] != "aborted"]
    with_cp = [r for r in strict if r["replay"] and not r["replay"].get("error")]
    passed = [r for r in with_cp if r["replay"]["pwml_written"]]
    replayable = [r for r in with_cp if not r["replay"]["pwml_written"]]
    fid = [r for r in with_cp if "fidelity_diffs" in r]
    return {
        "strict_legs": len(strict),
        "with_checkpoint": len(with_cp),
        "replay_errors": sum(1 for r in strict if r["replay"] and r["replay"].get("error")),
        "pwml_pass": len(passed),
        "pwml_pass_rate_of_strict": f"{len(passed)}/{len(strict)}",
        "pwml_pass_rate_of_replayed": f"{len(passed)}/{len(with_cp)}",
        "replayable_failures": len(replayable),
        "llm_originating_failures": sum(1 for r in strict if r["replay"] is None),
        "aborted_not_counted": sum(1 for r in records if r["recorded"] == "aborted"),
        "verdict_as_recorded": f"{sum(1 for r in fid if not r['fidelity_diffs'])}/{len(fid)}",
        "passing_ids": sorted(r["id"] for r in passed),
        "network_refused_total": sum(len(r["replay"].get("network_refused") or []) for r in with_cp),
    }


def print_summary(summary: Dict[str, Any], records: List[Dict[str, Any]], emit: Any = print) -> None:
    for name, s in summary.items():
        emit(f"\nSUMMARY ({name})")
        for k, v in s.items():
            if k != "passing_ids":
                emit(f"  {k:30s} {v}")
    emit("GOLD (replayed tail; certified gold only)")
    for r in records:
        g = (r["replay"] or {}).get("gold")
        if g:
            emit(f"  {r['id']:58s} {fmt_gold(g)}")


# ── ORCH-772 (D-114 § 1): PWML coverage per leg and over the corpus ─────────────────────


def leg_percent(row: Dict[str, Any], exclude_by_design: bool = False) -> Optional[float]:
    """The leg's coverage in percent. A strict leg with no replay (LLM-originating) or a replay error drew
    nothing: 0. ``None`` when the measured tree carries no coverage, or the leg's is not recorded."""

    r = row["replay"]
    if r is None or r.get("error"):
        return 0.0
    cov = r.get("pwml_coverage")
    if not cov or cov["drawn"] is None:
        return None
    total = cov["total"] - (cov["by_design_withheld"] if exclude_by_design else 0)
    return round(100.0 * cov["drawn"] / total, 4) if total else 0.0


def fmt_cov(row: Dict[str, Any]) -> str:
    r = row["replay"]
    if r is None:
        return "0 (no replay: LLM-originating or aborted)"
    if r.get("error"):
        return "0 (replay error)"
    cov = r.get("pwml_coverage")
    if not cov:
        return "n/a (the measured tree has no reaction_evidence pwml_coverage)"
    if cov["drawn"] is None:
        return f"not recorded of {cov['total']} ({cov['not_recorded']})"
    top = "; ".join(f"{n}x {ki} {reason[:60]}" for n, ki, _group, reason in cov["omitted_by_reason"][:2])
    return (f"{cov['drawn']}/{cov['total']} ({leg_percent(row):.1f}%; ex-by-design {leg_percent(row, True):.1f}%)"
            + (f" omitted: {top}" if top else ""))


def coverage_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean, median and share of legs >= 75%, on both denominators (D-113 rulings), both variants."""

    import statistics

    strict = [r for r in records if r["recorded"] != "aborted"]
    replayed = [r for r in strict if r["replay"] and not r["replay"].get("error")]
    out: Dict[str, Any] = {"target_percent": 75}
    if any(not r["replay"].get("pwml_coverage") for r in replayed):
        out["n/a"] = "the measured tree's reaction_evidence carries no pwml_coverage"
        return out
    for name, legs in (("strict_legs_with_an_outcome", strict), ("replayed", replayed)):
        for variant, ex in (("headline", False), ("excluding_by_design", True)):
            values = [leg_percent(r, ex) for r in legs]
            known = [v for v in values if v is not None]
            out[f"{name}.{variant}"] = {
                "legs": len(legs), "measured": len(known), "not_measured": len(values) - len(known),
                "mean": round(statistics.fmean(known), 4) if known else None,
                "median": round(statistics.median(known), 4) if known else None,
                "at_or_above_target": sum(v >= 75 for v in known),
                "share_at_or_above_target": f"{sum(v >= 75 for v in known)}/{len(known)}",
            }
    return out


def print_coverage(coverage: Dict[str, Any], records: List[Dict[str, Any]], emit: Any = print) -> None:
    emit("\nPWML COVERAGE (D-114 § 1: processes drawn in the PWML / processes in the text record; target 75%)")
    if "n/a" in coverage:
        emit(f"  n/a: {coverage['n/a']}")
        return
    for key, s in coverage.items():
        if isinstance(s, dict):
            mean = "n/a" if s["mean"] is None else f"{s['mean']:.1f}%"
            median = "n/a" if s["median"] is None else f"{s['median']:.1f}%"
            emit(f"  {key:48s} legs {s['legs']:3d} measured {s['measured']:3d}  mean {mean:>7s}  median {median:>7s}"
                 f"  >= 75%: {s['share_at_or_above_target']}")
    emit("LOWEST COVERAGE (replayed legs)")
    ranked = sorted((r for r in records if r["replay"] and not r["replay"].get("error") and leg_percent(r) is not None),
                    key=lambda r: (leg_percent(r), r["id"]))
    for r in ranked[:10]:
        emit(f"  {r['id']:58s} {fmt_cov(r)}")


def compare(args: argparse.Namespace) -> int:
    """D-112 § 2: pass rate up OR the target leg fixed; no passing leg turns failing; no gold recall lost."""

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base, tip = load(Path(args.base)), load(Path(args.tip))
    b = {r["id"]: r for r in base["legs"] if r["recorded"] != "aborted"}
    t = {r["id"]: r for r in tip["legs"] if r["recorded"] != "aborted"}
    assert set(b) == set(t), "the two runs replayed different legs"
    passing = lambda r: bool((r["replay"] or {}).get("pwml_written"))  # noqa: E731
    broke = sorted(i for i in b if passing(b[i]) and not passing(t[i]))
    fixed = sorted(i for i in b if not passing(b[i]) and passing(t[i]))
    lost = []
    for i in b:
        gb, gt = (b[i]["replay"] or {}).get("gold") or {}, (t[i]["replay"] or {}).get("gold") or {}
        for lvl in ("final", "pwml"):
            if lvl in gb and gb[lvl].get("R") is not None:
                if (gt.get(lvl) or {}).get("R") is None or gt[lvl]["R"] < gb[lvl]["R"]:
                    lost.append(f"{i} {lvl} R {gb[lvl]['R']:.3f} -> {(gt.get(lvl) or {}).get('R')}")
    nb, nt = sum(map(passing, b.values())), sum(map(passing, t.values()))
    target_fixed = bool(args.target) and args.target in fixed
    ok = (nt > nb or target_fixed) and not broke and not lost
    print(f"pass rate base {nb}/{len(b)} -> tip {nt}/{len(t)}")
    print(f"fixed: {fixed}")
    print(f"passing legs turned failing: {broke or 'none'}")
    print(f"gold recall lost: {lost or 'none'}")
    stage = lambda r: (r["replay"] or {}).get("failure_stage") or (r["replay"] or {}).get("error")  # noqa: E731
    moved = sorted(f"{i}: {stage(b[i])} -> {stage(t[i])}" for i in b
                   if not passing(b[i]) and not passing(t[i]) and stage(b[i]) != stage(t[i]))
    print(f"still failing, at a different step (information): {moved or 'none'}")
    # ORCH-772 (information): did anything but the coverage record move? Every verdict, PWML hash and
    # output-file digest is in the replay record, so equal records mean nothing else moved.
    strip = lambda r: json.dumps({k: v for k, v in (r["replay"] or {}).items() if k != "pwml_coverage"},  # noqa: E731
                                 sort_keys=True, default=str)
    differing = sorted(i for i in b if strip(b[i]) != strip(t[i]))
    print(f"replay records differing, ignoring pwml_coverage (information): {differing or 'none'} "
          f"({len(b) - len(differing)}/{len(b)} identical)")
    if args.target:
        print(f"target {args.target}: {'FIXED' if target_fixed else 'not fixed'}")
    print("D-112 § 2:", "ACCEPT" if ok else "REJECT")
    return 0 if ok else 1


def fixture(args: argparse.Namespace) -> int:
    """Copy the named legs' inputs, gzipped (mtime 0, so the bytes are reproducible), into ``--out``
    with a ``legs.json`` whose paths are fixture-relative and whose sha256 are the CORPUS manifest's:
    the test that decompresses them proves the fixture is the corpus input, byte for byte."""

    import gzip

    manifest = load(Path(args.manifest))
    out = Path(args.out)
    legs = []
    for leg in manifest["legs"]:
        if leg["id"] not in args.legs:
            continue
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", leg["id"])
        leg = deepcopy(leg)
        for name, meta in leg["inputs"].items():
            data = (PRIMARY / meta["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == meta["sha256"], meta["path"]
            if name.endswith(".pwml"):
                meta["path"] = None  # pinned by sha256 only; replaying does not read it
                continue
            rel = f"{slug}/{name}.gz"
            (out / slug).mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
            meta["path"] = rel
        legs.append(leg)
    assert len(legs) == len(args.legs), "a named leg is not in the manifest"
    doc = {"from_manifest": "docs/pwml_recovery_sprint/evidence/orch768_corpus_manifest.json", "legs": legs}
    (out / "legs.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{len(legs)} legs, {total} bytes -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--out", default=str(MANIFEST))
    rp = sub.add_parser("run")
    rp.add_argument("--src", help="<tree>/src to import t2pw from; omitted = sys.path as it stands")
    rp.add_argument("--out", required=True, help="scratch directory for per-leg outputs")
    rp.add_argument("--json", help="write the corpus record here")
    rp.add_argument("--txt", help="write the printed table here")
    rp.add_argument("--manifest", default=str(MANIFEST))
    rp.add_argument("--legs", nargs="*")
    rp.add_argument("--src-sha", help="only the legs the manifest records as run on this source tree (fidelity)")
    fx = sub.add_parser("fixture")
    fx.add_argument("--legs", nargs="+", required=True)
    fx.add_argument("--out", required=True)
    fx.add_argument("--manifest", default=str(MANIFEST))
    cp = sub.add_parser("compare")
    cp.add_argument("base")
    cp.add_argument("tip")
    cp.add_argument("--target", help="leg id the fix targets")
    args = ap.parse_args()
    if args.cmd == "inventory":
        doc = inventory()
        Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{len(doc['legs'])} legs, {len(doc['src_trees'])} source trees -> {args.out}")
        return 0
    if args.cmd == "compare":
        return compare(args)
    if args.cmd == "fixture":
        return fixture(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
