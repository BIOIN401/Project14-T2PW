"""FIX-TRNA decisive check: deterministic post-pipeline + strict PWML export of the stored chlorophyll
merged payload (legs4g/base1-PMC5130987), flag off or on, offline curator, no LLM audit, no LLM gap resolver.

usage (provider/DB env comes from live_launch.py, which loads .env; never read .env here):
  python live_launch.py --script postreplay.py off|on <out.json>
Writes <out.json> (verdict + tRNA evidence) and the PWML next to it.
"""
import faulthandler
import json
import os
import re
import sys
import traceback
from pathlib import Path

flag, outp = sys.argv[1], Path(sys.argv[2])
faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)
print("START", flag, flush=True)
os.environ["T2PW_OFFLINE_CURATOR"] = "1"
os.environ.pop("T2PW_AUDIT_NA_DECLARED", None)
if flag == "on":
    os.environ["T2PW_AUDIT_NA_DECLARED"] = "1"
for k in ("T2PW_REACTION_VERIFIER", "T2PW_INTERACTION_TYPING"):
    os.environ.pop(k, None)
sys.path.insert(0, "C:/t/fix-trna/src")

import functools  # noqa: E402

import t2pw.paths as paths  # noqa: E402
from t2pw.pipeline.export_mode import PATHWHIZ  # noqa: E402
from t2pw.service import defaults as d  # noqa: E402
from t2pw.service import extraction as ex  # noqa: E402
from t2pw.service import post_run as pr  # noqa: E402
from t2pw.service.export import run_export  # noqa: E402
from t2pw.service.reporter import CollectingReporter  # noqa: E402

assert str(paths.PROJECT_ROOT).replace("\\", "/").lower().startswith("c:/t/fix-trna"), paths.PROJECT_ROOT
SRC = next(Path("C:/t/acc-runs/legs4g/base1-PMC5130987").glob("*/papers/PMC5130987/strict/merged_payload.json"))
payload = json.loads(SRC.read_text(encoding="utf-8"))
state = {"export_mode": PATHWHIZ, "pathway_context": {"likely_organism": "Arabidopsis thaliana"}, "pwml_strict_db": True}
state[ex.DEADLINE_STATE_KEY] = None
reporter = CollectingReporter()
out = {"flag": flag, "source": str(SRC).replace("\\", "/"), "reaction20": None}
outputs_dir = paths.PROJECT_ROOT / "outputs"
pr.reset_quarantine_state(outputs_dir, state=state)
res = None
try:
    res = pr.run_post_pipeline(
        pr.PostRunRequest(
            final_payload=payload, use_llm_audit=False, use_sbml_overwatch=False,
            default_compartment=d.DEFAULT_COMPARTMENT, mapping_cache_text=d.MAPPING_CACHE_PATH,
            id_source_mode=d.id_source(), db_host=d.db_host(), db_port=d.db_port(), db_user=d.db_user(),
            db_password=d.db_password(), db_schema=d.db_schema(),
            audit_max_rounds=d.AUDIT_MAX_ROUNDS, audit_timeout_seconds=d.AUDIT_TIMEOUT_SECONDS,
            audit_candidate_count=d.AUDIT_CANDIDATE_COUNT, use_example_retrieval=d.USE_EXAMPLE_RETRIEVAL,
            example_index_path=d.EXAMPLE_INDEX_PATH, example_top_k=d.EXAMPLE_TOP_K,
            use_gap_resolver=d.USE_GAP_RESOLVER, use_llm_gap_resolver=False,
            gap_resolver_max_items=d.GAP_RESOLVER_MAX_ITEMS, use_stoich_agent=d.USE_STOICH_AGENT,
        ),
        state=state, reporter=reporter,
        open_review=functools.partial(pr.open_refinement_review, state=state),
        render_issue_detail=lambda _detail: None,
        render_contract_failure=lambda exc: reporter.error(f"stage boundary {exc.stage}: {exc}"),
    )
    out["post_run_failure"] = repr(res.failure) if res.failure is not None else None
except Exception:  # noqa: BLE001
    out["post_run_raised"] = traceback.format_exc()[-3000:]
out["reporter_errors"] = [str(e)[:600] for e in reporter.errors][:10]
out["reporter_exceptions"] = [str(e)[:600] for e in reporter.exceptions][:10]
arts = state.get("post_pipeline_artifacts") or {}
out["artifact_keys"] = sorted(arts)[:60] if isinstance(arts, dict) else None
out["gate_failed"] = bool(arts.get("gate_failed")) if isinstance(arts, dict) else None
work = Path("C:/t/vis-runs/scratch/fixtrna") / f"post-{flag}"
work.mkdir(parents=True, exist_ok=True)
if state.get("refinement_working_json") is not None:
    try:
        r = run_export(state=state, reporter=reporter, work_dir=work, project_root=paths.PROJECT_ROOT)
        out["export_ok"] = bool(r.get("ok"))
        out["export_error"] = str(r.get("error", ""))[:600]
        xml = r.get("pwml_xml") or r.get("xml") or ""
        if not xml and r.get("output_path") and Path(r["output_path"]).exists():
            xml = Path(r["output_path"]).read_text(encoding="utf-8", errors="replace")
        if isinstance(xml, bytes):
            xml = xml.decode("utf-8", "replace")
        out["export_keys"] = sorted(r)
        if xml:
            (outp.parent / f"{outp.stem}.pwml").write_text(xml, encoding="utf-8")
            out["pwml_bytes"] = len(xml)
            out["pwml_trna_lines"] = [ln.strip()[:240] for ln in xml.splitlines() if re.search(r"tRNA", ln, re.I)][:60]
            out["pwml_nucleic_acid_tags"] = sorted(set(re.findall(r"<(\w*[Nn]ucleic\w*)", xml)))
    except Exception:  # noqa: BLE001
        out["export_raised"] = traceback.format_exc()[-3000:]
else:
    out["export_ok"] = None
    out["export_note"] = "no refinement_working_json: the review step was not reached"
mapped = (arts.get("final_mapped") if isinstance(arts, dict) else None) or {}
if isinstance(mapped, dict) and mapped:
    ent = mapped.get("entities", {})
    out["mapped_trna_compounds"] = [c.get("name") for c in ent.get("compounds", []) if "trna" in str(c.get("name", "")).lower()]
    out["mapped_trna_nucleic_acids"] = [c.get("name") for c in ent.get("nucleic_acids", []) if "trna" in str(c.get("name", "")).lower()]
    rx = (mapped.get("processes") or {}).get("reactions", [])
    out["reaction20"] = next((r for r in rx if "tRNA Glu" in json.dumps(r) and "Glu to Glu-tRNA" in str(r.get("name"))), None)
outp.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print("WROTE", outp)
