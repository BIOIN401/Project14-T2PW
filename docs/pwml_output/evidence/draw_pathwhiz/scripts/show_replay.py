"""Showcase replay: stored final_mapped payloads -> run_pwml_export of the tree --src (main a6c07e16), no LLM, no network.
Adapted from C:/t/render-corr/scripts/ir_replay.py (same calls, same arguments). Per spec writes to --out/<slug>/:
  pathway.pwml, pwml_ir.json, final_payload.json (payload as exported), ours.sbml (clean-model SBML), replay.json (record).
spec fields: slug, run_dir, mode, pathway_name, organism, prefilled (true = live run, final_mapped already carries the
final-payload fill, so it is not re-applied), live_pwml (optional: PWML the live run wrote, compared byte for byte)."""
import argparse, copy, hashlib, json, os, shutil, sys, traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, a.src)
os.environ["T2PW_FINAL_PAYLOAD"] = "1"
os.environ["T2PW_PWML_LAYOUT"] = "v2"
os.environ["T2PW_OFFLINE_CURATOR"] = "1"

from t2pw.service import postpipeline  # noqa: E402
from t2pw.pipeline.export_mode import coerce_mode  # noqa: E402
from t2pw.clean_model.builder import read_run_meta, build_from_payload  # noqa: E402
from t2pw.clean_model.export.sbml import write_sbml  # noqa: E402
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.service import final_payload as fp  # noqa: E402

REPO = Path(a.src).resolve().parent
FILLCACHE = "C:/t/showcase2/_work/fillcache"  # copy of C:/t/ev-fp/fillcache (read here, never the original)


class _NoNet:
    n = 0

    def get_json(self, url, params=None):
        _NoNet.n += 1
        raise RuntimeError("network forbidden")
    get_text = get_json


def one(s):
    d = Path(s["run_dir"])
    out = Path(a.out) / s["slug"]
    out.mkdir(parents=True, exist_ok=True)
    rec = {"slug": s["slug"], "run_dir": s["run_dir"], "src": a.src}
    payload = json.load(open(d / "final_mapped.json", encoding="utf-8"))
    qr = json.load(open(d / "quarantine_report.json", encoding="utf-8")) if (d / "quarantine_report.json").exists() else {}
    di = qr.get("decision_inputs") or {}
    terms = list(dict.fromkeys(list(di.get("requested_core") or []) + list(di.get("pathway_context_core") or [])))
    ctx = {"key_compounds": terms} if terms else {}
    sdb = bool(qr.get("strict_db", True))
    rm = read_run_meta(str(d))
    if rm.get("pmid_source"):
        rm.pop("pmid", None)
    paper = {k: str(rm[k]) for k in ("pmid", "pmcid", "doi", "title", "year") if rm.get(k)}
    ctx["pathway_name"] = s["pathway_name"]
    ctx["likely_organism"] = s["organism"]
    payload = copy.deepcopy(payload)
    if not s.get("prefilled"):
        state = {"final_payload_paper": paper}
        txt = next((c for c in (d / "01_source_text.txt", d.parent / "01_source_text.txt") if c.exists()), None)
        if txt is not None:
            state["final_payload_paper_text"] = txt.read_text(encoding="utf-8", errors="replace")
            state["final_payload_paper_text_source"] = "01_source_text.txt"
        frep = fp.maybe_apply_final_id_fill(payload, ctx, state, fetcher=_NoNet(), cache=Cache(FILLCACHE))
        rec["fill_error"] = frep.get("error")
        rec["fill_written"] = frep.get("written")
    rec["network_attempts"] = _NoNet.n
    rec["paper"] = paper
    root = out / "_exportroot"
    shutil.rmtree(root, ignore_errors=True)
    (root / "reference").mkdir(parents=True)
    ref = next(p for p in (REPO / "reference/PW000001.pwml", REPO / "reference/PW012926.pwml") if p.exists())
    shutil.copy(ref, root / "reference" / ref.name)
    res = postpipeline.run_pwml_export(
        payload, "Generated Pathway", "", "Metabolic", root, root / "reference" / ref.name,
        3200, 1400, "#FFFFFF", None, sdb, None, None, ctx, coerce_mode(s["mode"]))
    rec["ok"] = bool(res.get("ok"))
    rec["error"] = str(res.get("error") or "")[:400]
    rec["counts"] = res.get("counts") or {}
    qa = res.get("qa") or {}
    rec["qa_ok"], rec["qa_errors"], rec["qa_warnings"] = qa.get("ok"), len(qa.get("errors") or []), len(qa.get("warnings") or [])
    xb = res.get("xml_bytes")
    if rec["ok"] and xb:
        (out / "pathway.pwml").write_bytes(xb)
        rec["pwml_sha256"] = hashlib.sha256(xb).hexdigest()
        lp = s.get("live_pwml")
        if lp and os.path.exists(lp):
            rec["live_pwml"] = lp
            rec["equals_live_pwml"] = open(lp, "rb").read() == xb
    if res.get("pwml_ir"):
        (out / "pwml_ir.json").write_text(json.dumps(res["pwml_ir"]), encoding="utf-8")
    with open(out / "final_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, default=str)
    meta = dict(rm)
    meta.setdefault("organism", s["organism"])
    m = build_from_payload(payload, meta, run_dir=str(d))
    write_sbml(m, str(out / "ours.sbml"))
    rec["ours_sbml"] = True
    shutil.rmtree(root, ignore_errors=True)
    json.dump(rec, open(out / "replay.json", "w", encoding="utf-8"), indent=1, default=str)
    return rec


specs = json.load(open(a.spec, encoding="utf-8"))
bad = 0
for s in specs:
    try:
        r = one(s)
        print("OK", s["slug"], r["ok"], r["counts"], "live_eq=%s" % r.get("equals_live_pwml"), flush=True)
        bad += 0 if r["ok"] else 1
    except Exception:
        bad += 1
        print("FAIL", s["slug"], traceback.format_exc()[-1500:], flush=True)
sys.exit(1 if bad else 0)
