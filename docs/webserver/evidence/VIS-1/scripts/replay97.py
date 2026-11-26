"""Adapted from C:/t/corr-assess/scripts/dump_replay.py (same run_pwml_export call, same per-run mode and strict_db,
flag 1, layout v2). Changes: no network (fill cache only, network attempts counted), writes pwml + IR + IR report +
IR validation per idx under --out/<label>/, and no dump into C:/t/corr-assess/base."""
import argparse, copy, csv, hashlib, json, os, shutil, sys, threading, time, traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--guard", type=float, default=240.0)
ap.add_argument("--idx", default="")
a = ap.parse_args()

sys.path.insert(0, a.src)
os.environ["T2PW_FINAL_PAYLOAD"] = "1"
os.environ["T2PW_PWML_LAYOUT"] = "v2"
os.environ["T2PW_OFFLINE_CURATOR"] = "1"

from t2pw.service import postpipeline  # noqa: E402
from t2pw.pipeline.export_mode import coerce_mode  # noqa: E402
from t2pw.clean_model.builder import read_run_meta  # noqa: E402
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.service import final_payload as fp  # noqa: E402

REPO = Path(a.src).resolve().parent
OUT = Path(a.out) / a.label
for sub in ("pwml", "ir"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
P = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
meta = {}
for r in csv.DictReader(open("C:/t/integ/docs/pwml_output/evidence/01/tidy_runs.csv", encoding="utf-8")):
    k = r["path"].rsplit("/final_mapped.json", 1)[0]
    meta[k[len(P):] if k.startswith(P) else k] = r
runs = [x for x in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if x["q"]]
FILLCACHE = "C:/t/showcase/_work/fillcache"  # same copy the showcase replay reads


class NoNet:
    n = 0

    def get_json(self, url, params=None):
        NoNet.n += 1
        raise RuntimeError("network forbidden")
    get_text = get_json


def key(run):
    return run["dir"][len(P):] if run["dir"].startswith(P) else run["dir"]


def one(idx, run):
    d = Path(run["dir"])
    m = meta[key(run)]
    rec = {"idx": idx, "dir": run["dir"], "mode": m["mode"], "ok": False, "error": "", "pwml_written": False}
    payload = json.load(open(d / "final_mapped.json", encoding="utf-8"))
    qr = json.load(open(d / "quarantine_report.json", encoding="utf-8")) if (d / "quarantine_report.json").exists() else {}
    di = qr.get("decision_inputs") or {}
    ctx_terms = list(dict.fromkeys(list(di.get("requested_core") or []) + list(di.get("pathway_context_core") or [])))
    ctx = {"key_compounds": ctx_terms} if ctx_terms else {}
    sdb = bool(qr.get("strict_db", True))
    rec["strict_db"] = sdb
    root = OUT / "work" / str(idx)
    shutil.rmtree(root, ignore_errors=True)
    (root / "reference").mkdir(parents=True)
    ref = next(p for p in (REPO / "reference/PW000001.pwml", REPO / "reference/PW012926.pwml") if p.exists())
    shutil.copy(ref, root / "reference" / ref.name)
    rm = read_run_meta(str(d))
    if rm.get("pmid_source"):
        rm.pop("pmid", None)
    paper = {k: str(rm[k]) for k in ("pmid", "pmcid", "doi", "title", "year") if rm.get(k)}
    ctx["pathway_name"] = m.get("pathway_name") or ""
    if m.get("organism"):
        ctx["likely_organism"] = m["organism"]
    state = {"final_payload_paper": paper}
    txt = next((c for c in (d / "01_source_text.txt", d.parent / "01_source_text.txt") if c.exists()), None)
    if txt is not None:
        state["final_payload_paper_text"] = txt.read_text(encoding="utf-8", errors="replace")
        state["final_payload_paper_text_source"] = "01_source_text.txt"
    payload = copy.deepcopy(payload)
    n0 = NoNet.n
    fp.maybe_apply_final_id_fill(payload, ctx, state, fetcher=NoNet(), cache=Cache(FILLCACHE))
    rec["network_attempts"] = NoNet.n - n0
    res = postpipeline.run_pwml_export(
        payload, "Generated Pathway", "", "Metabolic", root, root / "reference" / ref.name,
        3200, 1400, "#FFFFFF", None, sdb, None, None, ctx, coerce_mode(m["mode"]))
    rec["ok"] = bool(res.get("ok"))
    rec["error"] = str(res.get("error") or "")[:400]
    rec["counts"] = res.get("counts") or {}
    xb = res.get("xml_bytes")
    if res.get("ok") and xb:
        (OUT / "pwml" / f"{idx}.pwml").write_bytes(xb)
        rec.update(pwml_written=True, sha256=hashlib.sha256(xb).hexdigest(), size=len(xb))
    for k in ("pwml_ir", "pwml_ir_report", "pwml_ir_validation"):
        if res.get(k) is not None:
            (OUT / "ir" / f"{idx}.{k}.json").write_text(json.dumps(res[k], default=str), encoding="utf-8")
    shutil.rmtree(root, ignore_errors=True)
    return rec


sel = [int(x) for x in a.idx.split(",")] if a.idx else list(range(len(runs)))
out = open(OUT / "results.jsonl", "w", encoding="utf-8")
for idx in sel:
    t0 = time.time()
    box = {}

    def target():
        try:
            box["rec"] = one(idx, runs[idx])
        except BaseException as e:  # noqa: BLE001
            box["err"] = "".join(traceback.format_exception_only(type(e), e)).strip()[:400]

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(a.guard)
    rec = box.get("rec") or {"idx": idx, "ok": False, "pwml_written": False, "error": box.get("err", "TIMEOUT")}
    rec["seconds"] = round(time.time() - t0, 2)
    out.write(json.dumps(rec) + "\n")
    out.flush()
out.close()
print("REPLAY_DONE", a.label, "network_attempts", NoNet.n, flush=True)
os._exit(0)
