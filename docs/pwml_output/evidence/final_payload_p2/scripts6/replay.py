"""Flag-OFF/ON replay of stored final_mapped payloads through run_pwml_export. No LLM, no pipeline."""
import argparse, copy, csv, hashlib, json, os, shutil, sys, threading, time, traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--flag", type=int, default=0)
ap.add_argument("--label", required=True)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=1)
ap.add_argument("--guard", type=float, default=240.0, help="per-run seconds before the run is abandoned")
ap.add_argument("--strict-db", default="report", help="report (run's stored value, else True) | 1 | 0")
ap.add_argument("--pass", dest="pas", type=int, default=1)
ap.add_argument("--idx", default="", help="comma list of idx to run instead of start/count")
a = ap.parse_args()

sys.path.insert(0, a.src)
os.environ["T2PW_FINAL_PAYLOAD"] = str(a.flag)
os.environ["T2PW_OFFLINE_CURATOR"] = "1"

from t2pw.service import postpipeline  # noqa: E402
from t2pw.pipeline.export_mode import coerce_mode  # noqa: E402

REPO = Path(a.src).resolve().parent  # repo root of the code under test (reference/ lives here)
OUT = Path("C:/t/fp/docs/pwml_output/evidence/final_payload_p2/replay6") / a.label
(OUT / "pwml").mkdir(parents=True, exist_ok=True)
P = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
meta = {}
for r in csv.DictReader(open("C:/t/fp/docs/pwml_output/evidence/01/tidy_runs.csv", encoding="utf-8")):
    k = r["path"].rsplit("/final_mapped.json", 1)[0]
    meta[k[len(P):] if k.startswith(P) else k] = r
runs = [x for x in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if x["q"]]


FILLCACHE = "C:/t/ev-fp/fillcache"


class CountingFetcher:
    """Counts every call that reaches the network (cache hits never get here)."""

    def __init__(self, inner):
        self.inner, self.n, self.err = inner, 0, 0

    def _c(self, f, url, params):
        self.n += 1
        try:
            return f(url, params)
        except Exception:
            self.err += 1
            raise

    def get_json(self, url, params=None):
        return self._c(self.inner.get_json, url, params)

    def get_text(self, url, params=None):
        return self._c(self.inner.get_text, url, params)


def key(run):
    return run["dir"][len(P):] if run["dir"].startswith(P) else run["dir"]


def one(idx, run):
    d = Path(run["dir"])
    m = meta[key(run)]
    rec = {"idx": idx, "dir": run["dir"], "group": m["group"], "mode": m["mode"], "ok": False,
           "error": "", "pwml_written": False, "sha256": "", "size": 0, "counts": {}, "stage": ""}
    payload = json.load(open(d / "final_mapped.json", encoding="utf-8"))
    qr = {}
    if (d / "quarantine_report.json").exists():
        qr = json.load(open(d / "quarantine_report.json", encoding="utf-8"))
    rec["has_qreport"] = bool(qr)
    di = qr.get("decision_inputs") or {}
    # Stored decisions carry an old policy_version, so run_pwml_export would refuse them as
    # "controls changed". Pass NO report (boundary re-runs on the stored payload) and rebuild
    # the Stage-0 pathway_context from the core terms the old report recorded.
    ctx_terms = list(dict.fromkeys(list(di.get("requested_core") or []) + list(di.get("pathway_context_core") or [])))
    ctx = {"key_compounds": ctx_terms} if ctx_terms else None
    rec["ctx_terms"] = len(ctx_terms)
    sdb = bool(qr.get("strict_db", True)) if a.strict_db == "report" else a.strict_db == "1"
    rec["strict_db"] = sdb
    root = OUT / "work" / str(idx)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    (root / "reference").mkdir(parents=True)
    ref = next((p for p in (REPO / "reference/PW000001.pwml", REPO / "reference/PW012926.pwml") if p.exists()), None)
    if ref is None:
        raise RuntimeError("no reference pwml under " + str(REPO))
    shutil.copy(ref, root / "reference" / ref.name)
    if a.flag:
        from t2pw.clean_model.builder import read_run_meta
        from t2pw.clean_model.fill.cache import Cache
        from t2pw.service import final_payload as fp
        rm = read_run_meta(str(d))
        if rm.get("pmid_source"):  # guessed pmid (PMC header token): not an identifier, as apply_final_pathway_meta also drops it
            rm.pop("pmid", None)
        paper = {k: str(rm[k]) for k in ("pmid", "pmcid", "doi", "title", "year") if rm.get(k)}
        ctx = dict(ctx or {})
        ctx["pathway_name"] = m.get("pathway_name") or ""
        if m.get("organism"):
            ctx["likely_organism"] = m["organism"]
        state = {"final_payload_paper": paper}
        payload = copy.deepcopy(payload)
        fetcher = CountingFetcher(fp._live_fetcher())
        frep = fp.maybe_apply_final_id_fill(payload, ctx, state, fetcher=fetcher, cache=Cache(FILLCACHE))
        rec["paper_meta"] = paper
        rec["live_requests"] = fetcher.n
        rec["fetch_errors"] = fetcher.err
        rec["fill_error"] = frep.get("error")
        rec["fill_written"] = frep.get("written")
        rec["fill_summary"] = {k: frep.get(k) for k in ("written", "by_field", "refused", "meta_keys", "error")}
        rec["fill_summary"]["fill_requests"] = (frep.get("fill") or {}).get("requests")
        rec["fill_summary"]["fill_cache_hits"] = (frep.get("fill") or {}).get("cache_hits")
        rec["pathway_meta_report"] = frep.get("pathway_meta")
        rec["fill_entries"] = (payload.get("final_payload_fill") or {}).get("entries") or []
        rec["refused_detail"] = frep.get("refused_detail") or []
        fpp = payload.get("final_payload_pathway") or {}
        rec["title"] = (fpp.get("title") or {}).get("value")
        rec["title_source"] = (fpp.get("title") or {}).get("source")
        rec["reference"] = fpp.get("reference")
    res = postpipeline.run_pwml_export(
        payload, "Generated Pathway", "", "Metabolic", root, root / "reference" / ref.name,
        3200, 1400, "#FFFFFF", None, sdb, None, None, ctx, coerce_mode(m["mode"]))
    rec["ok"] = bool(res.get("ok"))
    rec["error"] = str(res.get("error") or "")[:400]
    rec["refusal_reasons"] = list(res.get("refusal_reasons") or [])[:5]
    rec["counts"] = res.get("counts") or {}
    rec["quarantine_reused"] = res.get("quarantine_reused")
    qa = res.get("qa") or {}
    rec["qa_ok"] = qa.get("ok")
    rec["qa_errors"] = len(qa.get("errors") or [])
    rec["qa_warnings"] = len(qa.get("warnings") or [])
    xb = res.get("xml_bytes")
    if res.get("ok") and xb:
        (OUT / "pwml" / f"{idx}.pwml").write_bytes(xb)
        rec.update(pwml_written=True, sha256=hashlib.sha256(xb).hexdigest(), size=len(xb))
    else:
        for k, st in (("stage3_contract_report", "stage3_revalidation"), ("required_gate_report", "required_field_gate")):
            if k in res:
                rec["stage"] = st
        if "prefreeze_resolution_error" in res:
            rec["stage"] = "prefreeze_resolution"
        rec["stage"] = rec["stage"] or ("quarantine_boundary" if res.get("refusal_reasons") else "exception_or_other")
    shutil.rmtree(root, ignore_errors=True)
    return rec


sel = [int(x) for x in a.idx.split(",")] if a.idx else list(range(a.start, min(a.start + a.count, len(runs))))
out = open(OUT / f"results_{a.start}.jsonl", "w", encoding="utf-8")
for idx in sel:
    t0 = time.time()
    box = {}

    def target():
        try:
            box["rec"] = one(idx, runs[idx])
        except BaseException as e:  # noqa: BLE001
            box["err"] = "".join(traceback.format_exception_only(type(e), e)).strip()[:400]
            box["tb"] = traceback.format_exc()[-800:]

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(a.guard)
    if "rec" in box:
        rec = box["rec"]
    else:
        # ponytail: a timed-out thread cannot be killed; it is abandoned and os._exit ends the process
        m = meta.get(key(runs[idx]), {})
        rec = {"idx": idx, "dir": runs[idx]["dir"], "group": m.get("group", ""), "mode": m.get("mode", ""),
               "ok": False, "pwml_written": False, "sha256": "", "size": 0, "counts": {},
               "error": box.get("err", "TIMEOUT>%ss" % a.guard),
               "stage": "harness_exception" if "err" in box else "harness_timeout"}
        if "tb" in box:
            rec["tb"] = box["tb"]
    rec["seconds"] = round(time.time() - t0, 2)
    rec["pass"] = a.pas
    with open(OUT / "requests.jsonl", "a", encoding="utf-8") as rf:
        rf.write(json.dumps({"idx": idx, "pass": a.pas, "live": rec.get("live_requests", 0), "errors": rec.get("fetch_errors", 0), "seconds": rec["seconds"]}) + chr(10))
    out.write(json.dumps(rec) + "\n")
    out.flush()
out.close()
os._exit(0)
