"""Per-run funnel scan. Read-only on runs. Writes runs_funnel.jsonl (+ names_funnel.jsonl for name-level stage diffs).
Usage: python scan_funnel.py OUTDIR
"""
import sys, json, os, re, collections
import xml.etree.ElementTree as ET

R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
ROWS = [json.loads(l) for l in open("C:/t/ev-dedup/out/runs.jsonl")]
EB = ("compounds", "proteins", "protein_complexes", "nucleic_acids", "element_collections")


def J(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def nm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().casefold())


def ents(p):
    e = (p or {}).get("entities") or {}
    out = {}
    for b in EB:
        out[b] = [r for r in (e.get(b) or []) if isinstance(r, dict) and str(r.get("name") or "").strip()]
    return out


def procs(p):
    pr = (p or {}).get("processes") or {}
    return {k: [r for r in (pr.get(k) or []) if isinstance(r, dict)] for k in ("reactions", "transports", "interactions")}


def pname(kind, r):
    if kind == "interactions":
        return nm(r.get("name")) or nm(str(r.get("entity_1")) + "|" + str(r.get("relationship")) + "|" + str(r.get("entity_2")))
    return nm(r.get("name")) or nm(json.dumps([r.get("inputs"), r.get("outputs"), r.get("cargo")], sort_keys=True))


def counts(p):
    if p is None:
        return None
    E = ents(p)
    P = procs(p)
    c = {b: len(E[b]) for b in EB}
    c.update({k: len(v) for k, v in P.items()})
    return c


def names(p):
    E = ents(p)
    P = procs(p)
    s = {}
    for b in EB:
        s[b] = sorted({nm(r["name"]) for r in E[b]})
    for k, v in P.items():
        s[k] = sorted({pname(k, r) for r in v})
    return s


def mapstate(r, bucket):
    mm = ((r.get("mapping_meta") or {}).get("resolution")) or {}
    return mm.get("status") or ("none")


def has_ext_id(r, bucket):
    ids = r.get("mapped_ids") if isinstance(r.get("mapped_ids"), dict) else {}
    if bucket == "compounds":
        return bool(r.get("pathbank_compound_id") or ids.get("pathbank_compound_id") or any(ids.get(k) for k in ("hmdb", "chebi", "kegg", "pubchem", "drugbank")))
    if bucket == "proteins":
        u = str(ids.get("uniprot") or r.get("uniprot") or "").strip()
        return bool((u and u.casefold() != "unknown") or r.get("pathbank_protein_id") or ids.get("pathbank_protein_id") or ids.get("drugbank"))
    return None


def lineage_origin(r):
    lin = r.get("provenance_lineage") or []
    stages = [(x.get("stage"), x.get("origin")) for x in lin if isinstance(x, dict)]
    first = stages[0] if stages else (None, None)
    return first, stages


def pwml_counts(path):
    try:
        root = ET.parse(path).getroot()
    except Exception as ex:
        return {"parse_error": str(ex)[:80]}
    pv = root.find("./pathway-visualization-contexts/pathway-visualization-context/pathway-visualization")
    if pv is None:
        return {"parse_error": "no pv"}
    out = {}
    for tag in ("compounds", "proteins", "protein-complexes", "nucleic-acids", "element-collections", "reactions",
                "reaction-coupled-transports", "transports", "interactions", "biological-states", "subcellular-locations"):
        n = pv.find(tag)
        out[tag] = len(list(n)) if n is not None else 0
    return out


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    fo = open(os.path.join(outdir, "runs_funnel.jsonl"), "w", encoding="utf-8")
    fn = open(os.path.join(outdir, "names_funnel.jsonl"), "w", encoding="utf-8")
    for r in ROWS:
        p = r["path"]
        full = p if os.path.isabs(p) else R + p
        d = os.path.dirname(full)
        rec = {"path": p, "root": r["root"], "qok": r["quarantine_ok"]}
        m = re.search(r"(PMC\d+)", p)
        rec["paper"] = m.group(1) if m else ("web:PMID32185015" if "e2e4" in p else "web:e2e5" if "e2e5" in p else "?")
        dm = re.search(r"(20\d\d-\d\d-\d\d)(_\d{4})?", p)
        rec["date"] = dm.group(1) if dm else ""
        s1 = J(d + "/stage1_payload.json")
        mg = J(d + "/merged_payload.json")
        fm = J(d + "/final_mapped.json")
        qr = J(d + "/quarantine_report.json")
        rem = J(d + "/removed_entity_report.json")
        gc = J(d + "/graph_closure_iterations.json")
        rec["s1"], rec["m"], rec["f"] = counts(s1), counts(mg), counts(fm)
        fn.write(json.dumps({"path": p, "s1": names(s1), "m": names(mg), "f": names(fm)}) + "\n")
        # merged entity admission
        ea = (mg or {}).get("entity_admission_report") or {}
        rec["ea_removed"] = len(ea.get("removed") or [])
        rec["ea_removed_rules"] = dict(collections.Counter((x.get("phase"), x.get("rule"), x.get("kind")) .__str__() for x in (ea.get("removed") or [])))
        rec["ea_demoted"] = len(ea.get("demoted") or [])
        rec["ea_demoted_reasons"] = dict(collections.Counter(str(x.get("reason") or x.get("rule") or x.get("to")) for x in (ea.get("demoted") or [])))
        rec["qlocked_m"] = len((mg or {}).get("quarantined_locked_reactions") or [])
        # final mapping status
        ms = collections.Counter()
        ext = collections.Counter()
        if fm:
            E = ents(fm)
            for b in ("compounds", "proteins", "protein_complexes"):
                for row in E[b]:
                    ms[b + ":" + mapstate(row, b)] += 1
                    h = has_ext_id(row, b)
                    if h is not None:
                        ext[b + (":id" if h else ":noid")] += 1
            # process provenance (first lineage origin)
            po = collections.Counter()
            P = procs(fm)
            for k, v in P.items():
                for row in v:
                    first, stages = lineage_origin(row)
                    last_stage = stages[-1] if stages else (None, None)
                    po[k + ":" + str(first[1])] += 1
                    for st, org in stages[1:]:
                        if st and st != "identifier_mapping":
                            po[k + ":touched_by:" + str(st)] += 1
            rec["f_proc_origin"] = dict(po)
            eo = collections.Counter()
            for b in ("compounds", "proteins", "protein_complexes", "nucleic_acids"):
                for row in E[b]:
                    first, stages = lineage_origin(row)
                    eo[b + ":" + str(first[1])] += 1
            rec["f_ent_origin"] = dict(eo)
        gen = collections.Counter()
        for tag, pp in (("m", mg), ("f", fm)):
            if pp:
                for b, rowsb in ents(pp).items():
                    for row in rowsb:
                        if row.get("generated") is True:
                            gen[tag + ":" + b + ":" + str(row.get("generation_reason"))] += 1
        rec["generated"] = dict(gen)
        mo = collections.Counter()
        for tag, pp in (("m", mg), ("f", fm)):
            if pp:
                for k, v in procs(pp).items():
                    for row in v:
                        L = row.get("provenance_lineage") or []
                        o = str(L[0].get("origin")) if L else "none"
                        sa = k == "interactions" and str(row.get("relationship")).upper() == "SAME_AS"
                        mo[f"{tag}:{k}:{'SAME_AS' if sa else o}"] += 1
        rec["proc_origin"] = dict(mo)
        rec["f_map"] = dict(ms)
        rec["f_ext"] = dict(ext)
        # quarantine
        if qr:
            rec["mode"] = qr.get("export_mode")
            rec["research_applied"] = (qr.get("research_mode") or {}).get("applied")
            rec["q_counts"] = qr.get("counts")
            qk = collections.Counter(); qa = collections.Counter(); qreason = collections.Counter()
            for x in qr.get("quarantined") or []:
                qk[x.get("process_kind")] += 1
                qreason[(x.get("process_kind"), x.get("state"), x.get("reason"))] += 1
            for x in qr.get("accepted") or []:
                qa[x.get("process_kind")] += 1
            rec["q_quar_kind"] = dict(qk)
            rec["q_acc_kind"] = dict(qa)
            rec["q_reasons"] = {"|".join(map(str, k)): v for k, v in qreason.items()}
            rec["q_release"] = (qr.get("release") or {}).get("status")
        else:
            rec["mode"] = "strict" if "/strict" in p or p.endswith("strict/final_mapped.json") else ("research" if "/research" in p else "?")
        if rem:
            rec["rem_counts"] = rem.get("counts")
            rr = collections.Counter()
            for x in rem.get("removed_entities") or []:
                rr[(x.get("bucket"), x.get("reason"))] += 1
            rec["rem_reasons"] = {"|".join(map(str, k)): v for k, v in rr.items()}
        if gc:
            rec["gc_iter"] = gc.get("iteration_count")
            tot = collections.Counter()
            for it in gc.get("iterations") or []:
                tot["ent"] += len(it.get("removed_entities") or [])
                tot["loc"] += len(it.get("removed_locations") or [])
                tot["bs"] += len(it.get("removed_biological_states") or [])
                tot["proc"] += len(it.get("quarantined_processes") or [])
            rec["gc_removed"] = dict(tot)
        # PWML
        for fnm in ("pathway.pwml", "pathway.review_required.pwml"):
            if os.path.exists(d + "/" + fnm):
                rec["pwml_file"] = fnm
                rec["pwml"] = pwml_counts(d + "/" + fnm)
        # pwml ir report
        ir = J(d + "/pwml_ir_report.json")
        if ir:
            rec["ir_counts"] = ir.get("counts")
            rec["ir_unresolved"] = {k: len(v) for k, v in (ir.get("unresolved") or {}).items() if isinstance(v, list)}
            rec["ir_errors"] = len(ir.get("errors") or [])
        rf = J(d + "/pwml_required_field_gate_report.json")
        if rf:
            rec["rf_ok"] = rf.get("ok")
            rec["rf_err_codes"] = (rf.get("summary") or {}).get("error_codes")
        cs = J(d + "/coverage_summary.json") or {}
        rc = cs.get("requested_context") or {}
        rec["pathway_name"] = rc.get("pathway_name")
        rec["context_type"] = rc.get("context_type")
        rec["document_type"] = rc.get("document_type")
        rec["organism"] = rc.get("likely_organism")
        # normalization stats (initial gate report) : drop-related counters
        ig = J(d + "/initial_stage3_gate_report.json") or {}
        rec["norm_stats"] = {k: v for k, v in (ig.get("normalization_stats") or {}).items() if isinstance(v, (int, float)) and v}
        # audit / gap
        au = J(d + "/audit_iteration_summary.json")
        if au:
            rd = au.get("rounds") or []
            rec["audit_rounds"] = len(rd)
            rec["audit_accepted_patches"] = sum((x.get("accepted_patch_count") or 0) for x in rd)
            rec["audit_rejected_patches"] = sum((x.get("rejected_patch_count") or 0) for x in rd)
            rec["audit_llm_ok_rounds"] = sum(1 for x in rd if x.get("llm_ok"))
            rec["audit_payload_changed"] = any(x.get("audit_payload_changed") for x in rd)
            rec["gap_payload_changed"] = any(x.get("gap_payload_changed") for x in rd)
        # run result
        rec["files"] = sorted(os.listdir(d))
        fo.write(json.dumps(rec) + "\n")
    fo.close()
    fn.close()


if __name__ == "__main__":
    main(sys.argv[1])
