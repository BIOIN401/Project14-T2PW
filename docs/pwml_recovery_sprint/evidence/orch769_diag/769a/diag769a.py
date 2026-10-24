"""ORCH-769a read-only diagnosis. Loads leg artifacts, runs pure gate/fold functions on in-memory
copies. Writes nothing (stdout only)."""
import json
from copy import deepcopy

from t2pw.pipeline.process_normalizer import run_strict_post_normalization_gates, GateValidationError
from t2pw.pipeline.gate_reports import payload_sha256
from t2pw.mapping.map_ids import _fold_unreferenced_duplicate_proteins

ROOT = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
L767 = ROOT + "/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/enterobactin-strict/2026-10-23_2224/papers/PMC12096016/strict"
L759P = ROOT + "/runs_papers/ORCH-759p/legs/enterobactin-strict/2026-10-22_2101/papers/PMC12096016/strict"
WRAP = "enterobactin synthase complex"
SYN = "enterobactin synthase"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def gate(label, payload):
    try:
        d = run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)
        errs, ok = d.get("errors") or [], True
    except GateValidationError as exc:
        errs, ok = (exc.details or {}).get("errors") or [], False
    print(f"[gate] {label}: ok={ok} errors={len(errs)}")
    for e in errs:
        print("    ", e.get("path"), "|", e.get("reason"))
    return ok


def drop_actor(payload, name):
    n = 0
    for bucket in ("reactions", "transports", "interactions"):
        for row in payload["processes"].get(bucket) or []:
            for key in ("enzymes", "modifiers", "transporters"):
                lst = row.get(key)
                if isinstance(lst, list):
                    keep = [a for a in lst if not (isinstance(a, dict) and (a.get("entity") or a.get("protein_complex")) == name)]
                    n += len(lst) - len(keep)
                    row[key] = keep
    return n


fm = load(L767 + "/final_mapped.json")
rep = load(L767 + "/final_stage3_gate_report.json")
print("[hash] final_mapped payload_sha256 =", payload_sha256(fm))
print("[hash] gate report payload_sha256  =", rep.get("payload_sha256"))

gate("767 final_mapped as shipped", fm)

a = deepcopy(fm)
a["entities"]["protein_complexes"][0]["components"][0]["stoichiometry"] = 1
gate("CF-A: wrapper component stoichiometry restored to 1 (the normalizer's own value)", a)

b = deepcopy(fm)
print("[cf-b] actor refs dropped:", drop_actor(b, WRAP))
b["entities"]["protein_complexes"] = [c for c in b["entities"]["protein_complexes"] if c.get("name") != WRAP]
gate("CF-B1: inferred actor + wrapper removed, synthetic protein row kept", b)

b2 = deepcopy(b)
rpt = {}
_fold_unreferenced_duplicate_proteins(b2, rpt)
print("[cf-b] 759a fold on CF-B1:", json.dumps(rpt.get("duplicate_protein_folds"), ensure_ascii=False)[:600])
ente = next(p for p in b2["entities"]["proteins"] if p.get("name") == "EntE")
print("[cf-b] EntE synonyms after fold:", ente.get("synonyms"))
gate("CF-B2: CF-B1 after the 759a fold", b2)

c = deepcopy(b)
c["entities"]["proteins"] = [p for p in c["entities"]["proteins"] if p.get("name") != SYN]
for k, v in c["element_locations"].items():
    if isinstance(v, list):
        c["element_locations"][k] = [r for r in v if not (isinstance(r, dict) and r.get("protein") == SYN)]
gate("CF-C: inferred actor + wrapper + synthetic row + its location removed", c)

rfm = load(L767 + "/final_mapped.json")
prots = rfm["entities"]["proteins"]
print("[ids] rows carrying P10378:", [(i, p.get("name")) for i, p in enumerate(prots) if (p.get("mapped_ids") or {}).get("uniprot") == "P10378"])
print("[ids] rows carrying P11454:", [(i, p.get("name")) for i, p in enumerate(prots) if (p.get("mapped_ids") or {}).get("uniprot") == "P11454"])

gate("759p final_mapped (passing control)", load(L759P + "/final_mapped.json"))
