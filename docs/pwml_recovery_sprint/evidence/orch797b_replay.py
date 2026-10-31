"""ORCH-797b offline leg replay: Stage 2 -> normalise -> Stage 6 -> strict quarantine -> tail (freeze, export), base vs tip.

usage: python orch797b_replay.py <tree_root> <workdir> <responses.json> [--record] [leg ...]

The ORCH-792/793 technique (``evidence/orch792_replay.py``) on the ORCH-796 legs (latest stamp per paper):
the leg's own ``merged_payload.json`` (its rows and spellings, never rebuilt ones), the run's own
``data/id_mapping_cache.json`` (copied) as the recorded answers, taxonomy ids from the leg's ``final_mapped.json``,
no PathBank DB, no LLM (``tests/orch762_harness.py`` doors). One change: EVERY taxon-scoped UniProt answer of the run
(``api-v9::`` and ``api-fallback-v2::``; the tip keys them ``api-v10`` / ``api-fallback-v3``) is removed from both cache
copies, so each measured tree's parser re-reads UniProt's answers itself. Those answers, and any family search a tree
issues, come from ``<responses.json>``, recorded once with ``--record`` (live read-only UniProt GETs, tip then base;
no tail) and replayed offline by both arms, so base and tip judge identical answers. PathBank (``db::``), taxonomy
and the run's family-search answers stay the leg's own. Offline, an unrecorded request is answered 404 and counted.

Tail: the Stage-6 payload is written as the leg's ``final_mapped.json`` with the leg's other files, and replayed
by ``evidence/orch768_replay_corpus.py`` ``replay_leg`` as a pre-freeze checkpoint (enrichment from the run's cache,
pre-freeze resolution, freeze/quarantine, PWML export). Approximation: the audit, curator and gap rounds between
Stage 3 and Stage 6 are not run (harness limit), so counts can differ from the live leg; both arms share it.
Read-only on the legs; writes only under <workdir> and <responses.json>.
"""
import importlib.util
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
RESP = Path(sys.argv[3]).resolve()
RECORD = "--record" in sys.argv[4:]
legs_wanted = [a for a in sys.argv[4:] if not a.startswith("--")]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent

T = H = None
if not RECORD:  # the tail harness blocks sockets; import it first so the measured tree is proven
    spec = importlib.util.spec_from_file_location("orch768_replay_corpus", HERE / "orch768_replay_corpus.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    T = H.load_tree(tree / "src")
sys.path[:0] = [str(tree / "src"), str(tree / "tests")]

import orch762_harness as h  # noqa: E402
from t2pw.mapping import map_ids  # noqa: E402
from t2pw.pipeline.process_normalizer import normalize_process_payload  # noqa: E402
from t2pw.pipeline.strict_quarantine import ACCEPTED_STATES, quarantine_and_close  # noqa: E402

RUN = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch796run")
LEGS = {
    "chlorophyll": "chlorophyll-strict/2026-10-29_1632/papers/PMC5130987/strict",
    "wnt_ca": "wnt_ca-strict/2026-10-29_1941/papers/PMC10425814/strict",
    "wnt_pcp": "wnt_pcp-strict/2026-10-29_1729/papers/PMC10425814/strict",
    "sour_taste_v2": "sour_taste_v2-strict/2026-10-29_1314/papers/PMC7059786/strict",
    "enterobactin": "enterobactin-strict/2026-10-29_1153/papers/PMC12096016/strict",
    "menaquinone": "menaquinone-strict/2026-10-29_1451/papers/PMC12421875/strict",
}


class Resp:
    def __init__(self, status, body, text):
        self.status_code, self._body = status, body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        return deepcopy(self._body)


class Http:
    """UniProt answers from the recording; live (UniProt only) in --record mode; everything else 404."""

    def __init__(self):
        self.responses = json.loads(RESP.read_text(encoding="utf-8")) if RESP.exists() else {}
        self.calls, self.unrecorded, self.recorded_now = [], [], []

    def get(self, url, params=None, headers=None):
        key = h.request_key(url, params)
        self.calls.append(key)
        if not url.startswith("https://rest.uniprot.org/"):
            return Resp(404, {}, "")
        hit = self.responses.get(key)
        if hit is None and RECORD:
            import requests
            r = requests.get(url, params=params, timeout=60)
            is_json = "json" in (r.headers.get("content-type") or "")
            hit = {"status": r.status_code, "json": r.json() if is_json else None, "text": None if is_json else r.text}
            self.responses[key] = hit
            self.recorded_now.append(key)
        if hit is None:
            self.unrecorded.append(key)
            return Resp(404, {}, "")
        return Resp(int(hit["status"]), hit.get("json"), hit.get("text"))


def norm(s):
    return map_ids._normalize_name(str(s or ""))


def shipped(payload):
    out = {}
    ents = payload.get("entities") or {}
    for row in ents.get("proteins") or []:
        if isinstance(row, dict):
            out[str(row.get("name"))] = (row.get("mapped_ids") or {}).get("uniprot") or ""
    for row in ents.get("protein_complexes") or []:
        if isinstance(row, dict):
            fam = ((row.get("mapping_meta") or {}).get("family_complex") or {})
            if fam:
                out[f"[family complex] {row.get('name')}"] = ",".join(
                    f"{c.get('name')}={c.get('uniprot') or '-'}" if isinstance(c, dict) else str(c)
                    for c in row.get("components") or []) or "(no members: Unknown fallback)"
    return out


summary = {}
for leg in legs_wanted or list(LEGS):
    d = RUN / "runs_papers/ORCH-796/legs" / LEGS[leg]
    lw = work / leg
    lw.mkdir(parents=True, exist_ok=True)
    merged = json.loads((d / "merged_payload.json").read_text(encoding="utf-8"))
    final = json.loads((d / "final_mapped.json").read_text(encoding="utf-8"))
    cache = json.loads((RUN / "data/id_mapping_cache.json").read_text(encoding="utf-8"))
    prot = cache.get("proteins") or {}
    # Every taxon-scoped UniProt answer is re-asked (the tip keys them api-v10 / api-fallback-v3 because the
    # parsed candidate gained protein_names); PathBank (db::), taxonomy and family-search answers stay the leg's.
    dropped = [k for k in prot if k.startswith(("api-v9::", "api-fallback-v2::"))]
    targets = sorted({k.split("::")[1] for k in dropped})
    for k in dropped:
        del prot[k]
    for name in ("id_mapping_cache.json", "id_mapping_cache_stage6.json"):
        (lw / name).write_text(json.dumps(cache), encoding="utf-8")
    del cache, prot
    taxa = {norm(s.get("name")): s.get("taxonomy_id") for s in (final.get("entities") or {}).get("species") or []
            if isinstance(s, dict) and s.get("taxonomy_id")}
    for s in (merged.get("entities") or {}).get("species") or []:
        if isinstance(s, dict) and not s.get("taxonomy_id") and taxa.get(norm(s.get("name"))):
            s["taxonomy_id"] = taxa[norm(s.get("name"))]
    http = Http()
    with h.Doors(http) as doors:
        first = h.stage2(merged, lw)
        normalized, _ = normalize_process_payload(first["payload"])
        st6 = map_ids.map_payload(deepcopy(normalized), cache_path=lw / "id_mapping_cache_stage6.json",
                                  id_source="hybrid", use_cache=True, allow_complex_wrapper_creation=True,
                                  allow_structural_cleanup=True, merge_identity_duplicates=True)["payload"]
    if RECORD:
        RESP.write_text(json.dumps(http.responses, sort_keys=True, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{leg}: recorded {len(http.recorded_now)} new answers, {len(http.calls)} requests")
        continue
    q = quarantine_and_close(st6, strict_db=True, pathway_context=None)
    kept = [a for a in q.admissions if a.get("state") in ACCEPTED_STATES]
    omitted = [a for a in q.admissions if a.get("state") not in ACCEPTED_STATES]
    print(f"\n######## {leg}  tree={tree.name}  requests={len(http.calls)} unrecorded={len(http.unrecorded)} "
          f"ai={len(doors.ai_calls)} cache-answers-removed={len(dropped)} targets={len(targets)}")
    for key in http.unrecorded[:5]:
        print("  UNRECORDED", key[:200])
    print(f"  quarantine (strict): accepted={len(kept)} omitted={len(omitted)}")
    ids = shipped(st6)
    print("  shipped:", json.dumps(ids, ensure_ascii=False, sort_keys=True))
    for a in omitted:
        print(f"  omitted {a.get('process_kind')} {a.get('name')!r}: {a.get('reason')}")
    # Tail: the Stage-6 payload as the leg's pre-freeze checkpoint. NCBI backfill is off offline, so each
    # species row takes the fields the leg's own backfill wrote (classification etc.), as the taxonomy id did.
    leg_species = {norm(s.get("name")): s for s in (final.get("entities") or {}).get("species") or [] if isinstance(s, dict)}
    for s in (st6.get("entities") or {}).get("species") or []:
        for key, value in (leg_species.get(norm(s.get("name"))) or {}).items():
            if s.get(key) in (None, "", [], {}):
                s[key] = deepcopy(value)
        # The Unknown placeholder's species is PathBank species 4; offline (no DB) it stays "novel".
        if norm(s.get("name")) == "unknown species" and not s.get("pathbank_species_id"):
            s["pathbank_species_id"] = s["species_id"] = map_ids._PATHBANK_UNKNOWN_SPECIES_ID
    scratch = lw / "inputs" / "leg"
    scratch.mkdir(parents=True, exist_ok=True)
    for name in H.LEG_FILES:
        if (d / name).is_file():
            shutil.copyfile(d / name, scratch / name)
    (scratch / "final_mapped.json").write_text(json.dumps(st6, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(RUN / "data/enrichment_cache.json", scratch / "enrichment_cache.json")
    names = [n for n in list(H.LEG_FILES) + ["enrichment_cache.json"] if (scratch / n).is_file()]
    inputs = {n: {"path": f"leg/{n}", "sha256": H.sha256_file(scratch / n)} for n in names}
    try:
        r = H.replay_leg(T, {"id": f"ORCH-796/{leg} (ORCH-797b stage-6 replay)", "paper": d.parts[-2], "mode": "strict",
                             "inputs": inputs}, lw / "work", primary=lw / "inputs")
    except Exception as exc:  # noqa: BLE001 -- recorded; the next leg still runs
        r = {"failure_stage": f"replay_exception: {type(exc).__name__}: {str(exc)[:200]}", "pwml_written": None}
    cov = r.get("pwml_coverage") or {}
    tail = {k: r.get(k) for k in ("checkpoint", "pwml_written", "failure_stage")}
    tail["quarantine"] = {k: (r.get("quarantine") or {}).get(k) for k in ("ok", "refusal", "counts")}
    tail["coverage"] = {k: cov.get(k) for k in ("drawn", "total", "by_design_withheld", "no_pwml_cause")}
    tail["export"] = {k: (r.get("export") or {}).get(k) for k in ("ok", "error", "required_gate_codes", "ir_blocking")}
    tail["driver_codes"] = (r.get("driver") or {}).get("codes")
    tail["network_refused"] = len(r.get("network_refused") or [])
    print("  tail:", json.dumps(tail, default=str))
    summary[leg] = {"accepted": len(kept), "pwml_written": r.get("pwml_written"), "drawn": cov.get("drawn"),
                    "total": cov.get("total"), "unrecorded": len(http.unrecorded)}
    (scratch / "enrichment_cache.json").unlink()
if summary:
    print("\nSUMMARY", json.dumps(summary, sort_keys=True))
