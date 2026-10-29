"""ORCH-790/791 offline Stage-6 replay of three legs, then strict quarantine, at one tree.

usage: python orch790_replay.py <tree_root> <workdir> <record|replay>

Legs (read-only): ORCH-785 wnt_ca and chlorophyll, ORCH-784 wnt_pcp, from merged_payload.json,
through ``tests/orch762_harness.py`` of the measured tree (Stage 2 -> Stage 3 normalisation ->
Stage 6), then ``strict_quarantine.quarantine_and_close`` (strict_db=True). Doors: no PathBank
DB, no LLM; protein rows the leg resolved are answered with the leg's own recorded resolution
(``recorded_resolutions(final_mapped.json)``, the ORCH-759a technique); every other UniProt REST
answer is served from ``evidence/orch790/uniprot_responses.json``. ``record`` fetches the missing
UniProt answers live ONCE (rest.uniprot.org only) and saves them; ``replay`` is offline and lists
any unrecorded query. The Stage-2 cache is a copy of the leg run's ``data/id_mapping_cache.json``.

wnt_ca: the gap resolver's PKC accession (Par-3, Q8TEW0, ``best_effort_fallback``, score 0.39,
``gap_iteration_summary.json``) is put back on the PKC row before Stage 6, as the leg had it.
The audit and curator rounds are not run.

Per leg it prints: EIF3G anywhere, each family complex (members, note), each still-empty complex,
and every SIGNED interaction after quarantine with its verdict: drawn (both endpoints
non-protein-row entities), bare_protein_endpoint (an endpoint still names a protein row), or the
quarantine reason.
"""

import json
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
mode = sys.argv[3]
sys.path[:0] = [str(tree / "src"), str(tree / "tests")]

import orch762_harness as h  # noqa: E402
from t2pw.pipeline.process_normalizer import normalize_process_payload  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
from t2pw.pwml.interaction_types import interaction_relationship_word, pwml_interaction_type  # noqa: E402

RESPONSES = Path(__file__).resolve().parent / "orch790" / "uniprot_responses.json"
WT = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees")
LEGS = {
    "wnt_ca": (WT / "orch785run", "ORCH-785", "wnt_ca-strict"),
    "chlorophyll": (WT / "orch785run", "ORCH-785", "chlorophyll-strict"),
    "wnt_pcp": (WT / "orch784run", "ORCH-784", "wnt_pcp-strict"),
}
PKC_GAP = {"mapped_ids": {"uniprot": "Q8TEW0"}, "mapping_meta": {
    "provider": "UniProt", "source": "api", "confidence": 0.3875, "chosen_rule": "best_effort_fallback"}}


class Recorder(h.ReplayHttp):
    def __init__(self, responses):
        super().__init__(responses=responses)
        import requests
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Project14-T2PW-ORCH790-replay-recorder/1.0"
        self.fetched = 0

    def get(self, url, params=None, headers=None):
        key = h.request_key(url, params)
        self.calls.append(key)
        if not url.startswith(h.RECORDED_HOSTS):
            self.mocked.append(key)
            return h.FakeResponse(404, {})
        if key not in self.responses:
            time.sleep(0.2)
            resp = self.session.get(url, params=params, timeout=60)
            try:
                body = resp.json()
            except ValueError:
                body = resp.text  # the TSV family answers
            self.responses[key] = {"status": resp.status_code, "json": body}
            self.fetched += 1
        hit = self.responses[key]
        return Tsv(int(hit["status"]), hit["json"])


class Tsv(h.FakeResponse):
    """A recorded TSV answer is a string; ``.text`` must be that string, not its JSON."""

    def __init__(self, status, body):
        super().__init__(status, body)
        if isinstance(body, str):
            self.text = body


class Replay(h.ReplayHttp):
    def get(self, url, params=None, headers=None):
        key = h.request_key(url, params)
        hit = self.responses.get(key)
        resp = super().get(url, params, headers)
        return Tsv(int(hit["status"]), hit["json"]) if hit is not None and url.startswith(h.RECORDED_HOSTS) else resp


def norm(text):
    return " ".join(str(text or "").casefold().split())


def verdicts(qres):
    payload = qres.payload
    ents = payload.get("entities") or {}
    protein_names = {norm(r.get("name")) for r in ents.get("proteins") or [] if isinstance(r, dict)}
    names = set()
    for rows in ents.values():
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict):
                names.add(norm(r.get("name")))
                names.update(norm(s) for s in r.get("synonyms") or [] if isinstance(s, str))
    out = {}
    for row in (payload.get("processes") or {}).get("interactions") or []:
        if not isinstance(row, dict) or not pwml_interaction_type(interaction_relationship_word(row)):
            continue
        ends = [row.get(next((f for f in fields if row.get(f)), fields[0]))
                for fields in (("left", "entity_1", "source"), ("right", "entity_2", "target"))]
        if any(norm(e) in protein_names for e in ends):
            verdict = "bare_protein_endpoint"
        elif all(norm(e) in names for e in ends):
            verdict = "drawn"
        else:
            verdict = "undeclared_endpoint"
        out.setdefault(str(row.get("name")), []).append(f"{verdict}  [{ends[0]} -> {ends[1]}]")
    for adm in qres.admissions:
        if str(adm.get("pointer") or "").startswith("/processes/interactions/") and "accepted" not in str(adm.get("state")):
            out.setdefault(str(adm.get("name")), []).append(f"omitted:{adm.get('reason')}")
    return out


def main():
    responses = json.loads(RESPONSES.read_text(encoding="utf-8")) if RESPONSES.exists() else {}
    print("tree:", tree, "| mode:", mode, "| recorded answers:", len(responses))
    for leg, (root, task, paper) in LEGS.items():
        [strict] = sorted((root / "runs_papers" / task / "legs" / paper).glob("*/papers/*/strict"))
        lw = work / leg
        lw.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "data" / "id_mapping_cache.json", lw / "id_mapping_cache.json")
        merged = json.loads((strict / "merged_payload.json").read_text(encoding="utf-8"))
        final = json.loads((strict / "final_mapped.json").read_text(encoding="utf-8"))
        # NCBI backfill is off offline; the species taxonomy id the leg's backfill wrote is put
        # back (the orch762 fixture convention), or no family UniProt search could run.
        taxa = {norm(r.get("name")): r.get("taxonomy_id") for r in final["entities"].get("species", [])
                if isinstance(r, dict) and r.get("taxonomy_id")}
        for row in merged["entities"].get("species", []):
            if isinstance(row, dict) and not row.get("taxonomy_id") and taxa.get(norm(row.get("name"))):
                row["taxonomy_id"] = taxa[norm(row.get("name"))]
        http = Recorder(responses) if mode == "record" else Replay(responses)
        with h.Doors(http, resolutions=h.recorded_resolutions(final)) as doors:
            first = h.stage2(merged, lw)
            normalized, _ = normalize_process_payload(first["payload"])
            if leg == "wnt_ca":
                for row in normalized["entities"]["proteins"]:
                    if row.get("name") == "PKC":
                        row["mapped_ids"] = dict(PKC_GAP["mapped_ids"])
                        row.setdefault("mapping_meta", {}).update(PKC_GAP["mapping_meta"])
                print("PKC rows in the Stage-6 input:", [r.get("name") for r in normalized["entities"]["proteins"]
                                                         if norm(r.get("name")) in ("pkc", "protein kinase c")])
                # Trace only: which PKC rows reach the family conversion, and with what identity.
                from t2pw.mapping import map_ids
                real_family = map_ids._apply_family_complexes

                def traced(mapped, **kw):
                    for r in mapped["entities"]["proteins"]:
                        if norm(r.get("name")) in ("pkc", "protein kinase c"):
                            m = r.get("mapping_meta") or {}
                            print("  at family conversion:", r.get("name"), r.get("mapped_ids"), m.get("chosen_rule"),
                                  (m.get("resolution") or {}).get("status"))
                    return real_family(mapped, **kw)
                map_ids._apply_family_complexes = traced
                try:
                    out = h.stage6(normalized, lw)
                finally:
                    map_ids._apply_family_complexes = real_family
            else:
                out = h.stage6(normalized, lw)
        stage6 = out["payload"]
        print(f"\n===== {leg}  ({strict.relative_to(root)})  species taxa: {sorted(taxa.values())}")
        print(f"http: {len(http.calls)} requests, unrecorded {len(http.unrecorded)}, mocked {len(http.mocked)}"
              + (f", fetched live {http.fetched}" if mode == "record" else "") + f" | ai calls {len(doors.ai_calls)}")
        ents = stage6["entities"]
        eif3g = [r.get("name") for r in ents.get("proteins", [])
                 if "O75821" in json.dumps(r.get("mapped_ids") or {}) or str(r.get("uniprot_id")) == "O75821"]
        print("EIF3G (O75821) protein rows:", eif3g or "none")
        print("stage-6 quarantined_proteins:", [(q.get("name"), q.get("reason")) for q in stage6.get("quarantined_proteins") or []])
        if leg == "wnt_ca":
            rep = json.dumps(out.get("report") or {}, ensure_ascii=False, default=str)
            def walk(o, path):
                if isinstance(o, dict):
                    if "PKC" in o.values():
                        print(f"  stage-6 report {path}:", json.dumps(o, ensure_ascii=False, default=str)[:300])
                    for k, v in o.items():
                        walk(v, f"{path}/{k}")
                elif isinstance(o, list):
                    for i, v in enumerate(o):
                        walk(v, f"{path}/{i}")
            walk(out.get("report") or {}, "")
            print("  'PKC' occurrences in the stage-6 report:", rep.count('"PKC"'))
            print("PKC rows after stage 6:", [(b, r.get("name"), (r.get("mapped_ids") or {}).get("uniprot"))
                                              for b in ("proteins", "protein_complexes") for r in ents.get(b, [])
                                              if norm(r.get("name")) in ("pkc", "protein kinase c")])
        for c in ents.get("protein_complexes", []):
            fam = (c.get("mapping_meta") or {}).get("family_complex")
            comps = [x.get("name") if isinstance(x, dict) else x for x in c.get("components") or []]
            if fam:
                print(f"  family complex: {c.get('name')} -> {comps} | synonyms {c.get('synonyms') or []} | {fam.get('note')}")
            elif not comps:
                print(f"  EMPTY complex: {c.get('name')}")
        qres = quarantine_and_close(deepcopy(stage6), strict_db=True)
        rows = verdicts(qres)
        drawn = sum(1 for v in rows.values() for x in v if x.startswith("drawn"))
        bare = sum(1 for v in rows.values() for x in v if x.startswith("bare_protein_endpoint"))
        print(f"signed interactions after quarantine: drawn {drawn}, bare_protein_endpoint {bare}")
        for name in sorted(rows):
            for v in rows[name]:
                print(f"    {name}: {v}")
        if mode == "record":
            RESPONSES.parent.mkdir(parents=True, exist_ok=True)
            RESPONSES.write_text(json.dumps(responses, indent=0, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        if http.unrecorded:
            print("UNRECORDED:", http.unrecorded[:10])


if __name__ == "__main__":
    main()
