"""ORCH-773 diagnosis: which surviving processes keep a failing entity row alive, and how.

Usage: python probe_keepers.py --src <tree>/src --cap <capture dir> --leg <slug> --entity <bucket>:<name> [...]

Runs the tree's quarantine_and_close on the captured freeze input, then, against the RESULT payload
(the graph closure left), lists for each named entity row: its name norms (name + synonyms, the
closure's own _entity_name_norms), every surviving process that references one of those norms
(slot, referenced name, actor provenance), and, for a protein, every surviving complex that lists it
as a component together with that complex's own referencing processes. Also prints, for each process
reference, what the quarantine registry resolves the name to. Read-only; offline.
"""

import argparse
import json
import socket
import sys
from pathlib import Path


def _block(*_a, **_k):
    raise OSError("ORCH773 probe: network blocked")


socket.socket.connect = _block
socket.create_connection = _block
socket.getaddrinfo = _block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--leg", required=True)
    ap.add_argument("--entity", nargs="+", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.src).resolve()))
    from t2pw.pipeline import strict_quarantine as SQ
    from t2pw.pipeline.process_normalizer import _entity_name_norms, _normalize

    doc = json.loads((Path(args.cap) / args.leg / "freeze_input.json").read_text(encoding="utf-8"))
    res = SQ.quarantine_and_close(doc["payload"], strict_db=True, pathway_context=doc["pathway_context"], mode="pathwhiz")
    p = res.payload
    reg = SQ._build_registry(p)
    procs = p.get("processes") or {}

    def refs(bucket, row):
        out = []
        for slot, names, _m in SQ._essential_participants(bucket, row):
            out += [(slot, n, None) for n in names]
        for slot in SQ._ACTOR_SLOTS.get(bucket, ()):
            for a in row.get(slot) or []:
                n = SQ._actor_name(a) if not isinstance(a, str) else SQ._canonical(a)
                if n:
                    out.append((slot, n, a.get("provenance") if isinstance(a, dict) else None))
        return out

    def keepers(norms):
        hits = []
        for bucket in SQ._PROCESS_KIND_BY_BUCKET:
            for i, row in enumerate(procs.get(bucket) or []):
                if not isinstance(row, dict):
                    continue
                for slot, n, prov in refs(bucket, row):
                    if _normalize(n) in norms:
                        r = reg.resolve(n)
                        hits.append(f"/processes/{bucket}/{i} {row.get('name')!r} {slot}={n!r} prov={prov} resolves->{(r[0], r[1].get('name')) if r else None}")
        return hits

    print(f"## {args.leg} ok={res.ok} refusal={res.refusal_reasons}")
    for spec in args.entity:
        bucket, name = spec.split(":", 1)
        rows = [r for r in (p.get("entities") or {}).get(bucket) or [] if isinstance(r, dict) and r.get("name") == name]
        for row in rows:
            norms = {n for n in _entity_name_norms([row]) if n}
            print(f"\n[{bucket}] {name!r} synonyms={row.get('synonyms')} generated={row.get('generated')} "
                  f"generation_reason={row.get('generation_reason')} provenance={row.get('provenance')}")
            print("   components:", json.dumps(row.get("components"), ensure_ascii=False, default=str)[:400] if row.get("components") is not None else None)
            for h in keepers(norms):
                print("   direct keeper:", h)
            if bucket == "proteins":
                for c in (p.get("entities") or {}).get("protein_complexes") or []:
                    if not isinstance(c, dict):
                        continue
                    comp = {_normalize(SQ._component_name(x)) for x in c.get("components") or []}
                    if comp & norms:
                        cn = {n for n in _entity_name_norms([c]) if n}
                        print(f"   component of complex {c.get('name')!r} generated={c.get('generated')}")
                        for h in keepers(cn):
                            print("      complex keeper:", h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
