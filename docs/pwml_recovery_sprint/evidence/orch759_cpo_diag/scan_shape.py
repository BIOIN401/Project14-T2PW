"""Stdlib-only, read-only: find the CPO shape across every archived final_mapped.json.

Shape: a reaction enzyme/modifier actor that is a bare PROTEIN (not a protein_complex) while
the protein row carries a real UniProt/DrugBank id. For each hit, record the row's
mapping_meta.chosen_rule (the CPO root cause is `api_uniprot_fallback`: identity first/again
established by the Phase-2 UniProt fallback, which runs after the Stage-6 wrapper rewrite),
whether a same-family placeholder wrapper sits in the same actor list, and the leg's
pwml_required_field_gate_report.json codes if one exists.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
DIRS = ["runs_papers", "runs_verify", "runs", "runs_validation", "runs_smoke"]
SENT = {"", "unknown", "n/a", "na", "none"}


def norm(s):
    return " ".join(str(s or "").casefold().split())


def ident(row):
    for c in (row, row.get("mapped_ids") or {}):
        if not isinstance(c, dict):
            continue
        for k in ("uniprot", "uniprot_id", "drugbank", "drugbank_id"):
            v = str(c.get(k) or "").strip()
            if v and v.casefold() not in SENT:
                return v
    return ""


def actor_name(a):
    if isinstance(a, str):
        return a, ""
    if not isinstance(a, dict):
        return "", ""
    for k in ("entity", "protein_complex", "protein", "name"):
        if a.get(k):
            t = a.get("entity_type") or ("protein_complex" if k == "protein_complex" else ("protein" if k == "protein" else ""))
            return str(a[k]), str(t)
    return "", ""


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
legs = hits = 0
by_rule = {}
for d in DIRS:
    for fm_path in sorted((ROOT / d).rglob("final_mapped.json")):
        if "/research/" in fm_path.as_posix() or "worktrees" in fm_path.as_posix():
            continue  # research mode never wraps enzymes by design
        try:
            p = json.loads(fm_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print("UNREADABLE", fm_path, exc)
            continue
        legs += 1
        ents = p.get("entities") or {}
        prots = {norm(r.get("name")): r for r in ents.get("proteins") or [] if isinstance(r, dict)}
        cplx = {norm(r.get("name")): r for r in ents.get("protein_complexes") or [] if isinstance(r, dict)}
        gate = fm_path.parent / "pwml_required_field_gate_report.json"
        gate_codes = None
        if gate.exists():
            try:
                g = json.loads(gate.read_text(encoding="utf-8"))
                gate_codes = sorted({e.get("code") for e in g.get("errors") or []})
            except Exception:  # noqa: BLE001
                gate_codes = ["unreadable"]
        leg_hits = []
        for i, rx in enumerate((p.get("processes") or {}).get("reactions") or []):
            if not isinstance(rx, dict):
                continue
            for key in ("enzymes", "modifiers"):
                acts = rx.get(key) or []
                for j, a in enumerate(acts):
                    name, typ = actor_name(a)
                    n = norm(name)
                    if not n or typ == "protein_complex" or (not typ and n in cplx):
                        continue
                    row = prots.get(n)
                    if row is None or not ident(row):
                        continue
                    if key == "modifiers" and isinstance(a, dict) and norm(a.get("role")) not in {"catalyst", "activator", "inhibitor", "enzyme", ""}:
                        continue
                    mm = row.get("mapping_meta") or {}
                    siblings = [actor_name(b)[0] for b in acts if b is not a]
                    ph = [s for s in siblings if norm(s) in cplx and (cplx[norm(s)].get("mapping_meta") or {}).get("chosen_rule") == "pathbank_unknown_protein_fallback"]
                    leg_hits.append((i, rx.get("name"), key, j, name, ident(row), mm.get("chosen_rule"), mm.get("identity_status"),
                                     bool(mm.get("rejected_mapped_ids")), ph))
        if leg_hits:
            hits += 1
            rel = fm_path.relative_to(ROOT).as_posix()
            print(f"\n{rel}  gate_report_codes={gate_codes}")
            seen = set()
            for h in leg_hits:
                by_rule[h[6]] = by_rule.get(h[6], 0) + 1
                k = (h[0], h[4])
                if k in seen:
                    continue
                seen.add(k)
                print(f"  rx{h[0]} {h[1]!r} [{h[2]}/{h[3]}] bare={h[4]!r} id={h[5]} chosen_rule={h[6]} status={h[7]} rejected_ids={h[8]} placeholder_siblings={h[9]}")
print(f"\nlegs scanned: {legs}; legs with a bare identified protein enzyme: {hits}; actor rows by chosen_rule: {by_rule}")
