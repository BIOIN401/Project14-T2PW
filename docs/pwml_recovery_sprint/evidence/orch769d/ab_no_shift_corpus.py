"""ORCH-769d A/B: apply_patch_with_policy on a seeded random corpus, on whichever t2pw
PYTHONPATH names. Run once on base src and once on tip src; ``--compare`` diffs them.

Two corpora, same seed on both trees:
  * ``no_shift``: every op is either unable to move an array index (dict-key add, append
    with "-", replace, dict-key remove) or an array remove/insert whose confidence is
    below its policy threshold, so _should_accept (a pure function) rejects it on any
    tree. Severity item (2): these must be byte-identical, payload and report.
  * ``shift``: the same plus accepted-confidence array removes and inserts. These are
    expected to differ exactly where an accepted remove/insert precedes an op it moves.

Prints one sha256 per case into --out. No network, no writes except --out.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.add_argument("--cases", type=int, default=3000)
ap.add_argument("--compare", nargs=2)
args = ap.parse_args()

if args.compare:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare)
    result = {"trees": [a["t2pw"], b["t2pw"]]}
    for corpus in ("no_shift", "shift"):
        diff = [i for i, (x, y) in enumerate(zip(a[corpus], b[corpus])) if x["h"] != y["h"]]
        with_shift = [i for i in diff if a[corpus][i]["accepted_shift"]]
        result[corpus] = {
            "cases": len(a[corpus]),
            "differing": len(diff),
            "differing_with_an_accepted_base_remove_or_insert": len(with_shift),
            "differing_without_one": [i for i in diff if i not in with_shift][:20],
            "cases_with_an_accepted_base_remove_or_insert": sum(1 for x in a[corpus] if x["accepted_shift"]),
        }
    print(json.dumps(result, indent=2))
    ok = result["no_shift"]["differing"] == 0 and not result["shift"]["differing_without_one"]
    sys.exit(0 if ok else 1)

import t2pw  # noqa: E402
from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402

BUCKETS = ("proteins", "compounds", "protein_complexes")


def payload(rng):
    proteins = [{"name": f"P{i}", "note": ""} for i in range(rng.randint(3, 7))]
    compounds = [{"name": f"C{i}"} for i in range(rng.randint(3, 6))]
    complexes = [{"name": f"X{i}", "components": [{"name": f"P{i}"}]} for i in range(rng.randint(1, 2))]
    for row in proteins + compounds:
        if rng.random() < 0.4:
            row["synonyms"] = [row["name"].lower()]
    reactions = [
        {
            "name": f"R{i}",
            "inputs": [rng.choice(compounds)["name"]],
            "outputs": [rng.choice(compounds)["name"]],
            "enzymes": [{"entity": rng.choice(proteins)["name"]}] if rng.random() < 0.7 else [],
        }
        for i in range(rng.randint(1, 3))
    ]
    interactions = [
        {"entity_1": rng.choice(proteins)["name"], "entity_2": rng.choice(compounds)["name"]}
        for _ in range(rng.randint(0, 2))
    ]
    return {
        "entities": {"proteins": proteins, "compounds": compounds, "protein_complexes": complexes},
        "processes": {"reactions": reactions, "interactions": interactions},
    }


def op(rng, p, shift):
    bucket = rng.choice(BUCKETS)
    n = len(p["entities"][bucket]) + 1  # one past the end, so some ops fail to apply
    i = rng.randrange(n)
    conf = round(rng.uniform(0.5, 1.0), 3)
    kinds = ["replace_note", "replace_name", "add_syns", "add_syn_dash", "append_row",
             "remove_key", "rxn_name", "rxn_input_dash", "low_remove", "low_insert"]
    if shift:
        kinds += ["remove_row", "remove_row", "insert_row", "remove_input"]
    kind = rng.choice(kinds)
    base = {"evidence": "ab corpus", "confidence": conf}
    if kind == "replace_note":
        return {"op": "replace", "path": f"/entities/{bucket}/{i}/note", "value": "n", **base}
    if kind == "replace_name":
        return {"op": "replace", "path": f"/entities/{bucket}/{i}/name", "value": f"{bucket[:1]}{i}x", **base}
    if kind == "add_syns":
        return {"op": "add", "path": f"/entities/{bucket}/{i}/synonyms", "value": [f"s{i}"], **base}
    if kind == "add_syn_dash":
        return {"op": "add", "path": f"/entities/{bucket}/{i}/synonyms/-", "value": f"d{i}", **base}
    if kind == "append_row":
        return {"op": "add", "path": f"/entities/{bucket}/-", "value": {"name": f"new{i}"}, **base}
    if kind == "remove_key":
        return {"op": "remove", "path": f"/entities/{bucket}/{i}/synonyms", **base}
    if kind == "rxn_name":
        r = rng.randrange(len(p["processes"]["reactions"]) + 1)
        return {"op": "replace", "path": f"/processes/reactions/{r}/name", "value": f"R{r}x", **base}
    if kind == "rxn_input_dash":
        r = rng.randrange(len(p["processes"]["reactions"]) + 1)
        return {"op": "add", "path": f"/processes/reactions/{r}/inputs/-", "value": "C0",
                **{**base, "confidence": max(conf, 0.985)}}
    if kind == "low_remove":  # below the 0.95 remove threshold on any tree
        return {"op": "remove", "path": f"/entities/{bucket}/{i}", **{**base, "confidence": min(conf, 0.94)}}
    if kind == "low_insert":  # below the 0.75 add threshold on any tree
        return {"op": "add", "path": f"/entities/{bucket}/{i}", "value": {"name": f"ins{i}"},
                **{**base, "confidence": min(conf, 0.69)}}
    if kind == "remove_row":
        return {"op": "remove", "path": f"/entities/{bucket}/{i}", **{**base, "confidence": max(conf, 0.96)}}
    if kind == "insert_row":
        return {"op": "add", "path": f"/entities/{bucket}/{i}", "value": {"name": f"ins{i}"}, **base}
    r = rng.randrange(len(p["processes"]["reactions"]))
    return {"op": "remove", "path": f"/processes/reactions/{r}/inputs/0", **{**base, "confidence": 0.99}}


def is_shift(o):
    leaf = o["path"].rsplit("/", 1)[-1]
    return o["op"] in {"add", "remove"} and leaf.isdigit() and o["path"].count("/") in (3, 5)


out = {"t2pw": str(Path(t2pw.__file__).parent)}
for corpus, shift in (("no_shift", False), ("shift", True)):
    rng = random.Random(769 if not shift else 7694)
    rows = []
    for _ in range(args.cases):
        p = payload(rng)
        patch = [op(rng, p, shift) for _ in range(rng.randint(1, 8))]
        patched, report = apply_patch_with_policy(p, patch)
        blob = json.dumps([patched, report], sort_keys=True, ensure_ascii=True, default=str)
        accepted_shift = any(is_shift(r["op"]) for r in report["accepted"])
        rows.append({"h": hashlib.sha256(blob.encode()).hexdigest()[:16], "accepted_shift": accepted_shift})
    out[corpus] = rows
    print(corpus, "cases", len(rows), "with an accepted remove/insert:", sum(r["accepted_shift"] for r in rows))
Path(args.out).write_text(json.dumps(out), encoding="utf-8")
