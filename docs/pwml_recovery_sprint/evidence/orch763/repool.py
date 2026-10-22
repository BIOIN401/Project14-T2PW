"""ORCH-763: re-pool kept Stage-1 draws offline with the code on PYTHONPATH and report every split.

Offline: no model, no network, no pipeline run. Run it once with the base tree and once with
the tip on PYTHONPATH (through bounded_run.py) and compare the two outputs.

Usage:  PYTHONPATH=<tree>/src python repool.py <repo root> [<primary checkout root>]

Draw sets (each a list of 3 Stage-1 payloads):
* ``*-759``: the ORCH-759 legs, REBUILT by ``build_fixture.py`` (tests/fixtures/orch763/). The
  dictionary is the leg's own chunk-1 text (re-chunked from the leg's ``01_source_text.txt``
  exactly as the app does; its length must equal the side file's ``paper_chars``). Needs the
  primary checkout's ``runs_papers/`` (read only); without it the committed full paper is used
  and the line says so. At base, the pooled result is checked against the leg's own side file.
* ``*-751``: the ORCH-751 draws ORCH-754a replayed, kept in full (``evidence/orch754a_inputs``,
  ``evidence/orch752_chlorophyll_payloads``), with the committed full paper as dictionary.

Per set it prints: the pooled reaction count; every pooled row (support, key, state, merges and
their bridge and flags); every SPLIT -- (a) a pair with one reaction key kept apart, with the
recorded reason, (b) pooled rows naming the same enzyme set under different keys, (c) pooled
rows the gold matches to one gold reaction -- and the gold score (eval-v2 ``score_leg``, as
``orch756_gold_accuracy.py`` calls it) where the case is certified. Last, an independent
chimera check: every pair of draw rows inside one pooled row is re-compared with
``_conflict`` on their OWN profiles; a conflicting pair would be a chimera.
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
PRIMARY = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None

import t2pw.pipeline.stage_one_ensemble as E  # noqa: E402
from t2pw.bench.eval_v2 import reactions as RX  # noqa: E402
from t2pw.bench.eval_v2.gold import load_fulltext, load_supplement  # noqa: E402
from t2pw.pipeline.name_keys import strict_key  # noqa: E402
from t2pw.pipeline.paper_lexicon import PaperLexicon  # noqa: E402
from t2pw.pipeline.pipeline import chunk_text  # noqa: E402

EV = REPO / "docs" / "pwml_recovery_sprint" / "evidence"
PAPERS = EV / "orch754a_inputs" / "papers"
FIX = REPO / "tests" / "fixtures" / "orch763"
R = "/processes/reactions"
print(f"stage_one_ensemble loaded from {E.__file__}")
print(f"nested-location rule present: {hasattr(E, 'NESTED_LOCATION_FLAG')}\n")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def leg_text(fixture):
    src = fixture["source"]
    want = src["side_file_paper_chars"]
    if PRIMARY is not None:
        source = Path(src["stage1_payload"]).parent.parent / "01_source_text.txt"
        if source.is_file():
            for c in chunk_text(source.read_text(encoding="utf-8"), 10000, 1600):
                if len(c["text"]) == want:
                    return c["text"], f"leg chunk text ({want} chars, = side file paper_chars)"
    return None, "leg source text unavailable"


SETS = []
for name, pid in (("menaquinone", "PMC12421875"), ("enterobactin", "PMC12096016"), ("chlorophyll", "PMC5130987")):
    fx = FIX / f"{name}_orch759_draws.json"
    if fx.is_file():
        fixture = load(fx)
        text, how = leg_text(fixture)
        if text is None:
            text, how = (PAPERS / f"{pid}.txt").read_text(encoding="utf-8"), f"committed full paper ({how})"
        SETS.append((f"{name}-759", pid, fixture["draws"], text, how, fixture["source"]))
    folder = {"menaquinone": EV / "orch754a_inputs" / "PMC12421875_menaquinone",
              "enterobactin": EV / "orch754a_inputs" / "PMC12096016_enterobactin",
              "chlorophyll": EV / "orch752_chlorophyll_payloads"}[name]
    draws = [load(folder / f"control__run{i}__payload.json") for i in (1, 2, 3)]
    SETS.append((f"{name}-751", pid, draws, (PAPERS / f"{pid}.txt").read_text(encoding="utf-8"),
                 "committed full paper", None))

CASES = load_supplement()
SUMMARY = []


def state_of(row, draw):
    label = row.get("biological_state")
    st = E._states_by_key(draw).get(strict_key(label)) if label else None
    loc = (st or {}).get("compartment_canonical") or (st or {}).get("subcellular_location") or ""
    return f"{label!r}" + (f" [{loc}]" if loc else "")


for label, pid, draws, text, how, source in SETS:
    merged, report = E.merge_draws(draws, paper_text=text)
    rows = report["buckets"][R]["rows"]
    # The side-file records are grouped by key; the pooled payload is in creation order. Pair
    # them k-th to k-th per key (as build_fixture.py does) and number everything R1.. in
    # side-file order, the order stage1_ensemble_report.txt prints.
    slots = {}
    for j, rec in enumerate(rows):
        slots.setdefault(rec["key"], []).append(j)
    order = [slots[E._label(E.row_key(R, row))].pop(0) for row in merged["processes"]["reactions"]]
    pooled = [None] * len(rows)
    for row, j in zip(merged["processes"]["reactions"], order):
        pooled[j] = row
    assert all(r is not None for r in pooled) and all(p["name"] == r["name"] for p, r in zip(pooled, rows))
    print(f"=================== {label} ({pid}); dictionary: {how}")
    print(f"draw rows {report['buckets'][R]['rows_per_draw']} -> pooled reactions {len(rows)}; "
          f"names rewritten {len(report['names_rewritten'])}; typo snaps {len(report['typo_snapped'])}")
    flags = Counter()
    for i, (rec, row) in enumerate(zip(rows, pooled), 1):
        print(f"  R{i} [{rec['support']} draws {rec['draws']}] {rec['name']!r}  key {rec['key']!r}  state {row.get('biological_state')!r}")
        for m in rec.get("merges") or []:
            marks = [m[k]["reason"] if isinstance(m[k], dict) else m[k] for k in ("flag", "location_flag", "typo_flag") if m.get(k)]
            flags.update(marks)
            extra = f"  locations {m['location_flag']['locations']}" if m.get("location_flag") else ""
            print(f"      + d{m['draw']} {m['name']!r} bridged {m['bridged_by']} flags {marks}{extra}")
        if rec.get("state_taken"):
            print(f"      state taken from a merged row: {rec['state_taken']}")
    print(f"  merge flags: {dict(flags)}")

    print("  SPLITS (a) one key, kept apart:")
    splits_a = [(i, rec, other) for i, rec in enumerate(rows, 1) for other in rec.get("kept_apart_from") or []]
    for i, rec, other in splits_a:
        print(f"      R{i} {rec['name']!r} (draws {rec['draws']}) apart from {other['name']!r} (draws {other['draws']}): {other['reason']}")
    print(f"      total {len(splits_a)}")
    print("  SPLITS (b) one enzyme set, different keys:")
    by_enz = {}
    for i, row in enumerate(pooled, 1):
        enz = frozenset(strict_key(n) for _s, _it, n in E._enzyme_entries(row))
        if enz:
            by_enz.setdefault(enz, []).append(i)
    splits_b = {k: v for k, v in by_enz.items() if len({rows[j - 1]["key"] for j in v}) > 1}
    for enz, idx in splits_b.items():
        print(f"      {{{', '.join(sorted(enz))}}}: " + "; ".join(f"R{j} {rows[j - 1]['key']!r}" for j in idx))
    print(f"      total {len(splits_b)} enzyme set(s)")

    case = CASES.get(pid)
    if case is not None and case.complete:
        paper = load_fulltext(case, PAPERS)
        m = RX.score_leg(case, merged, [], paper, expected_species=case.species)
        t1 = m["table1_paper_extraction"]
        line = (f"P {t1['precision']:.3f} R {t1['recall']:.3f} matched {len(t1['gold_reactions_matched'])}/{t1['gold_reactions']} "
                f"predicted {t1['rows_claimed_as_paper_extraction']} (rows {m['retained_reactions']}, dups collapsed {m['duplicate_rows_collapsed']})")
        print(f"  GOLD {line}")
        print("  SPLITS (c) one gold reaction, several pooled rows:")
        for g in case.reactions:
            hits = [order[r["row_index"]] + 1 for r in m["rows"] if r["class"] == RX.TP and g.id in r["gold_ids"]]
            if len(hits) > 1:
                print(f"      gold {g.id} {g.label}: " + "; ".join(f"R{h} {rows[h - 1]['key']!r}" for h in hits))
            if not hits:
                print(f"      gold {g.id} {g.label}: NOT MATCHED")
    else:
        line = "gold not certified complete: not scored"
        print(f"  GOLD {line}")

    # independent chimera check: re-compare every pair of draw rows inside one pooled row
    lex = PaperLexicon(text)
    member_rows = {}
    for d, draw in enumerate(draws, 1):
        for row in (draw.get("processes") or {}).get("reactions") or []:
            member_rows.setdefault((d, row.get("name"), row.get("evidence")), []).append((row, draw))
    chimeras = 0
    for i, rec in enumerate(rows, 1):
        members = [(rec["draws"][0], rec["name"], None)] + [(m["draw"], m["name"], m.get("evidence")) for m in rec.get("merges") or []]
        profs = []
        for d, nm, ev in members:
            cands = [x for (dd, n, e), lst in member_rows.items() if dd == d and n == nm and (ev is None or e == ev) for x in lst]
            if cands:
                row, draw = cands[0]
                profs.append((d, nm, E._profile(row, E._states_by_key(draw), lambda s: strict_key(lex.canonical(s)[0]))))
        for x in range(len(profs)):
            for y in range(x + 1, len(profs)):
                why = E._conflict(profs[x][2], profs[y][2])
                if why:
                    chimeras += 1
                    print(f"  PAIRWISE CONFLICT inside R{i}: d{profs[x][0]} {profs[x][1]!r} vs d{profs[y][0]} {profs[y][1]!r}: {why}")
    print(f"  pairwise conflicts inside pooled rows: {chimeras}")

    if source is not None:
        side = next(b for b in load(source["side_file"]) if (b.get("buckets") or {}).get(R, {}).get("rows"))
        theirs = [(r["key"], r["draws"], [k["reason"] for k in r.get("kept_apart_from") or []]) for r in side["buckets"][R]["rows"]]
        ours = [(r["key"], r["draws"], [k["reason"] for k in r.get("kept_apart_from") or []]) for r in rows]
        same = theirs == ours and len(report["names_rewritten"]) == len(side["names_rewritten"])
        print(f"  RECONSTRUCTION CHECK vs the leg's own side file ({len(theirs)} rows, {len(side['names_rewritten'])} renames): "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            for t, o in zip(theirs, ours):
                if t != o:
                    print(f"      leg {t}\n      now {o}")
            if len(theirs) != len(ours):
                print(f"      leg rows {len(theirs)} vs now {len(ours)}")
    SUMMARY.append(f"{label:18s} pooled {len(rows):3d}  splits(a) {len(splits_a):2d}  splits(b) {len(splits_b):2d}  "
                   f"flags {dict(flags)}  chimeras {chimeras}  gold {line}")
    print()

print("SUMMARY")
for s in SUMMARY:
    print("  " + s)
