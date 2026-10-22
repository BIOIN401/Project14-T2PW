"""ORCH-758: the chlorophyll 41 and the WNT chunk 3, offline against the code at a43a2b12. No model, no network.

Usage: PYTHONPATH=src python orch758_chlorophyll_lexicon.py <chlorophyll paper> <WNT paper> <chlorophyll ensemble side file json>

1. Re-chunks both papers exactly as the app's Stage 1 did (run_stage_one_with_chunking ->
   chunk_text with the UI defaults 10000 words / 1600 overlap) and prints every chunk's
   section, chars and words -- so the ensembles in the side files can be matched to chunks.
2. Builds the PaperLexicon the chlorophyll leg's first ensemble used (chunk 1 text; its
   paper_chars must equal the side file's 64053) and runs canonical() on every compound
   spelling any draw wrote (the side file's compound rows and their merges).
3. A counterfactual, NOT production code: lets long_form() also read the paper's own
   shared-locant phrase "X (Y) L" (the form abbreviation_pairs already parses into the pair
   "Y L" = "X L"), re-keys the 41 pooled reactions with the renames that produces, and
   counts distinct keys. That count is an UPPER BOUND on the collapse: the compatibility
   rule (_conflict) cannot be re-run without the per-draw payloads, which are not kept.
"""
import json
import re
import sys
from pathlib import Path

from t2pw.pipeline.name_keys import strict_key
from t2pw.pipeline.paper_lexicon import PaperLexicon, abbreviation_pairs, spelling_key
from t2pw.pipeline.pipeline import chunk_text

CHUNK_WORDS, OVERLAP = 10000, 1600
chl_text = Path(sys.argv[1]).read_text(encoding="utf-8")
wnt_text = Path(sys.argv[2]).read_text(encoding="utf-8")
side = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))

for label, text in (("chlorophyll PMC5130987", chl_text), ("WNT PMC10425814", wnt_text)):
    print(f"\n== chunks of {label}: {len(text)} chars, {len(text.split())} words")
    for c in chunk_text(text, CHUNK_WORDS, OVERLAP):
        head = " ".join(c["text"].split()[:12])
        print(f"   chunk {c['chunk_id']}: section={c['section']!r} relevance={c['relevance_score']} "
              f"chars={len(c['text'])} words={c['word_count']} | starts: {head[:110]!r}")

chunks = chunk_text(chl_text, CHUNK_WORDS, OVERLAP)
chunk1 = chunks[0]["text"]
lex = PaperLexicon(chunk1)
first = side[0]
print(f"\n== lexicon over chlorophyll chunk 1: paper_chars={lex.summary()['paper_chars']} "
      f"(side file: {first['paper_lexicon']['paper_chars']}), pairs={len(lex.pairs)} "
      f"(side file: {first['paper_lexicon']['abbreviation_pairs']})")
for short in ("Urogen III", "Coprogen III", "Protogen IX", "Proto IX", "MV-Chlide a", "Mg-Proto IX", "heme b",
              "Chl", "Glu-tRNA", "CPO", "PPO", "POR", "UROS"):
    print(f"   pair for {short!r}: {lex.pairs.get(spelling_key(short))}")
print("   Schwartz-Hearst candidates around 'MV-Chlide a':",
      [p for p in abbreviation_pairs(chunk1) if "MV" in p[0] or "vinyl" in p[1]])

_LOCANT_PHRASE = re.compile(r"^(?P<outer>.+?)\s*\((?P<inner>[^()]+)\)\s+(?P<loc>[IVX]{1,4}|[A-Za-z])$")


def counterfactual(name: str):
    """canonical(), plus the paper's own 'X (Y) L' phrase read as the pair 'Y L' = 'X L'."""
    out, rules, pattern = lex.canonical(name)
    if rules:
        return out, rules, pattern
    m = _LOCANT_PHRASE.match(name.strip())
    if m:
        short = f"{m['inner'].strip()} {m['loc']}"
        pair = lex.pairs.get(spelling_key(short))
        if pair and strict_key(pair[1]) == strict_key(f"{m['outer']} {m['loc']}"):
            return pair[1], ["abbreviation"], "shared_locant (verbatim X (Y) L)"
    return name, [], ""


names = []
for row in first["buckets"]["/entities/compounds"]["rows"]:
    names.append((row["draws"][0], row["name"]))
    names += [(m["draw"], m["name"]) for m in row.get("merges") or []]
print("\n== canonical() on every compound spelling in chunk 1 (draw: name -> production | counterfactual)")
rename = {}
for draw, name in names:
    prod = lex.canonical(name)
    cf = counterfactual(name)
    if prod[1] or cf[1]:
        print(f"   d{draw}: {name!r} -> {prod[0]!r} {prod[1]} | {cf[0]!r} {cf[1]}")
    if cf[1] and not prod[1]:
        rename[strict_key(name)] = strict_key(cf[0])
print("   counterfactual renames (strict keys):", rename)

rows = first["buckets"]["/processes/reactions"]["rows"]


def rekey(key: str) -> str:
    sides = []
    for side_text in key.split(" -> "):
        sides.append(" + ".join(sorted(rename.get(p, p) for p in side_text.split(" + "))))
    return " -> ".join(sides)


before = {r["key"] for r in rows}
after = {}
for i, r in enumerate(rows, 1):
    after.setdefault(rekey(r["key"]), []).append((i, r["draws"], r["name"]))
print(f"\n== pooled reaction rows {len(rows)}; distinct keys {len(before)}; distinct keys after the "
      f"counterfactual renames {len(after)} (rows sharing a key still need _conflict to merge)")
for key, members in after.items():
    if len(members) > 1:
        print(f"   {key}: rows {[m[0] for m in members]} draws {[m[1] for m in members]}")
