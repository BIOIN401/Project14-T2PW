"""ORCH-763 (b): why 'DMK' and 'demethylmenaquinone (DMK)' did not bridge in the ORCH-759 menaquinone leg.

Offline, read-only. No model, no network. Usage (through bounded_run.py):
    PYTHONPATH=<tree>/src python diag_dmk.py <leg 01_source_text.txt> <leg side file stage1_ensemble_report.json>

1. Re-chunks the leg's source text exactly as the app's Stage 1 did (chunk_text, UI defaults
   10000 words / 1600 overlap) and picks the chunk whose length equals the side file's
   paper_chars -- the text the leg's PaperLexicon was built from.
2. Builds that PaperLexicon and asks it about every DMK / MK spelling the draws wrote.
3. Replays Schwartz-Hearst's own steps on the paper's two defining phrases, so the step that
   rejects each pair is printed rather than inferred.
4. Prints the strict keys (case) and the biological-state profiles (location) of the MenA /
   MenG rows, so case and location are ruled in or out with evidence.
"""
import json
import sys
from pathlib import Path

from t2pw.pipeline import paper_lexicon as PL
from t2pw.pipeline.name_keys import strict_key
from t2pw.pipeline.pipeline import chunk_text

source = Path(sys.argv[1]).read_text(encoding="utf-8")
side = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
want = side[0]["paper_lexicon"]["paper_chars"]
chunks = chunk_text(source, 10000, 1600)
print(f"source {len(source)} chars; chunks {[len(c['text']) for c in chunks]}; side file paper_chars {want}")
text = next(c["text"] for c in chunks if len(c["text"]) == want)
lex = PL.PaperLexicon(text)
print(f"lexicon over the matching chunk: pairs={len(lex.pairs)} (side file {side[0]['paper_lexicon']['abbreviation_pairs']})")
for short, (s, long, pattern) in sorted(lex.pairs.items()):
    print(f"   pair {short!r} -> {long!r} [{pattern}]")

print("\n== the paper's own definitions")
for phrase in ("demethylmenaquinone (DMK)", "menaquinone (MK"):
    at = text.find(phrase)
    print(f"   {phrase!r} occurs at {at}: {text[max(0, at - 90):at + 40]!r}")

print("\n== Schwartz-Hearst step by step on each defining parenthesis")
for m in PL._PAREN.finditer(text):
    inner = m.group(1).split(", ")[0].split("; ")[0].strip()
    if inner not in ("DMK", "MK"):
        continue
    head = text[max(0, m.start() - 500):m.start()]
    cut = max(head.rfind(". "), head.rfind(", "), head.rfind("; "))
    long_cand = head[cut + 2:] if cut >= 0 else head
    best = PL._best_long_form(inner, long_cand.strip())
    print(f"   SF {inner!r}; parenthesis {m.group(0)!r}; long-form candidate {long_cand.strip()[-70:]!r}")
    print(f"      _best_long_form -> {best!r}; _valid_pair -> {PL._valid_pair(inner, long_cand.strip())!r}")
    missing = [ch for ch in inner if ch.lower() not in long_cand.lower()]
    print(f"      short-form characters absent from the whole candidate: {missing}")

print("\n== what the lexicon answers for every spelling the draws wrote")
for name in ("DMK", "demethylmenaquinone (DMK)", "demethylmenaquinone", "MK", "menaquinone (MK)", "menaquinone"):
    print(f"   {name!r}: strict_key {strict_key(name)!r}; spelling_key {PL.spelling_key(name)!r}; "
          f"pairs.get -> {lex.pairs.get(PL.spelling_key(name))!r}; long_form -> {lex.long_form(name)!r}; "
          f"canonical -> {lex.canonical(name)!r}; occurs -> {lex.occurs(name)}")
print("   case-folded lookups (would a case rule have found it?):",
      {k: v for k, v in lex.pairs.items() if k.casefold() in ("dmk", "mk")})

print("\n== side file: the MenA / MenG pooled rows and their keys")
for row in side[0]["buckets"]["/processes/reactions"]["rows"]:
    if row["name"].startswith(("MenA", "MenG")):
        print(f"   {row['name']!r} draws {row['draws']} key {row['key']!r} kept_apart_from {row.get('kept_apart_from')}")
