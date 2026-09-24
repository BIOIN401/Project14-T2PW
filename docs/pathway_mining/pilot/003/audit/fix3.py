from pathlib import Path
p = Path("C:/t/mining-pilot-runs3/audit/md3.py")
t = p.read_text(encoding="utf-8")
pairs = [
    ("{r['search']['round_metrics_papers_discovered']` against 64", "{r['search']['round_metrics_papers_discovered']}` against 64"),
    ("f\"answered with 122-209 output tokens in 4.5-8.3 s. That is about",
     "f\"answered with 122-209 output tokens (`_0001`, `_0003` and `_0014` in 4.5-8.3 s). That is about"),
    ("6. **Species states are still set by keywords (F3 / A3).** 16 of 17 `inferred_only` labels come from the Marchantia review's \"\n        f\"table header \\\"Arabidopsis Ortholog\\\" or from \\\"Marchantia lacks JAR1 orthologs\\\".",
     "6. **Species states are still set by keywords (F3 / A3).** 16 of 17 `inferred_only` labels come from the substring 'ortholog' \"\n        f\"in the review's passages: its table header \\\"Arabidopsis Ortholog\\\", \\\"lacks JAR1 orthologs\\\", \\\"orthology analysis\\\"."),
    ("Those six came from calls _0001, _0002, _0003, _0008 and _0013,", "Those six came from calls _0001, _0002, _0003, _0007, _0008 and _0013,"),
    ("add(\"The other twelve spent 457 to about 10,200 estimated reasoning tokens.",
     "add(\"The other twelve spent from about 100 (`_0009`, 11.5 s, borderline, yet it filled every field) to about 10,200 estimated reasoning tokens."),
    ("| wrong NCBI ids / duplicate titles | - (no NCBI) / 0 | 7+ / 6 | 0 / 0 |", "| wrong NCBI ids / duplicate titles | - (no NCBI) / 0 | 7+ proven / 6 | 0 of 5 checked / 0 |"),
    ("16 of 17 `inferred_only` labels come from 'Arabidopsis Ortholog' / 'lacks JAR1 orthologs', including the A. thaliana OPR3 step.",
     "16 of 17 `inferred_only` labels come from the substring 'ortholog' ('Arabidopsis Ortholog' table header, 'lacks JAR1 orthologs', 'orthology analysis'), including the A. thaliana OPR3 step."),
    ("\"**Not exercised.** All 4 papers are full text; no two cited spans share text.\"",
     "\"**Not exercised.** All 4 papers are full text; \" + str(r['duplicate_span_texts']) + \" pairs of spans share identical text.\""),
    ("\"among them the 13-LOX batches that gave", "\"among them two of the four 13-LOX batches that gave"),
]
for a, b in pairs:
    assert t.count(a) == 1, a[:80]
    t = t.replace(a, b)
p.write_text(t, encoding="utf-8")
q = Path("C:/t/mining-pilot-runs3/audit/build_audit3.py")
t = q.read_text(encoding="utf-8")
a = "    component_relevance=dict("
assert t.count(a) == 1
t = t.replace(a, "    duplicate_span_texts=sum(n - 1 for n in Counter(e['text'] for e in evidence.values()).values() if n > 1),\n" + a)
q.write_text(t, encoding="utf-8")
print("ok")
