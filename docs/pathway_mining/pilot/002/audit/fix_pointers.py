from pathlib import Path
p = Path("C:/t/mining-pilot-runs2/audit/md_text.py")
t = p.read_text(encoding="utf-8")
pairs = [
    ("`mining/acquisition.py` dedup by `paper_id` only", "`mining/acquisition.py:528-` `Acquirer.acquire` (dedup by `paper_id` only)"),
    ("`modelclient.py:50, 256-266`; `llm/client.py:703-756`", "`modelclient.py:50, 271-280`; `llm/client.py:682-756`"),
    ("`mining/species.py:108-115, 259-264`", "`mining/species.py:108-115, 276-281`"),
    ("`mining/budget.py:193-231`; `mining/controller.py:300-313, 383`", "`mining/budget.py:193-235`; `mining/controller.py:319-331, 383`"),
    ("`mining/controller.py:403-408, 438-446`", "`mining/controller.py:403-408, 443-446`"),
    ("`mining/retrieval.py:225-237`", "`mining/retrieval.py:227-238`"),
    ("`mining/acquisition.py:59-70, 285-305, 336-352`", "`mining/acquisition.py:62-70, 426-448, 487`"),
    ("`mining/queries.py:180-183`; `mining/ids.py:82-83`", "`mining/queries.py:184`; `mining/ids.py:82-83`"),
    ("`mining/modelclient.py:280-300, 371-374`; `mining/controller.py:524-560`", "`mining/modelclient.py:291-300, 371-374`; `mining/controller.py:552-566`"),
    ("`mining/extraction.py:40-50, 242-259`", "`mining/extraction.py:46-53, 237-259`"),
    ("(`flags.py:54-62`)", "(`flags.py:55-62`)"),
    ("(`modelclient.py:50, 263-266`)", "(`modelclient.py:50, 271-280`)"),
]
for a, b in pairs:
    assert t.count(a) == 1, a
    t = t.replace(a, b)
p.write_text(t, encoding="utf-8")
b = Path("C:/t/mining-pilot-runs2/audit/build_audit.py")
t = b.read_text(encoding="utf-8")
a = "    side_by_side=side,\n)"
assert t.count(a) == 1
t = t.replace(a, "    side_by_side=side,\n    span_texts={sp[\"evidence_id\"]: evidence[sp[\"evidence_id\"]][\"text\"] for x in reaction_rows for sp in x[\"spans\"]},\n)")
b.write_text(t, encoding="utf-8")
print("ok")
