import json, sys
from pathlib import Path
import t2pw.pipeline.stage_one_ensemble as E
from t2pw.pipeline.pipeline import chunk_text
fx = json.loads(Path("tests/fixtures/orch763/chlorophyll_orch759_draws.json").read_text(encoding="utf-8"))
src = Path(sys.argv[1]).read_text(encoding="utf-8")
text = next(c["text"] for c in chunk_text(src, 10000, 1600) if len(c["text"]) == fx["source"]["side_file_paper_chars"])
merged, report = E.merge_draws(fx["draws"], paper_text=text)
for row in merged["processes"]["reactions"]:
    rec = next(r for r in report["buckets"]["/processes/reactions"]["rows"] if r["name"] == row["name"] and r["key"] == E._label(E.row_key("/processes/reactions", row)))
    enz = sorted({n for _s, _i, n in E._enzyme_entries(row)})
    print(f"d{rec['draws']} {row['name']!r}: {E._label(E.row_key('/processes/reactions', row))} | enzymes {enz} | {row.get('biological_state')}")
