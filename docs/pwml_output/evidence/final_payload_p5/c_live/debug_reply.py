"""Why was PMC6112128 on_2 rejected? Re-ask the same Stage-1 payload 3 times and keep the raw model replies.

usage: python debug_reply.py <stage1_post.json> <paper_run_dir_with_01_source_text.txt> <out.json>
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "C:/t/fp/src")
import t2pw.llm.client as client  # noqa: E402  (loads .env)

for k in ("T2PW_FINAL_PAYLOAD", "T2PW_OFFLINE_CURATOR", "T2PW_TRANSPORT_SIDES"):
    os.environ.pop(k, None)
from t2pw.service import transport_sides as ts  # noqa: E402

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
text = Path(sys.argv[2], "01_source_text.txt").read_text(encoding="utf-8")
out = []
for i in range(3):
    raw = []

    def chat(messages, **kw):
        res = client.chat_detailed(messages, **kw)
        raw.append(getattr(res, "text", res))
        return res

    fixed, report = ts.repair_transport_sides(payload, text, chat_fn=chat)
    out.append({"try": i + 1, "raw_reply": raw, "repairs": report["repairs"], "rejected": report["rejected"], "error": report["error"]})
    print(i + 1, "repairs", len(report["repairs"]), "rejected", report["rejected"])
Path(sys.argv[3]).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
