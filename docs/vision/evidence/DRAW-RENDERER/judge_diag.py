"""One judge call, full error. usage: judge_diag.py <tree> <model> <json 0|1>"""
import os
import sys

sys.path.insert(0, os.path.join(sys.argv[1], "src"))
from t2pw.llm import client as C  # noqa: E402

msgs = [{"role": "user", "content": [{"type": "text", "text": 'Return ONLY JSON {"ok": true}'}]}]
try:
    r = C.chat_detailed(msgs, temperature=0.0, max_tokens=50, response_json=sys.argv[3] == "1", model_override=sys.argv[2],
                        stage_name="draw_judge", timeout=60)
    print("OK", r.text[:200])
except Exception as e:  # noqa: BLE001
    print("ERR", str(e)[:1200])
print(C._token_stats)
