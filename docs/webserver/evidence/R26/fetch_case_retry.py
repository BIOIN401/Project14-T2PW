"""Fetch the R26 case once, retrying Europe PMC's transient 500s (same fetch path as the recorder)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "C:/t/ws-R26/tests")
from equivalence import harness  # noqa: E402

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
for attempt in range(25):
    try:
        case = harness.fetch_case(Path(f"C:/t/wsR26tmp/fetchtry/{work.name}/{attempt}"), harness.SECOND_CASE)
    except AssertionError as exc:
        print(attempt, "no text yet:", str(exc)[:120], flush=True)
        time.sleep(5)
        continue
    (work / "case.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    print("ok", case["paper_id"], len(case["full_text"]), case["full_text_sha256"][:16], flush=True)
    sys.exit(0)
print("gave up", flush=True)
sys.exit(1)
