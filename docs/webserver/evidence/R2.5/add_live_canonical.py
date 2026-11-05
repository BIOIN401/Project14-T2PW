"""Scratch: add the live run's own canonical payload (tmp/final.mapped.json of the completing segment, whose
payload_sha256 equals the recorded live hash) to the fixture as recorded.canonical_payload. Entries untouched."""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/t/ws-R25/tests")
from dotenv import dotenv_values  # noqa: E402

from equivalence import harness, llm_replay  # noqa: E402
from t2pw.pipeline.gate_reports import payload_sha256  # noqa: E402

live = json.loads(Path("C:/t/wsR25tmp/rec3/record-03/tmp/final.mapped.json").read_text(encoding="utf-8"))
fixture = harness.load_fixture()
before_entries = json.dumps(fixture["entries"], sort_keys=True)
assert payload_sha256(live) == fixture["recorded"]["canonical_payload_sha256"], "not the live canonical payload"
fixture["recorded"]["canonical_payload"] = live
fixture["recorded"]["canonical_payload_source"] = "record-03/tmp/final.mapped.json (payload_sha256 == canonical_payload_sha256)"
assert json.dumps(fixture["entries"], sort_keys=True) == before_entries
text = json.dumps(fixture, indent=1, sort_keys=True, ensure_ascii=False)
values = dotenv_values("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.env")
llm_replay.assert_no_secret_values(text, {k: str(v) for k, v in values.items() if k in llm_replay.SECRET_ENV and v})
harness.FIXTURE.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
print("ok", harness.FIXTURE.stat().st_size, len(text))
