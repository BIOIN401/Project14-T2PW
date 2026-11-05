"""Scratch (review F3, no re-recording): add requested_model to every LLM entry of the committed fixture,
derived only from what the fixture holds. Every model the pinned env can request through OpenRouter is one id
except OPENROUTER_EXTRACTION_FALLBACK_MODEL; every recorded answer came from that one id, never from the
fallback; so each entry's requested model is that id. Refuses if any of that does not hold."""
import gzip
import json
import sys

sys.path.insert(0, "C:/t/ws-R25/tests")
from dotenv import dotenv_values  # noqa: E402

from equivalence import harness, llm_replay  # noqa: E402

fixture = harness.load_fixture()
env = fixture["env"]
assert env["LLM_PROVIDER"] == "openrouter"
primary = {v for k, v in env.items() if k.startswith("OPENROUTER_") and k.endswith("_MODEL")
           and k != "OPENROUTER_EXTRACTION_FALLBACK_MODEL"}
assert len(primary) == 1, primary
(model,) = primary
fallback = env.get("OPENROUTER_EXTRACTION_FALLBACK_MODEL")
answered = {row["response"].get("model") for row in fixture["entries"]["llm"]}
assert answered == {model} and fallback not in answered, (answered, fallback)
before = json.dumps({k: v for k, v in fixture["entries"].items() if k != "llm"}, sort_keys=True)
for row in fixture["entries"]["llm"]:
    assert "requested_model" not in row
    row["requested_model"] = model
fixture["recorded"]["requested_model_source"] = (
    "backfilled (review F3, no re-recording): every OPENROUTER_*_MODEL in env except the extraction fallback is "
    f"{model!r}, and all {len(answered and fixture['entries']['llm'])} answers came from it")
assert json.dumps({k: v for k, v in fixture["entries"].items() if k != "llm"}, sort_keys=True) == before
text = json.dumps(fixture, indent=1, sort_keys=True, ensure_ascii=False)
values = dotenv_values("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.env")
llm_replay.assert_no_secret_values(text, {k: str(v) for k, v in values.items() if k in llm_replay.SECRET_ENV and v})
harness.FIXTURE.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
print("ok", model, len(fixture["entries"]["llm"]), harness.FIXTURE.stat().st_size, len(text))
