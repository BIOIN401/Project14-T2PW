"""Scratch: replay the fixture through the app arm, capture the first full missed LLM request, diff it
against the recorded requests with the same system prompt."""
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/t/ws-R25/tests")
import pytest  # noqa: E402

from equivalence import harness, llm_replay  # noqa: E402

fixture = harness.load_fixture()
mp = pytest.MonkeyPatch()
harness.replay_env(mp, fixture)
full = []
real_lookup = llm_replay.Cassette._lookup


def lookup(self, kind, key, request):
    try:
        return real_lookup(self, kind, key, request)
    except llm_replay.ReplayMiss:
        full.append((kind, request))
        raise


mp.setattr(llm_replay.Cassette, "_lookup", lookup)
app = harness.run_app_arm(Path(sys.argv[1]), llm_replay.Cassette("replay", fixture["entries"]), 240.0)
print("OUTCOME", app["outcome"]["status"], app["outcome"]["stage"], "misses", len(app["misses"]), [m["kind"] for m in app["misses"]])
kind, miss = full[0]
print("FIRST MISS kind", kind, {k: (v if k != "messages" else len(v)) for k, v in miss.items()})
recorded = [row["request"] for row in fixture["entries"][kind]]
if kind == "llm":
    import difflib as _d
    sys0 = miss["messages"][0]["content"][:80]
    same = [r for r in recorded if r["messages"][0]["content"][:80] == sys0]
    print("recorded same-system temps:", [r.get("temperature") for r in same])
    cand = [r for r in same if r.get("temperature") == miss.get("temperature")] or same
    rec_u, rep_u = cand[0]["messages"][1]["content"], miss["messages"][1]["content"]
    print("system equal:", cand[0]["messages"][0] == miss["messages"][0], "user len", len(rec_u), len(rep_u))
    sm = _d.SequenceMatcher(None, rec_u, rep_u, autojunk=False)
    for op, a0, a1, b0, b1 in [o for o in sm.get_opcodes() if o[0] != "equal"][:12]:
        print(op, "REC:", repr(rec_u[max(0, a0 - 80):a1 + 40])[:400])
        print("   REP:", repr(rep_u[max(0, b0 - 80):b1 + 40])[:400])
else:
    print(json.dumps(miss)[:1000])
