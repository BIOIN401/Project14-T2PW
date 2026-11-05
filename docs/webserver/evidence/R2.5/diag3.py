"""Scratch: two replayed app arms; print every leaf of the canonical payload that differs between them."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/t/ws-R25/tests")
import pytest  # noqa: E402

from equivalence import harness, llm_replay  # noqa: E402

fixture = harness.load_fixture()
mp = pytest.MonkeyPatch()
harness.replay_env(mp, fixture)
base = Path(sys.argv[1])
arms = [harness.run_app_arm(base / f"a{i}", llm_replay.Cassette("replay", fixture["entries"]), 240.0, fixture["case"])
        for i in (1, 2)]


def flat(x, p=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flat(v, f"{p}/{k}"))
        return out
    if isinstance(x, list):
        out = {}
        for i, v in enumerate(x):
            out.update(flat(v, f"{p}/{i}"))
        return out
    return {p: x}


a, b = (flat(arm["canonical_payload"]) for arm in arms)
print("hashes", [arm["canonical_payload_sha256"][:12] for arm in arms], "xml equal", arms[0]["xml"] == arms[1]["xml"])
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print("differing leaves:", len(diff))
for k in diff[:40]:
    print(k, "|", repr(a.get(k))[:160], "|", repr(b.get(k))[:160])
