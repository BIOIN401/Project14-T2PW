"""ORCH-762: record, ONCE, the live UniProt answers the replay tests serve offline.

Usage (from the worktree root; ``--src`` picks the code whose queries are recorded):

    python docs/pwml_recovery_sprint/evidence/orch762/record_fixtures.py --src <tree>/src --label base|tip

Runs every replay scenario of ``tests/test_orch762_protein_identity.py`` through
``tests/orch762_harness.py`` with a recording client in place of the replay client. Only
``https://rest.uniprot.org/`` (UniProtKB search, UniParc, UniProt taxonomy) is fetched live;
every other host gets the same mocked 404 the tests give it. Nothing else is called: no
model, no PathBank, no NCBI (``T2PW_SPECIES_NCBI=0``), no EuropePMC (mocked alias table).

Answers are merged into ``tests/fixtures/orch762/uniprot_responses.json`` keyed exactly as
``orch762_harness.request_key`` keys them; a key already present is not fetched again, so the
base and tip runs together fetch each distinct query once. Fixture-building only: the tests
never touch the network.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.src).resolve()))
import t2pw  # noqa: E402,F401  - imported first so the harness keeps this tree

WORKTREE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE / "tests"))
import requests  # noqa: E402

import orch762_harness as H  # noqa: E402


print("t2pw from", Path(t2pw.__file__).resolve())
responses = H.load_responses()


class Recorder(H.ReplayHttp):
    def __init__(self) -> None:
        super().__init__(responses=responses)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Project14-T2PW-ORCH762-fixture-recorder/1.0"})
        self.fetched = 0

    def get(self, url, params=None, headers=None):  # type: ignore[override]
        key = H.request_key(url, params)
        self.calls.append(key)
        if not url.startswith(H.RECORDED_HOSTS):
            self.mocked.append(key)
            return H.FakeResponse(404, {})
        if key not in self.responses:
            time.sleep(0.2)
            resp = self.session.get(url, params=params, timeout=60)
            try:
                body = resp.json()
            except ValueError:
                body = None
            self.responses[key] = {"status": resp.status_code, "json": body}
            self.fetched += 1
        hit = self.responses[key]
        return H.FakeResponse(int(hit["status"]), hit["json"])


total = 0
for name, run in H.SCENARIOS.items():
    http = Recorder()
    run(http, Path(f"C:/t/bt-orch762/record-{args.label}-{name}"))
    total += http.fetched
    print(f"{args.label} {name}: {len(http.calls)} requests, {http.fetched} fetched live, {len(http.mocked)} mocked")
    H.write_responses(responses)
print(f"{args.label}: fetched {total} live; fixture now holds {len(responses)} answers")
