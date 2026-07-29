"""Concurrency guarantees for Stage 7 enrichment (added 2026-07-28).

``enrich_entities.enrich_payload`` used to make every UniProt / ChEBI / HMDB / KEGG
lookup serially on the main thread. For run ``runs/2026-07-28_0919`` leg
``PMC12444477`` (strict) that was ~22 of the leg's 49m48s: 68 distinct lookups (21
UniProt, 17 ChEBI, 17 KEGG, 13 HMDB) taken one at a time against four slow public
endpoints. It is now a three-phase plan/prefetch/merge, with one bounded thread pool
per service.

The whole change was scoped to throughput -- "correctness must not change" -- so
these tests exist to pin exactly that. The load-bearing assertions are:

  * the concurrent path and ``max_workers=1`` (the literal old code path) produce
    identical payloads, identical reports and identical cache files;
  * the number of upstream requests does not go up, in particular that a repeated
    accession is still fetched once and still recorded as a ``cache_hits``
    increment rather than a second ``api_calls`` increment;
  * cache keys are unchanged, and the cache survives concurrent writers;
  * ``EnrichmentCache.save`` cannot leave a half-written 25.9 MB file behind.

Nothing here touches the network: every test monkeypatches the four ``_fetch_*``
functions, which is also how ``tests/test_pathbank_unknown_fallback.py`` has always
kept enrichment offline.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping import enrich_entities  # noqa: E402
from t2pw.mapping.enrich_entities import EnrichmentCache, enrich_payload  # noqa: E402
from t2pw.pipeline.entity_identity import (  # noqa: E402
    PATHBANK_UNKNOWN_PROTEIN_ID,
    PATHBANK_UNKNOWN_PROTEIN_NAME,
    PATHBANK_UNKNOWN_PROTEIN_UNIPROT,
)


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _FetchRecorder:
    """Deterministic stand-ins for the four ``_fetch_*`` functions.

    Records every call with its thread name so a test can prove work really did run
    on more than one thread, and returns payloads whose ``retrieved_at`` values are
    fixed. Fixed timestamps matter: ``_utc_now_iso()`` has one-second resolution, so
    a serial run that straddles a second boundary and a concurrent run that does not
    would differ in provenance for reasons that have nothing to do with threading.
    """

    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: List[Tuple[str, str]] = []
        self.threads: Dict[str, set] = {}
        self._lock = threading.Lock()

    def _record(self, service: str, entity_id: str) -> None:
        with self._lock:
            self.calls.append((service, entity_id))
            self.threads.setdefault(service, set()).add(threading.current_thread().name)

    def ids_for(self, service: str) -> List[str]:
        return [entity_id for svc, entity_id in self.calls if svc == service]

    def uniprot(self, _client: Any, uniprot_id: str) -> Dict[str, Any]:
        self._record("uniprot", uniprot_id)
        if self.delay:
            time.sleep(self.delay)
        return {
            "status": "ok",
            "parsed": {
                "status": "ok",
                "uniprot_id": uniprot_id,
                "recommended_name": f"Recommended {uniprot_id}",
                "alternative_names": [f"Alt {uniprot_id}"],
                "gene_names": [f"gene{uniprot_id}"],
                "enzyme_commission": ["1.2.3.4"],
                "provenance": {
                    "source": "UniProt",
                    "query": uniprot_id,
                    "source_url": f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json",
                    "retrieved_at": "2026-07-28T00:00:00Z",
                },
            },
            "raw": {"primaryAccession": uniprot_id},
        }

    def chebi(self, _client: Any, chebi_id: str) -> Dict[str, Any]:
        self._record("chebi", chebi_id)
        if self.delay:
            time.sleep(self.delay)
        return {
            "status": "ok",
            "source": "ChEBI",
            "id": chebi_id,
            "primary_name": f"chebi name {chebi_id}",
            "synonyms": [f"chebi syn {chebi_id}"],
            "inchi": f"InChI={chebi_id}",
            "inchikey": f"KEY-{chebi_id}",
            "smiles": "CCO",
            "formula": "C2H6O",
            "charge": 0,
            "mass_monoisotopic": 46.0419,
            "mass_average": 46.07,
            "chebi_roles": ["solvent"],
            "ontology_parents": ["CHEBI:1:parent"],
            "cross_references": {"chebi_id": chebi_id},
            "provenance": {
                "source": "ChEBI",
                "id": chebi_id,
                "source_url": "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity",
                "retrieved_at": "2026-07-28T00:00:00Z",
                "status": "ok",
            },
        }

    def hmdb(self, _client: Any, hmdb_id: str) -> Dict[str, Any]:
        self._record("hmdb", hmdb_id)
        if self.delay:
            time.sleep(self.delay)
        return {
            "status": "ok",
            "source": "HMDB",
            "id": hmdb_id,
            "accession": hmdb_id,
            "primary_name": f"hmdb name {hmdb_id}",
            "synonyms": [f"hmdb syn {hmdb_id}"],
            "description": "",
            "taxonomy": {"class": "Alcohols"},
            "ontology_classification": {"class": "Alcohols"},
            "pathways": ["Ethanol degradation"],
            "cellular_locations": ["Cytoplasm"],
            "tissue_locations": ["Liver"],
            "biofluid_locations": ["Blood"],
            "cross_references": {"hmdb_id": hmdb_id},
            "provenance": {
                "source": "HMDB",
                "id": hmdb_id,
                "source_url": f"https://hmdb.ca/metabolites/{hmdb_id}",
                "retrieved_at": "2026-07-28T00:00:00Z",
                "status": "ok",
            },
        }

    def kegg(self, _client: Any, kegg_id: str) -> Dict[str, Any]:
        self._record("kegg", kegg_id)
        if self.delay:
            time.sleep(self.delay)
        return {
            "status": "ok",
            "source": "KEGG",
            "id": kegg_id,
            "primary_name": f"kegg name {kegg_id}",
            "synonyms": [f"kegg syn {kegg_id}"],
            "formula": "C2H6O",
            "mass_monoisotopic": 46.0419,
            "mass_average": 46.07,
            "pathways": ["Glycolysis"],
            "kegg_class": ["Lipids"],
            "cross_references": {"kegg_id": kegg_id},
            "provenance": {
                "source": "KEGG",
                "id": kegg_id,
                "source_url": f"https://rest.kegg.jp/get/cpd:{kegg_id}",
                "retrieved_at": "2026-07-28T00:00:00Z",
                "status": "ok",
            },
        }


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _FetchRecorder:
    rec = _FetchRecorder()
    monkeypatch.setattr(enrich_entities, "_fetch_uniprot_enrichment", rec.uniprot)
    monkeypatch.setattr(enrich_entities, "_fetch_chebi_enrichment", rec.chebi)
    monkeypatch.setattr(enrich_entities, "_fetch_hmdb_enrichment", rec.hmdb)
    monkeypatch.setattr(enrich_entities, "_fetch_kegg_enrichment", rec.kegg)
    return rec


def _payload(*, proteins: int = 8, compounds: int = 10) -> Dict[str, Any]:
    """A payload shaped like the PMC12444477 leg: proteins with UniProt ids, compounds
    with a mix of ChEBI/HMDB/KEGG ids, deliberate duplicate accessions, plus the rows
    enrichment is supposed to skip entirely."""

    protein_rows: List[Dict[str, Any]] = []
    for i in range(proteins):
        # Every third protein reuses the previous accession so the de-duplication
        # path (one fetch, later rows counted as cache hits) is exercised.
        accession = f"P{(i - i % 3):05d}"
        protein_rows.append(
            {
                "name": f"Protein {i}",
                "mapped_ids": {"uniprot": accession},
            }
        )
    # A row with no name and a row with no mapped accession: both must be skipped by
    # planner and merge pass alike.
    protein_rows.append({"name": "", "mapped_ids": {"uniprot": "P99999"}})
    protein_rows.append({"name": "No accession", "mapped_ids": {}})
    protein_rows.append("not a dict")

    compound_rows: List[Dict[str, Any]] = []
    for i in range(compounds):
        mapped: Dict[str, Any] = {}
        if i % 2 == 0:
            mapped["chebi"] = str(15000 + (i // 2))
        if i % 3 == 0:
            mapped["hmdb"] = f"{1000 + (i // 3)}"
        if i % 2 == 1:
            mapped["kegg"] = f"cpd:C{i:05d}"
        compound_rows.append({"name": f"Compound {i}", "mapped_ids": mapped})
    if compound_rows:
        # Duplicate of compound 0 -> same three cache keys, must not refetch.
        duplicate = deepcopy(compound_rows[0])
        duplicate["name"] = "Compound 0 again"
        compound_rows.append(duplicate)
    compound_rows.append({"name": "Unmappable", "mapped_ids": {"pubchem": "5957"}})
    compound_rows.append({"name": "", "mapped_ids": {"chebi": "99999"}})

    return {
        "entities": {"proteins": protein_rows, "compounds": compound_rows},
        "element_locations": {
            "compound_locations": [
                {"compound": "Compound 0", "biological_state": "cytosol", "biological_state_id": 7}
            ]
        },
    }


_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _scrub(value: Any) -> Any:
    """Blank every ISO timestamp so two runs a second apart still compare equal.

    ``_utc_now_iso()`` is called directly in a couple of places ``_FetchRecorder``
    cannot reach (the protein error branch, the ``sources`` fallback), and it has
    one-second resolution. Without this, a comparison would flake whenever a run
    straddled a second boundary -- a false failure that says nothing about threading.
    """

    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return _TIMESTAMP.sub("<TS>", value)
    return value


# ---------------------------------------------------------------------------
# Identity of concurrent vs serial
# ---------------------------------------------------------------------------


def test_concurrent_and_serial_paths_produce_identical_results(
    tmp_path: Path, recorder: _FetchRecorder
) -> None:
    """The headline guarantee: threading changed the clock, not the answer."""

    payload = _payload()

    serial_cache = tmp_path / "serial_cache.json"
    concurrent_cache = tmp_path / "concurrent_cache.json"

    serial = enrich_payload(deepcopy(payload), cache_path=serial_cache, max_workers=1)
    serial_calls = list(recorder.calls)
    recorder.calls.clear()
    concurrent = enrich_payload(deepcopy(payload), cache_path=concurrent_cache)

    assert _scrub(concurrent["payload"]) == _scrub(serial["payload"])
    assert _scrub(concurrent["report"]) == _scrub(serial["report"])
    assert _scrub(json.loads(concurrent_cache.read_text(encoding="utf-8"))) == _scrub(
        json.loads(serial_cache.read_text(encoding="utf-8"))
    )

    # Same set of upstream requests, and no more of them. Order within a service may
    # differ (that is the point of a pool); the multiset per service may not.
    for service in ["uniprot", "chebi", "hmdb", "kegg"]:
        concurrent_ids = sorted(recorder.ids_for(service))
        serial_ids = sorted(entity_id for svc, entity_id in serial_calls if svc == service)
        assert concurrent_ids == serial_ids, service


def test_report_entity_order_is_unchanged_by_threading(
    tmp_path: Path, recorder: _FetchRecorder
) -> None:
    """``report["entities"]`` is a positional audit trail (it carries json_pointers),
    so it must stay in payload order no matter which thread answered first."""

    payload = _payload()
    serial = enrich_payload(deepcopy(payload), cache_path=tmp_path / "s.json", max_workers=1)
    concurrent = enrich_payload(deepcopy(payload), cache_path=tmp_path / "c.json")

    pointers = [row["json_pointer"] for row in concurrent["report"]["entities"]]
    assert pointers == [row["json_pointer"] for row in serial["report"]["entities"]]

    # Independently of the serial comparison: every protein pointer precedes every
    # compound pointer, and within each group the payload indices only increase.
    # That is the walk order of the merge pass, and it is what makes json_pointer
    # usable as an audit trail back into the payload.
    kinds = [p.split("/")[2] for p in pointers]
    assert kinds == sorted(kinds, key=["proteins", "compounds"].index)
    for kind in ["proteins", "compounds"]:
        indices = [int(p.split("/")[3]) for p in pointers if p.split("/")[2] == kind]
        assert indices == sorted(indices), kind


def test_duplicate_accessions_are_fetched_once_and_counted_as_cache_hits(
    tmp_path: Path, recorder: _FetchRecorder
) -> None:
    """The subtle way a prefetch could corrupt the report.

    In the serial code the second protein carrying an already-seen accession is a
    ``cache_hits`` increment, because the first one's ``cache.set`` landed before the
    second one's ``cache.get``. A prefetch that queued every occurrence instead of
    every distinct key would make two requests while the report still claimed one.
    """

    payload = _payload(proteins=9, compounds=4)
    result = enrich_payload(payload, cache_path=tmp_path / "cache.json")

    uniprot_ids = recorder.ids_for("uniprot")
    assert len(uniprot_ids) == len(set(uniprot_ids))
    assert result["report"]["calls"]["uniprot"] == len(uniprot_ids)
    # 9 proteins collapse to 3 accessions -> 3 calls and 6 hits.
    assert result["report"]["summary"]["proteins_with_uniprot"] == 9
    assert result["report"]["calls"]["uniprot"] == 3
    assert result["report"]["summary"]["cache_hits"] >= 6

    total_calls = sum(result["report"]["calls"].values())
    assert result["report"]["summary"]["api_calls"] == total_calls
    assert total_calls == len(recorder.calls)


def test_warm_cache_makes_no_requests_at_all(tmp_path: Path, recorder: _FetchRecorder) -> None:
    """A second run over the same paper must stay entirely offline -- the planner
    filters cache hits out before it ever reaches the pool."""

    cache_path = tmp_path / "cache.json"
    payload = _payload()
    enrich_payload(deepcopy(payload), cache_path=cache_path)
    first_call_count = len(recorder.calls)
    assert first_call_count > 0

    recorder.calls.clear()
    warm = enrich_payload(deepcopy(payload), cache_path=cache_path)
    assert recorder.calls == []
    assert warm["report"]["summary"]["api_calls"] == 0
    assert warm["report"]["summary"]["cache_hits"] > 0


def test_cache_keys_are_unchanged_by_the_concurrent_path(
    tmp_path: Path, recorder: _FetchRecorder
) -> None:
    """Cache-key construction is what makes the 25.9 MB shared cache reusable across
    runs; a normalization drift in the planner would silently orphan every old entry."""

    cache_path = tmp_path / "cache.json"
    enrich_payload(_payload(proteins=3, compounds=6), cache_path=cache_path)
    written = json.loads(cache_path.read_text(encoding="utf-8"))

    assert all(key.startswith("uniprot:") for key in written["proteins"])
    assert set(written["proteins"]) == {"uniprot:P00000"}
    for key in written["compounds"]:
        prefix = key.split(":", 1)[0]
        assert prefix in {"chebi", "hmdb", "kegg"}
    assert "chebi:CHEBI:15000" in written["compounds"]
    assert "hmdb:HMDB1000" in written["compounds"]
    assert "kegg:C00001" in written["compounds"]


def test_planner_covers_every_miss_so_nothing_falls_back_to_a_serial_fetch(
    tmp_path: Path, recorder: _FetchRecorder
) -> None:
    """The planner and the merge pass share ``_protein_uniprot_accession`` and
    ``_compound_candidate_ids`` precisely so they cannot disagree. If they ever do,
    the merge pass silently refetches inline and the run gets slow again without
    getting wrong -- this test is the alarm for that."""

    payload = _payload()
    entities = payload["entities"]
    cache = EnrichmentCache(tmp_path / "empty.json")
    targets = enrich_entities._plan_enrichment_fetches(
        entities["proteins"], entities["compounds"], cache
    )

    result = enrich_payload(payload, cache_path=tmp_path / "cache.json")
    assert len(targets) == len(recorder.calls)
    assert len(targets) == result["report"]["summary"]["api_calls"]
    assert len({t.cache_key for t in targets}) == len(targets)


def test_the_merge_pass_really_calls_the_shared_predicates(
    tmp_path: Path, recorder: _FetchRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "cannot drift apart" claim, asserted instead of assumed.

    Review on 2026-07-28 found that ``_protein_uniprot_accession`` and
    ``_compound_candidate_ids`` were called ONLY by the planner: the merge pass
    re-derived both inline (``_canonical(mapped_ids.get("uniprot")).upper()`` and
    a fresh three-key ``_normalize_*`` dict), so the module docstring's "phase 1
    builds keys with the same helpers phase 3 does, so the two cannot drift
    apart" was describing an intention rather than the code. Only the
    ``_normalize_*`` functions were genuinely shared.

    The two directions of drift are not equally visible, which is why this is
    worth a test of its own rather than a comment. Under-planning is loud enough
    to be harmless: the merge pass falls through to its inline fetch, the run gets
    slow again, and ``test_planner_covers_every_miss...`` above catches it.
    OVER-planning is silent and is the one that matters -- phase 2 issues real
    requests to UniProt/ChEBI/HMDB/KEGG on worker threads, and
    ``summary.api_calls`` is incremented only in phase 3, so the enrichment report
    would understate the traffic actually sent to four free public academic
    services that owe us nothing.

    Replacing the helpers with refusals is the sharpest available probe: if either
    loop still had its own copy of the predicate it would keep fetching, and the
    counters below would be non-zero. The unpatched run first is the positive
    control -- without it, "zero fetches" would also be satisfied by a payload
    that never had anything to fetch.
    """

    control = enrich_payload(_payload(), cache_path=tmp_path / "control.json")
    control_summary = control["report"]["summary"]
    assert control_summary["api_calls"] > 0
    assert control_summary["proteins_with_uniprot"] > 0
    assert control_summary["compounds_with_ids"] > 0
    recorder.calls.clear()

    monkeypatch.setattr(enrich_entities, "_protein_uniprot_accession", lambda _protein: "")
    monkeypatch.setattr(enrich_entities, "_compound_candidate_ids", lambda _compound: {})

    result = enrich_payload(_payload(), cache_path=tmp_path / "patched.json")
    summary = result["report"]["summary"]

    assert recorder.calls == [], "phase 3 fetched a row the shared predicate excluded"
    assert summary["api_calls"] == 0
    assert summary["proteins_with_uniprot"] == 0
    assert summary["compounds_with_ids"] == 0
    assert summary["cache_hits"] == 0, "no row reached a cache lookup either"


def test_pathbank_unknown_sentinel_is_not_prefetched(
    tmp_path: Path, recorder: _FetchRecorder
) -> None:
    """Commit 92e1192's guarantee -- the PathBank Unknown sentinel never reaches
    UniProt -- has to hold in the planner too, which is a second place that decides
    what to fetch. A planner that skipped the sentinel check would send it to UniProt
    on a worker thread where the existing test could not see it."""

    payload = _payload(proteins=1, compounds=0)
    payload["entities"]["proteins"].append(
        {
            "name": PATHBANK_UNKNOWN_PROTEIN_NAME,
            "pathbank_protein_id": PATHBANK_UNKNOWN_PROTEIN_ID,
            "mapped_ids": {"uniprot": PATHBANK_UNKNOWN_PROTEIN_UNIPROT},
            "mapping_meta": {"cross_species_placeholder": True},
        }
    )
    result = enrich_payload(payload, cache_path=tmp_path / "cache.json")

    assert result["report"]["summary"]["proteins_skipped_pathbank_unknown"] == 1
    assert "Unknown" not in recorder.ids_for("uniprot")


# ---------------------------------------------------------------------------
# The pool actually runs concurrently, and politely
# ---------------------------------------------------------------------------


def test_prefetch_really_uses_multiple_threads_and_beats_the_serial_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof the pool is doing something.

    Each fake fetch sleeps 250ms, chosen to sit in the same range as the real
    latencies measured on 2026-07-28 (HMDB 0.35s, UniProt 1.10s, KEGG 1.79s, ChEBI
    4.80s including its three retries) rather than far below them. That distinction
    matters: ``_ServiceLane.min_interval_seconds`` is 0.15-0.50s, so with a *tiny*
    fake latency the pacing floor -- not the fetches -- would dominate and the
    concurrent path could legitimately come out slower than serial on a toy payload.
    With realistic per-request cost the pacing never binds and the pool wins by
    roughly the lane width.

    Asserted as a ratio rather than an absolute wall time so it does not flake on a
    loaded box.
    """

    rec = _FetchRecorder(delay=0.25)
    monkeypatch.setattr(enrich_entities, "_fetch_uniprot_enrichment", rec.uniprot)
    monkeypatch.setattr(enrich_entities, "_fetch_chebi_enrichment", rec.chebi)
    monkeypatch.setattr(enrich_entities, "_fetch_hmdb_enrichment", rec.hmdb)
    monkeypatch.setattr(enrich_entities, "_fetch_kegg_enrichment", rec.kegg)

    payload = _payload(proteins=6, compounds=6)

    start = time.monotonic()
    enrich_payload(deepcopy(payload), cache_path=tmp_path / "serial.json", max_workers=1)
    serial_seconds = time.monotonic() - start
    lookup_count = len(rec.calls)
    rec.calls.clear()
    rec.threads.clear()

    start = time.monotonic()
    enrich_payload(deepcopy(payload), cache_path=tmp_path / "concurrent.json")
    concurrent_seconds = time.monotonic() - start

    assert lookup_count >= 10
    assert concurrent_seconds < serial_seconds * 0.8, (concurrent_seconds, serial_seconds)
    # At least one service must have spread its work over more than one worker.
    assert any(len(names) > 1 for names in rec.threads.values()), rec.threads
    # Every worker thread belongs to a named per-service pool, never the main thread
    # pool of some other subsystem.
    for service, names in rec.threads.items():
        assert all(name.startswith(f"t2pw-enrich-{service}") for name in names), (service, names)


def test_service_lane_paces_request_starts() -> None:
    """The rate limiter, tested without the network.

    ``max_workers`` alone does not bound the request *rate*: hmdb.ca answers our
    User-Agent with HTTP 403 in 0.35s and a fast-failing host would otherwise let two
    workers fire many requests a second at a service that is already refusing us.
    ``wait_turn`` reserves the next start slot under a lock, so N calls take at least
    (N-1) x interval regardless of how many threads make them.
    """

    lane = enrich_entities._ServiceLane("test", max_workers=4, min_interval_seconds=0.05)
    start = time.monotonic()
    threads = [threading.Thread(target=lane.wait_turn) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.05 * 5 * 0.9, elapsed


def test_service_lane_budgets_stay_conservative() -> None:
    """These numbers are a promise to four free public academic endpoints. Raising
    one should be a deliberate, reviewed act, not a drive-by edit, so the ceiling is
    pinned here with the reasoning in ``_build_service_lanes``."""

    lanes = enrich_entities._build_service_lanes()
    assert set(lanes) == {"uniprot", "chebi", "hmdb", "kegg"}
    for lane in lanes.values():
        assert 1 <= lane.max_workers <= 4, lane.service
        assert lane.min_interval_seconds > 0.0, lane.service
        # <= ~7 request starts per second per service, everywhere.
        assert lane.max_workers / max(lane.min_interval_seconds, 1e-9) <= 30.0
    # HMDB is an HTML scrape of a small academic site: it must never be the widest
    # lane, and it must be paced at least as gently as any API lane.
    assert lanes["hmdb"].max_workers <= 2
    assert lanes["hmdb"].min_interval_seconds >= max(
        lanes["uniprot"].min_interval_seconds,
        lanes["chebi"].min_interval_seconds,
        lanes["kegg"].min_interval_seconds,
    )
    # KEGG's REST terms exclude bulk harvesting; keep it at ~3 request starts/second.
    assert lanes["kegg"].max_workers / lanes["kegg"].min_interval_seconds <= 6.5


def test_each_worker_thread_gets_its_own_http_client() -> None:
    """``requests.Session`` is not documented thread-safe; sharing one across a pool
    is how you get a response handed to the wrong caller. ``_ClientPool`` is the
    guard, so it gets a direct test rather than only an implicit one."""

    pool = enrich_entities._ClientPool()
    seen: Dict[str, int] = {}
    lock = threading.Lock()

    def _grab() -> None:
        client = pool.get()
        again = pool.get()
        assert client is again  # same thread -> same session, so pooling still works
        with lock:
            seen[threading.current_thread().name] = id(client)

    threads = [threading.Thread(target=_grab, name=f"w{i}") for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(seen) == 4
    assert len(set(seen.values())) == 4
    pool.close()


# ---------------------------------------------------------------------------
# Cache integrity
# ---------------------------------------------------------------------------


def test_cache_survives_concurrent_writers(tmp_path: Path) -> None:
    """``EnrichmentCache.set`` is a ``setdefault``-then-assign pair, which is not
    atomic under the GIL even though each half is. The shipped design keeps every
    cache mutation on the main thread, but the class is shared mutable state backing
    a 25.9 MB file used by every run, so the lock is verified rather than assumed."""

    cache = EnrichmentCache(tmp_path / "cache.json")
    writers = 8
    per_writer = 200
    barrier = threading.Barrier(writers)

    def _write(worker: int) -> None:
        barrier.wait()
        for i in range(per_writer):
            cache.set("compounds", f"kegg:C{worker:02d}{i:04d}", {"status": "ok", "id": i})

    threads = [threading.Thread(target=_write, args=(w,)) for w in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(cache.data["compounds"]) == writers * per_writer
    cache.save()
    reloaded = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert len(reloaded["compounds"]) == writers * per_writer


def test_cache_save_is_atomic_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old ``write_text`` truncated the real 25.9 MB file and then streamed into
    it, so an interrupted save left a half-written document that ``__init__``
    swallows as "no cache at all" -- every subsequent paper then re-fetches
    everything. ``save`` now writes a sibling temp and ``os.replace``s it."""

    cache_path = tmp_path / "cache.json"
    cache = EnrichmentCache(cache_path)
    cache.set("proteins", "uniprot:P19367", {"status": "ok"})
    cache.save()

    original = cache_path.read_text(encoding="utf-8")
    assert json.loads(original)["proteins"]["uniprot:P19367"] == {"status": "ok"}

    # A save that dies inside os.replace must leave the previous file intact.
    # monkeypatch, not a bare assignment: ``enrich_entities.os`` IS the stdlib os
    # module, so assigning through it would replace os.replace for the whole
    # interpreter and poison every later test in the session.
    calls = {"n": 0}

    def _boom(src: Any, dst: Any) -> None:
        calls["n"] += 1
        raise OSError("simulated share violation")

    monkeypatch.setattr(enrich_entities.os, "replace", _boom)
    cache.set("proteins", "uniprot:P00000", {"status": "ok"})
    with pytest.raises(RuntimeError, match="atomically replace"):
        cache.save()
    monkeypatch.undo()

    assert calls["n"] == 3  # retried before giving up
    assert cache_path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob("cache.json.tmp-*")) == []


def test_enrichment_writes_no_stray_temp_files(tmp_path: Path, recorder: _FetchRecorder) -> None:
    cache_path = tmp_path / "cache.json"
    enrich_payload(_payload(proteins=3, compounds=4), cache_path=cache_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cache.json"]


# ---------------------------------------------------------------------------
# Failure behaviour
# ---------------------------------------------------------------------------


def test_worker_exception_surfaces_at_the_same_entity_as_the_serial_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``_fetch_*`` functions turn network and parse errors into ``status:
    error`` dicts, so an exception escaping one is a real bug. Prefetch deliberately
    drops such a key instead of recording it, which makes the merge pass call the
    fetcher inline and raise at exactly the entity the serial code would have --
    same exception, same point, one wasted request."""

    attempts: List[str] = []

    def _explode(_client: Any, uniprot_id: str) -> Dict[str, Any]:
        attempts.append(uniprot_id)
        raise ValueError(f"boom:{uniprot_id}")

    monkeypatch.setattr(enrich_entities, "_fetch_uniprot_enrichment", _explode)

    payload = {
        "entities": {
            "proteins": [
                {"name": "First", "mapped_ids": {"uniprot": "P00001"}},
                {"name": "Second", "mapped_ids": {"uniprot": "P00002"}},
            ]
        }
    }

    with pytest.raises(ValueError, match="boom:P00001"):
        enrich_payload(deepcopy(payload), cache_path=tmp_path / "cache.json")

    # P00001 is attempted in the pool and again inline; the pool may also have
    # attempted P00002 before the merge pass reached it, but the exception raised is
    # the first entity's, exactly as in the serial walk.
    assert attempts.count("P00001") == 2

    attempts.clear()
    with pytest.raises(ValueError, match="boom:P00001"):
        enrich_payload(deepcopy(payload), cache_path=tmp_path / "cache2.json", max_workers=1)
    assert attempts == ["P00001"]


def test_error_statuses_from_workers_are_recorded_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real 2026-07-28 shape of the world: every ChEBI lookup returns
    ``request_failed`` (the EBI endpoint answers HTTP 500) and almost every HMDB
    lookup returns HTTP 403. Those error dictionaries are cached and merged exactly
    like successes, and threading must not change that."""

    def _chebi_error(_client: Any, chebi_id: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "error_message": "request_failed:HTTP request failed after retries",
            "provenance": {
                "source": "ChEBI",
                "id": chebi_id,
                "retrieved_at": "2026-07-28T00:00:00Z",
            },
        }

    def _hmdb_403(_client: Any, hmdb_id: str) -> Dict[str, Any]:
        return {
            "status": "not_found",
            "error_message": "HTTP 403",
            "provenance": {
                "source": "HMDB",
                "id": hmdb_id,
                "retrieved_at": "2026-07-28T00:00:00Z",
            },
        }

    rec = _FetchRecorder()
    monkeypatch.setattr(enrich_entities, "_fetch_uniprot_enrichment", rec.uniprot)
    monkeypatch.setattr(enrich_entities, "_fetch_chebi_enrichment", _chebi_error)
    monkeypatch.setattr(enrich_entities, "_fetch_hmdb_enrichment", _hmdb_403)
    monkeypatch.setattr(enrich_entities, "_fetch_kegg_enrichment", rec.kegg)

    payload = _payload()
    serial = enrich_payload(deepcopy(payload), cache_path=tmp_path / "s.json", max_workers=1)
    concurrent = enrich_payload(deepcopy(payload), cache_path=tmp_path / "c.json")

    assert _scrub(concurrent["payload"]) == _scrub(serial["payload"])
    assert _scrub(concurrent["report"]) == _scrub(serial["report"])
    statuses = {row["status"] for row in concurrent["report"]["entities"]}
    assert "ok_with_warnings" in statuses or "error" in statuses
