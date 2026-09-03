"""F-177 -- canonical and fallback verdicts must not be conflatable.

WHY A CONSTRUCTED FIXTURE AND NOT ONLY THE ARCHIVED CORPUS. The archived corpus
happens to split 10 canonical / 9 fallback on T-109, and every number below could
be reproduced from it. But a corpus is a draw: if a future run produced only
canonical legs, a reporter that had quietly gone back to summing would still look
right. So the load-bearing test here builds two leg directories BY HAND, gives them
payloads whose verdicts DISAGREE, and asserts the disagreement survives into the
report. A reporter that conflated them could not pass this.

The reporter itself is evaluation-only and lives under ``docs/.../evidence/``, so it
is loaded by path -- it is deliberately not importable production code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPORTER = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "eval_semantic_populations.py"


def _load_reporter() -> Any:
    spec = importlib.util.spec_from_file_location("eval_semantic_populations", REPORTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_semantic_populations"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reporter() -> Any:
    if not REPORTER.is_file():
        pytest.skip(f"reporter not present: {REPORTER}")
    return _load_reporter()


# ---------------------------------------------------------------------------
# The classification itself.
# ---------------------------------------------------------------------------
def test_population_is_decided_by_the_payload_FILE_and_never_guessed(reporter: Any) -> None:
    """The two filenames are different KINDS of object, so they get different names.

    An unrecognised source is its own population rather than being folded into
    either. Folding it into ``canonical`` would overstate the product evidence;
    folding it into ``fallback`` would understate a real export. Guessing in either
    direction is the defect.
    """

    payload = {"processes": {"reactions": [{"name": "r"}]}}
    assert reporter.classify_population("final_mapped.json", payload) == reporter.CANONICAL
    assert reporter.classify_population("merged_payload.json", payload) == reporter.FALLBACK
    assert reporter.classify_population("something_else.json", payload) == reporter.UNKNOWN
    assert reporter.classify_population("final_mapped.json", None) == reporter.NO_PAYLOAD
    assert reporter.classify_population("final_mapped.json", {}) == reporter.NO_PAYLOAD


def test_inapplicable_is_read_BEFORE_ok_so_it_can_never_score_as_a_pass(
    reporter: Any,
) -> None:
    """THE misreading this finding family exists to prevent, pinned mechanically.

    An inapplicable check carries ``ok=True`` by design -- absence of evidence must
    never be a failure. So a classifier that consulted ``ok`` first would score
    every unevaluable check as PASSED, which is exactly how "four of five biology
    checks pass" was once read off an applicability column.
    """

    class _Check:
        def __init__(self, ok: Any, reason: str = "") -> None:
            self.ok = ok
            self.inapplicable_reason = reason

    # ok=True AND inapplicable -> must NOT be PASSED.
    assert reporter.classify_verdict(_Check(True, "some reason")) != reporter.PASSED
    assert reporter.classify_verdict(_Check(True)) == reporter.PASSED
    assert reporter.classify_verdict(_Check(False)) == reporter.FAILED
    assert reporter.classify_verdict(_Check(None)) == reporter.INAPPLICABLE
    # The two artifact states stay apart, keyed off bench.semantic's corrected text.
    assert reporter.classify_verdict(
        _Check(True, "no RAG admission report was supplied to this evaluation call")
    ) == reporter.MISSING
    assert reporter.classify_verdict(
        _Check(True, "this is a MALFORMED artifact, not a missing one")
    ) == reporter.MALFORMED


# ---------------------------------------------------------------------------
# The load-bearing test: two populations, verdicts that DISAGREE.
# ---------------------------------------------------------------------------
def _write_leg(run: Path, slug: str, mode: str, filename: str, payload: Dict[str, Any]) -> None:
    leg = run / "papers" / slug / mode
    leg.mkdir(parents=True, exist_ok=True)
    (leg / filename).write_text(json.dumps(payload), encoding="utf-8")


def _payload(organism: str) -> Dict[str, Any]:
    """A payload whose ORGANISM decides one gating check's verdict.

    ``organism_compatible`` is used as the discriminator because it is decided by a
    single field, so the two legs differ in exactly one respect and the resulting
    verdict difference cannot be attributed to anything else.
    """

    return {
        "entities": {"proteins": [], "compounds": []},
        "processes": {
            "reactions": [
                {
                    "name": "r1",
                    "inputs": ["a"], "outputs": ["b"],
                    "enzymes": [{"entity": "E1"}],
                    "organism": organism,
                    "source": {"quote": "a is converted to b by E1"},
                }
            ]
        },
    }


def test_a_canonical_PASS_and_a_fallback_FAIL_stay_apart_in_the_report(
    reporter: Any, tmp_path: Path
) -> None:
    """THE F-177 PROOF. Same check, two populations, opposite verdicts, both visible.

    Construction: one leg stores its payload as ``final_mapped.json`` with the gold
    case's own organism; the other stores an otherwise identical payload as
    ``merged_payload.json`` with a FORBIDDEN organism. The only difference between
    the legs is the filename and the organism, so any difference in the report is
    attributable to those two things and nothing else.

    What must hold:
      1. the legs land in DIFFERENT populations;
      2. ``organism_compatible`` reads PASSED under one and FAILED under the other;
      3. neither count is reachable by summing -- there is no combined row;
      4. the failure is ATTRIBUTED to its population, not merely counted.
    Any reporter that merged the two would fail 1, 2 and 4 together.
    """

    from t2pw.bench.goldset import load_gold_set

    cases = {case.paper_id: case for case in load_gold_set().cases}
    slug = "PMC12312563"
    case = cases[slug]
    forbidden = (case.forbidden_organisms or [None])[0]
    assert forbidden, "this gold case must declare a forbidden organism for the fixture"

    run = tmp_path / "run"
    _write_leg(run, slug, "strict", "final_mapped.json",
               _payload(case.actual_organism or "Listeria monocytogenes"))
    _write_leg(run, slug, "research", "merged_payload.json",
               _payload(str(forbidden)))

    tally = reporter.Tally()
    reporter.evaluate_run(ROOT, run, tally)

    # 1. different populations
    assert tally.legs[reporter.CANONICAL] == [f"{slug}/strict"]
    assert tally.legs[reporter.FALLBACK] == [f"{slug}/research"]

    # 2. the SAME check disagrees across them
    organism = tally.counts["organism_compatible"]
    assert organism[reporter.CANONICAL][reporter.PASSED] == 1
    assert organism[reporter.CANONICAL][reporter.FAILED] == 0
    assert organism[reporter.FALLBACK][reporter.FAILED] == 1
    assert organism[reporter.FALLBACK][reporter.PASSED] == 0

    # 3. no combined row exists to be misread
    assert set(organism) <= set(reporter.POPULATION_ORDER)
    assert "total" not in organism and "combined" not in organism

    # 4. the failure is attributed, not merely counted
    attributed = [f for f in tally.failures if f["check"] == "organism_compatible"]
    assert len(attributed) == 1
    assert attributed[0]["population"] == reporter.FALLBACK
    assert attributed[0]["leg"] == f"{slug}/research"


def test_the_structured_report_never_emits_a_denominator_free_headline(
    reporter: Any, tmp_path: Path
) -> None:
    """Every count in the serialized report is reachable only through a population.

    This is the machine-readable half of the same guarantee: a consumer that wants a
    combined number has to construct it deliberately and name what it combined,
    because none is offered.
    """

    run = tmp_path / "run"
    _write_leg(run, "PMC12312563", "strict", "final_mapped.json",
               _payload("Listeria monocytogenes"))
    tally = reporter.Tally()
    reporter.evaluate_run(ROOT, run, tally)

    document = tally.to_dict()
    assert set(document) == {"legs_by_population", "counts", "failures"}
    for _check, populations in document["counts"].items():
        assert populations, "a check with no population is an unattributed count"
        for population in populations:
            assert population in reporter.POPULATION_ORDER
    for failure in document["failures"]:
        assert failure["population"] in reporter.POPULATION_ORDER
