"""C-116 — the D-088 diagnostics as their OWN artifact; the verdict is unchanged.

**G9 CLASSIFICATION: NEW ACCEPTANCE TESTS, explicitly labelled.** Every test accepts
a NEW CAPABILITY -- a diagnostics document that did not exist before this card. G9
says such a capability carries labelled new acceptance tests and needs NO fabricated
base-SHA failure, and none is manufactured here. Proved behaviourally on committed
artifacts: the verdict is BYTE-IDENTICAL on four archived legs and still has 15 keys;
the document agrees with it on all 83; the writer emits both in one call; the three
populations separate; 374 / 60 / 90 reproduces."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.pipeline.strict_quarantine as SQ  # noqa: E402
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    COVERAGE_DIAGNOSTICS_FILENAME,
    COVERAGE_REPORT_FILENAME,
    evaluate_core_coverage,
    quarantine_and_close,
    write_quarantine_artifacts,
)

from test_strict_quarantine import _glutathione_payload  # noqa: E402

#: The verdict's shape, in order. This card adds NOTHING to it -- that is the card.
PRE_EXISTING_KEYS: Tuple[str, ...] = (
    "schema_version", "requested_core_terms", "requested_core_declared",
    "requested_context", "requested_core_source", "matched_terms",
    "unmatched_terms", "coverage_ratio", "core_accepted_processes",
    "auxiliary_accepted_processes", "surviving_processes",
    "quarantined_processes", "thresholds", "minimum_core_satisfied", "reasons",
)

#: What the DOCUMENT carries. None of these may ever appear in the verdict.
DOCUMENT_KEYS: Tuple[str, ...] = (
    "d088_diagnostics_version", "diagnostics_rule", "diagnostics_computed",
    "unmatched_terms", "unmatched_anchor_diagnostics", "subprocess_coverage",
    "subprocess_coverage_ratio", "subprocess_source_declared",
)

ARCHIVED_LEGS: Tuple[str, ...] = (
    "runs_verify/2026-09-01_1612/papers/PMC12096016/strict",
    "runs_verify/2026-09-01_1612/papers/PMC12782028/strict",
    "runs_verify/2026-09-01_1612/papers/PMC12452463/strict",  # TRAP-1: nothing moves it
    "runs_verify/2026-08-21_2239/papers/PMC12782028/strict",  # F-168 thin-context draw
)

#: ``evidence/f167_history_census.log`` (G11 ``T-108/13``) is the SOURCE OF TRUTH for
#: these three: asserted, never re-derived -- if they stop reproducing, the rule or
#: the census is wrong, and a green test may not settle which.
CENSUS_LOG = "docs/pwml_recovery_sprint/evidence/f167_history_census.log"
CENSUS_ANCHORS, CENSUS_ALIGNED, CENSUS_IN_PAYLOAD = 374, 60, 90


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


#: ``write_quarantine_artifacts``'s own serializer call, so equality IS file equality.
def _serialize(document: Any) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False, default=str)


#: ``(archived verdict, archived report, the verdict this tree produces)``. Nothing
#: is re-run: every input comes from that leg's own committed artifacts.
def _replay(leg: str):
    directory = ROOT / leg
    archived = _load(directory / COVERAGE_REPORT_FILENAME)
    report = _load(directory / "quarantine_report.json")
    payload_path = directory / "final_mapped.json"
    thresholds = archived.get("thresholds") or {}
    produced = evaluate_core_coverage(
        _load(payload_path) if payload_path.is_file() else {},
        report.get("admissions") or [],
        pathway_context=archived.get("requested_context"),
        min_core_processes=int(thresholds.get("min_core_processes", 1)),
        min_core_coverage=float(thresholds.get("min_core_coverage", 0.5)),
    )
    return archived, report, produced


def _require(leg: str) -> None:
    if not (ROOT / leg / COVERAGE_REPORT_FILENAME).is_file():
        pytest.skip(f"archived leg not present: {leg}")


def _row(document: Dict[str, Any], term: str) -> Dict[str, Any]:
    for row in document["unmatched_anchor_diagnostics"]:
        if row["term"] == term:
            return row
    raise AssertionError(f"{term!r} is not among the unmatched anchors")


def _committed_legs() -> List[Path]:
    return sorted(path for root in ("runs", "runs_verify")
                  for path in (ROOT / root).rglob(COVERAGE_REPORT_FILENAME))


@pytest.mark.parametrize("leg", ARCHIVED_LEGS)
def test_the_coverage_verdict_is_byte_identical_to_base_on_archived_legs(leg: str) -> None:
    """NEW ACCEPTANCE TEST (G9). THE test of this card: replaying each leg's own
    committed inputs reproduces the coverage document production wrote, BYTE FOR BYTE
    under the writer's serializer, with no sixteenth key -- a reorder, a re-rounded
    ratio or an added key all fail here."""

    _require(leg)
    archived, report, produced = _replay(leg)

    assert list(produced) == list(PRE_EXISTING_KEYS), f"{leg}: verdict shape moved"
    assert len(produced) == 15 and dict(produced) == archived, f"{leg}: a value moved"
    assert _serialize(produced) == (ROOT / leg / COVERAGE_REPORT_FILENAME).read_text(
        encoding="utf-8"), f"{leg}: the verdict's serialized bytes moved"

    # The diagnostics exist and are invisible to serialization -- both halves matter.
    assert getattr(produced, SQ._D088_DIAGNOSTICS_ATTR, None) is not None
    assert "d088" not in _serialize(produced)
    assert SQ._D088_DIAGNOSTICS_ATTR not in dict(produced)
    # The one route by which a verdict key could move a decision id is the policy
    # version, which IS a decision input and this card must not have touched it.
    assert SQ.QUARANTINE_POLICY_VERSION == report["policy_version"]
    assert SQ.decision_identifier(report) == report["decision_id"], f"{leg}: decision id"


def test_the_diagnostics_document_agrees_with_the_verdict_on_every_archived_leg() -> None:
    """NEW ACCEPTANCE TEST (G9) -- the anti-drift proof the C-114 ruling requires.
    Two documents CAN disagree, so over EVERY committed leg under both trees the
    document's ``unmatched_terms`` must equal the verdict's, verbatim and in order.
    """

    legs = _committed_legs()
    assert len(legs) == 83, "the committed corpus the census measured"
    for path in legs:
        leg = path.parent.relative_to(ROOT).as_posix()
        archived, _report, produced = _replay(leg)
        document = SQ._d088_diagnostics_document(produced)
        assert document["unmatched_terms"] == list(produced["unmatched_terms"]), leg
        assert document["unmatched_terms"] == list(archived["unmatched_terms"]), leg
        assert document["d088_diagnostics_version"] == 1
        assert document["diagnostics_computed"] is True, leg
        assert [row["term"] for row in document["unmatched_anchor_diagnostics"]] == (
            document["unmatched_terms"]), leg


def test_the_writer_emits_both_or_neither(tmp_path: Path) -> None:
    """NEW ACCEPTANCE TEST (G9) -- "verdict written, diagnostics absent" is
    unreachable. One REAL ``quarantine_and_close`` + ``write_quarantine_artifacts``
    call, not a mimicked result: the claim is about the producer.
    """

    written = write_quarantine_artifacts(
        quarantine_and_close(_glutathione_payload(), strict_db=True), tmp_path)
    verdict_path, document_path = (tmp_path / COVERAGE_REPORT_FILENAME,
                                   tmp_path / COVERAGE_DIAGNOSTICS_FILENAME)
    assert verdict_path.is_file() and document_path.is_file()
    document = _load(document_path)
    assert set(document) == set(DOCUMENT_KEYS)
    assert document["diagnostics_computed"] is True
    assert document["unmatched_terms"] == _load(verdict_path)["unmatched_terms"]
    # The RETURNED map is deliberately the four-name artifact set: two tests this card
    # does not own assert it by equality, and the diagnostics are a record on disk.
    assert COVERAGE_DIAGNOSTICS_FILENAME not in written
    assert COVERAGE_REPORT_FILENAME in written and len(written) == 4
    # A verdict NOT from evaluate_core_coverage still yields a document that says so.
    absent = SQ._d088_diagnostics_document({"unmatched_terms": ["x"]})
    assert absent["diagnostics_computed"] is False
    assert absent["unmatched_terms"] == ["x"]
    assert absent["unmatched_anchor_diagnostics"] is None


def test_entd_is_in_payload_and_not_wired() -> None:
    """NEW ACCEPTANCE TEST (G9) -- D-088 clause 6's distinction, plus the rule marker.

    Catches diagnostics that cannot separate "never extracted" from "extracted and not
    wired": ``EntD`` is in the payload and its process was quarantined, while ``NADH``
    and ``Fur`` are not in the payload at all (F-169). Matching anchors against the
    entity list would call ``EntD`` satisfied and the cap would stop meaning anything.
    The ``ATP`` row is the other half: the census rule records ``in_payload: false``,
    and F-169's amendment, confirmed by an independent bio-auditor, establishes BOTH
    HALVES are false -- it is present under its spelled-out name and IS wired into an
    admitted reaction. It is kept DELIBERATELY so the totals stay comparable, which is
    why ``diagnostics_rule`` exists and why the field docstring must call
    ``in_payload`` a CENSUS-RULE result: clause 7 protects evidence, and an unlabelled
    preserved falsehood is not evidence. This pins the LABEL, not the falsehood."""

    leg = "runs_verify/2026-09-01_1612/papers/PMC12096016/strict"
    _require(leg)
    _archived, _report, produced = _replay(leg)
    document = SQ._d088_diagnostics_document(produced)
    assert document["unmatched_terms"] == ["NADH", "ATP", "EntD", "Fur"]

    entd = _row(document, "EntD")
    assert entd["in_payload"] is True and entd["payload_names"] == ["EntD"]
    assert entd["wired_to_admitted_core"] is False
    for term in ("NADH", "Fur"):
        assert _row(document, term)["in_payload"] is False, term
        assert _row(document, term)["payload_names"] == [], term
    # False by construction for every UNMATCHED anchor, recorded anyway: the day one
    # comes back True the matcher has a bug, and this is how it is found.
    assert all(row["wired_to_admitted_core"] is False
               for row in document["unmatched_anchor_diagnostics"])
    assert document["diagnostics_rule"] == "f167_census_v1"
    assert _row(document, "ATP")["in_payload"] is False  # the census rule's answer
    doc = SQ._d088_anchor_diagnostics.__doc__ or ""
    assert "CENSUS-RULE RESULT" in doc and "NEVER" in doc, (
        "the field docstring must say plainly that in_payload is a rule result")


def test_pmc12782028_uncovered_subprocesses() -> None:
    """NEW ACCEPTANCE TEST (G9) -- catches a rule blind to a missing upstream arm.
    The T-108 draw leaves two Stage-0 subprocesses uncovered; the 2026-08-21 draw of
    the same paper/mode reads 1.0. That is F-168's instability RECORDED rather than
    papered over -- a gate keyed to this number would have released that leg.
    """

    leg = "runs_verify/2026-09-01_1612/papers/PMC12782028/strict"
    _require(leg)
    _archived, _report, produced = _replay(leg)
    document = SQ._d088_diagnostics_document(produced)
    coverage = {row["subprocess"]: row["covered"] for row in document["subprocess_coverage"]}

    assert len(coverage) == 5 and sorted(
        name for name, covered in coverage.items() if not covered) == [
        "methylsterol demethylation", "mevalonate pathway"]
    assert document["subprocess_coverage_ratio"] == pytest.approx(0.6)
    assert document["subprocess_source_declared"] is True
    for row in document["subprocess_coverage"]:
        assert bool(row["covering_processes"]) is row["covered"]

    earlier = "runs_verify/2026-08-21_2239/papers/PMC12782028/strict"
    _require(earlier)
    drifted = SQ._d088_diagnostics_document(_replay(earlier)[2])
    assert drifted["subprocess_coverage_ratio"] == pytest.approx(1.0)
    assert drifted["unmatched_terms"], "the entity-anchor cap still fires on that draw"


def test_the_diagnostics_move_no_field_a_gate_reads() -> None:
    """NEW ACCEPTANCE TEST (G9) -- "records nothing that gates" (C-056c, F-168). The
    same admissions and requested core, with and without Stage-0 ``main_subprocesses``:
    every verdict key must be identical except ``requested_context``, the echo of the
    varied input -- and the DOCUMENT must differ, or the comparison is vacuous.
    """

    admissions = [
        {"state": "core_accepted", "name": "EntC reaction",
         "core_terms": ["chorismate", "isochorismate", "entc reaction"]},
        {"state": "auxiliary_accepted", "name": "some interaction", "core_terms": ["entb"]},
        {"state": "quarantined_unmapped_entity", "name": "EntD step", "core_terms": ["entd"]},
    ]
    payload = {"entities": {"proteins": [{"name": "EntD"}],
                            "compounds": [{"name": "chorismate"}]}}
    context = {"pathway_name": "siderophore biosynthesis",
               "key_compounds": ["chorismate", "NADH"], "key_proteins": ["EntC", "EntD"]}
    with_subs = dict(context, main_subprocesses=["chorismate isomerization", "assembly"])

    without = evaluate_core_coverage(payload, admissions, pathway_context=context)
    withs = evaluate_core_coverage(payload, admissions, pathway_context=with_subs)

    assert without["unmatched_terms"], "the fixture must actually have unmatched anchors"
    for key in PRE_EXISTING_KEYS:
        if key != "requested_context":
            assert withs[key] == without[key], f"{key} moved when subprocesses appeared"

    a, b = SQ._d088_diagnostics_document(without), SQ._d088_diagnostics_document(withs)
    assert a["subprocess_source_declared"] is False and a["subprocess_coverage"] == []
    assert a["subprocess_coverage_ratio"] == 0.0
    assert b["subprocess_source_declared"] is True and b["subprocess_coverage"] != []
    # No document key is named by a reason line, and no threshold grew.
    assert all(key not in " ".join(withs["reasons"]) for key in DOCUMENT_KEYS)
    assert set(withs["thresholds"]) == {"min_core_processes", "min_core_coverage"}
    # Alignment is a record: it never promotes an anchor into matched_terms.
    for row in b["unmatched_anchor_diagnostics"]:
        if row["subprocess_aligned"]:
            assert row["term"] in withs["unmatched_terms"]
            assert row["term"] not in withs["matched_terms"]


def test_census_populations_reproduce() -> None:
    """NEW ACCEPTANCE TEST (G9) -- the committed corpus census triple, exactly: 374
    unmatched anchors, 60 subprocess-aligned, 90 in payload. IF THESE DIFFER, DO NOT
    TUNE THE RULE -- the census or the port is wrong, and which matters more."""

    assert (ROOT / CENSUS_LOG).is_file(), "the census log is the cited source of truth"
    legs = _committed_legs()
    anchors = aligned = in_payload = testable = 0
    for path in legs:
        archived = _load(path)
        context = archived.get("requested_context") or {}
        if not archived.get("requested_core_declared"):
            continue
        if not (archived.get("unmatched_terms") or []):
            continue
        if not (context.get("main_subprocesses") or []):
            continue
        testable += 1
        produced = _replay(path.parent.relative_to(ROOT).as_posix())[2]
        assert produced["unmatched_terms"] == list(archived["unmatched_terms"]), str(path)
        for row in SQ._d088_diagnostics_document(produced)["unmatched_anchor_diagnostics"]:
            anchors += 1
            aligned += 1 if row["subprocess_aligned"] else 0
            in_payload += 1 if row["in_payload"] else 0

    assert testable == 73
    assert (anchors, aligned, in_payload) == (
        CENSUS_ANCHORS, CENSUS_ALIGNED, CENSUS_IN_PAYLOAD)


def test_no_gold_read() -> None:
    """NEW ACCEPTANCE TEST (G9) -- PRODUCT_CONTRACT 12 and the gold-identity rule,
    scoped honestly. The module ALREADY carries one deliberate, documented
    ``t2pw.bench.semantic_production`` import inside ``quarantine_and_close`` that
    predates this card, so the import assertion covers exactly the functions this card
    OWNS, and the identity assertion exactly the prose it WROTE."""

    source = (SRC / "t2pw" / "pipeline" / "strict_quarantine.py").read_text(encoding="utf-8")
    owned = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and (node.name in {"evaluate_core_coverage", "write_quarantine_artifacts"}
             or node.name.startswith("_d088_"))
    ]
    assert len(owned) >= 7, "the owned functions must be found by name"
    for node in owned:
        for child in ast.walk(node):
            imported = ([alias.name for alias in child.names]
                        if isinstance(child, ast.Import) else
                        [str(child.module or "")] if isinstance(child, ast.ImportFrom) else [])
            assert not any(n.startswith("t2pw.bench") for n in imported), node.name
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                assert "gold" not in child.value.lower(), (node.name, child.value)

    # The paper-identity rule is enforced over ALL of ``src/`` by
    # ``tests/test_c074_strict_core_floor.py``, which this card does NOT amend; what
    # is asserted here is narrower and duplicate-free -- the prose THIS card authored.
    lines = source.splitlines()
    index = next(n for n, x in enumerate(lines) if x.startswith("COVERAGE_DIAGNOSTICS_"))
    authored = "\n".join([
        source.split("D-088 diagnostics", 1)[1].split("\ndef evaluate_core_coverage(", 1)[0],
        source.split("**D-088 DIAGNOSTICS:", 1)[1].split('"""', 1)[0],
        "\n".join(lines[index - 4:index + 1]),
    ])
    assert "_d088_diagnostics_document" in authored, "the authored region must be found"
    for needle in ("PMC", "pinned_v1", "goldset", "bench/gold", "mevalonate"):
        assert needle not in authored, needle
    for needle in ("bench/gold", "bench\\gold", "pinned_v1.json", "goldset"):
        assert needle not in source, needle
