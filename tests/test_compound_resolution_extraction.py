"""C-040 acceptance tests for ``t2pw.pwml.compound_resolution``.

**Every test here is a NEW ACCEPTANCE test for a genuinely new module, not a
regression proof.** C-040 is a mechanical extraction: it moves four functions
out of ``ir.py`` and changes no caller's behaviour. There is therefore no
base-SHA behavioural failure to demonstrate, and none was manufactured. Nothing
below is written to fail on ``ImportError`` at the base SHA either -- symbol
absence is not behavioural proof.

The load-bearing assertion is the opposite of a regression proof:
``test_build_pwml_ir_matches_the_pre_extraction_golden`` asserts that **nothing**
changed beyond a documented delta. Its digests no longer come from the raw sweep
at the dispatch base ``e4eeef429468ef42cfdfd12295ea86447f0c674f``; they come from
a **pre-freeze-routed** sweep at the C-051 stack tip.

That pin has been moved twice, each time deliberately and with a documented
delta: by C-045a (D-016 put species canonicalization before the freeze) and by
C-051b (C-051 made ``build_pwml_ir`` refuse unresolved compound rows, so the
sweep must route through ``run_prefreeze_resolution``). Both deltas are recorded
above ``GOLDEN``. It must never be moved to make an accidental drift go green.

That count was already stale when C-093 read it -- C-050g, C-050h, C-050i and
C-050k each moved it afterwards, and C-068 admitted a leg. **Do not read a move
count from this paragraph; every move is recorded, in order, in the comment block
above ``GOLDEN``, which is the authority.** C-093's is the sixth: 56 committed
legs admitted and one excluded, with **no existing digest moved** -- 0 of the 34
already pinned changed under the sweep that measured the new ones.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml import compound_resolution as cr  # noqa: E402
from t2pw.pwml import ir as ir_mod  # noqa: E402
from t2pw.pwml.ir import DuplicateNamedRowError, build_pwml_ir  # noqa: E402
from t2pw.pwml.name_index import PathwhizNameIndex, default_name_index  # noqa: E402
from t2pw.pwml.prefreeze_resolution import (  # noqa: E402
    PrefreezeResolutionError,
    run_prefreeze_resolution,
)

#: The nine helpers copied verbatim from ir.py:43-96, :183-193, :244-260.
LEAF_HELPERS = (
    "_safe_dict", "_safe_list", "_canonical", "_norm", "_to_int",
    "_first_nonempty", "_db_id", "_dedupe_aliases", "_add_issue",
)

ENTRY_POINTS = ("_resolve_compound_rows", "_canonicalize_compound_offline")

MOVED = ENTRY_POINTS + ("_normalize_compound_external_ids", "_compound_external_ids")


@pytest.fixture(autouse=True)
def _no_live_db(monkeypatch: Any) -> None:
    """No test in this module may reach the live PathBank DB.

    ``db_resolver=None`` makes ``_resolve_compound_rows`` call
    ``PathBankDbResolver.from_env()``. SPIKE-002 §7 measured that DB answering
    on this host, which would make every assertion below network-dependent and
    the golden digests unreproducible. The message is load-bearing: it lands in
    ``db_resolution.reason`` and is hashed into GOLDEN.
    """
    import t2pw.mapping.map_ids as map_ids

    class _ForcedUnavailable:
        @classmethod
        def from_env(cls) -> Any:
            raise RuntimeError("harvest_forced_unavailable")

    monkeypatch.setattr(map_ids, "PathBankDbResolver", _ForcedUnavailable)


def _index() -> PathwhizNameIndex:
    """A two-row offline index: KEGG C00037 -> the canonical name 'Glycine'."""
    return PathwhizNameIndex({
        "compounds": {
            "by_id": {"78": {"name": "Glycine", "kegg": "C00037"}},
            "kegg": {"C00037": "78"},
        }
    })


# ---------------------------------------------------------------------------
# Leaf-helper equality pin (SPIKE-002 F-3)
# ---------------------------------------------------------------------------


def test_leaf_helper_copies_are_source_identical_to_the_ir_originals() -> None:
    """NEW ACCEPTANCE. The copies must never drift from ir.py's originals.

    Duplication is forced -- ir.py imports compound_resolution, so importing
    back would be a cycle -- so it is pinned instead: editing either side alone
    fails here.
    """
    for name in LEAF_HELPERS:
        original = inspect.getsource(getattr(ir_mod, name))
        copy = inspect.getsource(getattr(cr, name))
        assert copy == original, f"{name} has drifted from its ir.py original"


def test_leaf_helper_copies_behave_identically_to_the_ir_originals() -> None:
    """NEW ACCEPTANCE. Source equality, corroborated behaviourally."""
    values: List[Any] = [
        None, "", "  ", 0, 1, True, False, 3.7, "12", "  CHEBI:15428 ",
        "Glycine", "glycine", "succinyl-CoA", "N-acetyl  glutamate",
        [], {}, {"a": 1}, "HMDB0000123", "cpd:C00037", "1e3",
    ]
    for name in ("_safe_dict", "_safe_list", "_canonical", "_norm", "_to_int"):
        original, copy = getattr(ir_mod, name), getattr(cr, name)
        for value in values:
            assert copy(value) == original(value), f"{name}({value!r}) diverged"

    rows: List[Dict[str, Any]] = [
        {}, {"hmdb_id": "HMDB0000123"}, {"mapped_ids": {"kegg": "C00037"}},
        {"mapping_meta": {"candidates": [{"chebi_id": "15428"}]}},
        {"pathwhiz_id": "78", "name": "Glycine"},
    ]
    keys = ["hmdb_id", "hmdb", "kegg_id", "chebi_id", "pathwhiz_id"]
    for row in rows:
        assert cr._first_nonempty(row, keys) == ir_mod._first_nonempty(row, keys)
        assert cr._db_id(row, keys) == ir_mod._db_id(row, keys)
    assert cr._dedupe_aliases(values) == ir_mod._dedupe_aliases(values)

    mine: Dict[str, Any] = {}
    theirs: Dict[str, Any] = {}
    cr._add_issue(mine, "error", "c", "m", pointer="/p", extra=1)
    ir_mod._add_issue(theirs, "error", "c", "m", pointer="/p", extra=1)
    assert mine == theirs and mine["ok"] is False


# ---------------------------------------------------------------------------
# The move itself
# ---------------------------------------------------------------------------


def test_the_four_moved_functions_now_live_in_the_new_module() -> None:
    """NEW ACCEPTANCE. All four are defined in compound_resolution, not ir."""
    for name in MOVED:
        func = getattr(cr, name)
        assert func.__module__ == "t2pw.pwml.compound_resolution", name

    # ir.py keeps working references, by re-import rather than by definition:
    # :920 (the preflight, which stays) and the build_pwml_ir call site.
    for name in ("_compound_external_ids", "_resolve_compound_rows"):
        assert getattr(ir_mod, name) is getattr(cr, name), name

    source = inspect.getsource(ir_mod)
    for name in MOVED:
        assert f"\ndef {name}(" not in source, f"ir.py still defines {name}"


def test_extraction_creates_no_import_cycle() -> None:
    """NEW ACCEPTANCE. ir imports compound_resolution, so the reverse is a cycle.

    Checked over the module's actual import statements, module-level and nested,
    rather than over its text -- the docstring names ir.py legitimately.
    """
    imported = set()
    for node in ast.walk(ast.parse(inspect.getsource(cr))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert "t2pw.pwml.ir" not in imported, f"cycle: compound_resolution imports {imported}"
    assert imported == {"__future__", "re", "typing", "t2pw.pwml.db_resolver",
                        "t2pw.pwml.name_index", "t2pw.mapping.map_ids"}


# ---------------------------------------------------------------------------
# Adapter part 1 -- report shape
# ---------------------------------------------------------------------------


def test_bare_report_no_longer_raises_on_either_hard_indexed_path() -> None:
    """NEW ACCEPTANCE. A pre-freeze caller holds ``{}``, never _new_report().

    Covers both reachable raise sites -- the legacy-id row and the non-legacy
    unresolved row -- which between them index db_resolution.compounds and
    unresolved.db_identities.
    """
    report: Dict[str, Any] = {}
    rows = [{"name": "Glycine", "pathwhiz_id": "78"}, {"name": "novel thing"}]
    resolved = cr._resolve_compound_rows(
        rows, db_resolver=None, strict_db=False, report=report,
        pointer_prefix="/entities/compounds", name_index=None,
    )
    assert len(resolved) == 2
    assert len(report["db_resolution"]["compounds"]) == 2
    assert report["unresolved"]["db_identities"], "unresolved row was not recorded"


def test_ensure_resolution_report_is_a_noop_on_a_real_report() -> None:
    """NEW ACCEPTANCE. The hardening must not perturb the in-IR call path."""
    report = ir_mod._new_report()
    before = json.dumps(report)  # value AND key order
    returned = cr.ensure_resolution_report(report)
    assert returned is report
    assert json.dumps(report) == before

    populated = {"db_resolution": {"compounds": [{"x": 1}]},
                 "unresolved": {"db_identities": [{"y": 2}]}}
    cr.ensure_resolution_report(populated)
    assert populated["db_resolution"]["compounds"] == [{"x": 1}]
    assert populated["unresolved"]["db_identities"] == [{"y": 2}]


# ---------------------------------------------------------------------------
# Adapter part 2 -- apply_canonical_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_apply_canonical_name_is_keyword_only_and_defaults_true(name: str) -> None:
    """NEW ACCEPTANCE. Required on BOTH moved entry points."""
    parameter = inspect.signature(getattr(cr, name)).parameters["apply_canonical_name"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True


def test_default_true_renames_and_records_exactly_as_before() -> None:
    """NEW ACCEPTANCE. The default is today's behaviour, unchanged."""
    row: Dict[str, Any] = {"name": "gly", "kegg_id": "C00037"}
    report: Dict[str, Any] = {}
    cr._canonicalize_compound_offline(row, name_index=_index(), report=report)

    assert row["name"] == "Glycine"
    assert row["raw_name"] == "gly"
    assert row["db_row"] == {"id": 78, "name": "Glycine"}
    assert row["db_status"] == "matched_offline_name_index"
    assert "gly" in row["aliases"]
    entries = report["name_canonicalization"]["compounds"]
    assert entries[0]["from"] == "gly" and entries[0]["to"] == "Glycine"


def test_false_suppresses_the_rename_and_leaves_a_provenance_record() -> None:
    """NEW ACCEPTANCE. Lets a caller attach identifiers without renaming.

    The rename is what needs reference propagation across the payload's process
    strings; declining it is how a pre-freeze caller stays safe. Declining is
    still recorded, so the suppression is traceable.
    """
    row: Dict[str, Any] = {"name": "gly", "kegg_id": "C00037"}
    report: Dict[str, Any] = {}
    cr._canonicalize_compound_offline(
        row, name_index=_index(), report=report, apply_canonical_name=False,
    )

    assert row == {"name": "gly", "kegg_id": "C00037"}, "row must be untouched"
    assert "compounds" not in report.get("name_canonicalization", {})
    suppressed = report["name_canonicalization"]["compounds_suppressed"]
    assert suppressed == [{
        "from": "gly", "to": "Glycine", "matched_on": "kegg",
        "db_id": 78, "source": "pathwhiz_id_db.json", "applied": False,
    }]


def test_resolve_compound_rows_forwards_apply_canonical_name() -> None:
    """NEW ACCEPTANCE. The knob must reach the canonicalization it controls."""
    def _run(**kwargs: Any) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        rows = [{"name": "gly", "kegg_id": "C00037"}]
        resolved = cr._resolve_compound_rows(
            rows, db_resolver=None, strict_db=False, report=report,
            pointer_prefix="/entities/compounds", name_index=_index(), **kwargs,
        )
        return {"row": resolved[0], "report": report}

    renamed = _run()
    assert renamed["row"]["name"] == "Glycine"

    suppressed = _run(apply_canonical_name=False)
    assert suppressed["row"]["name"] == "gly"
    assert suppressed["report"]["name_canonicalization"]["compounds_suppressed"]
    # Identifier resolution still happened; only the rename was declined.
    assert suppressed["report"]["db_resolution"]["compounds"]


# ---------------------------------------------------------------------------
# Adapter part 3 -- row set and idempotency
# ---------------------------------------------------------------------------


def test_is_idempotent_and_tolerates_undeduped_unpruned_rows() -> None:
    """NEW ACCEPTANCE. Pre-freeze the module sees a strict superset of rows.

    Duplicate names survive the in-IR ``_dedupe_named_rows`` step that runs
    before today's call site, so the module must neither collapse them nor
    change its answer when re-run over rows it has already resolved.
    """
    rows = [
        {"name": "gly", "kegg_id": "C00037"},
        {"name": "gly", "kegg_id": "C00037"},
        {"name": "ferric enterobactin"},
    ]
    first_report: Dict[str, Any] = {}
    first = cr._resolve_compound_rows(
        rows, db_resolver=None, strict_db=False, report=first_report,
        pointer_prefix="/entities/compounds", name_index=_index(),
    )
    assert len(first) == 3, "row set must not be collapsed"
    assert [row["name"] for row in first] == ["Glycine", "Glycine", "ferric enterobactin"]

    second_report: Dict[str, Any] = {}
    second = cr._resolve_compound_rows(
        first, db_resolver=None, strict_db=False, report=second_report,
        pointer_prefix="/entities/compounds", name_index=_index(),
    )
    assert [row["name"] for row in second] == [row["name"] for row in first]
    # Already-canonical rows carry a db_row name, so no second rename is logged.
    assert "compounds" not in second_report.get("name_canonicalization", {})


# ---------------------------------------------------------------------------
# PURE-MOVE PIN -- the primary acceptance criterion
# ---------------------------------------------------------------------------

#: sha256 per committed leg fixture over build_pwml_ir's (ir, report) tuple
#: under all five configurations below, hashed both key-sorted and in insertion
#: order. MOVED ONCE, DELIBERATELY, by C-045a on 2026-08-15 under permanent merge
#: rule 4 -- not a behavioural correction. D-016 (LOCKED) put the species
#: canonicalization before the freeze, so a *standalone* build_pwml_ir -- which is
#: what this pins -- cannot know a species was only deterministically normalized
#: and stops publishing report["preflight"]["species"] and its collision warning.
#: The measured delta, the negative that nothing under ``ir`` moved, and the proof
#: that the production path preserves the preflight unchanged are
#: evidence/c045a_{standalone,production}_delta.json; every digest below
#: regenerates via evidence/c045a_golden_rebaseline.py --mode digest.
#:
#: WHAT THIS DOES AND DOES NOT COVER -- corrected by C-045b, 2026-08-15. The
#: sentence above used to read "and no longer the production path". That was
#: false when it was written: at C-045a's tip only Streamlit ran the pre-freeze
#: sequence, and the CLI entry point (README.md:40 -> scripts/run_pwml.py ->
#: writer.run_pwml_pipeline_export) still reached build_pwml_ir with a payload no
#: pre-freeze stage had touched -- so this configuration WAS a production path,
#: and it exported the un-normalized organism name. C-045b wired the seam into
#: run_pwml_pipeline_export, which is what makes the claim true. Accurately, now:
#: this golden pins build_pwml_ir called WITHOUT the pre-freeze sequence, which is
#: no longer how either production entry point reaches it; both entry points are
#: covered behaviourally instead -- Streamlit by
#: tests/test_streamlit_quarantine_boundary.py, the CLI by
#: tests/test_pwml_writer.py::test_cli_export_emits_the_canonical_organism_and_
#: keeps_its_provenance and ::test_cli_export_runs_every_registered_prefreeze_
#: canonicalizer. The 32 digests below are unchanged by C-045b and were
#: independently reproduced at both SHAs by REV-045a.
#:
#: MOVED A SECOND AND FINAL TIME by C-051b on 2026-08-16, again under permanent
#: merge rule 4 and again NOT a behavioural correction. C-051 made
#: ``build_pwml_ir`` refuse a compound row that carries no resolution verdict, so
#: the configuration the paragraph above describes -- ``build_pwml_ir`` on a raw
#: leg fixture -- no longer produces a digest at all: it raises
#: ``UnresolvedCompoundRowError`` on all 160 (leg, configuration) pairs, measured.
#: ``_leg_digest`` therefore routes each configuration through
#: ``run_prefreeze_resolution`` first, which is what both production entry points
#: do. The move was decomposed rather than asserted, at
#: evidence/c051b_golden_move_attribution.json:
#:
#:   A raw @328862a -> B pre-freeze @328862a  the ROUTING plus the per-config
#:                                            payload isolation it forces, at
#:                                            pre-C-051 code
#:   B            -> C pre-freeze @tip        the C-051 stack's CODE alone
#:   A            -> C                        the whole move; every digest below
#:
#: REV-051b split A->B further: isolation alone moves ``db_match/reason`` on 23
#: pairs and routing alone moves it back exactly, so A->B is FOUR IR paths of
#: which one cancels and the bucket correctly reports three. No digest is
#: affected -- A->C is the two ``synonyms`` paths only.
#:
#: P1: under a hash-verified export of 328862a -- the last SHA before C-051, where
#: this file is byte-identical to its C-045a state -- the superseded 32 digests
#: reproduced 32/32, so the delta below is the whole delta.
#: P3, the explicit negative, measured over the 156 pairs that produce IR on both
#: sides: A->C moves exactly TWO paths under ``ir``, ``entities/compounds[]/
#: synonyms`` and its length, on 12 pairs across 4 legs -- the original supported
#: name preserved as a synonym, which D-015 clause 5 requires and which the
#: standalone configuration never did. The compound verdict field ``db_status``
#: moves A->B on 54 pairs and moves BACK to its A value B->C, net zero: the
#: exporter's post-freeze second resolution pass had been overwriting the
#: pre-freeze verdict, and deleting it restores it. Nothing else under ``ir``
#: moves on any leg or configuration. The remaining delta is all under the
#: report, where the exporter no longer publishes a resolution it no longer
#: performs. Regenerate with evidence/c045a_golden_rebaseline.py --mode digest.
#:
#: MOVED A THIRD TIME by C-050g on 2026-08-16, on ONE leg only. C-050g repaired
#: the source-collision comparison in ``_reject_ambiguous_renames``, which
#: changes the code that stops ``PMC12444477…/strict`` under three of the five
#: configurations -- and ``_leg_digest`` hashes the stop code in, so that leg's
#: digest moves with it. Re-derived with --mode digest at the tip: **1 digest
#: moved, 31 byte-identical, 0 stops removed, 3 codes substituted.** See
#: GOLDEN_PREFREEZE_STOPS below for why the code changed and why that leg still
#: does not export.
#:
#: MOVED A FOURTH TIME by C-050i on 2026-08-17, again on ONE leg only, again under
#: permanent merge rule 4 and again NOT a behavioural correction of anything the
#: golden was pinning. C-050i made ``ir._dedupe_named_rows`` REFUSE a post-freeze
#: ``_norm`` collision in an **entity** bucket instead of dropping a row first-wins
#: on a warning. ``PMC12444477…/strict``'s remaining two building configurations,
#: B and E, hit F-039's ``lipid iv a`` collision (PathBank 40738 against 40982) and
#: now raise ``PWML_IR_DUPLICATE_NAMED_ROW``; ``_leg_digest`` hashes that under a
#: DISTINCT ``#ir_refusal:`` marker, so the digest moves with it. Measured under
#: pytest over the full 32 x 5 sweep: **1 digest moved, 31 byte-identical, 0
#: pre-freeze stops added or removed, 0 codes substituted.** See
#: GOLDEN_IR_REFUSALS below.
#:
#: **Instrument note, recorded because it cost a diagnosis. F-047.** Do NOT read
#: these digests by importing this module outside pytest. An out-of-pytest harness
#: reported different digests for every IR-BUILDING leg -- and reported all 32 as
#: moved at the **base SHA** as well as at the tip, which is how the perturbation
#: was located in the instrument rather than in the code. **pytest is the
#: authority**, and the operational rule is: measure under pytest, and if you must
#: use another harness, measure the base with that same harness first.
#:
#: **The cause is UNKNOWN. Do not reason from a mechanism here.** This note first
#: blamed the ``default=_nonjson`` / ``repr`` fallback in the ``json.dumps`` below.
#: **REV-050i falsified that**: it instrumented ``_nonjson`` and the hook fires
#: **zero times** on the leg it measured, while independently reproducing the
#: divergence (out-of-pytest ``fc587e03…`` against pytest/``GOLDEN``
#: ``64038a74…``). The phenomenon is confirmed and its mechanism is not identified.
#: Recorded as an admitted unknown rather than quietly reworded, because an
#: authoritative-sounding wrong mechanism is worse than none -- the next card would
#: try to reason from it.
#:
#: The one leg C-050i moved is **immune** to the hazard and its digest is therefore
#: safe: after the move none of its five configurations builds an IR at all, so its
#: digest is composed purely of configuration names and stop/refusal codes. Both
#: instruments agreed on it exactly, and REV-050i additionally hand-computed it from
#: first principles and matched the committed value.
#:
#: (The ``64038a74…`` quoted just above is REV-050i's *historical* measurement of
#: ``PMC12312563/strict``. C-050k moved that leg to ``69d9da7b…`` -- see below. The
#: quotation is left as measured, because it is a record of what that instrument
#: reported on that day.)
#:
#: **MOVED, DELIBERATELY, ON EIGHT LEGS by C-050k on 2026-08-17** under permanent
#: merge rule 4, orchestrator-authorised by **D-044 §1**. Measured **under pytest**,
#: never by a direct-import harness (F-047, above).
#:
#: The eight: ``PMC12312563/strict`` ``64038a74…``->``69d9da7b…`` · ``1306
#: PMC12452463/research`` ``dd9a2f5c…``->``dbf62298…`` · ``1358 PMC12096016/research``
#: ``f1f6ff4d…``->``20cbe56b…`` · ``1754 PMC12096016/research``
#: ``33112778…``->``c04624aa…`` · ``1754 PMC12180156/strict``
#: ``e28efcf1…``->``1427c040…`` · ``1754 PMC12452463/research``
#: ``a75cb748…``->``1b503ae4…`` · ``1754 PMC12452463/strict``
#: ``5e40a7ca…``->``219fdbfe…`` · ``1754 PMC12856317/strict``
#: ``32ab0313…``->``a6ca91c5…``.
#:
#: **24 of the 32 legs are byte-identical**, and the eight that moved are **exactly**
#: the eight C-050k's independent census identified as carrying a reference that
#: resolves through an ambiguous alias key. That correspondence is the ratification
#: argument (D-044 §1): the delta is the precise footprint of the change, not a
#: diffuse drift, and two measurements taken by different instruments agree on the
#: same set with no overlap error.
#:
#: **Bounded by measurement, not by assertion.** On the R3 control leg, captured with
#: the same harness on both trees (``evidence/g11/C-050k/07-g9-base.json`` and
#: ``08-g9-tip.json``): the **IR digest did not move**, ``errors`` is empty on both
#: sides, ``report["ok"]`` is ``True`` on both sides -- so **no leg changes whether it
#: exports**, which is D-041 §2 limit 3 -- and the whole delta is **two added
#: ``ambiguous_entity_row_reference`` warnings**. No binding moved: C-050k records the
#: choice and rebinds nothing (D-043 §4).
#:
#: **MOVED A SEVENTH TIME, DELIBERATELY, ON 73 OF 89 LEGS by ORCH-754b on 2026-10-19**
#: under permanent merge rule 4, the re-pin orchestrator-authorised for ORCH-754b. It is the
#: CUMULATIVE move ``c9265fb8`` -> ORCH-754b's tip: ORCH-747/749 (``70d05e03``) moved these
#: digests and the re-pin was owed (``evidence/orch747_ir_digest_delta.json``,
#: ``orch749_ir_digest_delta.json``); ORCH-754b moves them again (the withheld record now
#: names a missing species / identity and carries its evidence; ``competes_with`` is no
#: longer signed). Measured over all 89 legs x all five configurations by
#: ``evidence/orch754b_cumulative_repin.py`` -> ``orch754b_cumulative_repin.json``
#: (G11 ``ORCH-754b/18``): **every moved pointer is an interaction annotation** --
#: ``processes.interactions`` rows, the geometry of the interactions PathWhiz can draw, and
#: the warnings / counts reporting them; **no entity, reaction, transport, biological state,
#: non-interaction visualization, error, ``ok`` or stop moved on any leg**. The new values
#: are pytest's own (``evidence/orch754b_pins_before_repin.pytest.log``), and the
#: out-of-pytest instrument reproduced every committed pin at the base and every pytest
#: digest at the tip once it applied this module's autouse ``_no_live_db`` patch itself.
GOLDEN = {
    "runs/2026-07-27_1623/papers/PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/strict/final_mapped.json": "a8880057346845b22a1be8ce3ef7e5acd2332db0e7893e36beacc14bbba62b59",
    # C-050h delta: f0dd12d5… -> e5a40385…, then the ONLY moved digest again under
    # C-050i: e5a40385… -> d22b58c8…. ``_leg_digest`` hashes each stop's code, and
    # this leg's LAST TWO building configurations (B, E) now refuse post-freeze with
    # ``PWML_IR_DUPLICATE_NAMED_ROW`` -- see GOLDEN_IR_REFUSALS. A / C / D still stop
    # pre-freeze with ``PREFREEZE_DUPLICATE_CANONICAL_ROWS``, unchanged.
    #
    # Measured under pytest over all 32 legs x 5 configurations: **this digest is
    # the only one that moves**, and no other leg's stops change. After this move no
    # configuration of this leg builds an IR at all, so its digest is now composed
    # purely of configuration names and stop/refusal codes.
    "runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict/final_mapped.json": "d22b58c8b308cbd83f0fd10327c7453405fd542e3d997e0c7179e4120de44bd1",
    "runs/2026-07-28_0919/papers/PMC13278307__an-overview-of-mobile-colistin-resistance-mcr-g/strict/final_mapped.json": "e64f42db94c40e1eebe681674fdff2262ce971ed0970bc61c90176fef8592dc8",
    "runs/2026-08-02_2130/papers/PMC12096016/research/final_mapped.json": "e5e027d9e7272dffb033e39392fbe79e55ceda4fc2c43b1d80d3b618345ca505",
    "runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json": "c0b2e21db349c0e28209b0e821080511792d6fce9360b7e959bd137ac811b93d",
    "runs/2026-08-02_2130/papers/PMC12180156/research/final_mapped.json": "3c5f9303b234c08d09e670f9d62de3ca580ba34ffe98a84fda321f22a2b808c8",
    "runs/2026-08-02_2130/papers/PMC12180156/strict/final_mapped.json": "b7b353119c974288b27b4167e2a4ebf9431781d8cc5f1d42d5987ff7fef6d50b",
    "runs/2026-08-02_2130/papers/PMC12444477/research/final_mapped.json": "b90f294e17729f1a8e2b4e3c20c3b8d6d9bbdb498f8e82e3c3af703a5e23a240",
    "runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json": "8bf7913e776db4761451f3750f7ff8407318fa3b32adb1322a24b03a9db6a06a",
    "runs/2026-08-02_2130/papers/PMC12452463/research/final_mapped.json": "7af07be350aded81fc9861171911267f357c130b9b4d9d38f45108f670f6f417",
    "runs/2026-08-02_2130/papers/PMC12782028/strict/final_mapped.json": "5ddfb6c4d04653c58f0332aaeb60d88496c4fce62de27acba8edea1e44c468e0",
    "runs/2026-08-02_2130/papers/PMC12856317/research/final_mapped.json": "4b34a0fac1b710fbc45b52189e6f17c1bafed1cf462642b089f78e8b773cd512",
    "runs/2026-08-02_2130/papers/PMC12856317/strict/final_mapped.json": "1c24675f286c8678407a89610e5901f07d1e580000524a3bc47c720bb39d5c46",
    "runs/2026-08-02_2130/papers/PMC13231680/strict/final_mapped.json": "ce4b7b36597a9ae43bfe0174674f75b2d53224e42708bd32088f87bb27564b50",
    "runs_verify/2026-08-04_1148/papers/PMC13231680/research/final_mapped.json": "c0f61e7a2102a0c9aba81e7ea2a05155defd5acfef51f7bd25b8507da9da25af",
    "runs_verify/2026-08-04_1207/papers/PMC12452463/strict/final_mapped.json": "77bea8f6e872d63cd34e7ebe5aa701d96cb4989dc35bbeef818eb0ef9e4e626c",
    "runs_verify/2026-08-04_1234/papers/PMC12096016/strict/final_mapped.json": "f6d4edb1c14709b73aca3f78070a737aaf85a19e86b47a1c3788bcbbf9e6febe",
    "runs_verify/2026-08-04_1234/papers/PMC12856317/strict/final_mapped.json": "9b45b2f78e0cf8f6f6164c1dc6a24c76708f69762870ce1450bb8dbbb57e399b",
    "runs_verify/2026-08-04_1306/papers/PMC12096016/research/final_mapped.json": "c6caa117c0b6d0e8499a1f3d7fadd9aefdee2fd0ace73ccb3759724e4c884681",
    "runs_verify/2026-08-04_1306/papers/PMC12452463/research/final_mapped.json": "b1ae1a97a4f17eb6618cdc3a6a39170f2841d746559a1322a56d477728aaf7e4",
    "runs_verify/2026-08-04_1358/papers/PMC12096016/research/final_mapped.json": "e0ef63b16e128228605824734786c719314a822173cce3bbd3f6e1374be3ca62",
    "runs_verify/2026-08-04_1504/papers/PMC12856317/strict/final_mapped.json": "4c9d3c3ee47e0966e577a744b3b262a170f80c55d31ccd5198771c8f2f9be185",
    "runs_verify/2026-08-04_1647/papers/PMC12856317/strict/final_mapped.json": "428fce1aa7e9a21f337158a70ead719a05570b163c0e9b1e3458dd2b60fc460d",
    "runs_verify/2026-08-04_1754/papers/PMC12096016/research/final_mapped.json": "6cbe8e113571225f3bfdedfedd84e557f7c01824b4d23c222f679efa92e73212",
    "runs_verify/2026-08-04_1754/papers/PMC12096016/strict/final_mapped.json": "c12b098c9bcbaf000aed439cd4d8ab53f0e90830e4725139fc36c8361c255be6",
    "runs_verify/2026-08-04_1754/papers/PMC12180156/research/final_mapped.json": "7c74400d2ec3d9c0e798551712d481ab43f35414a9796f6bf7d6de18cf68bc3b",
    "runs_verify/2026-08-04_1754/papers/PMC12180156/strict/final_mapped.json": "877243c19d5bb6f85b1950c2b0c85d1207bfd7eb9dab7c832e7ad60d34c76935",
    "runs_verify/2026-08-04_1754/papers/PMC12452463/research/final_mapped.json": "d07b2a2df4a9fc41442bb43e3cbb4003cda0cf796f495b84587a29042e48e8c7",
    "runs_verify/2026-08-04_1754/papers/PMC12452463/strict/final_mapped.json": "f170a48033af977f578ea80f76c47cb751661b5c57d0ed3d94e348423887c587",
    "runs_verify/2026-08-04_1754/papers/PMC12782028/research/final_mapped.json": "2c8897c47475836b45a581258a211a6b039217b73795b6aed68b1f0085c8ad1e",
    "runs_verify/2026-08-04_1754/papers/PMC12856317/research/final_mapped.json": "2f2387efa1437a551e42bb114475cf92630694188600c82a87098880a37dcd5b",
    "runs_verify/2026-08-04_1754/papers/PMC12856317/strict/final_mapped.json": "fb2aab05be6d25626469307dac4a581cc858989c7cae639d14512ec1e7d1b6fc",
    # ADMITTED by C-068 on 2026-08-21 (F-069), the ONE of T-100's three survivors
    # that is an export fixture at all: ``quarantine_report.json -> ok`` is
    # ``true`` with ``refusal_reasons: []`` and ``degree_zero_exports: []``,
    # ``prefreeze_db_resolution`` is ``{"available": true}``, it is accessioned
    # (10 ``"enrichment"`` / 5 ``"ec_number"`` occurrences), and it contributes
    # **0** rows to the C-030 identity census. Measured **under pytest** (F-047):
    # all five configurations BUILD -- ``stops == {}``, no pre-freeze stop and no
    # IR refusal -- so this leg is in neither ``GOLDEN_PREFREEZE_STOPS`` nor
    # ``GOLDEN_IR_REFUSALS``, and over the same 33-leg sweep the other 32 legs'
    # digests AND stops are byte-identical: the differing key set is exactly
    # ``{runs_verify/2026-08-18_1328/…/PMC12096016/research/final_mapped.json}``.
    #
    # THE DIGEST IS A PIN, NOT AN ENDORSEMENT. This payload carries three
    # biological defects, named here so no later reader reads "it is in GOLDEN"
    # as "it is correct":
    #   1. ``processes.reactions[3]`` "EntE-catalyzed adenylation of 2,3-DHB" has
    #      ``outputs: ["enterobactin"]``. **Adenylation produces DHB-AMP.** A
    #      named enzyme is asserted to make a product it does not make -- and the
    #      row's own ``evidence`` calls EntE a "2,3-dihydroxybenzoyl-AMP ligase;
    #      EC 6.2.1.71", so the payload contradicts itself.
    #   2. ``processes.transports[0]`` "enterobactin secretion" carries
    #      ``transporters: [{entity: "EntE", provenance: "inferred"}]`` on a span
    #      reading "secreted ... by a **TolC**-dependent process". The transporter
    #      is not the protein the cited span names. **F-058.**
    #   3. ``processes.reactions[4]``, the EntF assembly, lists ``EntE`` as a
    #      catalyst whose entire ``evidence`` is the reaction's own name
    #      concatenated with a truncated fragment ("EntF-catalyzed enterobactin
    #      assembly with L-serine Enterobactin production from activated
    #      2,3-DHB t"). That is not evidence.
    #
    # **F-079** is registered against this same leg -- classified
    # ``release_ready`` with ``semantic_evaluation: passed`` despite defect 1.
    # **F-079 is NOT fixed here**: it is unowned and needs its own card and its
    # own current-source measurement. Pinning this digest freezes the defect so
    # that card can prove its fix moved something.
    "runs_verify/2026-08-18_1328/papers/PMC12096016/research/final_mapped.json": "6dd347d33e0d47f3bd972c12b293be61cd3244a83336f1fb9ed62c3646ddb9c6",
    # ── C-093, 2026-08-27: 56 legs admitted, one excluded, on the record ──
    #
    # **MOVED A SIXTH TIME, and this is the largest move this golden has taken.**
    # Between C-068 (35 committed legs) and C-093's base the corpus reached **92**:
    # T-101, T-103, T-104, T-105 and T-106 each committed their legs, as did the
    # C-072/C-073 paper validation, the two affected-paper cohorts and the C-081 /
    # C-082 validation cohort. 57 legs arrived; the coverage tripwire had been red
    # ever since the first of them, which is precisely the decay F-069 diagnosed.
    #
    # **Every one of the 57 was decided individually, and both decisions rest on the
    # SAME criterion C-068 used** -- ``quarantine_report.json -> ok``, corroborated by
    # accession occurrence counts and ``prefreeze_db_resolution``. Measured, per leg,
    # from the committed artifacts (``evidence/g11/C-093/05-compound-audit.json`` and
    # ``06-baseline-audit-control.json``):
    #
    #   * **56 legs: report PRESENT, ``ok`` TRUE, ``refusal_reasons: []``,
    #     ``prefreeze_db_resolution: {"available": true}``, ``"enrichment"`` present
    #     (2-23 occurrences).** These are the canonical payloads of completed runs --
    #     exactly the shape of the ONE leg C-068 admitted -- so each is PINNED here.
    #   * **1 leg: report PRESENT, ``ok`` FALSE.** Excluded, in ``EXCLUDED`` below,
    #     with its own refusal named. See that entry: it carries a registered
    #     production finding, and pinning it would have recorded that defect as
    #     expected.
    #
    # There is no blanket rule and no pattern-widening here: the criterion is applied
    # to each leg's own measured verdict, and it happens to separate them 56 / 1.
    #
    # **Nothing was re-baselined to make this green.** Measured under pytest over the
    # whole 92-leg corpus (``evidence/c093_leg_digest_sweep.py``,
    # ``evidence/g11/C-093/03-digest-sweep.json``): **0 of the 34 pre-existing GOLDEN
    # digests moved** -- byte-identical, which is also the control proving the
    # instrument that produced the 56 new digests agrees with the committed ones.
    # **0 of the 57 new legs stops pre-freeze or refuses post-freeze**, so
    # ``GOLDEN_PREFREEZE_STOPS`` and ``GOLDEN_IR_REFUSALS`` did not have to move
    # either: the only two legs in this corpus that stop are the same two as before.
    # Also measured across all 57: **no compound row lacks a resolution verdict** and
    # **no leg carries a ``_norm`` collision among its compound names**, so neither of
    # the two refusal shapes this golden records is latent in them.
    #
    # **THE DIGESTS ARE PINS, NOT ENDORSEMENTS** -- as C-068 wrote of the leg it
    # admitted, and it is worth repeating at this scale. One standing observation,
    # recorded so no later reader mistakes silence for approval: 18 of these 56 legs
    # export a compound identity drawn from a resolution whose own
    # ``mapping_meta.resolution.status`` is ``ambiguous`` or ``fallback`` (e.g.
    # ``isochorismate`` -> 40741, ``2,3-dihydro-2,3-dihydroxybenzoate`` -> 40770).
    # **These legs did not introduce that**: the same shape is present in the frozen
    # 35-leg cohort, on legs already pinned above -- including ``intermediate I`` and
    # ``intermediate II`` on ``PMC12312563__structures-...``, the first entry in this
    # dict. It is a standing property of the pipeline, out of C-093's scope, and it is
    # NOT a per-leg exclusion criterion; treating it as one here would have been the
    # blanket rule this card is forbidden to write.
    "runs_verify/2026-08-21_1822/papers/PMC12782028/research/final_mapped.json": "a43a711197c5b943a23563cda5df709b7e4b79a27713021f9c472f8447171e31",
    "runs_verify/2026-08-21_1822/papers/PMC12782028/strict/final_mapped.json": "e2c0857d7255cc66d5bff5065604f6243da720fb283d8875b414dd24b332b0ec",
    "runs_verify/2026-08-21_2014/papers/PMC12312563/research/final_mapped.json": "9c4ad0d7cade5e864a216b1a43477b79b2878465faa0ed673acffa1701ab1af7",
    "runs_verify/2026-08-21_2014/papers/PMC12312563/strict/final_mapped.json": "241791e904540c9e8dd1c45dc6359629d5a7025f93e26617086472eb02251bc8",
    "runs_verify/2026-08-21_2057/papers/PMC12096016/research/final_mapped.json": "c172ba80b243b408b3917b8f6afce092773d26ae5301df810ffcdc37d1331436",
    "runs_verify/2026-08-21_2057/papers/PMC12096016/strict/final_mapped.json": "f363caed4d26993858100e17e02cb9728ad59f1d74382ab487fa69e547fc9f6d",
    "runs_verify/2026-08-21_2057/papers/PMC12452463/research/final_mapped.json": "836f4c80624c0ab529f844c27c626518163e79ec94f37a0972a012dde67e788b",
    "runs_verify/2026-08-21_2057/papers/PMC12452463/strict/final_mapped.json": "c110566b27508e719f07c968335cc74b1c68da19fe638551cfc9b9835d3ab4dc",
    "runs_verify/2026-08-21_2239/papers/PMC12096016/research/final_mapped.json": "e3c333bf9c55c4d679f508548e6f081b0038e8633a754eba08bde54696a5405e",
    "runs_verify/2026-08-21_2239/papers/PMC12096016/strict/final_mapped.json": "ff40de3bdba44b6b03eb14c3838d41648a00fd0b58aa8f1d9e1805d3185ecb39",
    "runs_verify/2026-08-21_2239/papers/PMC12180156/research/final_mapped.json": "5139cd5226b3cd351cc91b764878753ca2717ba8be586cf996a5c72ab4501342",
    "runs_verify/2026-08-21_2239/papers/PMC12180156/strict/final_mapped.json": "b62b4d50a9db4bd2c3525774bab405e271cc8189f581a45d2a2f52c07565c28b",
    "runs_verify/2026-08-21_2239/papers/PMC12452463/research/final_mapped.json": "2a2ea555db8c5d64a763721d72912d21b255fe6d90fae37a05492d26dbbdd69d",
    "runs_verify/2026-08-21_2239/papers/PMC12452463/strict/final_mapped.json": "b7d1026240b951b1c35e281382ac63188c4e7c7fdc5044437278791f21bec1f1",
    "runs_verify/2026-08-21_2239/papers/PMC12782028/research/final_mapped.json": "0025be2cc106f8507f9701c1a89a7395824b8b69c95980c374e2715e7656c1df",
    "runs_verify/2026-08-21_2239/papers/PMC12782028/strict/final_mapped.json": "c0afe61de14749159cac438919a811509e248ebdab90c0f399d8ad08d082d75e",
    "runs_verify/2026-08-21_2239/papers/PMC12856317/research/final_mapped.json": "4803ed3361fdd6e3f4e005cde63341e880412753ea8cc1552cf078a1ee09c12f",
    "runs_verify/2026-08-21_2239/papers/PMC12856317/strict/final_mapped.json": "647e8eb17244cbb2ea553d295dd75e96f6ce2076e767bd4061cda9592e33173d",
    "runs_verify/2026-08-22_1821/papers/PMC12452463/strict/final_mapped.json": "9c16316597690f8915df6f8806468b048d3deb5b22b7b6725f553313fa05ddce",
    "runs_verify/2026-08-22_2017/papers/PMC12452463/research/final_mapped.json": "0d8ee6d2640c1963f05e4c851b1538ec8f7e84fd7f27722357d3ee94ba643dfa",
    "runs_verify/2026-08-22_2017/papers/PMC12452463/strict/final_mapped.json": "5d02fa2320655f35ea1c0d66f6f20d8a5850af9e800af63d94956d887f49e830",
    "runs_verify/2026-08-22_2017/papers/PMC12856317/research/final_mapped.json": "857e917d68608c0b3c448ca03389bc44e168d412c2f03e657bab15cd850798c0",
    "runs_verify/2026-08-22_2017/papers/PMC12856317/strict/final_mapped.json": "7d05b1d405914846fdadeedd7665e084c2b319027cd00c5c8e2e284774710658",
    "runs_verify/2026-08-22_2147/papers/PMC12096016/research/final_mapped.json": "0f180147b489367a586cf4d9ff87e7be78c6e11cfc94dbd7597515bf91e521cb",
    "runs_verify/2026-08-22_2147/papers/PMC12096016/strict/final_mapped.json": "129ec6823dd1638f0082dfa48c08c077e43ab2152a4e4c584c033ef613bbf699",
    "runs_verify/2026-08-22_2147/papers/PMC12180156/research/final_mapped.json": "66e2537177be299d96da36a83c641b1744ea22ba1e111094425380f1315b928d",
    "runs_verify/2026-08-22_2147/papers/PMC12180156/strict/final_mapped.json": "bdbf5721c736ab0cdd00a57a44aa70b7e2201cbeec66feb8f1b6ac645e294d66",
    "runs_verify/2026-08-22_2147/papers/PMC12452463/research/final_mapped.json": "50cd42bd3c11d9a323dd3cbb92d986865ae8074f2da2f7381df79f33597ab751",
    "runs_verify/2026-08-22_2147/papers/PMC12452463/strict/final_mapped.json": "7e55b2923f54f8348a8e8b60d33ad9bc6aa921696891b16fbbff42f3fffab87d",
    "runs_verify/2026-08-22_2147/papers/PMC12782028/research/final_mapped.json": "53a81d401c8d63cb70011cb0fb01fd30f1409e4c07e9bd3bf4390bbf80c82bf0",
    "runs_verify/2026-08-22_2147/papers/PMC12782028/strict/final_mapped.json": "086bade2ddeb3c55561e279115d5cae0d790fef89ad2bf5aa6e9b218120d709e",
    "runs_verify/2026-08-22_2147/papers/PMC12856317/research/final_mapped.json": "ab9abf0e72137f47ede0299e99eb3378ed3fc6857f8e383def1cc46040d78cc1",
    "runs_verify/2026-08-22_2147/papers/PMC12856317/strict/final_mapped.json": "dd532850ef42306b62c8dc8607825c9eb90b341a2f90fbd63e8582d283339525",
    "runs_verify/2026-08-22_2147/papers/PMC13231680/research/final_mapped.json": "2417b7b15fc7494b5655639d921acfb749e1c67eaa147cbafe15c66de8b992b0",
    "runs_verify/2026-08-22_2147/papers/PMC13231680/strict/final_mapped.json": "d6104ed20299cd949a58b6ce44374d42c36ba0f0116a6d251b6d95e8a8d661af",
    "runs_verify/2026-08-24_1203/papers/PMC12096016/research/final_mapped.json": "1ce8e7a205031c1aab038fc82dd3464d3eddd9e1304021cc84ed6ac336bc7e44",
    "runs_verify/2026-08-24_1203/papers/PMC12096016/strict/final_mapped.json": "42f4f9a2b4805085d030b424ec3b785ea29c24b40585eec46f6bdd12fe5be0e0",
    "runs_verify/2026-08-24_1203/papers/PMC12452463/research/final_mapped.json": "f25fa1e4c3f1e92d3531d4de1b82b98f0f7f5a6342441057dcfd16a30192e46a",
    "runs_verify/2026-08-24_1203/papers/PMC12452463/strict/final_mapped.json": "e68f35aa0355fa536fcf982877da79394dac32368a0e77d15ff7560bb12e2e7f",
    "runs_verify/2026-08-24_1203/papers/PMC12856317/research/final_mapped.json": "bfba6300f2153a9131df3a67018918c6dc5b0fb352e381ab0c2ed816f52dc151",
    "runs_verify/2026-08-24_1203/papers/PMC13231680/strict/final_mapped.json": "04b55d50fea40eec68ac69bfdd85e6036c2d90f985f0896edba2d3d1ec54b0b9",
    "runs_verify/2026-08-24_1402/papers/PMC12180156/research/final_mapped.json": "54f964feeedffdb36394bf70b3da7981d0c52a5b337cfbfc19a329033acd496b",
    "runs_verify/2026-08-24_1402/papers/PMC12782028/research/final_mapped.json": "837983ac4c7725b2583bbe1ce8529099334840964217f9d1ee8986fd8ad0812f",
    "runs_verify/2026-08-24_1428/papers/PMC12096016/research/final_mapped.json": "3b9b15bd738ccad5de3efbb81da7759618fb8ab9a23ac05ee1d8d10583816703",
    "runs_verify/2026-08-24_1428/papers/PMC12096016/strict/final_mapped.json": "a25e647331057b52481f90418ddd5169eb79e78efaab6d74eb435de8696e8bad",
    "runs_verify/2026-08-24_1428/papers/PMC12180156/strict/final_mapped.json": "72fbf5eebad7d51e32a69f9e5121c596c37ccf16e32bbc7dc885ed44a08cf61a",
    "runs_verify/2026-08-24_1428/papers/PMC12444477/strict/final_mapped.json": "268d1912b5d63f5ce3a78c392fd9374baa8a21e6f83744a75bbe141fda9e2523",
    "runs_verify/2026-08-24_1428/papers/PMC12452463/research/final_mapped.json": "4b399cfe764df081c65c1f6077c58ac2edf00909112f135a8e3c699f6d0258fc",
    "runs_verify/2026-08-24_1428/papers/PMC12452463/strict/final_mapped.json": "70af28aa537d2c846e02272bd5177cfb780339032593cb2968929b67ca6f4ac4",
    "runs_verify/2026-08-24_1428/papers/PMC12782028/research/final_mapped.json": "b2b111229e3dc06e06747c8a8ff5b06d05cff1d67e324ad8444dda6156d90412",
    "runs_verify/2026-08-24_1428/papers/PMC12782028/strict/final_mapped.json": "907f26ff79348d31b80058b2304f83b28e37e48a92a7c4bcfafd81dfd81009eb",
    "runs_verify/2026-08-24_1428/papers/PMC12856317/research/final_mapped.json": "11e7794a4c144e1a5ddd48ace812b22f89e5eed735d25051bb138e2049994802",
    "runs_verify/2026-08-24_1428/papers/PMC12856317/strict/final_mapped.json": "11dc2a08cfb41cca4410098bfb88af69a3c0d9537f87d232d733d2dee5edd098",
    "runs_verify/2026-08-25_1216/papers/PMC12444477/research/final_mapped.json": "3704d47128ea70aeaaecc7c44cdb00d800a28aae0ebccbe7369ea773bcea0b5b",
    "runs_verify/2026-08-25_1216/papers/PMC12444477/strict/final_mapped.json": "9208f3bb30d1375889b5c28b5a8ea7d58056eee812b0d9b1be1d082455491ff3",
    "runs_verify/2026-08-25_1216/papers/PMC12856317/strict/final_mapped.json": "90c5b5f09444a5a1d5cd16d8f0784e1c4715eedf0870d6ae0bc726960ae910c1",
}

#: Minimum length, in stripped characters, of an ``EXCLUDED`` reason.
#:
#: Not a style rule. C-068 § 3b requires each reason to state TWO facts, and a
#: string too short to hold them is a silencing dressed as a record. The bound is
#: far below the length of a real reason: it exists to make ``""``, ``" "`` and
#: ``"n/a"`` structurally impossible, not to police prose.
MIN_EXCLUSION_REASON_CHARS = 120


class ExclusionReasonMissing(ValueError):
    """A committed leg was excluded from ``GOLDEN`` without a stated reason."""


def _excluded(*entries: Tuple[str, str]) -> Dict[str, str]:
    """Build :data:`EXCLUDED` so a leg CANNOT enter it without a reason.

    **Structural, not conventional.** There is no dict literal below for a later
    editor to append a bare key to: the register is built from ``(leg, reason)``
    pairs, and a bare string, a 1-tuple, a non-string reason, an empty reason, a
    whitespace reason, a reason shorter than
    :data:`MIN_EXCLUSION_REASON_CHARS`, or a duplicated leg raises
    :exc:`ExclusionReasonMissing` **at import time** -- so the module fails to
    collect rather than one test failing somewhere downstream.

    Proved by :func:`test_excluded_cannot_silence_a_leg_without_a_reason`.
    """
    register: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ExclusionReasonMissing(
                f"an EXCLUDED entry is a (leg, reason) pair; got {entry!r}")
        leg, reason = entry
        if not isinstance(reason, str) or len(reason.strip()) < MIN_EXCLUSION_REASON_CHARS:
            raise ExclusionReasonMissing(
                f"{leg}: excluded with no usable reason ({reason!r}). State why "
                f"this committed leg is not an export fixture -- C-068 § 3b "
                f"requires the quarantine fact AND the refusal-trigger fact.")
        if leg in register:
            raise ExclusionReasonMissing(f"{leg}: excluded twice")
        register[leg] = " ".join(reason.split())
    return register


def _recorded_refusal(leg: str) -> Dict[str, Any]:
    """The quarantine verdict a leg's OWN ``quarantine_report.json`` records.

    C-093. The exclusion register says why a leg is not an export fixture; this
    is what the artifact says, so the two can be compared instead of the reason
    being taken on trust. Missing or unreadable reports come back as
    ``ok=None`` with no triggers, which
    :func:`test_excluded_cannot_silence_a_leg_without_a_reason` treats as a
    failure rather than a pass -- an exclusion whose evidence is gone is not
    justified.
    """
    report = ROOT / leg
    report = report.parent / "quarantine_report.json"
    if not report.is_file():
        return {"ok": None, "refusal_reasons": []}
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ok": None, "refusal_reasons": []}
    reasons = data.get("refusal_reasons")
    return {"ok": data.get("ok"),
            "refusal_reasons": [str(r) for r in reasons]
            if isinstance(reasons, list) else []}


#: Committed leg fixtures deliberately kept OUT of ``GOLDEN``, each with its reason.
#:
#: C-068, closing F-069. Two of T-100's three survivors are not payloads the
#: exporter is ever handed on a completed run, so pinning their digests would pin
#: the output of a path production never takes. They are excluded ON THE RECORD
#: rather than by silence: the coverage tripwire asserts against
#: ``set(GOLDEN) | set(EXCLUDED)``, so it is green on today's corpus and **still
#: fires on a genuinely new committed leg** -- the property that matters.
#: Re-pointing it to pass in both configurations would have destroyed that
#: property (REV-051); leaving it permanently red would have decayed it into
#: noise, the second-order cost F-069 itself diagnoses.
#:
#: **Exclusion is not a biological judgement and must never become one.** Neither
#: entry is out because its biology is disliked; both are out because the file is
#: a pre-quarantine fallback. **An entry here is not a licence to ignore the
#: artifact** -- F-055…F-064 reason about exactly these two legs.
EXCLUDED: Dict[str, str] = _excluded(
    ("runs_verify/2026-08-18_1328/papers/PMC12096016/strict/final_mapped.json",
     "quarantine_report.json -> ok is FALSE with refusal_reasons "
     "['degree_zero_export:1'], so this final_mapped.json is the PRE-QUARANTINE "
     "FALLBACK and not the canonical payload; corroborated by 0 occurrences each "
     "of 'enrichment' and 'ec_number' and a null prefreeze_db_resolution, against "
     "10, 5 and {'available': true} on the same paper's research leg. It is "
     "therefore never a payload build_pwml_ir is handed on a completed run. And "
     "its ONLY structural refusal is strict_invariants.degree_zero_exports == "
     "[{'bucket': 'proteins', 'name': 'Isochorismatase (EntB)'}] -- the exact row "
     "C-059's REASON_ALREADY_COVERED was written to reject (F-075) -- so this "
     "artifact may no longer reflect pipeline behaviour and must not be pinned as "
     "though it did."),
    ("runs_verify/2026-08-18_1328/papers/PMC12452463/strict/final_mapped.json",
     "quarantine_report.json -> ok is FALSE with refusal_reasons "
     "['degree_zero_export:1'], so this final_mapped.json is the PRE-QUARANTINE "
     "FALLBACK and not the canonical payload. Directly WITNESSED, not inferred: "
     "removed_entity_report.json lists 2,3-dihydroxybenzoate, DHB-AMP, Fe2+, EntF "
     "and Fur as removed, and all five are still present in this final_mapped.json "
     "(5, 5, 7, 7 and 5 occurrences). Corroborated by 0 occurrences each of "
     "'enrichment' and 'ec_number' and a null prefreeze_db_resolution. It is "
     "therefore never a payload build_pwml_ir is handed on a completed run. And "
     "its ONLY structural refusal is strict_invariants.degree_zero_exports == "
     "[{'bucket': 'proteins', 'name': 'Isochorismatase (EntB)'}] -- the exact row "
     "C-059's REASON_ALREADY_COVERED was written to reject (F-075) -- so this "
     "artifact may no longer reflect pipeline behaviour and must not be pinned as "
     "though it did."),
    # C-093. The ONE of 2026-08-24 cohort A's legs that is not an export fixture,
    # and the only one of the 57 legs committed since C-068 that is excluded.
    ("runs_verify/2026-08-24_1203/papers/PMC12856317/strict/final_mapped.json",
     "quarantine_report.json -> ok is FALSE with refusal_reasons "
     "['unexportable_entity:2'], so this final_mapped.json is the PRE-QUARANTINE "
     "FALLBACK and not the canonical payload; corroborated by 0 occurrences each "
     "of 'enrichment' and 'ec_number' and a null prefreeze_db_resolution, against "
     "7, 1 and {'available': true} on the same paper's research leg of the same "
     "run. It is therefore never a payload build_pwml_ir is handed on a completed "
     "run. Its ONLY structural refusal is "
     "strict_invariants.unexportable_entities == the two proteins rows "
     "'ATP-dependent Clp protease ATP-binding subunit clpX-like, mitochondrial' "
     "and 'Putative ATP-dependent Clp protease proteolytic subunit, "
     "mitochondrial', each with reason 'protein_missing_external_identity' -- and "
     "THAT REFUSAL IS A REGISTERED FINDING, which is why this leg is excluded "
     "rather than pinned. Both rows carry a correct human accession: CLPX has "
     "pathbank_protein_id 8580 in mapping_meta and uniprot O76031 in "
     "mapping_meta.candidates[0]; CLPP has 3923 and Q16740 in the same two "
     "places; both resolutions are status 'matched' at confidence 1.0. "
     "entity_identity.protein_external_identity scans row, mapped_ids, ids and "
     "mapping_meta and stops there, while ir._first_nonempty also reaches "
     "mapping_meta.candidates[0] -- so the gate reports as absent exactly the "
     "identifier the exporter would have exported. ON THIS TIER the gate is "
     "stricter than the exporter, which is the safe direction -- but that is a "
     "MEASUREMENT OVER THIS CORPUS, NOT A PROPERTY OF THE CODE. The asymmetry "
     "runs both ways: ir._first_nonempty never reads row['ids'] while "
     "protein_external_identity does, so a uniprot or drugbank value present ONLY "
     "under 'ids' would pass the gate and leave the exporter with no identity, "
     "which is the UNSAFE direction. REV-093 measured both over ALL 92 committed "
     "final_mapped.json -- the whole corpus, 14 under runs/ and 78 under "
     "runs_verify/ -- 1011 protein and protein_complex rows including components: "
     "unsafe 0, safe 2, exactly O76031 and Q16740. (An earlier draft said 78, "
     "which is the runs_verify/ subtree and not the corpus; it was re-measured "
     "corpus-wide rather than relabelled, and a git grep for '\"ids\"' across all "
     "92 legs returns zero, so the unsafe tier has no exposure today.) The "
     "measurement is an INTEGRATION-BRANCH artifact, task ORCH-093 label "
     "ladder-all92, at e9aa5c8; it is not in this branch's tree, so look for it "
     "under g11/ORCH-093/ after merge. So no biological gate is weakened TODAY, "
     "the 'ids' tier is latent, and the follow-up card must close both tiers "
     "rather than only the one that fired. Pinning this digest would record the "
     "defect as the expected result. It is the same census the 'proteins' bucket "
     "in tests/test_c030_canonical_identity_fallback.py's CENSUS_ADMISSIONS is "
     "admitted under, and it needs its own card and its own current-source "
     "measurement; C-093 may not touch src/."),
)

#: The (leg, configuration) pairs whose pre-freeze stage STOPS, by code.
#:
#: A stop is a **result of that configuration, not a failure and not a digest**.
#: D-015 clause 6 requires the pre-freeze stage to "fail visibly on ambiguous or
#: dangling references". Refusing is the system working, so these are recorded
#: rather than skipped, deselected, xfailed or swallowed: a change that silently
#: stopped raising, or that raised a different code, or that started raising on a
#: fifth pair, fails ``test_build_pwml_ir_matches_the_pre_extraction_golden``.
#:
#: C-051b measured all four raising identically at 328862a and at its tip, and
#: REV-045a had observed the same four. Both readings recorded them as one
#: phenomenon. **C-050g measured them individually and they are TWO, with
#: different causes and different correct outcomes.**
#:
#: PMC13278307 · C_canned -- ``AMBIGUOUS_RENAME_TARGET``, UNCHANGED and CORRECT.
#: The sources are ``PEtN-lipid A`` and ``modified Lipid A``, two genuinely
#: distinct compounds the canned resolver sends to one target. Applying that
#: rename would merge them, which is inventing biology, so the guard must keep
#: raising here -- and it does. Note this pair stops ONLY under the canned
#: resolver; at production defaults the leg resolves and exports at both SHAs.
#:
#: PMC12444477 · A, C, D -- code SUBSTITUTED by C-050g, ``AMBIGUOUS_RENAME_TARGET``
#: -> ``PREFREEZE_CONNECTIVITY_BROKEN``. The old code was a FALSE diagnosis. Its
#: two sources were ``sn -glycerol 3-phosphate`` and ``sn-glycerol 3-phosphate``:
#: one molecule, two spellings, which ``_norm`` read as two only because its
#: ``[^a-z0-9:+ ]+ -> " "`` substitution re-introduced a double space after
#: ``_canonical`` had already collapsed whitespace. C-050g collapses that at the
#: comparison, so the guard correctly stops firing.
#:
#: **The leg still does not export, and that is a ratified position, not an
#: unfinished fix.** Measured at production defaults, both ``strict_db`` modes:
#: the rename target ``Glycerol 3-phosphate`` normalizes to
#: ``glycerol 3 phosphate``, which is ALSO the ``_norm`` of compound row #20,
#: ``glycerol-3-phosphate`` -- a row that is **not in the rename map**.
#: ``_reject_ambiguous_renames`` groups only over rename-map sources, so neither
#: half of it can see that collision; the rename proceeds, rows 20/36/38 end up
#: sharing one name, a participant reference that resolved to ``compounds#20``
#: starts resolving to ``compounds#20|compounds#36|compounds#38``, and the
#: connectivity signature check stops the run. That refusal is CORRECT under
#: today's rules: D-015 clause 5 requires participant connectivity to be
#: preserved, and the payload holds four spellings of this one molecule (#20,
#: #36, #37, #38). Merging duplicate ROWS is a policy this codebase does not have
#: -- pre-freeze may not (``PREFREEZE_ROW_COUNT_CHANGED``) and post-freeze may not
#: (permanent merge rule 8) -- and it is routed as a separate card. The
#: no-PWML outcome is an accepted ``PRODUCT_CONTRACT`` §1 cost until then.
#:
#: The pre-existing collision ``lipid iv a`` (rows 5 and 23) does NOT stop the
#: run: it is identical before and after the rename, so the signature does not
#: move. It is why 44 committed rows became 43 IR compounds at the integration
#: base -- the post-freeze exporter merged that pair silently.
#: C-050h · PMC12444477 · A, C, D -- code SUBSTITUTED again,
#: ``PREFREEZE_CONNECTIVITY_BROKEN`` -> ``PREFREEZE_DUPLICATE_CANONICAL_ROWS``.
#: **The refusal is not lifted and was not meant to be** (D-034 clause 1,
#: ratified; D-036 records the census measuring this group NOT-PROVEN under
#: D-035 clause 3, KEGG ``C03189`` against ``C00093``). What moved is the
#: diagnosis: the connectivity check reported a truncated diff of two resolved
#: signature strings; ``_reject_duplicate_canonical_rows`` now refuses first and
#: names rows 20 / 36 / 38, their before and after spellings, which of them
#: canonicalization moved, and each row's payload-carried identifiers.
#:
#: **Measured scope of the C-050h delta over this whole table: three
#: (leg, configuration) pairs, all on this leg, code only.** No pair started or
#: stopped stopping and no built IR moved -- including PMC13278307 · C_canned
#: below, still raising ``AMBIGUOUS_RENAME_TARGET`` on two genuinely distinct
#: compounds as D-035 clause 7 requires.
GOLDEN_PREFREEZE_STOPS: Dict[str, Dict[str, str]] = {
    "runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict/final_mapped.json": {
        "A_dbdown_defaultindex_strict": "PREFREEZE_DUPLICATE_CANONICAL_ROWS",
        "C_canned_defaultindex_lenient": "PREFREEZE_DUPLICATE_CANONICAL_ROWS",
        "D_emptydb_defaultindex_strict": "PREFREEZE_DUPLICATE_CANONICAL_ROWS",
    },
    "runs/2026-07-28_0919/papers/PMC13278307__an-overview-of-mobile-colistin-resistance-mcr-g/strict/final_mapped.json": {
        "C_canned_defaultindex_lenient": "AMBIGUOUS_RENAME_TARGET",
    },
}

#: Prefix that distinguishes a POST-freeze exporter refusal from a PRE-freeze stop
#: inside ``_leg_digest``'s single ``stops`` map. See that function's docstring.
IR_REFUSAL_PREFIX = "ir_refusal:"

#: The (leg, configuration) pairs whose **post-freeze IR build** REFUSES, by code.
#:
#: Introduced by **C-050i**, and the fourth deliberate move of this golden under
#: permanent merge rule 4. Like ``GOLDEN_PREFREEZE_STOPS`` above, a refusal here is
#: **a RESULT of that configuration, not a failure and not a digest**: C-050i made
#: ``ir._dedupe_named_rows`` refuse a post-freeze ``_norm`` collision in an ENTITY
#: bucket rather than drop a row first-wins on a warning.
#:
#: **The measured delta is exactly two slots, on one leg**, over the full 32 legs x
#: 5 configurations sweep:
#:
#:   PMC12444477…/strict · B_dbdown_noindex_strict          NEW  ir_refusal
#:   PMC12444477…/strict · E_fromenv_raises_emptyindex_…    NEW  ir_refusal
#:
#: Everything else is untouched, and that is asserted rather than assumed: configs
#: A / C / D on this same leg still stop pre-freeze with
#: ``PREFREEZE_DUPLICATE_CANONICAL_ROWS``, PMC13278307 · C_canned still stops with
#: ``AMBIGUOUS_RENAME_TARGET``, and the other **31 legs' digests do not move**.
#: Only this leg's digest moves, because only its two built configurations became
#: refusals.
#:
#: **Why these two and no others.** The collision is rows 5 and 23,
#: ``'lipid IV_A'`` (PathBank 40982 / ChEBI 58603) against ``'lipid IV A'``
#: (PathBank 40738 / ChEBI 60365 / KEGG C06025) -- F-039's pair, one ``_norm`` key
#: ``lipid iv a``. At the integration base the exporter dropped row 23 and re-bound
#: reaction 9 ``"lipid IV A -> lipid A precursor"`` to ``cmp_6``, a **different
#: database compound**, after the canonical hash, reporting one warning and no
#: error: 44 committed compound rows became 43 IR compounds. Configs A / C / D
#: never reach the exporter because pre-freeze stops them first, so only B and E
#: can move. C-050i re-measured every committed leg across all nine buckets and
#: found this to be the **only** ``_norm`` collision in the corpus.
#:
#: **Component buckets are NOT in this table and must not be added to it.** C-050i's
#: guard binds the entity call site only; the component call site keeps its warning
#: because ``prefreeze_resolution._canonicalize_species_rows`` deliberately
#: converges a ``_norm`` group onto its leader and those rows share a
#: ``taxonomy_id``. That residual is **F-046**, owned by **C-050j**.
GOLDEN_IR_REFUSALS: Dict[str, Dict[str, str]] = {
    "runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict/final_mapped.json": {
        "B_dbdown_noindex_strict": "PWML_IR_DUPLICATE_NAMED_ROW",
        "E_fromenv_raises_emptyindex_lenient": "PWML_IR_DUPLICATE_NAMED_ROW",
    },
}


def _expected_outcomes(leg: str) -> Dict[str, str]:
    """The two tables, merged the way ``_leg_digest`` reports them.

    Kept as one function so a leg can never be checked against one table and
    silently skipped by the other.
    """
    expected = dict(GOLDEN_PREFREEZE_STOPS.get(leg, {}))
    for config, code in GOLDEN_IR_REFUSALS.get(leg, {}).items():
        assert config not in expected, (
            f"{leg}/{config} is recorded as BOTH a pre-freeze stop and an IR "
            f"refusal; a configuration cannot stop twice")
        expected[config] = f"{IR_REFUSAL_PREFIX}{code}"
    return expected


def _configs() -> List[Any]:
    """Five deterministic, wholly offline configurations.

    The reason strings below are load-bearing: they are written into the report
    as ``db_resolution.reason`` and are therefore hashed into GOLDEN.
    """

    class _DownDb:
        last_error = "harvest_db_down"

        def available(self) -> bool:
            return False

    class _EmptyDb:
        def available(self) -> bool:
            return True

        def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:
            return []

    class _CannedDb:
        def available(self) -> bool:
            return True

        def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:
            return [{
                "id": 78, "name": "Glycine", "short_name": "Gly",
                "hmdb_id": "HMDB0000123", "kegg_id": "C00037", "chebi_id": "15428",
                "pubchem_cid": "750", "cas": "56-40-6", "biocyc_id": "GLY",
                "chemspider_id": "730", "drugbank_id": "DB00145",
                "pwc_id": "PW_C000123", "description": "canned",
                "synonyms": "Glycine; Gly",
            }]

    index = default_name_index()
    assert index is not None, "data/pathwhiz_id_db.json is missing"
    return [
        ("A_dbdown_defaultindex_strict",
         dict(db_resolver=_DownDb(), strict_db=True, name_index=index)),
        ("B_dbdown_noindex_strict",
         dict(db_resolver=_DownDb(), strict_db=True, name_index=None)),
        ("C_canned_defaultindex_lenient",
         dict(db_resolver=_CannedDb(), strict_db=False, name_index=index)),
        ("D_emptydb_defaultindex_strict",
         dict(db_resolver=_EmptyDb(), strict_db=True, name_index=index)),
        ("E_fromenv_raises_emptyindex_lenient",
         dict(db_resolver=None, strict_db=False, name_index=PathwhizNameIndex({}))),
    ]


def _nonjson(obj: Any) -> Dict[str, str]:
    return {"__nonjson__": type(obj).__name__, "repr": repr(obj)}


def _leg_digest(payload: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """Digest ``build_pwml_ir`` over the five configurations, pre-freeze first.

    Returns ``(digest, stops)`` -- **arity deliberately unchanged**, because
    ``evidence/c045a_golden_rebaseline.py:173`` serializes this tuple whole.
    ``stops`` maps a configuration name to the code that prevented a digest, from
    either of **two distinct classes**:

    * a bare code -- a **pre-freeze** :class:`PrefreezeResolutionError`, asserted
      against ``GOLDEN_PREFREEZE_STOPS``;
    * ``ir_refusal:<code>`` -- a **post-freeze** :class:`DuplicateNamedRowError`
      from inside ``build_pwml_ir``, asserted against ``GOLDEN_IR_REFUSALS``
      (C-050i).

    The prefix is what keeps them apart in one map, so the existing
    ``GOLDEN_PREFREEZE_STOPS`` values did not have to move.

    The pre-freeze call is not decoration: since C-051 ``build_pwml_ir``
    refuses a compound row that carries no resolution verdict, so a raw leg
    fixture no longer reaches a digest at all. Routing through
    ``run_prefreeze_resolution`` is what production does at both export entry
    points, and it is the same re-pointing C-051 applied to the other refused
    test nodes.

    **Each configuration gets its own deep copy of the fixture, and its own
    resolver and index.** The pre-freeze stage rewrites the payload in place --
    renames, aliases, the verdict field, the ``prefreeze_db_resolution``
    carrier -- so sharing one payload object across the five, as this function
    did while the sweep was read-only, would feed each configuration the
    previous one's renames and measure a sequence rather than five
    configurations.
    """
    digest = hashlib.sha256()
    stops: Dict[str, str] = {}
    for name, kwargs in _configs():
        digest.update(name.encode())
        staged = copy.deepcopy(payload)
        try:
            run_prefreeze_resolution(
                staged,
                strict_db=kwargs["strict_db"],
                db_resolver=kwargs["db_resolver"],
                name_index=kwargs["name_index"],
            )
        except PrefreezeResolutionError as stop:
            # D-015 clause 6 -- "fail visibly on ambiguous or dangling
            # references" -- is a RESULT of this configuration, not a failure of
            # the golden and not a digest. It is hashed in by code as well as
            # asserted below, so a change that silently stopped raising would
            # move the digest too and could not pass by matching one of them.
            stops[name] = stop.code
            digest.update(f"#prefreeze_stop:{stop.code}".encode())
            continue
        try:
            built = build_pwml_ir(staged, **kwargs)
        except DuplicateNamedRowError as refusal:
            # C-050i. A POST-freeze refusal, and a different class from the
            # pre-freeze stop above: this one happened inside the exporter, on a
            # payload the pre-freeze stage passed. Recorded under a DISTINCT
            # ``ir_refusal:`` prefix and hashed with a DISTINCT digest marker so
            # the two can never be conflated -- neither by the table below nor by
            # a digest that happened to collide.
            stops[name] = f"{IR_REFUSAL_PREFIX}{refusal.code}"
            digest.update(f"#ir_refusal:{refusal.code}".encode())
            continue
        for sort_keys in (True, False):
            blob = json.dumps(list(built), sort_keys=sort_keys, indent=1, default=_nonjson)
            digest.update(hashlib.sha256(blob.encode()).hexdigest().encode())
    return digest.hexdigest(), stops


def test_build_pwml_ir_matches_the_pre_extraction_golden() -> None:
    """NEW ACCEPTANCE, and C-040's primary acceptance criterion.

    Asserts that nothing changed that C-051's documented delta does not
    account for: over every committed leg fixture, ``build_pwml_ir``'s output on
    the pre-freeze-routed payload is byte-identical to the re-baselined sweep,
    the ``(leg, configuration)`` pairs that stop pre-freeze stop with exactly the
    recorded code, and the pairs that refuse **post-freeze** refuse with exactly
    the recorded code (``GOLDEN_IR_REFUSALS``, C-050i). A non-empty diff here is a
    drift, not a licence to re-baseline.
    """
    mismatched = []
    for leg, expected in GOLDEN.items():
        path = ROOT / leg
        assert path.is_file(), f"committed leg fixture is missing: {leg}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual, stops = _leg_digest(payload)
        expected_stops = _expected_outcomes(leg)
        if stops != expected_stops:
            mismatched.append(
                f"{leg}\n  expected stops {expected_stops}"
                f"\n  actual stops   {stops}")
        if actual != expected:
            mismatched.append(f"{leg}\n  expected {expected}\n  actual   {actual}")
    assert not mismatched, "build_pwml_ir output drifted:\n" + "\n".join(mismatched)


def _leg_coverage_gap(found: Set[str]) -> Tuple[List[str], List[str]]:
    """``(unaccounted, stale)`` for a set of discovered leg paths.

    Split out of the tripwire by C-068 so its predicate can be exercised on a
    **synthesized** leg path without writing into ``runs/`` or ``runs_verify/``,
    which are read-only evidence (D-055, F-055…F-064).
    """
    accounted = set(GOLDEN) | set(EXCLUDED)
    return sorted(found - accounted), sorted(accounted - found)


def test_the_golden_covers_every_committed_leg_fixture() -> None:
    """NEW ACCEPTANCE. A new committed leg must be ACCOUNTED FOR deliberately.

    **C-068 widened the accounting set, not the assertion** (F-069). Every
    committed leg must appear in ``GOLDEN`` -- pinned -- or in ``EXCLUDED`` with
    a stated reason, which :func:`_excluded` makes structurally impossible to
    skip. A leg in neither still fails here, and that is the property the test
    exists for; :func:`test_the_coverage_tripwire_fires_on_an_unaccounted_leg`
    proves the widened form is not vacuous.
    """
    found = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for root in ("runs", "runs_verify")
        for path in (ROOT / root).rglob("final_mapped.json")
    }
    assert found, "no committed leg fixtures found under runs/ or runs_verify/"
    both = sorted(set(GOLDEN) & set(EXCLUDED))
    assert not both, f"a leg is BOTH pinned and excluded: {both}"
    unaccounted, stale = _leg_coverage_gap(found)
    assert (unaccounted, stale) == ([], []), (
        f"unaccounted -- add to GOLDEN, or to EXCLUDED with a reason: "
        f"{unaccounted}; stale in GOLDEN/EXCLUDED: {stale}"
    )


def test_the_coverage_tripwire_fires_on_an_unaccounted_leg() -> None:
    """NEW ACCEPTANCE (C-068, RULING 13). The widened tripwire is NOT vacuous.

    ``found == set(GOLDEN) | set(EXCLUDED)`` would look green forever if the
    accounting set silently absorbed whatever it was given. It does not: a leg
    path in neither register is reported unaccounted, and a register key with no
    file on disk is still reported stale. The synthetic path is never written to
    disk -- ``runs/`` and ``runs_verify/`` are read-only evidence.
    """
    real = set(GOLDEN) | set(EXCLUDED)
    assert _leg_coverage_gap(real) == ([], [])

    synthetic = "runs_verify/9999-99-99_9999/papers/PMCNONVACUITY/research/final_mapped.json"
    assert synthetic not in real
    assert _leg_coverage_gap(real | {synthetic}) == ([synthetic], [])

    # and an EXCLUDED key does not become a place to hide a deleted fixture.
    dropped = sorted(EXCLUDED)[0]
    assert _leg_coverage_gap(real - {dropped}) == ([], [dropped])


def test_the_repaired_coverage_and_digest_pins_themselves_go_red(
        monkeypatch: Any) -> None:
    """NON-VACUITY (C-093), permanent, and the strongest form of it.

    :func:`test_the_coverage_tripwire_fires_on_an_unaccounted_leg` exercises
    :func:`_leg_coverage_gap`, the predicate. This drives **the two repaired test
    functions themselves** and asserts they RAISE, which is the only thing that
    answers "did admitting 56 legs turn the pin into a rubber stamp?":

    1. drop one leg from ``GOLDEN`` -- the corpus scan must report it unaccounted,
       so the tripwire has not been widened into an unconditional pass;
    2. corrupt one digest -- ``build_pwml_ir``'s output pin must still bite, so
       the 56 admitted digests are load-bearing values and not decoration;
    3. blank ``EXCLUDED`` -- the excluded leg must come back as unaccounted, so
       exclusion is a record and not a deletion.

    (2) is run over a ONE-leg ``GOLDEN`` on purpose: the assertion under test is
    per-leg, and sweeping all 90 to prove it would cost seconds for no more proof.
    """
    real_golden, real_excluded = dict(GOLDEN), dict(EXCLUDED)
    test_the_golden_covers_every_committed_leg_fixture()  # control: green

    dropped = sorted(real_golden)[0]
    monkeypatch.setitem(globals(), "GOLDEN",
                        {k: v for k, v in real_golden.items() if k != dropped})
    with pytest.raises(AssertionError, match="unaccounted"):
        test_the_golden_covers_every_committed_leg_fixture()

    monkeypatch.setitem(globals(), "GOLDEN", real_golden)
    monkeypatch.setitem(globals(), "EXCLUDED", {})
    with pytest.raises(AssertionError, match="unaccounted"):
        test_the_golden_covers_every_committed_leg_fixture()

    monkeypatch.setitem(globals(), "EXCLUDED", real_excluded)
    admitted = "runs_verify/2026-08-24_1203/papers/PMC12452463/strict/final_mapped.json"
    assert admitted in real_golden, "the C-093 admission this proves is load-bearing"
    monkeypatch.setitem(globals(), "GOLDEN", {admitted: "0" * 64})
    with pytest.raises(AssertionError, match="drifted"):
        test_build_pwml_ir_matches_the_pre_extraction_golden()

    monkeypatch.setitem(globals(), "GOLDEN", {admitted: real_golden[admitted]})
    test_build_pwml_ir_matches_the_pre_extraction_golden()  # and green again


def test_excluded_cannot_silence_a_leg_without_a_reason() -> None:
    """NEW ACCEPTANCE (C-068 § 6). A leg cannot enter EXCLUDED reason-free.

    The register is a constructor, not a literal, so there is no bare key to
    append: every reason-free shape raises, and it raises at import time.

    **RE-BASED BY C-093, from three literals to the property they were standing
    in for.** C-068 asserted that every reason contains the strings
    ``degree_zero_export`` and ``C-059``, which was true of the two legs it
    excluded because both were refused by the same trigger. That is an accident of
    a two-element register, not a rule: the third exclusion is refused with
    ``unexportable_entity:2``, and under the old assertion an accurate reason for
    it would have FAILED while a copy-pasted inaccurate one would have passed --
    the exact inversion a silencing test must not have. The property those
    literals approximate is that a reason must be checkable against the artifact,
    so it is now asserted directly: **every reason must name the quarantine
    verdict AND quote that leg's OWN recorded ``refusal_reasons``, read from its
    ``quarantine_report.json`` on disk.** A reason describing the wrong refusal
    now fails, which the literal form could not detect.
    """
    leg = "runs/2026-01-01_0000/papers/PMCFAKE/strict/final_mapped.json"
    for entry in (leg, (leg,), (leg, ""), (leg, "   "), (leg, "n/a"),
                  (leg, None), (leg, 0), (leg, "excluded, see the ledger"),
                  (leg, "x" * (MIN_EXCLUSION_REASON_CHARS - 1))):
        with pytest.raises(ExclusionReasonMissing):
            _excluded(entry)  # type: ignore[arg-type]

    good = "z" * MIN_EXCLUSION_REASON_CHARS
    assert _excluded((leg, good)) == {leg: good}
    with pytest.raises(ExclusionReasonMissing):
        _excluded((leg, good), (leg, good))

    for path, reason in EXCLUDED.items():
        assert len(reason) >= MIN_EXCLUSION_REASON_CHARS, path
        assert "quarantine_report.json -> ok is FALSE" in reason, path
        recorded = _recorded_refusal(path)
        assert recorded["ok"] is False, (
            f"{path} is excluded as a pre-quarantine fallback, but its "
            f"quarantine_report.json does not say ok is false")
        assert recorded["refusal_reasons"], path
        for trigger in recorded["refusal_reasons"]:
            assert trigger in reason, (
                f"{path}: excluded without naming its own refusal {trigger!r}. "
                f"A reason that does not match the artifact is a silencing.")


def test_an_exclusion_reason_that_names_the_wrong_refusal_is_caught() -> None:
    """NON-VACUITY (C-093), permanent. The re-based reason check can go RED.

    The predicate above reads each leg's ``refusal_reasons`` off disk, so it would
    be vacuous if every reason trivially contained every trigger. It does not: a
    reason carrying a DIFFERENT leg's trigger is rejected, and so is one that
    names none. Exercised against the committed register without writing anything
    under ``runs/`` or ``runs_verify/``, which are read-only evidence.
    """
    triggers = {leg: _recorded_refusal(leg)["refusal_reasons"] for leg in EXCLUDED}
    assert all(triggers.values()), "an excluded leg records no refusal at all"

    distinct = {t for values in triggers.values() for t in values}
    # SOFT COUPLING, recorded rather than engineered away (REV-093). This needs
    # TWO different triggers in the register to be able to forge a wrong reason at
    # all -- today 'degree_zero_export:1' and 'unexportable_entity:2'. If the
    # register ever collapses to one trigger it fails with the message below
    # rather than silently going vacuous, and the fix is to build the forged
    # reason from a literal instead of from a sibling entry.
    assert len(distinct) > 1, (
        "the register no longer holds two different refusal triggers, so this "
        "test can no longer distinguish the property from the old literal")

    for leg, reason in EXCLUDED.items():
        mine = set(triggers[leg])
        others = distinct - mine
        assert others, leg
        # the real reason names its own trigger ...
        assert all(t in reason for t in mine), leg
        # ... and a reason built from someone else's would be caught
        forged = " ".join(sorted(others)) + " " + "z" * MIN_EXCLUSION_REASON_CHARS
        assert not all(t in forged for t in mine), leg
