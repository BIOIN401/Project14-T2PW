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
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
GOLDEN = {
    "runs/2026-07-27_1623/papers/PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/strict/final_mapped.json": "64038a74f18848499a00b3ce4ea95555b4f568f78484f5e7bab07abf54af6a8d",
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
    "runs/2026-07-28_0919/papers/PMC13278307__an-overview-of-mobile-colistin-resistance-mcr-g/strict/final_mapped.json": "7954a4c9ae7a2905923b97194e620e1440888f83c4152c023ac7625a381b9e01",
    "runs/2026-08-02_2130/papers/PMC12096016/research/final_mapped.json": "f1a6a4d381e97d31cd09bbf037ed67da0e4a3fe45456906135cab736fb35603b",
    "runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json": "bdcb0fb81a19b8c2f956a959e20fb747d60ebd87aed9ff4bb540d041c66cfa80",
    "runs/2026-08-02_2130/papers/PMC12180156/research/final_mapped.json": "3c5f9303b234c08d09e670f9d62de3ca580ba34ffe98a84fda321f22a2b808c8",
    "runs/2026-08-02_2130/papers/PMC12180156/strict/final_mapped.json": "11718a90bd71b6ed8933186966781e3027043494366cef45ffc809d4a5afcd90",
    "runs/2026-08-02_2130/papers/PMC12444477/research/final_mapped.json": "583c6a72e13ad69c5f8824254fb23e6fb58896680e66cff37c4fdbc040f2513d",
    "runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json": "6c012487141ec0c84966852403cb09c17d63bccb57a7491e9a9e5c5a5cd0537c",
    "runs/2026-08-02_2130/papers/PMC12452463/research/final_mapped.json": "b67d9b705ce2823c1287bfdec1ed23c22151a8efdf72b9225c38a6a1aeabe2ed",
    "runs/2026-08-02_2130/papers/PMC12782028/strict/final_mapped.json": "5ddfb6c4d04653c58f0332aaeb60d88496c4fce62de27acba8edea1e44c468e0",
    "runs/2026-08-02_2130/papers/PMC12856317/research/final_mapped.json": "8f58319162589c80244cf7f4bff45e9f46f66ffe83b8776136fa727eaec2591a",
    "runs/2026-08-02_2130/papers/PMC12856317/strict/final_mapped.json": "9a5eabdc11cc18df0e709f6527a9a676a0a8b31c57f2f6efa82803b93a80c5d5",
    "runs/2026-08-02_2130/papers/PMC13231680/strict/final_mapped.json": "0b1f6a9b3e4281c83c746e6082669a12587a155aa95e67d76ecf2e95c4c85598",
    "runs_verify/2026-08-04_1148/papers/PMC13231680/research/final_mapped.json": "68728359df34dbf5a30ebd4dad8421fef71ee4b1635e68e087411f82a16b2802",
    "runs_verify/2026-08-04_1207/papers/PMC12452463/strict/final_mapped.json": "5c2b04f8b372a648337c1c5b12d72ab5a4ceb64ee86dac40612a12380c2096c8",
    "runs_verify/2026-08-04_1234/papers/PMC12096016/strict/final_mapped.json": "2432316a575c173c98ccc9abb287906bff650051b5047626059dd49fb5baf549",
    "runs_verify/2026-08-04_1234/papers/PMC12856317/strict/final_mapped.json": "c7dfe651a28eaea3729e1b38da4968c4ed8f09195aac4373397f50011896ff30",
    "runs_verify/2026-08-04_1306/papers/PMC12096016/research/final_mapped.json": "4213f64f6f5dd6d9e0fd09b69b4e319138618b5b9fa98e259741930096ff3fce",
    "runs_verify/2026-08-04_1306/papers/PMC12452463/research/final_mapped.json": "dd9a2f5cea146384ea45dab5667c6cdf4afaa49a0f5b59244a84c34e86e8778b",
    "runs_verify/2026-08-04_1358/papers/PMC12096016/research/final_mapped.json": "f1f6ff4d9a235149274a1a2cec0bd51777176973f157f3761f2e5f7fd26a6615",
    "runs_verify/2026-08-04_1504/papers/PMC12856317/strict/final_mapped.json": "f9ac6acd6b8a9728d1cc9995594d861b7f70e4f867bcba7da3bbcb50a3b4365f",
    "runs_verify/2026-08-04_1647/papers/PMC12856317/strict/final_mapped.json": "3fa1cd47e28be8b29a3e4fc5909db94fa4daa33bb8f6c7943506ca8b535707e8",
    "runs_verify/2026-08-04_1754/papers/PMC12096016/research/final_mapped.json": "33112778ffac13bc18c97f4333a1b9b23b0a5d7bd44247609ee42540ddb9ea11",
    "runs_verify/2026-08-04_1754/papers/PMC12096016/strict/final_mapped.json": "7ac1c6bbfbdf9ba0c1e6b91b1e697ad373b4cdd67b5cd89eb347931405355174",
    "runs_verify/2026-08-04_1754/papers/PMC12180156/research/final_mapped.json": "a2540f701344d92753f59b2bbcfb6122bd8c34684c427d8c2c23f5395d5f7401",
    "runs_verify/2026-08-04_1754/papers/PMC12180156/strict/final_mapped.json": "e28efcf175ab89293e987502ab88e6513c9d530a6c078378befd55cb9d1a7d24",
    "runs_verify/2026-08-04_1754/papers/PMC12452463/research/final_mapped.json": "a75cb748ed26640f91db16508e1d081e0ed2207850097932fb2cc4f673de9e68",
    "runs_verify/2026-08-04_1754/papers/PMC12452463/strict/final_mapped.json": "5e40a7cab6ee37d6d3ad265f3cb079906f041e2f7266d46d85a4973fb0fb600e",
    "runs_verify/2026-08-04_1754/papers/PMC12782028/research/final_mapped.json": "2c8897c47475836b45a581258a211a6b039217b73795b6aed68b1f0085c8ad1e",
    "runs_verify/2026-08-04_1754/papers/PMC12856317/research/final_mapped.json": "5ca749ae322a5e0b1998b934a945d3a0e41ca61a921197be66eaa9085d32dd38",
    "runs_verify/2026-08-04_1754/papers/PMC12856317/strict/final_mapped.json": "32ab0313dffd1e0b295a92256e46d0f50024d88ca7c33aa8c9b13990b171a3a6",
}

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


def test_the_golden_covers_every_committed_leg_fixture() -> None:
    """NEW ACCEPTANCE. A new committed leg must be added to GOLDEN deliberately."""
    found = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for root in ("runs", "runs_verify")
        for path in (ROOT / root).rglob("final_mapped.json")
    }
    assert found, "no committed leg fixtures found under runs/ or runs_verify/"
    assert found == set(GOLDEN), (
        f"missing from GOLDEN: {sorted(found - set(GOLDEN))}; "
        f"stale in GOLDEN: {sorted(set(GOLDEN) - found)}"
    )
