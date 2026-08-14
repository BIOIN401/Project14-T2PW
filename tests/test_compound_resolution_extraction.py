"""C-040 acceptance tests for ``t2pw.pwml.compound_resolution``.

**Every test here is a NEW ACCEPTANCE test for a genuinely new module, not a
regression proof.** C-040 is a mechanical extraction: it moves four functions
out of ``ir.py`` and changes no caller's behaviour. There is therefore no
base-SHA behavioural failure to demonstrate, and none was manufactured. Nothing
below is written to fail on ``ImportError`` at the base SHA either -- symbol
absence is not behavioural proof.

The load-bearing assertion is the opposite of a regression proof:
``test_build_pwml_ir_matches_the_pre_extraction_golden`` asserts that **nothing**
changed. Its digests were derived from a sweep captured at the dispatch base
``e4eeef429468ef42cfdfd12295ea86447f0c674f`` *before* the extraction, so it pins
pre-extraction behaviour and will fail if the move was not pure.

That pin is deliberate and it is expected to be moved -- once, deliberately, with
a documented delta -- by C-051, which deletes the in-IR resolution call. It must
never be moved to make an accidental drift go green.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml import compound_resolution as cr  # noqa: E402
from t2pw.pwml import ir as ir_mod  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir  # noqa: E402
from t2pw.pwml.name_index import PathwhizNameIndex, default_name_index  # noqa: E402

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
#: order. DERIVED FROM A SWEEP AT BASE e4eeef42, BEFORE the extraction.
GOLDEN = {
    "runs/2026-07-27_1623/papers/PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/strict/final_mapped.json": "f7b90d9c316551e7ea5a4926e44ccf56bc06526c0f16bf3430f00681ab37464f",
    "runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict/final_mapped.json": "4776eaf3fa304f427be50f01d1647c18d2cc129ecbb1e57b475a109cdf2e9b4f",
    "runs/2026-07-28_0919/papers/PMC13278307__an-overview-of-mobile-colistin-resistance-mcr-g/strict/final_mapped.json": "636e7a9cca659cca4e365404ff1d27ae29776660bcf7e2fff84134e4be6f9529",
    "runs/2026-08-02_2130/papers/PMC12096016/research/final_mapped.json": "66469f59bc720627eeb9acd5b2a9d1d67db2ef201a8c1fe5684c94260c499917",
    "runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json": "9d9adcad5a124045750c99b6a37562e3d402cef8c84b627753563330961aaff4",
    "runs/2026-08-02_2130/papers/PMC12180156/research/final_mapped.json": "b355d20d744eba2d4e3efe0b909480926893440a4071f6be6317702f6deae3be",
    "runs/2026-08-02_2130/papers/PMC12180156/strict/final_mapped.json": "76f9186a80caf753b644693348a0532af8cbc986b0888847d9accba4908df90b",
    "runs/2026-08-02_2130/papers/PMC12444477/research/final_mapped.json": "55337f73f621ec067d328a7beabc563d53cb2339d74a02dda3462d770d1ff0ad",
    "runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json": "df5d14202ec1109d98bf5e210338b096bd96c09c12a361fdcda419c1592d9d2d",
    "runs/2026-08-02_2130/papers/PMC12452463/research/final_mapped.json": "37ecb1279d0635128b608707b29df8c01b2c792e58d0c9658a85fc5be3f2c877",
    "runs/2026-08-02_2130/papers/PMC12782028/strict/final_mapped.json": "0aae3942f7a384106549ca7dc5c8f47d70853477925d53d6d155d15474eff360",
    "runs/2026-08-02_2130/papers/PMC12856317/research/final_mapped.json": "ed0efe5ef1002bf5e3ec7c25dd07a2b75c7ac421bb36821dd121739a768d9801",
    "runs/2026-08-02_2130/papers/PMC12856317/strict/final_mapped.json": "e439b998b658df3507169db4b393ea43084eefaecef77075958b4b61abde5f7b",
    "runs/2026-08-02_2130/papers/PMC13231680/strict/final_mapped.json": "15a1cc3b2454d03a76ed98ef3c1e7d135540ac806d1c0da6a9629b201c8422bd",
    "runs_verify/2026-08-04_1148/papers/PMC13231680/research/final_mapped.json": "84b5f7d96333ad19f29112fe20bf90876b166d6be063e9bf54b964bc8359fe70",
    "runs_verify/2026-08-04_1207/papers/PMC12452463/strict/final_mapped.json": "754736facd1d7745d7dbf074ff0d1fc528c742830560e9ad9dd3fcacc73f4ba7",
    "runs_verify/2026-08-04_1234/papers/PMC12096016/strict/final_mapped.json": "f4c7e27b60df966d43b0a5cf66b1ce8602f6253de65de19658ebe98451115d01",
    "runs_verify/2026-08-04_1234/papers/PMC12856317/strict/final_mapped.json": "020e3af1b35eab20911e31128040cacc49600b579bc63a095b572ecf4f902043",
    "runs_verify/2026-08-04_1306/papers/PMC12096016/research/final_mapped.json": "14034d1ebca67cc28b572b3eead411b10db00fbe4e3ede9e5c9698aab9ff0c09",
    "runs_verify/2026-08-04_1306/papers/PMC12452463/research/final_mapped.json": "fbd84dbe7222b2a007c89cc15e6823ae69e3dcafa07ebf6820411e9f52d03a82",
    "runs_verify/2026-08-04_1358/papers/PMC12096016/research/final_mapped.json": "ce6b5f23bf6b6fcbf7b23ae8971f9f36468b61ee8e6ebcdb69ca0f6743a5ebde",
    "runs_verify/2026-08-04_1504/papers/PMC12856317/strict/final_mapped.json": "281875fc54583cb78e83c3a97510a04fb365300e40dd7aaaff901032aa817384",
    "runs_verify/2026-08-04_1647/papers/PMC12856317/strict/final_mapped.json": "f13b25e1d30f8c86bca8ff40ff8ca704b81294e4ac09f7c6611868dfde1af328",
    "runs_verify/2026-08-04_1754/papers/PMC12096016/research/final_mapped.json": "cf6eb786ae0180e26f547e930d50ce8ef3455cb495659191b5ab88a93e762618",
    "runs_verify/2026-08-04_1754/papers/PMC12096016/strict/final_mapped.json": "7f23596a8d47bfc9fb75cb81a357df4d9d5102c195be61172b43a87747379a89",
    "runs_verify/2026-08-04_1754/papers/PMC12180156/research/final_mapped.json": "b62db93e3206d2e17d751cf53c1115f640eb6407a5eb89a2108c6ae4d8e095fc",
    "runs_verify/2026-08-04_1754/papers/PMC12180156/strict/final_mapped.json": "fb3c0a4787b5b3d298f222485e19d454d706a3b4bafa6075e4a9b6282f9059ce",
    "runs_verify/2026-08-04_1754/papers/PMC12452463/research/final_mapped.json": "7b61228e9c9ea71846b7fa7e6fe7bc9da276c5199129a57e2b26b8e24e05b6ab",
    "runs_verify/2026-08-04_1754/papers/PMC12452463/strict/final_mapped.json": "0393db5ef8c9f6065a85067fdcc6ec143b018a90b5d137dd8229de6ebbd637dc",
    "runs_verify/2026-08-04_1754/papers/PMC12782028/research/final_mapped.json": "5c72760a9e6d300eae50ea7c87c892f07c82db114f2e35fad59b14d18656a556",
    "runs_verify/2026-08-04_1754/papers/PMC12856317/research/final_mapped.json": "ee59deb6df99861940633713185849c5e2db3631a1f7d93d77426bb5696bfac1",
    "runs_verify/2026-08-04_1754/papers/PMC12856317/strict/final_mapped.json": "ddc900f9731d69a03f197b083a38a2bc3b4c05d77d7534f827fa498df13da9f1",
}


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


def _leg_digest(payload: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, kwargs in _configs():
        built = build_pwml_ir(payload, **kwargs)
        digest.update(name.encode())
        for sort_keys in (True, False):
            blob = json.dumps(list(built), sort_keys=sort_keys, indent=1, default=_nonjson)
            digest.update(hashlib.sha256(blob.encode()).hexdigest().encode())
    return digest.hexdigest()


def test_build_pwml_ir_matches_the_pre_extraction_golden() -> None:
    """NEW ACCEPTANCE, and C-040's primary acceptance criterion.

    Asserts that NOTHING changed: build_pwml_ir's output is byte-identical to
    the pre-extraction sweep on every committed leg fixture. A non-empty diff
    here means the move was not pure.
    """
    mismatched = []
    for leg, expected in GOLDEN.items():
        path = ROOT / leg
        assert path.is_file(), f"committed leg fixture is missing: {leg}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = _leg_digest(payload)
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
