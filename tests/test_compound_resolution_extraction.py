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
#: order. MOVED ONCE, DELIBERATELY, by C-045a on 2026-08-15 under permanent merge
#: rule 4 -- not a behavioural correction. D-016 (LOCKED) put the species
#: canonicalization before the freeze, so a *standalone* build_pwml_ir -- what
#: this pins, and no longer the production path -- cannot know a species was only
#: deterministically normalized and stops publishing report["preflight"]["species"]
#: and its collision warning. The measured delta, the negative that nothing under
#: ``ir`` moved, and the proof that the production path preserves the preflight
#: unchanged are evidence/c045a_{standalone,production}_delta.json; every digest
#: below regenerates via evidence/c045a_golden_rebaseline.py --mode digest.
GOLDEN = {
    "runs/2026-07-27_1623/papers/PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/strict/final_mapped.json": "55d6c3940a715d5b708099916270311805928d68a7a0fc5f6cbb8b81b29dc7ab",
    "runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict/final_mapped.json": "43a165d24f4d743222ff6a53822e25e33ee8fb38e2c89cda6caed907df10cf04",
    "runs/2026-07-28_0919/papers/PMC13278307__an-overview-of-mobile-colistin-resistance-mcr-g/strict/final_mapped.json": "02d1e354daabb6600afa1fad88cf3bb870ee9ac64d9c717194f9fb7f067bc362",
    "runs/2026-08-02_2130/papers/PMC12096016/research/final_mapped.json": "ad84f2f4b9d87a9c8c9a42b2132f50c3a239047ecac5196c18436c289efab73b",
    "runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json": "2cbd51157a3106f140ed9a91eda2910895853b50bac958c37e1345bf11ca3d05",
    "runs/2026-08-02_2130/papers/PMC12180156/research/final_mapped.json": "4cb0fef1571c3262825879571b7e754fcf95e6c1b0581de93445b7d69f621311",
    "runs/2026-08-02_2130/papers/PMC12180156/strict/final_mapped.json": "ad160d65c2d1303ba58b1b300948030931e466127a85f6e078a68671953ce6f3",
    "runs/2026-08-02_2130/papers/PMC12444477/research/final_mapped.json": "e2add078c442e2a31fcc515944651b12773a1acbabfc4e771ef6ac4e8e596cf1",
    "runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json": "bd200e361a5b60bbc34ae59b25f1bc3a9e34bd637bec74a23cc983b4dfdb30e8",
    "runs/2026-08-02_2130/papers/PMC12452463/research/final_mapped.json": "0fab3c5364a0072e3973c002a319f251ecd338b16a477b43179f5d546d34531e",
    "runs/2026-08-02_2130/papers/PMC12782028/strict/final_mapped.json": "5d95036ca6783f93bfa590d9ac9b0621c88b327881f3be0ba611c13e9bd4ccf3",
    "runs/2026-08-02_2130/papers/PMC12856317/research/final_mapped.json": "ca0c4b43723f785a1a7830d914056167f6c53ddc09ca4ebee5db47793b4b4958",
    "runs/2026-08-02_2130/papers/PMC12856317/strict/final_mapped.json": "43bd1e6048daf42a9a67126bc7a2556f38ab03b9f53b465bc582537955856d03",
    "runs/2026-08-02_2130/papers/PMC13231680/strict/final_mapped.json": "91e237161790b6f73745e8aaedfb3eb99d80b70306739e2c970c5386d4e65e85",
    "runs_verify/2026-08-04_1148/papers/PMC13231680/research/final_mapped.json": "c76a509af1ecd8d3229c64bbb961a0c0b1cdfe9d5c525731a02f6f1d74e6b984",
    "runs_verify/2026-08-04_1207/papers/PMC12452463/strict/final_mapped.json": "a6a79fbeb4f18fc44050de8a1a7578379193cebb8fef9f45bafc7dce594ae45b",
    "runs_verify/2026-08-04_1234/papers/PMC12096016/strict/final_mapped.json": "0f7f68682e48b7d28bdf308d4f75f8b74483231d476755884a6a0230be228675",
    "runs_verify/2026-08-04_1234/papers/PMC12856317/strict/final_mapped.json": "5485f752a7eddfdcfb32b402be39ef22dfa107d0d2213ede5b92aded436d4218",
    "runs_verify/2026-08-04_1306/papers/PMC12096016/research/final_mapped.json": "b3b2b2b79c3292a5d0da3947904eb3e266d2b1d6a016c0a30084dd45d87bd804",
    "runs_verify/2026-08-04_1306/papers/PMC12452463/research/final_mapped.json": "b48b2f35097af8d244b4f2a31b53420533981e54534f6153564f70c2791cafaa",
    "runs_verify/2026-08-04_1358/papers/PMC12096016/research/final_mapped.json": "9714cf38774406a7cfacad89e7bbdf3eb7778a948b2927840cf9069f945d85a6",
    "runs_verify/2026-08-04_1504/papers/PMC12856317/strict/final_mapped.json": "da0a7b2a805bbf162b148399efe17e1f1841fdf5e61d3a56d76247c1af171f9f",
    "runs_verify/2026-08-04_1647/papers/PMC12856317/strict/final_mapped.json": "49eb933dd26f086ed831497c012e4538952ab900d37a2697eb85b67e641985fc",
    "runs_verify/2026-08-04_1754/papers/PMC12096016/research/final_mapped.json": "22fb2b6c25a14816acc3b4aea96a30d412a7e63119683966c4dad0a2f50b4fb7",
    "runs_verify/2026-08-04_1754/papers/PMC12096016/strict/final_mapped.json": "f4b738e66651ba680a57683976bd607406699fbc355a620e45163441effd6e4d",
    "runs_verify/2026-08-04_1754/papers/PMC12180156/research/final_mapped.json": "fa7c19d58fa908e7289971434ac64469454951967956d55bc6e5bb323d7b11e8",
    "runs_verify/2026-08-04_1754/papers/PMC12180156/strict/final_mapped.json": "15e417a78b8ec03fbb193a47e7d67b435cb4aadbcca37e79d627065efe226c81",
    "runs_verify/2026-08-04_1754/papers/PMC12452463/research/final_mapped.json": "99b2c87451690db27fb4c75315970eacc4bc4164edb21cc86b60517da077cf04",
    "runs_verify/2026-08-04_1754/papers/PMC12452463/strict/final_mapped.json": "e1c787a38e59e81ec70434e9a79ea84f280c05d99d136db1b3c04848c174e5d2",
    "runs_verify/2026-08-04_1754/papers/PMC12782028/research/final_mapped.json": "6cd0da47635552261daf6fc8bb46768be04a93a59326f90513d1022363d71df5",
    "runs_verify/2026-08-04_1754/papers/PMC12856317/research/final_mapped.json": "443fe9f59b19935747ff45baec488bbbef29fea470d883ea33a186406739384f",
    "runs_verify/2026-08-04_1754/papers/PMC12856317/strict/final_mapped.json": "ae853353b520df0b57a243270477d7904953c0b3e3387dd8343963a8454a1651",
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
