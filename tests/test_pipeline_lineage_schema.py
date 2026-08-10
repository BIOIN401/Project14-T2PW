"""C-015 schema tests for ``t2pw.pipeline.lineage``: the ``PRODUCT_CONTRACT`` § 3
categories, closed vocabularies, append-only history, loud rejection, order
independence -- the contract seven branches build on."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from t2pw.pipeline import lineage as L

_SOURCE = {"source_id": "PMC12444477", "source_type": "paper"}

#: origin -> (stage, support, paper_explicit) a real writer would pair it with.
_BY_ORIGIN = {
    "paper_stated": ("paper_extraction", "direct", "explicit"),
    "rag_literature": ("rag_admission", "direct", "not_explicit"),
    "database_grounded": ("identifier_mapping", "direct", "not_evaluated"),
    "inferred": ("deterministic_inference", "derived", "not_evaluated"),
    "audit_modified": ("audit_repair", "indirect", "not_evaluated"),
    "unresolved": ("gap_resolution", "unsupported", "not_evaluated"),
    "excluded": ("quarantine", "unsupported", "not_evaluated"),
}

_FACTS = {"stage", "origin", "support", "paper_explicit", "reason", "review_required",
          "uncertainty", "sources"}


def entry(**over) -> L.LineageEntry:
    base = dict(stage="paper_extraction", origin="paper_stated", support="direct",
                paper_explicit="explicit", reason="stated in Results",
                review_required=False, sources=[_SOURCE])
    base.update(over)
    return L.LineageEntry(**base)


def test_all_six_contract_categories_round_trip_with_all_six_facts() -> None:
    assert len(L.CONTRACT_CATEGORIES) == 6
    for category, origins in L.CONTRACT_CATEGORIES.items():
        for origin in origins:
            stage, support, explicit = _BY_ORIGIN[origin]
            before = L.Lineage((entry(stage=stage, origin=origin, support=support,
                                      paper_explicit=explicit),))
            rows = json.loads(json.dumps(before.as_list()))
            assert L.Lineage(rows) == before, category
            assert set(rows[0]) == _FACTS, category
            assert (rows[0]["stage"], rows[0]["origin"]) == (stage, origin)
            assert rows[0]["paper_explicit"] == explicit  # never collapsed to a bool


@pytest.mark.parametrize("over, field", [
    ({"stage": "Stage 1"}, "stage"), ({"stage": "extraction"}, "stage"),
    ({"origin": "rag"}, "origin"), ({"origin": ""}, "origin"),
    ({"support": "high"}, "support"), ({"paper_explicit": "unknown"}, "paper_explicit"),
    ({"paper_explicit": True}, "paper_explicit"), ({"reason": "   "}, "reason"),
    ({"review_required": 1}, "review_required"),
    ({"review_required": True, "uncertainty": ""}, "uncertainty"),
    ({"support": "direct", "sources": []}, "sources"),
    ({"support": "indirect", "sources": []}, "sources"),
    ({"sources": [{"source_type": "paper"}]}, "sources"),
    ({"sources": [{"pmid": "1"}]}, "sources"), ({"sources": "PMC1"}, "sources"),
])
def test_malformed_or_unknown_value_is_rejected_naming_the_field(over, field) -> None:
    with pytest.raises(L.LineageError) as err:
        entry(**over)
    assert str(err.value).startswith(field + ":"), str(err.value)


def test_missing_or_unknown_entry_field_is_rejected() -> None:
    good = entry().as_dict()
    with pytest.raises(L.LineageError, match="stage"):
        L.Lineage([{k: v for k, v in good.items() if k != "stage"}])
    with pytest.raises(L.LineageError, match="note"):
        L.Lineage([dict(good, note="x")])
    with pytest.raises(L.LineageError, match=L.LINEAGE_KEY):
        L.read({L.LINEAGE_KEY: "not-a-list"})


def test_append_preserves_every_earlier_attribution() -> None:
    row = {"name": "glycine"}
    first, second = entry(), entry(
        stage="identifier_mapping", origin="database_grounded",
        paper_explicit="not_evaluated", reason="PathBank exact identifier",
        sources=[{"source_id": "PW_C000123", "source_type": "pathbank"}])
    L.record(row, first)
    L.record(row, second)
    assert L.read(row).canonical() == (first, second)
    base = L.Lineage((first,))
    assert base.append(second).entries == (first, second)
    assert base.entries == (first,)  # append returned a new value, it did not mutate
    assert not [n for n in dir(L.Lineage) if n.startswith(("remove", "pop", "clear", "del"))]


def test_entry_order_never_changes_the_value_but_content_does() -> None:
    a = entry()
    b = entry(stage="quarantine", origin="excluded", support="unsupported",
              paper_explicit="not_evaluated", reason="no supporting evidence",
              review_required=True, uncertainty="removed from the core", sources=[])
    assert L.Lineage((a, b)) == L.Lineage((b, a))
    assert L.Lineage((a, b)).as_list() == L.Lineage((b, a)).as_list()
    assert hash(L.Lineage((a, b))) == hash(L.Lineage((b, a)))
    other = {"source_id": "PMC12898747", "source_type": "paper"}
    assert entry(sources=[_SOURCE, other]) == entry(sources=[other, _SOURCE])
    assert L.Lineage((a, b)) != L.Lineage((a,))
    assert L.Lineage((a,)) != L.Lineage((entry(reason="a different reason"),))


def test_vocabularies_are_closed_and_frozen() -> None:
    assert L.LINEAGE_KEY == "provenance_lineage"
    assert L.ORIGINS == ("paper_stated", "rag_literature", "database_grounded",
                         "inferred", "audit_modified", "unresolved", "excluded")
    assert L.SUPPORT_LEVELS == ("unsupported", "derived", "indirect", "direct")
    assert L.PAPER_EXPLICIT_VALUES == ("explicit", "not_explicit", "not_evaluated")
    assert set(L.ORIGINS) == {o for v in L.CONTRACT_CATEGORIES.values() for o in v}
    assert {s for s, _, _ in _BY_ORIGIN.values()} | {"normalization"} <= set(L.STAGES)
    assert len(set(L.STAGES)) == len(L.STAGES) and "export" not in L.STAGES


def test_typed_source_is_validated_exactly_like_a_mapping() -> None:
    # LineageSource is what seven producer cards import; it must not be a way around
    # the checks a mapping gets, or support="direct" names a source that does not exist.
    for build in (lambda **kw: L.LineageSource(**kw), lambda **kw: entry(sources=[kw])):
        with pytest.raises(L.LineageError, match=r"^sources: .*source_id or a uri"):
            build(source_type="uniprot")
        with pytest.raises(L.LineageError, match=r"^sources\.source_id: expected a str"):
            build(source_id=123)  # a LineageError naming the field, not a bare TypeError
    typed = entry(sources=[L.LineageSource(source_id=" PMC1 ", source_type="paper")])
    assert typed == entry(sources=[{"source_id": "PMC1", "source_type": "paper"}])
    row: dict = {}
    L.record(row, typed)
    assert L.read(row) == L.Lineage((typed,))  # what writes must read back, not raise


@pytest.mark.parametrize("origin", ["rag_literature", "database_grounded"])
def test_externally_added_origin_always_needs_a_source(origin) -> None:
    assert L.SOURCED_ORIGINS == ("rag_literature", "database_grounded")
    kw = dict(origin=origin, support="unsupported", paper_explicit="not_evaluated")
    with pytest.raises(L.LineageError, match="^sources: origin=" + repr(origin)):
        entry(sources=[], **kw)        # § 3 binds it to a source at EVERY support level
    assert entry(sources=[_SOURCE], **kw).sources
    assert entry(origin="unresolved", support="unsupported", sources=[],  # a failed
                 paper_explicit="not_evaluated").origin  # lookup, not a sourceless claim


def test_module_imports_no_stage_module() -> None:
    tree = ast.parse(Path(L.__file__).read_text(encoding="utf-8"))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not [m for m in mods if m.split(".")[0] == "t2pw"], sorted(mods)
