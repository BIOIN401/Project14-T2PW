"""A process row's lineage record must survive ``_clean_processes``'s rebuild.

Why this file exists
--------------------
``pipeline._clean_processes`` rebuilds every process row from a key whitelist.
``_carry_rag_provenance`` already copies the three namespaced RAG pointers onto the
rebuilt row, but nothing copied the lineage record delivered by
``t2pw.pipeline.lineage`` -- so it was silently dropped by the rebuild.

Silent is the operative word. ``lineage.LineageEntry`` has **no entity or reaction id
field**: attribution is POSITIONAL, it means "the row this record is stored on". A step
that rebuilds a row without carrying unknown keys therefore does not produce a row with
missing provenance, it produces a row that now *asserts* provenance it does not have --
"imported from PMC12898747 at rag_admission, review required" silently becomes "the seed
paper stated this". ``_clean_entities`` copies entity rows key-for-key so entity lineage
already survived; the four process buckets rebuilt here did not.

What is pinned here
-------------------
1. A lineage record on a reaction / transport / reaction-coupled transport /
   interaction row survives ``_clean_processes`` and ``clean_stage_one``.
2. It survives **verbatim**. A carrier carries; it does not validate, normalize,
   re-order or repair. The record that arrives is the record that leaves -- same list
   object, same entry order, even when that order is not canonical and even when the
   record is one ``lineage.read`` would reject. Re-canonicalizing or rejecting in
   transit would be exporter-repair in a new place.
3. The three existing RAG carriers are untouched: same keys, same values, same key
   ORDER on the rebuilt row, and ``evidence`` is still flattened to a string and still
   not a carrier key.
4. Seam S5 holds: ``pipeline.py`` still imports nothing from ``t2pw.rag``.

Offline / deterministic: no chromadb, no network, no live LLM. The ``openai`` stub
mirrors tests/test_pipeline_reaction_rag_provenance.py because importing the pipeline
pulls the mapping/LLM stack at import time.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *_: object, **__: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **__: None)
            )

    openai_stub.OpenAI = _OpenAI
    openai_stub.RateLimitError = RuntimeError
    openai_stub.APIError = RuntimeError
    openai_stub.APITimeoutError = RuntimeError
    openai_stub.AuthenticationError = RuntimeError
    openai_stub.BadRequestError = RuntimeError
    sys.modules["openai"] = openai_stub

from t2pw.pipeline import lineage  # noqa: E402
from t2pw.pipeline.pipeline import (  # noqa: E402
    _RAG_ROW_CARRIER_KEYS,
    _clean_processes,
    clean_stage_one,
)

LINEAGE_KEY = lineage.LINEAGE_KEY

# The pointer shape the reference run's imported rows carry (see
# tests/test_pipeline_reaction_rag_provenance.py for the measurement).
_IMPORTED_POINTER = {
    "source_id": "PMC12898747",
    "source_title": "Lipid A modification systems in Gram-negative bacteria",
    "source_type": "paper",
    "source_uri": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12898747/",
    "section": "results",
    "chunk_id": "PMC12898747::results::0007",
}

_ADMISSION_ENTRY = lineage.LineageEntry(
    stage="rag_admission",
    origin="rag_literature",
    support="direct",
    paper_explicit="not_explicit",
    reason="admitted from a retrieved passage to fill a pathway gap",
    review_required=True,
    uncertainty="the seed paper never states this step",
    sources=(
        lineage.LineageSource(
            source_id="PMC12898747",
            source_type="paper",
            uri=_IMPORTED_POINTER["source_uri"],
            locator="results::0007",
        ),
    ),
)


def _with_lineage(row: dict) -> dict:
    """``row`` with a real, validated lineage record written by ``lineage.record``."""
    lineage.record(row, _ADMISSION_ENTRY)
    return row


def _reaction(**overrides: object) -> dict:
    row = {
        "name": "lipid IVA palmitoylation",
        "inputs": ["lipid IVA", "palmitoyl-ACP"],
        "outputs": ["hepta-acylated lipid A", "ACP"],
        "evidence": "PagP transfers a palmitate to lipid A.",
    }
    row.update(overrides)  # type: ignore[arg-type]
    return row


def _rag_reaction() -> dict:
    """A reaction carrying all three namespaced RAG pointers and no lineage."""
    return _reaction(
        rag_provenance=dict(_IMPORTED_POINTER),
        source_papers=[{"source_id": "PMC12898747", "title": "Lipid A modification systems"}],
        rag_confidence=0.82,
    )


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _all_four_buckets() -> dict:
    """One row per process bucket, every one carrying a lineage record."""
    return {
        "reactions": [_with_lineage(_reaction())],
        "transports": [
            _with_lineage(
                {"name": "lipid A flipping", "cargo": "lipid A", "transporters": [{"protein": "MsbA"}]}
            )
        ],
        "reaction_coupled_transports": [
            _with_lineage(
                {"name": "coupled export", "reaction": "ATP hydrolysis", "transport": "lipid A flipping"}
            )
        ],
        "interactions": [
            _with_lineage({"entity_1": "PhoP", "relationship": "activates", "entity_2": "pmrD"})
        ],
    }


# ---------------------------------------------------------------------------
# 1 — G9 REGRESSION: the record was dropped by the rebuild.
# ---------------------------------------------------------------------------
def test_reaction_lineage_record_survives_the_whitelist_rebuild():
    """G9 regression. On the base SHA the rebuilt reaction row has no lineage at all."""
    processes = {"reactions": [_with_lineage(_reaction())]}
    expected = _canonical(processes["reactions"][0][LINEAGE_KEY])

    cleaned = _clean_processes(copy.deepcopy(processes))
    row = cleaned["reactions"][0]

    assert LINEAGE_KEY in row, (
        f"the whitelist rebuild dropped {LINEAGE_KEY!r}; the row now silently reads as "
        f"seed-paper content. Rebuilt keys: {sorted(row)}"
    )
    assert _canonical(row[LINEAGE_KEY]) == expected
    # The attribution itself, not merely the key, arrived intact.
    entry = lineage.read(row).canonical()[0]
    assert entry.stage == "rag_admission"
    assert entry.review_required is True
    assert entry.sources[0].source_id == "PMC12898747"


def test_every_process_bucket_keeps_its_lineage_record():
    """G9 regression. All four buckets are rebuilt from a whitelist, all four lost it."""
    source = _all_four_buckets()
    expected = {
        bucket: _canonical(rows[0][LINEAGE_KEY]) for bucket, rows in source.items()
    }

    cleaned = _clean_processes(copy.deepcopy(source))

    for bucket, want in expected.items():
        row = cleaned[bucket][0]
        assert LINEAGE_KEY in row, f"{bucket}: rebuild dropped {LINEAGE_KEY!r} ({sorted(row)})"
        assert _canonical(row[LINEAGE_KEY]) == want, bucket


def test_the_carrier_copies_the_record_verbatim_and_never_rewrites_it():
    """G9 regression + the carry-never-edit invariant.

    The stored order is deliberately NOT canonical and one entry is deliberately
    unparseable. A carrier that round-tripped the value through ``lineage.Lineage``
    would re-sort the first and raise on the second; both are rewrites of another
    stage's attribution in transit, which is the exporter-repair shape in a new place.
    """
    later = {
        "stage": "audit_repair",
        "origin": "audit_modified",
        "support": "derived",
        "paper_explicit": "not_evaluated",
        "reason": "reconnected an orphaned product",
        "review_required": False,
        "uncertainty": "",
        "sources": [],
    }
    earlier = {
        "stage": "paper_extraction",
        "origin": "paper_stated",
        "support": "direct",
        "paper_explicit": "explicit",
        "reason": "stated in the results section",
        "review_required": False,
        "uncertainty": "",
        "sources": [{"source_id": "PMC12444477", "source_type": "paper", "uri": "", "locator": ""}],
    }
    unparseable = {"stage": "no-such-stage", "why": "a future writer's field"}
    stored = [later, earlier, unparseable]

    processes = {"reactions": [_reaction(**{LINEAGE_KEY: stored})]}
    cleaned = _clean_processes(processes)
    carried = cleaned["reactions"][0][LINEAGE_KEY]

    # Same object: copied by reference, exactly as the three RAG keys are.
    assert carried is stored
    # Same entries, same order -- not canonicalized, not sorted, not deduped.
    assert carried == [later, earlier, unparseable]


# ---------------------------------------------------------------------------
# 2 — NEW ACCEPTANCE: the carrier's key list.
# ---------------------------------------------------------------------------
def test_the_carrier_key_list_is_the_rag_keys_plus_lineage_and_nothing_else():
    """New acceptance. ``_ROW_CARRIER_KEYS`` does not exist on the base SHA."""
    from t2pw.pipeline.pipeline import _ROW_CARRIER_KEYS

    assert _ROW_CARRIER_KEYS == _RAG_ROW_CARRIER_KEYS + (LINEAGE_KEY,)
    # The lineage key is appended, never folded into the RAG tuple: that tuple's
    # documented identity is ``RAG_ADDITIVE_KEYS`` minus ``evidence`` (F-016).
    assert LINEAGE_KEY not in _RAG_ROW_CARRIER_KEYS
    # Neither exclusion was reversed to make lineage work.
    assert "evidence" not in _ROW_CARRIER_KEYS
    assert "source_refs" not in _ROW_CARRIER_KEYS


def test_clean_stage_one_carries_lineage_on_processes_and_entities():
    """G9 regression. The other ``_clean_processes`` door, plus the entity side.

    A chunked run re-cleans the merged payload through ``clean_stage_one``, so both
    doors have to carry the record or a chunked run drops it again. On the base SHA
    the entity keeps its record and the reaction does not, which is the asymmetry.
    """
    entity = _with_lineage({"name": "lipid IVA", "rag_provenance": dict(_IMPORTED_POINTER)})
    cleaned = clean_stage_one(
        {"entities": {"compounds": [entity]}, "processes": {"reactions": [_with_lineage(_reaction())]}}
    )

    reaction = cleaned["processes"]["reactions"][0]
    compound = cleaned["entities"]["compounds"][0]
    # Process rows and entity rows now agree: both keep the record, in the same shape.
    assert _canonical(reaction[LINEAGE_KEY]) == _canonical(compound[LINEAGE_KEY])
    assert lineage.read(reaction).canonical() == lineage.read(compound).canonical()


def test_lineage_is_appended_after_the_rag_carriers_and_changes_nothing_else():
    """G9 regression. The rebuilt row gains the record, and gains only the record."""
    rag_only = _rag_reaction()
    both = _with_lineage(_rag_reaction())

    cleaned = _clean_processes({"reactions": [rag_only, both]})
    rag_row, both_row = cleaned["reactions"]

    # Exactly one new key, strictly last, so a RAG row's byte order is preserved.
    assert list(both_row) == list(rag_row) + [LINEAGE_KEY]
    for key in _RAG_ROW_CARRIER_KEYS:
        assert both_row[key] == rag_row[key] == rag_only[key]


# ---------------------------------------------------------------------------
# 3 — PRESERVATION: unchanged on both sides of this patch.
# ---------------------------------------------------------------------------
def test_the_three_rag_carriers_are_unchanged():
    """Preservation. A RAG row with no lineage is what it always was, key order included."""
    rag_only = _rag_reaction()
    row = _clean_processes({"reactions": [copy.deepcopy(rag_only)]})["reactions"][0]

    assert list(row) == [
        "inputs", "outputs", "name", "evidence",
        "rag_provenance", "source_papers", "rag_confidence",
    ]
    assert LINEAGE_KEY not in row
    for key in _RAG_ROW_CARRIER_KEYS:
        assert row[key] == rag_only[key]


def test_a_row_with_no_carrier_still_gains_no_key():
    """Preservation. The emptiness rule is the one the RAG carriers already used."""
    cleaned = _clean_processes(
        {
            "reactions": [
                _reaction(name="seed reaction"),
                _reaction(name="empty lineage", **{LINEAGE_KEY: []}),
            ]
        }
    )
    seed_row, empty_row = cleaned["reactions"]

    assert list(seed_row) == ["inputs", "outputs", "name", "evidence"]
    # An empty record carries no attribution -- ``lineage.read`` cannot tell it from an
    # absent one -- so it is skipped like an empty RAG pointer rather than materialized.
    assert list(empty_row) == ["inputs", "outputs", "name", "evidence"]
    assert lineage.read(empty_row).entries == ()


def test_evidence_is_still_flattened_to_a_string_and_is_not_a_carrier_key():
    """Preservation of F-016: the ``evidence`` exclusion was not reversed.

    ``evidence`` is excluded from the carrier on purpose -- the process cleaners
    flatten it through ``_evidence_text`` because ``ReactionModel.evidence`` is typed
    ``str``. Re-emitting the record list would hand every downstream consumer back the
    list shape that seam exists to remove. Widening the carrier for lineage must not
    smuggle that reversal in.
    """
    row = _with_lineage(
        _reaction(
            evidence=[{"text": "PagP transfers a palmitate.", "source_id": "PMC12898747"}],
            rag_provenance=dict(_IMPORTED_POINTER),
        )
    )
    cleaned = _clean_processes({"reactions": [row]})["reactions"][0]

    assert isinstance(cleaned["evidence"], str)
    assert "PagP transfers a palmitate." in cleaned["evidence"]
    # The list shape the process cleaners exist to remove did not come back.
    assert not isinstance(cleaned["evidence"], list)


def test_pipeline_module_still_imports_nothing_from_the_rag_package():
    """Preservation of seam S5. Asserted on the parsed module, not on a grep."""
    tree = ast.parse((SRC / "t2pw" / "pipeline" / "pipeline.py").read_text(encoding="utf-8"))

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert not [name for name in imported if name.split(".")[:2] == ["t2pw", "rag"]], imported
