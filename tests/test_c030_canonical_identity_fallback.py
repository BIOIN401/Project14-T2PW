"""A0-C1: the graph hash covers the identity fallback the EXPORTER consumes.

``ir._first_nonempty`` settles an entity's exported identity over four tiers --
the record, ``mapping_meta``, ``mapped_ids``, then the FIRST
``mapping_meta.candidates`` entry. ``graph_projection`` reached tiers 1 and 3
only, so on committed artifacts an identifier that decides what gets exported
could be rewritten without moving ``canonical_graph_sha256`` -- the digest
exporters bind to. This file measures those rows and proves all three directions:
the consumed value moves the hash, ranking/transient noise does not, and a
reorder moves it exactly when it moves WHICH value is consumed.

``ir.py`` is used here strictly as a READ-ONLY ORACLE. Nothing imported from it
is modified, and the projection deliberately does not import it: exporters bind
to this hash, so what the hash covers may not be defined by the exporter it is
checked against.

Labels, each verified by running THIS file against base e4eeef4's sources
(24 failed, 6 passed there; 30 passed at the tip):
  G9 REGRESSION  -- red on base on a VALUE the pipeline computed, never on an
                    import: ``test_every_committed_gap_row_is_now_covered``
                    (19 ids, on the graph hash),
                    ``test_mutating_the_consumed_fallback_moves_the_graph_hash``
                    and ``test_a_reorder_that_changes_the_consumed_value_moves_the_hash``
                    (on the graph hash), and
                    ``test_the_published_graph_hash_is_what_the_verdict_checks``
                    (on the verdict REASON: base reports
                    ``canonical_payload_mismatch_graph_intact`` for an edit that
                    moved the exported identity).
  NEW ACCEPTANCE -- everything else. Two of them are also red on base --
                    ``test_the_allowlist_names_every_identity_key_ir_consumes``
                    on allowlist membership, and
                    ``test_the_seam_resolves_its_hash_schema_import_when_exec_d_alone``
                    on a missing statement. Neither is offered as a G9 proof:
                    a symbol that is absent is not behaviour.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in (ROOT / "src",) if str(p) not in sys.path]

from t2pw.pipeline import canonical_hash as ch  # noqa: E402
from t2pw.pipeline import gate_reports as gr  # noqa: E402
from t2pw.pwml import ir  # noqa: E402

APP_REL = "src/t2pw/app/streamlit_app.py"
#: The worked counterexample: ``isochorismate`` carries no identifier of its own
#: and no ``mapped_ids``, so the 40741 that is exported comes from candidate 0.
COUNTEREXAMPLE = ROOT / "runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json"
COMPOUND = "isochorismate"
COMPOUND_KEYS = ["pathbank_compound_id", "pw_compound_id", "pathwhiz_id"]

#: Every ordered identity key list ``ir.py`` resolves, by the bucket it is applied
#: to. ``components`` is the nested list at ``ir.py:1204`` / ``:1212``.
BUCKET_KEYS: dict[str, tuple[tuple[str, ...], ...]] = {
    "cell_types": (("pathbank_cell_type_id", "pw_cell_type_id", "pathwhiz_id"),),
    "compounds": (tuple(COMPOUND_KEYS), ("hmdb_id", "hmdb"), ("kegg_id", "kegg"),
                  ("pubchem_cid", "pubchem"), ("pwc_id",)),
    "element_collections": (("pathbank_element_collection_id",
                             "pw_element_collection_id", "pathwhiz_id"),),
    "nucleic_acids": (("pathbank_nucleic_acid_id", "pw_nucleic_acid_id", "pathwhiz_id"),),
    "protein_complexes": (("pathbank_complex_id", "pathbank_protein_complex_id",
                           "pw_complex_id", "pathwhiz_id"),),
    "proteins": (("pathbank_protein_id", "pw_protein_id", "pathwhiz_id"),
                 ("uniprot", "uniprot_id", "uniprot-id"),
                 ("drugbank", "drugbank_id", "drugbank-id")),
    "species": (("pathbank_species_id", "pw_species_id", "pathwhiz_id"),),
    "subcellular_locations": (("pathbank_subcellular_location_id",
                               "pw_subcellular_location_id", "pathwhiz_id"),),
    "tissues": (("pathbank_tissue_id", "pw_tissue_id", "pathwhiz_id"),),
    "components": (("pathbank_protein_id", "pw_protein_id", "pathwhiz_id", "protein_id"),
                   ("uniprot", "uniprot_id", "uniprot-id")),
}


def _corpus() -> list[str]:
    listed = subprocess.run(["git", "ls-files", "*final_mapped.json"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return sorted(listed.stdout.split())


def _rows(payload: Any) -> Any:
    """Every (bucket, keys, row) ``ir.py`` resolves an identity for."""
    entities = payload.get("entities") if isinstance(payload, dict) else None
    for bucket, key_lists in BUCKET_KEYS.items():
        if bucket == "components":
            continue
        for row in (entities or {}).get(bucket) or []:
            if isinstance(row, dict):
                yield bucket, key_lists, row
                for component in (row.get("components") or []) if bucket == \
                        "protein_complexes" else []:
                    if isinstance(component, dict):
                        yield "components", BUCKET_KEYS["components"], component


def _blind(row: dict) -> dict:
    """``row`` as the OLD projection saw it: tiers 1 and 3, never ``mapping_meta``.

    Dropping the one container is exact because every identity key is in
    :data:`GRAPH_FIELDS` -- asserted by
    :func:`test_the_allowlist_names_every_identity_key_ir_consumes`, without which
    a tier-1 hit on an unnamed key would be counted visible when it is not.
    """
    return {key: value for key, value in row.items() if key != "mapping_meta"}


def _slot(row: dict, keys: tuple[str, ...]) -> tuple[Any, str]:
    """The (container, key) the ladder consumes, for a row that lost tiers 1 and 3.
    ``mapping_meta`` before candidate 0, because a whole tier is scanned first."""
    meta = row.get("mapping_meta") or {}
    candidates = meta.get("candidates") or []
    first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    for container in (meta, first):
        for key in keys:
            if container.get(key) not in (None, ""):
                return container, key
    raise AssertionError("a gap row with no fallback slot")


@lru_cache(maxsize=None)
def _gap_rows() -> tuple[tuple[str, str, int, str], ...]:
    """THE CENSUS. One entry per committed row whose EXPORTED identity comes from
    a container the projection could not see: (file, bucket, row index, key)."""
    found = []
    for relative in _corpus():
        payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for index, (bucket, key_lists, row) in enumerate(_rows(payload)):
            for keys in key_lists:
                consumed = ir._first_nonempty(row, list(keys))
                if consumed in (None, "") or consumed == ir._first_nonempty(
                        _blind(row), list(keys)):
                    continue
                found.append((relative, bucket, index, _slot(row, keys)[1]))
    return tuple(found)


def _mutated(value: Any) -> Any:
    return value + 1 if isinstance(value, int) and not isinstance(value, bool) \
        else f"{value}-c030"


def _counterexample() -> tuple[dict, dict]:
    payload = json.loads(COUNTEREXAMPLE.read_text(encoding="utf-8"))
    row = next(c for c in payload["entities"]["compounds"] if c.get("name") == COMPOUND)
    return payload, row


def _edit(edit) -> tuple[dict, dict, dict]:
    """``(base payload, moved payload, moved row)`` -- ``edit`` gets the moved row."""
    base, _ = _counterexample()
    moved, row = deepcopy(base), None
    row = next(c for c in moved["entities"]["compounds"] if c.get("name") == COMPOUND)
    edit(row)
    return base, moved, row


# ── the census ──────────────────────────────────────────────────────────────


def test_the_census_reproduces_over_the_committed_corpus() -> None:
    """NEW ACCEPTANCE. Pins the measurement A0-C1's acceptance is scoped to.
    The ledger's "60" does not reproduce; 49 is what the corpus holds."""
    gaps = _gap_rows()
    assert len(_corpus()) == 32
    assert len(gaps) == 49
    assert len({relative for relative, *_ in gaps}) == 19
    buckets: dict[str, int] = {}
    keys: dict[str, int] = {}
    for _relative, bucket, _index, key in gaps:
        buckets[bucket] = buckets.get(bucket, 0) + 1
        keys[key] = keys.get(key, 0) + 1
    assert buckets == {"compounds": 38, "protein_complexes": 11}
    assert keys == {"pathbank_compound_id": 38, "pathbank_complex_id": 11}


def test_the_allowlist_names_every_identity_key_ir_consumes() -> None:
    """NEW ACCEPTANCE (also red on base, on allowlist membership -- not a G9
    proof). GRAPH_FIELDS is the ONE gate on what reaches the graph hash, so an
    identity key it does not name is an identifier the exporter can change
    silently -- and it also makes :func:`_blind` an exact oracle."""
    every = {key for lists in BUCKET_KEYS.values() for keys in lists for key in keys}
    assert every <= ch.GRAPH_FIELDS, sorted(every - ch.GRAPH_FIELDS)


@pytest.mark.parametrize("relative", sorted({g[0] for g in _gap_rows()}))
def test_every_committed_gap_row_is_now_covered(relative: str) -> None:
    """G9 REGRESSION. For EVERY census row, rewriting the identifier the exporter
    consumes moves the graph hash. On base e4eeef4 not one of the 49 moves it."""
    payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    before = ch.canonical_graph_sha256(payload)
    covered = 0
    for _file, bucket, index, key in [g for g in _gap_rows() if g[0] == relative]:
        moved = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        _bucket, key_lists, row = list(_rows(moved))[index]
        assert _bucket == bucket
        keys = next(k for k in key_lists if key in k)
        container, slot = _slot(row, keys)
        container[slot] = _mutated(container[slot])
        # the slot really is the consumed one, and the export really moved
        assert ir._first_nonempty(row, list(keys)) == container[slot]
        assert ch.canonical_graph_sha256(moved) != before
        covered += 1
    assert covered


# ── the three directions, on the committed counterexample ───────────────────


def test_mutating_the_consumed_fallback_moves_the_graph_hash() -> None:
    """G9 REGRESSION. 40741 -> 40742 at candidate 0 changes what ``ir.py`` exports
    for ``isochorismate``. On base the graph hash does not move; here it must."""
    base, moved, row = _edit(
        lambda r: r["mapping_meta"]["candidates"][0].update(pathbank_compound_id=40742))
    original = next(c for c in base["entities"]["compounds"] if c.get("name") == COMPOUND)
    assert all(original.get(k) is None for k in COMPOUND_KEYS)
    assert "mapped_ids" not in original
    # the EXPORTED identity moved with it, at ir.py's own record builder
    assert ir._db_id(original, COMPOUND_KEYS) == 40741
    assert ir._db_id(row, COMPOUND_KEYS) == 40742
    exported = ir._entity_record(row, "cmp1", COMPOUND_KEYS, "pathbank_compound_id")
    assert exported["pathwhiz_id"] == exported["pathbank_compound_id"] == 40742
    assert ch.canonical_graph_sha256(moved) != ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) != ch.canonical_payload_sha256(base)


def test_a_reorder_that_changes_the_consumed_value_moves_the_hash() -> None:
    """G9 REGRESSION. A0-C1's second half. Promoting candidate 1 changes which
    identifier is exported, so the hash MUST follow it."""
    base, moved, row = _edit(
        lambda r: r["mapping_meta"]["candidates"].insert(
            0, r["mapping_meta"]["candidates"].pop(1)))
    assert ir._db_id(row, COMPOUND_KEYS) != 40741
    assert ch.canonical_graph_sha256(moved) != ch.canonical_graph_sha256(base)


def test_a_score_edit_is_ranking_noise_and_never_moves_the_graph_hash() -> None:
    """NEW ACCEPTANCE. The other half of A0-C1: ranking is not biology."""
    base, moved, row = _edit(
        lambda r: r["mapping_meta"]["candidates"][0].update(score=0.123456))
    assert ir._db_id(row, COMPOUND_KEYS) == 40741
    assert ch.canonical_graph_sha256(moved) == ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) != ch.canonical_payload_sha256(base)


@pytest.mark.parametrize("case", ["reorder_the_tail", "drop_the_tail", "append_a_candidate"])
def test_ranking_that_leaves_the_consumed_value_in_front_never_moves_the_hash(
        case: str) -> None:
    """NEW ACCEPTANCE. The candidate LIST, its ORDER, its LENGTH and its other
    entries stay out: only the one value ``ir.py`` consumes is hashed."""
    edits = {
        "reorder_the_tail": lambda c: c.__setitem__(slice(1, None), list(reversed(c[1:]))),
        "drop_the_tail": lambda c: c.__setitem__(slice(1, None), []),
        "append_a_candidate": lambda c: c.append({"pathbank_compound_id": 999999,
                                                  "score": 0.01}),
    }
    base, moved, row = _edit(lambda r: edits[case](r["mapping_meta"]["candidates"]))
    assert ir._db_id(row, COMPOUND_KEYS) == 40741
    assert ch.canonical_graph_sha256(moved) == ch.canonical_graph_sha256(base)


def test_provenance_beside_the_consumed_value_stays_out_of_the_graph_hash() -> None:
    """NEW ACCEPTANCE. Transient metadata sitting in the same container as a value
    that IS hashed must not be dragged in with it."""
    base, moved, _row = _edit(lambda r: r["mapping_meta"].update(
        resolution={"status": "novel"}, chosen_rule="rewritten", confidence=0.1))
    assert ch.canonical_graph_sha256(moved) == ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) != ch.canonical_payload_sha256(base)


# ── the seam publishes the hashes, and they are load-bearing ────────────────


def _seam() -> Any:
    """``freeze_canonical_payload`` alone, the way the AST harnesses load it."""
    source = (ROOT / APP_REL).read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "freeze_canonical_payload")
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, APP_REL, "exec"), namespace)  # noqa: S102
    return namespace["freeze_canonical_payload"]


def test_the_seam_resolves_its_hash_schema_import_when_exec_d_alone() -> None:
    """NEW ACCEPTANCE (also red on base, on a missing statement -- not a G9
    proof). The import is function-local precisely so the harnesses, which exec
    this FunctionDef with a hand-built globals dict, can still run it."""
    assert _seam() is not None
    body = ast.parse((ROOT / APP_REL).read_text(encoding="utf-8"))
    node = next(n for n in body.body if isinstance(n, ast.FunctionDef)
                and n.name == "freeze_canonical_payload")
    imports = [n for n in ast.walk(node) if isinstance(n, ast.ImportFrom)]
    assert [(n.module, [a.name for a in n.names]) for n in imports] == [
        ("t2pw.pipeline.canonical_hash", ["HASH_SCHEMA_VERSION"])]


def test_the_published_graph_hash_is_what_the_verdict_checks() -> None:
    """G9 REGRESSION, on the verdict REASON. The wiring is only worth anything if
    the consumer reads it: an edit that moved the exported identity must be caught
    as a GRAPH mismatch, and an evidence-only edit as the payload mismatch that
    leaves the graph intact. On base the first is misreported as the SECOND --
    "the biology is intact, only the evidence moved" -- about a payload whose
    exported compound identity had changed."""
    payload, _row = _counterexample()
    report = gr.stamp_report({"stage": "final", "ok": True, "errors": []},
                             phase=gr.PHASE_FINAL_PRE_EXPORT,
                             payload=payload, payload_hash=gr.payload_sha256(payload),
                             hash_schema=ch.HASH_SCHEMA_VERSION)

    def verdict(candidate: dict) -> gr.GateVerdict:
        return gr.gate_verdict(gr.stamp_artifact_set({
            gr.FINAL_GATE_REPORT_KEY: report, gr.CANONICAL_PAYLOAD_KEY: candidate}))

    assert verdict(payload).failed is False
    _base, biology, _r = _edit(
        lambda r: r["mapping_meta"]["candidates"][0].update(pathbank_compound_id=40742))
    _base, evidence, _r = _edit(lambda r: r.update(lineage=[{"stage": "rag"}]))
    assert verdict(biology).reason == gr.REASON_CANONICAL_GRAPH_MISMATCH
    assert verdict(evidence).reason == gr.REASON_CANONICAL_PAYLOAD_MISMATCH_GRAPH_INTACT
