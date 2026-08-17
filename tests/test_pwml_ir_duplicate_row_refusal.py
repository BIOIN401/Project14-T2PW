"""C-050i acceptance -- the post-freeze row dedupe fails closed (F-039).

**EVERY TEST IN THIS FILE IS AN EXPLICITLY LABELLED NEW ACCEPTANCE TEST.**

Under permanent merge rule **G9** this is a *genuinely new capability*, not a
correction of pre-existing observable behaviour: ``ir._dedupe_named_rows`` gains a
refusal it never had. **No base-SHA behavioural failure is claimed and none is
fabricated.** There is no honest *production* leg at which a probe of the corpus
fails, because the one committed leg carrying a live ``_norm`` collision
(``PMC12444477…/strict``) now stops in **pre-freeze** at production defaults
(``PREFREEZE_DUPLICATE_CANONICAL_ROWS``). Live production exposure is **zero of 32
legs**, measured -- and zero exposure is not by itself a reason to narrow the guard:
F-039 proved the mechanism real, reachable and unguarded.

**Scope, as narrowed by the orchestrator in correction round 1.** The guard binds
``ir.py``'s **entity** call site only. The **component** call site keeps its
pre-existing warning, because ``prefreeze_resolution._canonicalize_species_rows``
*deliberately* converges a ``_norm`` group onto its leader **because this dedupe
collapses it**, and those rows share a ``taxonomy_id`` -- proven identity, which
D-035 permits, unlike F-039's coincident spelling over conflicting identifiers.
Both halves of that boundary are pinned under "Arm 6" below, and the component-side
residual is registered as **F-046**, owned by **C-050j**, not by this card.

What *is* demonstrated, by ``evidence/probe_c050i_dedupe_refusal.py --mode g9`` and
pinned below, is the **latent defect** run through both real implementations (base
tree exported at ``8f7514f`` by ``c051a_base_tree_batch.py``, and this tree): a
**constructed fixture** where the base drops a row and silently re-binds a reaction,
and the **real committed F-039 leg** under the two golden-sweep configurations
(``B``, ``E``) that still reach the exporter, where the base turns 44 payload
compound rows into 43 IR compounds and binds reaction 9 to the wrong molecule. Both
are labelled for what they are -- a constructed fixture and a test-configuration
replay demonstrating a latent defect -- **not** a corpus regression.

**Why a raise, not a blocking report issue.** ``_add_issue(report, "error", …)``
sets ``report["ok"] = False`` but does **not** stop IR construction, so the row
would still be dropped and an invalid IR still returned to a caller free to ignore
``ok``. Only raising fails before an invalid IR can be emitted. The structured
``duplicate_named_record`` diagnostic is *preserved*, promoted to ``error``, and now
names **both** conflicting rows.

**Why it blocks and never repairs.** Consolidating the rows would be post-freeze
biological repair of a different kind. **D-035 is unamended and D-036 defers the
consolidation engine**, and F-043 is the standing reason identifier equality is not
evidence of sameness: ``PG``, ``PG phosphate`` and ``(PGP)`` all carry PathBank 193,
which is UDP-glucose and biologically wrong for all three.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.ir import (  # noqa: E402
    DuplicateNamedRowError,
    _canonical,
    _dedupe_named_rows,
    _new_report,
    _norm,
    build_pwml_ir,
)
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution  # noqa: E402

#: A "no IR was produced" sentinel. Asserting on a sentinel rather than on
#: ``pytest.raises`` alone is what makes "nothing usable is returned" a *measured*
#: property instead of an inference from the traceback.
NOTHING = object()


def _compound(name: str, **extra: Any) -> Dict[str, Any]:
    """A compound row that already carries a pre-freeze resolution verdict.

    Without ``db_status`` ``build_pwml_ir`` refuses with
    ``UnresolvedCompoundRowError`` (C-051 / D-021) before reaching the dedupe, and
    every arm below would pass for the wrong reason.
    """

    row: Dict[str, Any] = {"name": name, "db_status": "unmatched"}
    row.update(extra)
    return row


def _payload(compounds: List[Dict[str, Any]],
             extra_buckets: Any = None) -> Dict[str, Any]:
    """A minimal payload. ``extra_buckets`` may itself override ``compounds``,
    which is why it is one mapping argument rather than ``**kwargs``."""

    entities: Dict[str, Any] = {"compounds": compounds, "proteins": []}
    entities.update(extra_buckets or {})
    return {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": entities,
        "processes": {"reactions": []},
    }


def _build(payload: Dict[str, Any], **kwargs: Any) -> Any:
    kwargs.setdefault("strict_db", False)
    kwargs.setdefault("name_index", None)
    return build_pwml_ir(copy.deepcopy(payload), pathway_name="P",
                         pathway_subject="Metabolic", **kwargs)


# ---------------------------------------------------------------------------
# Arm 1 + 2 -- a duplicate blocks, and nothing usable comes back
# ---------------------------------------------------------------------------


def test_new_acceptance_duplicate_compound_norm_keys_block() -> None:
    """NEW ACCEPTANCE (arm 1). F-039's real pair refuses instead of merging.

    ``'lipid IV_A'`` and ``'lipid IV A'`` both normalize to ``'lipid iv a'`` --
    ``_norm`` substitutes a space for the non-``[a-z0-9:+ ]`` class -- and they
    carry different PathBank and ChEBI identifiers. They are the pair the exporter
    silently collapsed on the one committed leg that reached it.
    """

    payload = _payload([
        _compound("lipid IV_A", pathbank_compound_id=40982, chebi_id="CHEBI:58603"),
        _compound("lipid IV A", pathbank_compound_id=40738, chebi_id="CHEBI:60365"),
    ])

    with pytest.raises(DuplicateNamedRowError) as excinfo:
        _build(payload)

    assert excinfo.value.code == "PWML_IR_DUPLICATE_NAMED_ROW"
    assert excinfo.value.norm_key == "lipid iv a"
    assert excinfo.value.pointer_prefix == "/entities/compounds"


def test_new_acceptance_no_ir_is_emitted_when_the_guard_fires() -> None:
    """NEW ACCEPTANCE (arm 2). The refusal precedes any invalid IR.

    This is the whole reason R2 requires a raise. A blocking report issue leaves
    ``build_pwml_ir`` returning an ``(ir, report)`` pair whose ``ir`` has already
    lost the row, and every caller that does not consult ``report["ok"]`` exports
    it. Here there is no pair to consult.
    """

    payload = _payload([
        _compound("lipid IV_A", pathbank_compound_id=40982),
        _compound("lipid IV A", pathbank_compound_id=40738),
    ])

    produced: Any = NOTHING
    with pytest.raises(DuplicateNamedRowError):
        produced = _build(payload)

    assert produced is NOTHING, "an IR object escaped the refusal"


# ---------------------------------------------------------------------------
# Arm 3 -- the diagnostic names BOTH rows
# ---------------------------------------------------------------------------


def test_new_acceptance_the_diagnostic_identifies_both_conflicting_rows() -> None:
    """NEW ACCEPTANCE (arm 3). Survivor *and* intruder, with names, keys, pointers.

    The superseded warning named only the second row and only its pointer: it said
    something was dropped without saying what it was dropped *onto*. Both halves
    are asserted on the exception and, separately, on the structured report issue
    -- ``_dedupe_named_rows`` takes the report as a parameter, so a caller owning
    that dict still sees the machine-readable account under the code it has always
    had, now at severity ``error``.
    """

    rows = [
        {"name": "lipid IV_A", "db_status": "unmatched"},
        {"name": "lipid IV A", "db_status": "unmatched"},
    ]
    report = _new_report()

    with pytest.raises(DuplicateNamedRowError) as excinfo:
        _dedupe_named_rows(rows, key_prefix="cmp", report=report,
                           pointer_prefix="/entities/compounds",
                           refuse_duplicates=True)

    error = excinfo.value
    assert error.names == ["lipid IV_A", "lipid IV A"]
    # The survivor was keyed; the intruder never reached the key assignment.
    assert error.keys == ["cmp_1", ""]
    assert error.pointers == ["/entities/compounds/0", "/entities/compounds/1"]
    for fragment in ("lipid IV_A", "lipid IV A", "cmp_1",
                     "/entities/compounds/0", "/entities/compounds/1", "lipid iv a"):
        assert fragment in str(error), f"the message does not name {fragment!r}"

    # The structured diagnostic is preserved, and promoted warning -> error.
    assert report["ok"] is False
    assert not any(i.get("code") == "duplicate_named_record" for i in report["warnings"])
    issues = [i for i in report["errors"] if i.get("code") == "duplicate_named_record"]
    assert len(issues) == 1
    assert issues[0]["norm_key"] == "lipid iv a"
    assert issues[0]["rows"] == [
        {"name": "lipid IV_A", "key": "cmp_1", "pointer": "/entities/compounds/0"},
        {"name": "lipid IV A", "key": "", "pointer": "/entities/compounds/1"},
    ]


# ---------------------------------------------------------------------------
# Arm 4 -- references cannot silently repoint
# ---------------------------------------------------------------------------

#: What the BASE implementation does with the fixture below, measured at ``8f7514f``
#: through ``evidence/probe_c050i_dedupe_refusal.py --mode g9`` against a
#: hash-verified base tree (2414/2414 blobs). **Recorded, not asserted as a base
#: failure** -- see this module's G9 note. The base built a two-compound IR, dropped
#: ``'lipid IV A'`` (PathBank 40738 / ChEBI 60365) entirely, and bound the reaction's
#: left side to ``cmp_1`` = ``'lipid IV_A'``, PathBank **40982** / ChEBI **58603**: a
#: different database compound, chosen after the freeze, on one warning and no error.
BASE_REBIND = {"compound_names": ["lipid IV_A", "lipid A precursor"],
               "reaction_left_entity_key": "cmp_1", "bound_row_pathwhiz_id": 40982,
               "duplicate_issue_severity": "warning"}


def _rebind_fixture() -> Dict[str, Any]:
    """The reaction consumes the spelling on the row the base drops."""
    payload = _payload([
        _compound("lipid IV_A", pathbank_compound_id=40982, chebi_id="CHEBI:58603"),
        _compound("lipid IV A", pathbank_compound_id=40738, chebi_id="CHEBI:60365",
                  kegg_id="C06025"),
        _compound("lipid A precursor", pathbank_compound_id=111),
    ])
    payload["processes"]["reactions"] = [
        {"name": "lipid IV A -> lipid A precursor",
         "inputs": ["lipid IV A"], "outputs": ["lipid A precursor"]},
    ]
    return payload


def test_new_acceptance_a_reference_cannot_be_silently_repointed() -> None:
    """NEW ACCEPTANCE (arm 4). The harm is a repoint, not a dangling reference.

    The mechanism, asserted rather than described: ``_dedupe_named_rows`` groups on
    ``_norm`` and ``entity_by_name`` -- which ``resolve_entity`` consults -- is keyed
    on the **same** ``_norm``, so a reference to the dropped spelling *resolves*, to
    the survivor. ``unresolved_entity_reference`` never fires and no downstream gate
    can catch it, which is why it must be caught here.
    """

    assert _norm("lipid IV A") == _norm("lipid IV_A") == "lipid iv a", (
        "the premise of the whole card: these two spellings share one exporter key")

    with pytest.raises(DuplicateNamedRowError) as excinfo:
        _build(_rebind_fixture())

    # The row the base bound the reaction to, and the row it destroyed, are both
    # named -- so the operator can see which molecule would have been substituted.
    assert excinfo.value.names == ["lipid IV_A", "lipid IV A"]
    assert BASE_REBIND["bound_row_pathwhiz_id"] == 40982, (
        "the base bound reaction 'lipid IV A -> …' to PathBank 40982, while the "
        "frozen payload declared its substrate as PathBank 40738")


# ---------------------------------------------------------------------------
# Arm 5 + 7 -- the non-colliding path is byte-identical (R3)
# ---------------------------------------------------------------------------

#: A committed leg with **no** ``_norm`` collision in any bucket, and the sha256 of
#: ``build_pwml_ir``'s ``ir`` and ``report`` over it, **measured at the base SHA
#: 8f7514f** through a hash-verified base tree. The R3 pin: the refusal must cost
#: exactly nothing off the colliding path. A move here is a regression, never a
#: licence to re-baseline.
R3_CONTROL_LEG = ("runs/2026-07-27_1623/papers/"
                  "PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/strict/"
                  "final_mapped.json")
R3_BASE_IR_DIGEST = "32bf7893e62fcb8ac5799cbf7d910a8076ff51ca63693f66262c9f1f7b1b8dc2"
R3_BASE_REPORT_DIGEST = "476e41daeb97aa593f8b4141ac3ba21879972440799d2a921fc5a0677a3ae19a"


class _DownDb:
    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


def test_new_acceptance_a_non_colliding_leg_is_byte_identical_to_the_base() -> None:
    """NEW ACCEPTANCE (arms 5 and 7). R3, on real committed data.

    Deleting the dedupe was rejected because ``record["key"] =
    f"{key_prefix}_{len(out) + 1}"`` would renumber every IR key and move goldens.
    This pins that promoting the collision branch renumbers nothing: over a real leg
    the whole IR and the whole report hash to what the base produced.
    """

    path = ROOT / R3_CONTROL_LEG
    assert path.is_file(), f"committed leg fixture is missing: {R3_CONTROL_LEG}"
    staged = json.loads(path.read_text(encoding="utf-8"))
    run_prefreeze_resolution(staged, strict_db=True, db_resolver=_DownDb(), name_index=None)

    ir, report = build_pwml_ir(staged, db_resolver=_DownDb(), strict_db=True, name_index=None)

    assert hashlib.sha256(
        json.dumps(ir, sort_keys=True, indent=1, default=repr).encode()
    ).hexdigest() == R3_BASE_IR_DIGEST, "the non-colliding IR moved"
    assert hashlib.sha256(
        json.dumps(report, sort_keys=True, indent=1, default=repr).encode()
    ).hexdigest() == R3_BASE_REPORT_DIGEST, "the non-colliding report moved"


def test_new_acceptance_distinct_rows_keep_their_key_numbering() -> None:
    """NEW ACCEPTANCE (arm 5). The focused form of R3: ``idx`` / ``len(out)``.

    Rows that are skipped -- non-dicts and blank names -- must still not consume a
    key, exactly as before, or every downstream key shifts by one.
    """

    rows: List[Any] = [
        {"name": "alpha"}, "not a dict", {"name": "   "}, {"name": "beta"},
        {"no_name_field": 1}, {"name": "gamma"},
    ]
    report = _new_report()
    out, by_norm = _dedupe_named_rows(rows, key_prefix="cmp", report=report,
                                      pointer_prefix="/entities/compounds",
                                      refuse_duplicates=True)

    assert [row["key"] for row in out] == ["cmp_1", "cmp_2", "cmp_3"]
    assert [row["name"] for row in out] == ["alpha", "beta", "gamma"]
    assert sorted(by_norm) == ["alpha", "beta", "gamma"]
    assert report["ok"] is True
    assert report["errors"] == [] and report["warnings"] == []


# ---------------------------------------------------------------------------
# Arm 6 -- R1 AS NARROWED: entity buckets refuse, component buckets warn
# ---------------------------------------------------------------------------

#: The **five entity buckets** -- ``ir.py``'s entity call site, which refuses. All
#: five feed ``entity_by_name``, keyed on the same ``_norm`` the dedupe groups on,
#: so all five carry the identical silent-repoint mechanism.
ENTITY_BUCKETS_UNDER_TEST = ["compounds", "proteins", "nucleic_acids",
                             "element_collections", "protein_complexes"]

#: The **four component buckets** -- ``ir.py``'s component call site, which keeps
#: its warning. See ``test_new_acceptance_component_buckets_still_warn``.
COMPONENT_BUCKETS = ["species", "subcellular_locations", "cell_types", "tissues"]


@pytest.mark.parametrize("bucket", ENTITY_BUCKETS_UNDER_TEST)
def test_new_acceptance_every_entity_bucket_refuses(bucket: str) -> None:
    """NEW ACCEPTANCE (arm 6). R1 as narrowed: all five entity buckets, not
    compounds only."""
    rows = [{"name": "Escherichia coli", "db_status": "unmatched"},
            {"name": "escherichia  coli", "db_status": "unmatched"}]
    payload = _payload([_compound("glycine")], {bucket: rows})

    with pytest.raises(DuplicateNamedRowError) as excinfo:
        _build(payload)

    assert excinfo.value.pointer_prefix == f"/entities/{bucket}"
    assert excinfo.value.norm_key == "escherichia coli"
    # ``_canonical`` collapses the whitespace run before the name is recorded.
    assert excinfo.value.names == ["Escherichia coli", "escherichia coli"]


@pytest.mark.parametrize("bucket", COMPONENT_BUCKETS)
def test_new_acceptance_component_buckets_still_warn(bucket: str) -> None:
    """NEW ACCEPTANCE (arm 6, the other half). **The narrowed boundary, pinned.**

    R1 as first issued bound the guard to *every* caller. That was wrong for the
    component call site and the orchestrator **narrowed it on measurement**:
    ``prefreeze_resolution._canonicalize_species_rows`` (``:1180-1197``)
    deliberately converges a ``_norm`` group onto its leader **because this dedupe
    collapses it**, and a row that stopped being a duplicate would become "a second
    species in the IR that the exporter never emitted" -- inventing biology in the
    exact direction that module exists to prevent. Converged rows share a
    ``taxonomy_id``: **proven identity**, which D-035 permits, unlike F-039's
    compound pair (coincident spelling over PathBank 40738 vs 40982).

    **The residual is named here rather than left silent: F-046, owned by C-050j,
    not this card.** A component collision the pre-freeze converger did *not*
    create still drops first-wins on a warning, and ``component_by_name``
    (``ir.py:995-1000``) is keyed on the same ``_norm``, so the same repoint
    applies. Measured live exposure over all 32 committed legs and all nine buckets
    is **zero**: the only collision in the corpus is F-039's, in an entity bucket.
    """
    payload = _payload([_compound("glycine")], {
        bucket: [{"name": "Escherichia coli"}, {"name": "escherichia  coli"}]})

    ir, report = _build(payload)

    # It collapses, exactly as before this card, and says so at severity warning.
    assert [row["name"] for row in ir[bucket]] == ["Escherichia coli"]
    # Scoped to THIS code: the minimal payload legitimately raises unrelated
    # ``biological_state_*`` errors, and asserting on the whole bucket would pin
    # those by accident.
    assert not [i for i in report["errors"] if i.get("code") == "duplicate_named_record"]
    warned = [i for i in report["warnings"] if i.get("code") == "duplicate_named_record"]
    assert len(warned) == 1
    assert warned[0]["message"] == (
        "Duplicate record for 'escherichia coli' ignored in PWML IR.")
    assert warned[0]["pointer"] == f"/entities/{bucket}/1"


# ---------------------------------------------------------------------------
# Arm 9 -- the adversarial arm: the guard cannot be bypassed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first,second,expected_norm",
    [
        # The real F-039 pair -- underscore against space. MANDATORY.
        ("lipid IV_A", "lipid IV A", "lipid iv a"),
        # Case only: ``_norm`` casefolds.
        ("Lipid IV A", "lipid iv a", "lipid iv a"),
        # Punctuation: the non-``[a-z0-9:+ ]`` class becomes a space.
        ("lipid-IV-A", "lipid IV A", "lipid iv a"),
        # Whitespace runs: ``_canonical`` collapses them before ``_norm`` sees them.
        ("lipid  IV   A", "lipid IV A", "lipid iv a"),
        # Trailing/leading punctuation, stripped by ``_norm``.
        ("(lipid IV A)", "lipid IV A", "lipid iv a"),
        # Mixed: every trick at once.
        ("  LIPID--IV__a  ", "lipid iv a", "lipid iv a"),
    ],
)
def test_new_acceptance_the_guard_cannot_be_bypassed_by_spelling(
        first: str, second: str, expected_norm: str) -> None:
    """NEW ACCEPTANCE (arm 9). MANDATORY adversarial arm.

    Every variant that lands on the same exporter ``_norm`` must refuse. A guard
    that any of these walked past would be decoration: the *only* reason
    ``ir._dedupe_named_rows`` ever fired on committed data is that ``_norm`` and
    ``process_normalizer._normalize`` disagree, so the pre-freeze dedupe cannot be
    relied on to have removed these first.
    """

    assert _norm(first) == _norm(second) == expected_norm

    with pytest.raises(DuplicateNamedRowError) as excinfo:
        _build(_payload([_compound(first), _compound(second)]))

    assert excinfo.value.norm_key == expected_norm
    # The recorded names are ``_canonical``-ed -- stripped, whitespace collapsed --
    # which is the spelling the payload row would have carried into the IR.
    assert excinfo.value.names == [_canonical(first), _canonical(second)]


def test_new_acceptance_aliases_do_not_provide_an_escape_hatch() -> None:
    """NEW ACCEPTANCE (arm 9, aliases path). Two halves, both asserted.

    ``entity_by_name`` is populated from ``[name, raw_name, short_name,
    common_name, *synonyms]`` (``ir.py``), all keyed on ``_norm``, so the alias
    path is where a bypass would hide.

    1. **No escape.** A colliding ``name`` still refuses when the rows also carry
       aliases -- decorating a row cannot buy it past the guard.
    2. **The residual, described as it actually behaves.** ``_dedupe_named_rows``
       groups on ``name`` alone, so a row whose *alias* collides with another row's
       *name* is **not** a dedupe collision and does **not** refuse. The guard is
       right not to fire: those two rows are genuinely distinct entities, and
       refusing would itself be wrong.

       **CORRECTED 2026-08-17 (REV-050i, F-048).** This docstring previously said
       the overlap was "resolved last-writer-wins" and "pinned here so the residue
       is attributable instead of latent". **Measured, every part of that was
       wrong.** ``entity_by_name`` is a ``defaultdict(list)`` populated with
       ``.append``, and ``resolve_entity``'s ``preferred_order`` loop **returns the
       first matching candidate**, so:

       * it is **payload row order** that wins, not the last writer;
       * the early ``return`` happens *before* the ``ambiguous_entity_reference``
         branch, so **no warning is emitted at all**;
       * the residue is therefore **exactly as latent as before** -- not
         attributable.

       That is **F-048**, owned by **C-050k**, not by this card. It is the same
       harm class this card exists to prevent -- a reference binding to a
       biologically different row with no diagnostic -- reached through the
       *aliases* surface rather than the *name* surface. It is pinned below in both
       row orders so the claim is measured rather than described.
    """

    # 1 -- aliases present, names still collide: refuses.
    with pytest.raises(DuplicateNamedRowError) as excinfo:
        _build(_payload([
            _compound("lipid IV_A", synonyms=["Lipid IVA"], raw_name="Lipid-IV-A"),
            _compound("lipid IV A", synonyms=["compound 40738"], short_name="LIVA"),
        ]))
    assert excinfo.value.norm_key == "lipid iv a"

    # 2 -- alias-only overlap: distinct ``name`` keys, so no dedupe collision, and
    # the binding follows PAYLOAD ROW ORDER with no diagnostic. Both orders are
    # exercised, because one order alone cannot tell "first wins" from "last wins".
    serine = dict(_compound("serine"), synonyms=["Glycine"])
    for rows, expected_bound in (
        ([serine, _compound("glycine")], "serine"),
        ([_compound("glycine"), serine], "glycine"),
    ):
        payload = _payload([dict(row) for row in rows])
        payload["processes"]["reactions"] = [
            {"name": "R1", "inputs": ["Glycine"], "outputs": []}]
        ir, report = _build(payload)

        assert not any(i.get("code") == "duplicate_named_record"
                       for i in report["errors"] + report["warnings"])
        # The reaction input 'Glycine' binds to whichever row came FIRST...
        bound_key = ir["processes"]["reactions"][0]["left"][0]["entity_key"]
        bound = next(r for r in ir["entities"]["compounds"] if r["key"] == bound_key)
        assert bound_key == "cmp_1"
        assert bound["name"] == expected_bound
        # ...and F-048's sting: nothing anywhere says the reference was ambiguous.
        assert not any(i.get("code") == "ambiguous_entity_reference"
                       for i in report["errors"] + report["warnings"])
