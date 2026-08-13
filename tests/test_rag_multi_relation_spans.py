"""C-061 — a span that states two reactions must support the claim it states.

The defect, from ``PMC12421875``. This sentence is the paper's own description of
two consecutive menaquinone steps::

    Subsequently, MenA joins DHNA and prenyl diphosphate to produce
    demethylmenaquinone (DMK), and MenG demethylates DMK to generate MK
    ( Fig. 1A ).

RAG retrieved the first reaction correctly — ``DHNA + prenyl diphosphate -> DMK
[MenA]``, confidence 0.92, pathway match, organism match — and the evidence
predicate refused it fifteen times, once per gap, on two counts: that the span
"states ``['dmk'] -> ['mk ( fig']``" and that "MenA is not the catalyst the span
attaches to this reaction". Both statements are about a sentence that names MenA
and states its reaction verbatim. Two independent causes:

1. the parser returned the FIRST template match anywhere in the body, so a
   sentence stating two reactions could only ever be read as one — and here the
   one it found was the *other* clause;
2. product trimming ran to the statement terminator, and the terminator was the
   period inside "Fig.", so the product was the non-compound ``MK ( Fig``.

``PRODUCT_CONTRACT.md`` § 2: "Missing detail initiates targeted retrieval and gap
resolution." Retrieval ran, found the reaction, and the gate refused content the
paper states word for word — a ``product_contract_violation``.

**What this must NOT do.** Merge gate 6 forbids weakening a biological gate to
increase PWML production. Widening what a span can be read as SAYING is the fix;
widening what counts as AGREEMENT would be the defect. The refusal tests below
(A3, A4, A5, A8) are the load-bearing half of this file.

Offline / deterministic: no chromadb, no network, no live LLM.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EVIDENCE = ROOT / "docs" / "pwml_recovery_sprint" / "evidence"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EVIDENCE) not in sys.path:
    sys.path.insert(0, str(EVIDENCE))

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

from t2pw.rag.admission import (  # noqa: E402
    REASON_DIRECTION_DISAGREES,
    REASON_NO_RELATION,
    REASON_RELATION_DISAGREES,
    REASON_ROLES_UNASSIGNABLE,
    REASON_UNSUPPORTED_CATALYST,
    parse_span_relation,
    parse_span_relations,
    validate_evidence_span,
)

#: The R-004 F-B2 fixture, verbatim from the committed admission report.
SPAN = (
    "Subsequently, MenA joins DHNA and prenyl diphosphate to produce "
    "demethylmenaquinone (DMK), and MenG demethylates DMK to generate MK ( Fig. 1A )."
)

LEG = ROOT / "runs" / "2026-08-02_2130" / "papers" / "PMC12421875"


def _check(span=SPAN, *, inputs, outputs, enzymes=(), reversible=False):
    return validate_evidence_span(
        span,
        inputs=list(inputs),
        outputs=list(outputs),
        enzymes=list(enzymes),
        reversible=reversible,
    )


# ===========================================================================
# A1 / A2 — both stated reactions are supported by the sentence stating them.
# ===========================================================================
def test_the_first_of_two_stated_reactions_is_supported() -> None:
    """A1 — the base failure. The claim IS the span's first clause, verbatim."""
    verdict = _check(
        inputs=["DHNA", "prenyl diphosphate"],
        outputs=["demethylmenaquinone (DMK)"],
        enzymes=["MenA"],
    )
    assert verdict.ok, verdict.reasons


def test_the_second_of_two_stated_reactions_stays_supported() -> None:
    """A2 — MenG's clause. Refused at base too, over ``MK ( Fig``."""
    verdict = _check(inputs=["DMK"], outputs=["MK"], enzymes=["MenG"])
    assert verdict.ok, verdict.reasons


# ===========================================================================
# A3 / A4 / A5 — the anti-gate-6 clauses. Reading two relations must not let a
# claim be assembled out of parts of both, or run either of them backwards.
# ===========================================================================
def test_a_catalyst_taken_from_the_other_clause_is_still_refused() -> None:
    """A3 — MenA's chemistry with MenG's name is not what the span says."""
    verdict = _check(
        inputs=["DHNA", "prenyl diphosphate"],
        outputs=["demethylmenaquinone (DMK)"],
        enzymes=["MenG"],
    )
    assert not verdict.ok
    assert any(r.startswith(REASON_UNSUPPORTED_CATALYST) for r in verdict.reasons), (
        verdict.reasons
    )


def test_a_reversed_reaction_is_still_refused() -> None:
    """A4 — the "B -> A supports A -> B" hole stays shut on both clauses."""
    first = _check(
        inputs=["demethylmenaquinone (DMK)"],
        outputs=["DHNA", "prenyl diphosphate"],
        enzymes=["MenA"],
    )
    assert not first.ok
    assert any(r.startswith(REASON_DIRECTION_DISAGREES) for r in first.reasons), (
        first.reasons
    )

    second = _check(inputs=["MK"], outputs=["DMK"], enzymes=["MenG"])
    assert not second.ok


def test_dropping_a_stated_substrate_is_still_refused() -> None:
    """A5 — the span has ``prenyl diphosphate``; a claim may not lose it.

    Currency exemption runs one way only. Dropping a real co-substrate changes
    the chemistry, and this is the exact shape the incomplete Stage-1 row at
    ``/processes/reactions/14`` has.
    """
    verdict = _check(
        inputs=["DHNA"], outputs=["demethylmenaquinone (DMK)"], enzymes=["MenA"]
    )
    assert not verdict.ok
    assert any(r.startswith(REASON_RELATION_DISAGREES) for r in verdict.reasons), (
        verdict.reasons
    )


# ===========================================================================
# A6 — the product is a compound, not a figure callout.
# ===========================================================================
def test_a_figure_callout_is_not_part_of_the_product_name() -> None:
    """A6 — ``MK ( Fig`` was a parse artifact, never a metabolite."""
    relations = parse_span_relations(SPAN)
    products = {p for relation in relations for p in relation.outputs}
    assert "MK" in products
    assert not any("fig" in p.casefold() for p in products), products

    # And the alias in "demethylmenaquinone (DMK)" is NOT trimmed with it: that
    # parenthesis is the paper naming one compound twice.
    assert "demethylmenaquinone (DMK)" in products


def test_both_stated_reactions_are_returned_with_their_own_catalysts() -> None:
    """The parse the two clauses deserve, from one span handed over whole."""
    relations = parse_span_relations(SPAN)
    readings = {
        (tuple(r.inputs), tuple(r.outputs), tuple(r.catalysts)) for r in relations
    }
    assert (("DHNA", "prenyl diphosphate"), ("demethylmenaquinone (DMK)",), ("MenA",)) in readings
    assert (("DMK",), ("MK",), ("MenG",)) in readings


# ===========================================================================
# A8 — no fail-open. Every branch where a parse failure could reach a permissive
# default, stated and shown not to.
# ===========================================================================
@pytest.mark.parametrize(
    "inputs,outputs,enzymes",
    [
        # A third reaction the span never states, built from names it does state.
        (["DHNA"], ["MK"], ["MenA"]),
        # Product of clause one, product of clause two, no relation between them.
        (["demethylmenaquinone (DMK)"], ["MK"], ["MenA"]),
        # A compound the span never mentions at all.
        (["chorismate"], ["demethylmenaquinone (DMK)"], ["MenA"]),
        # An enzyme the span never mentions.
        (["DMK"], ["MK"], ["UbiE"]),
    ],
)
def test_a_claim_the_span_does_not_state_is_refused(inputs, outputs, enzymes) -> None:
    """A8 — more readings must not mean more agreement."""
    verdict = _check(inputs=inputs, outputs=outputs, enzymes=enzymes)
    assert not verdict.ok
    assert verdict.reasons


def test_a_span_stating_no_relation_still_refuses_with_the_same_codes() -> None:
    """A8 — the two no-relation branches are untouched.

    ``parse_span_relations`` returning an empty list reaches exactly the code
    ``parse_span_relation`` returning ``None`` used to reach: a cue present means
    "roles unassignable", a cue absent means "no relation stated". Neither is a
    permissive default — both refuse.
    """
    cue = validate_evidence_span(
        "A and B were both produced in the assay",
        inputs=["A"],
        outputs=["B"],
        enzymes=[],
    )
    assert not cue.ok
    assert cue.reasons[0].startswith(REASON_ROLES_UNASSIGNABLE)

    no_cue = validate_evidence_span(
        "Strains carrying A and B were grown overnight",
        inputs=["A"],
        outputs=["B"],
        enzymes=[],
    )
    assert not no_cue.ok
    assert no_cue.reasons[0].startswith(REASON_NO_RELATION)


def test_an_unreadable_template_match_yields_no_relation_at_all() -> None:
    """A8 — a match whose roles cannot be assigned is DROPPED, never trimmed.

    Enumerating every start position means a template can now match in places the
    single-match parser never reached. Each such match is judged by the same
    rules, so an unreadable one adds nothing rather than adding a guess.
    """
    assert parse_span_relations("which is phosphorylated by LpxK to produce lipid IVA") == []
    assert parse_span_relation("which is phosphorylated by LpxK to produce lipid IVA") is None


def test_the_relation_cap_can_only_refuse_more() -> None:
    """A8 — truncation is on the SUPPORT side, so it cannot open the gate."""
    from t2pw.rag.admission import _MAX_SPAN_RELATIONS

    assert _MAX_SPAN_RELATIONS >= 1
    assert len(parse_span_relations(SPAN)) <= _MAX_SPAN_RELATIONS


# ===========================================================================
# A7 — replay against the committed production artifact.
# ===========================================================================
def _report(leg: str) -> dict:
    path = LEG / leg / "rag_admission_report.json"
    assert path.exists(), path
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_committed_rejection_replays_as_an_admission() -> None:
    """A7 — ``rejected[1]`` of the committed research-leg admission report.

    The recorded ``reasons`` are exactly the two span reasons and nothing else:
    no other gate objected to this candidate, so the span verdict is what
    rejected it and the span verdict is what admits it.
    """
    candidate = _report("research")["rejected"][1]
    assert candidate["evidence"]["span"] == SPAN
    assert candidate["status"] == "rejected"
    assert {r.split(":")[0] for r in candidate["reasons"]} == {
        REASON_RELATION_DISAGREES,
        REASON_UNSUPPORTED_CATALYST,
    }

    verdict = validate_evidence_span(
        candidate["evidence"]["span"],
        inputs=candidate["inputs"],
        outputs=candidate["outputs"],
        enzymes=candidate["enzymes"],
        reversible=bool(candidate.get("reversible")),
    )
    assert verdict.ok, verdict.reasons


def test_admitting_it_closes_the_legs_missing_supported_reaction() -> None:
    """A7 — ``missing_supported_reactions`` for that leg drops 1 -> 0.

    Measured, not asserted: the gold signature set for PMC12421875 is scored
    against the leg's committed ``merged_payload.json`` as it stands, and then
    against the same payload with the admitted candidate's row added. The only
    difference is the row this card unblocks.
    """
    from t2pw.bench import semantic
    from t2pw.bench.goldset import load_gold_set

    case = next(
        c for c in load_gold_set().cases if c.paper_id.startswith("PMC12421875")
    )
    payload = json.loads(
        (LEG / "research" / "merged_payload.json").read_text(encoding="utf-8")
    )
    paper = (LEG / "01_source_text.txt").read_text(encoding="utf-8")

    processes = semantic._processes(payload)
    _r, metrics, _u, before = semantic._check_supported_reactions(case, processes, paper)
    assert before == 1
    assert metrics["missing"][0]["signature"].startswith(
        "DHNA + prenyl diphosphate -> demethylmenaquinone"
    )

    candidate = _report("research")["rejected"][1]
    with_row = dict(processes)
    with_row["reactions"] = list(processes["reactions"]) + [
        {
            "name": candidate["name"],
            "inputs": list(candidate["inputs"]),
            "outputs": list(candidate["outputs"]),
            "enzymes": [{"protein": e, "evidence": e} for e in candidate["enzymes"]],
            "reversible": bool(candidate.get("reversible")),
        }
    ]
    _r2, _m2, _u2, after = semantic._check_supported_reactions(case, with_row, paper)
    assert after == 0


def test_the_incomplete_stage_one_row_is_not_what_matched() -> None:
    """The duplicate-row question, stated as a fact rather than assumed away.

    ``/processes/reactions/14`` already carries ``DHNA -> demethylmenaquinone
    [MenA]`` — the same reaction drawn WITHOUT its co-substrate. It does not match
    the gold signature (that is why the leg was missing one), and admitting the
    RAG candidate does not repair it. Whether the two rows are then unified is a
    merge/dedup question, owned elsewhere. This test pins the fact so the next
    reader does not have to rediscover it.
    """
    payload = json.loads(
        (LEG / "research" / "merged_payload.json").read_text(encoding="utf-8")
    )
    row = payload["processes"]["reactions"][14]
    assert row["inputs"] == ["DHNA"]
    assert row["outputs"] == ["demethylmenaquinone"]

    # And the span refuses that row's chemistry, correctly: it dropped a stated
    # substrate. Admission adds the complete row; it does not bless the partial one.
    verdict = _check(
        inputs=row["inputs"], outputs=row["outputs"], enzymes=["MenA"]
    )
    assert not verdict.ok


# ===========================================================================
# A9 — preservation, over every candidate the gate has actually judged.
# ===========================================================================
def test_every_unchanged_candidate_answers_exactly_as_it_did_at_base() -> None:
    """A9 — 2000 real candidate spans, nine papers, both legs.

    The committed base golden was generated from the dispatch base
    ``472293c`` with the same script. Every entry whose digest still matches is
    byte-identical in relation, verdict, reason strings and reversibility
    normalization. Every entry that moved must be a candidate the production run
    REJECTED — this card may unblock a refusal, never break an admission.
    """
    import c061_relation_golden as golden

    base = json.loads(
        (EVIDENCE / "c061_relation_golden_base.json").read_text(encoding="utf-8")
    )
    tip = golden.build()
    assert tip["n"] == base["n"] == 2000

    base_digests = dict(base["digests"])
    changed = [
        entry["key"]
        for entry in tip["entries"]
        if base_digests[entry["key"]] != entry["digest"]
    ]
    assert len(changed) == 115, changed[:10]
    assert all("/rejected[" in key for key in changed), changed[:10]
    assert all(
        entry["ok"] for entry in tip["entries"] if entry["key"] in set(changed)
    )

    # Nothing that used to be admitted moved, and no refusal changed its wording.
    unchanged = [e for e in tip["entries"] if e["key"] not in set(changed)]
    assert len(unchanged) == 1885


def test_the_delta_is_five_paper_verbatim_reactions() -> None:
    """A9 — what moved, in biology rather than in counts.

    Five distinct claims, all of them a clause of a sentence that states several:
    MenA's and MenG's steps in PMC12421875 (both legs), and MenA's and UbiE's
    steps in PMC12657337's three-reaction sentence. Every one names substrates,
    product and catalyst that its own span states word for word. No claim gained
    admission on a reaction its span does not state.
    """
    import c061_relation_golden as golden

    base_digests = dict(
        json.loads(
            (EVIDENCE / "c061_relation_golden_base.json").read_text(encoding="utf-8")
        )["digests"]
    )
    tip = {e["key"]: e for e in golden.build()["entries"]}

    distinct = {}
    for row in golden._rows():
        key = f"{row['paper']}/{row['leg']}/{row['bucket']}[{row['index']}]"
        if base_digests[key] == tip[key]["digest"]:
            continue
        distinct[
            (row["span"], tuple(row["inputs"]), tuple(row["outputs"]), tuple(row["enzymes"]))
        ] = row

    assert len(distinct) == 5, [k[1:] for k in distinct]
    for span, inputs, outputs, enzymes in distinct:
        verdict = validate_evidence_span(
            span,
            inputs=list(inputs),
            outputs=list(outputs),
            enzymes=list(enzymes),
            reversible=False,
        )
        assert verdict.ok
        # The reaction the span was read as stating is the reaction claimed —
        # not a neighbouring clause's, and not a mixture of two.
        agreeing = [
            r
            for r in parse_span_relations(span)
            if validate_evidence_span(
                span,
                inputs=list(inputs),
                outputs=list(outputs),
                enzymes=list(enzymes),
            ).ok
            and set(map(str.casefold, r.catalysts)) == set(map(str.casefold, enzymes))
        ]
        assert agreeing, (span[:60], inputs, outputs, enzymes)
