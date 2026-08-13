"""Reversibility symmetry, and the recall cost of requiring a parsed relation.

Two things live here because they are the same question from opposite sides.

**Reversibility** is an agreement rule, and an agreement rule that only fires in
one direction is not one. A one-way claim backed by reversible evidence used to
ship as one-way — silently dropping a direction the paper asserted — while the
mirror case was refused.

**Recall** is what tightening costs. Requiring a *parsed relation with assignable
roles* (rather than "the names co-occur") is the correction that stopped
``B -> A`` supporting ``A -> B``. It also refuses sentences no implemented
template covers, and that number has to be visible: a gate that admits nothing is
trivially free of false positives and useless.

How the measurement is kept honest
----------------------------------
* **Every labelled sentence carries an artifact path and is asserted to occur
  verbatim in that file.** :func:`test_every_labelled_sentence_is_verbatim_source`
  fails the whole module if one does not. An earlier version of this file quoted
  the ``evidence`` fields of ``stage1_payload.json`` and called them paper text;
  five of its seventeen "source sentences" were LLM-written paraphrases that
  appear in no paper, which made the measurement an evaluation of the extractor's
  prose style rather than of the parser's coverage of real writing.
* **The only normalization is whitespace collapsing.** Nothing else is repaired —
  the extractor's spaced parentheses (``acetyl-CoA synthetase ( acs )``) and
  detached commas (``The enzyme Idi , which``) are exactly what the production
  parser has to survive, so they are left in.
* **A correct parse means correct substrates, products, catalysts AND direction.**
  Judging on substrates and products alone (as the first version did) scored
  "an enzyme that converts acetate into acetyl-CoA" as fully correct while the
  parser was reporting no catalyst for it, so a candidate naming ``acs`` was
  refused. Both numbers are reported: metabolite-role recall, and full-role
  recall including catalysts and reversibility.
* **Catalysts are compared the way the gate compares them** — through
  :func:`t2pw.rag.admission.actor_spellings`, which is what
  :func:`validate_evidence_span` uses to decide whether a claimed enzyme is the
  one the span attached to the reaction. So "correct catalysts" means precisely
  "a candidate naming these enzymes, and no others, would be admitted".

Offline / deterministic: reads checked-in text and calls no model.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

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

from t2pw.rag.admission import (  # noqa: E402
    actor_spellings,
    parse_span_relation,
    parse_span_relations,
    validate_evidence_span,
)


# ===========================================================================
# 7. Reversibility agreement is symmetric.
# ===========================================================================
def _check(span, *, inputs, outputs, enzymes=(), reversible=False):
    return validate_evidence_span(
        span,
        inputs=list(inputs),
        outputs=list(outputs),
        enzymes=list(enzymes),
        reversible=reversible,
    )


def test_a_reversible_claim_backed_by_one_way_evidence_is_refused() -> None:
    """The claim may not be LESS constrained than the sentence backing it."""
    verdict = _check("A -> B", inputs=["A"], outputs=["B"], reversible=True)
    assert not verdict.ok
    assert any(
        r.startswith("evidence_direction_disagrees_with_claim") for r in verdict.reasons
    )


def test_a_one_way_claim_backed_by_reversible_evidence_is_normalized() -> None:
    """Explicitly normalized, not silently kept one-way and not refused.

    The evidence supports strictly more than the claim, so the honest outcome is
    to record what the evidence said — and to say that it was recorded.
    """
    verdict = _check("A <=> B", inputs=["A"], outputs=["B"], reversible=False)
    assert verdict.ok
    assert verdict.normalized_reversible is True


def test_reversible_evidence_with_swapped_sides_is_accepted_only_as_reversible() -> None:
    """``B <=> A`` supports ``A -> B`` — but only by making it reversible."""
    verdict = _check("B <=> A", inputs=["A"], outputs=["B"], reversible=False)
    assert verdict.ok
    assert verdict.normalized_reversible is True

    # The same swap without reversibility in the span stays refused.
    one_way = _check("B -> A", inputs=["A"], outputs=["B"], reversible=False)
    assert not one_way.ok


def test_prose_reversibility_is_read_off_the_sentence() -> None:
    """"catalyzes a reversible reaction" is a bidirectional claim."""
    relation = parse_span_relation(
        "MenF catalyzes a reversible reaction converting chorismate to isochorismate"
    )
    assert relation is not None
    assert relation.reversible is True
    assert relation.inputs == ["chorismate"]
    assert relation.outputs == ["isochorismate"]
    assert relation.catalysts == ["MenF"]

    verdict = _check(
        "MenF catalyzes a reversible reaction converting chorismate to isochorismate",
        inputs=["isochorismate"],
        outputs=["chorismate"],
        enzymes=["MenF"],
    )
    assert verdict.ok, verdict.reasons
    assert verdict.normalized_reversible is True


# ===========================================================================
# 8. Recall over sentences taken verbatim from checked-in source artifacts.
# ===========================================================================
#: The six papers the labelled sentences come from, and the checked-in artifact
#: each sentence must occur in verbatim.
PAPERS = {
    "PMC12421875": "PMC12421875__the-growth-benefits-and-toxicity-of-quinone-bio",
    "PMC12657337": "PMC12657337__high-level-production-of-vitamin-k2-in-lt-i-gt",
    "PMC12444477": "PMC12444477__the-regulation-of-lipid-a-biosynthesis",
    "PMC12312563": "PMC12312563__structures-of-listeria-monocytogenes-mend-in-th",
    "PMC12856317": "PMC12856317__a-reversible-feedback-mechanism-regulating-mito",
    "PMC13278307": "PMC13278307__an-overview-of-mobile-colistin-resistance-mcr-g",
    "PMC12096016": "PMC12096016__the-enterobactin-biosynthetic-intermediate-2-3",
}
RUN = ROOT / "runs" / "2026-07-28_0919" / "papers"


def artifact_for(pmc: str) -> Path:
    """The ``01_source_text.txt`` a labelled sentence claims to come from."""
    return RUN / PAPERS[pmc] / "01_source_text.txt"


def _norm(text: str) -> str:
    """Collapse whitespace. The ONLY normalization applied to source text."""
    return " ".join(str(text or "").split())


def _expect(inputs, outputs, catalysts=(), reversible=False) -> dict:
    return {
        "inputs": list(inputs),
        "outputs": list(outputs),
        "catalysts": list(catalysts),
        "reversible": bool(reversible),
    }


#: ``(pmc, sentence, expected)``. ``expected`` is ``None`` when the sentence
#: states no single transformation with assignable roles — a pathway summary, a
#: list of enzymes, a group transfer with no product named. Otherwise it is the
#: full reading a biologist would take from the sentence: substrates, products,
#: catalysts *as the sentence names them*, and direction.
LABELLED: tuple = (
    # --- PMC12421875, menaquinone biosynthesis in Lactococcus lactis ---------
    (
        "PMC12421875",
        "The enzyme MenF catalyzes a reversible reaction converting chorismate (A) "
        "to isochorismate (B)",
        _expect(["chorismate (A)"], ["isochorismate (B)"], ["MenF"], reversible=True),
    ),
    (
        "PMC12421875",
        "Subsequently, MenA joins DHNA and prenyl diphosphate to produce "
        "demethylmenaquinone (DMK)",
        _expect(
            ["DHNA", "prenyl diphosphate"], ["demethylmenaquinone (DMK)"], ["MenA"]
        ),
    ),
    (
        "PMC12421875",
        "MenG demethylates DMK to generate MK",
        _expect(["DMK"], ["MK"], ["MenG"]),
    ),
    (
        "PMC12421875",
        "MK biosynthesis starts from chorismate and proceeds through a seven-enzyme "
        "pathway consisting of MenF, MenD, MenH, MenC, MenE, MenB, and MenI to "
        "produce DHNA",
        None,  # a pathway summary, not a step
    ),
    # --- PMC12657337, MK-7 production in E. coli ----------------------------
    (
        "PMC12657337",
        "HepPPS catalyzes the conversion of FPP to heptaprenyl pyrophosphate (HepPP)",
        _expect(["FPP"], ["heptaprenyl pyrophosphate (HepPP)"], ["HepPPS"]),
    ),
    (
        "PMC12657337",
        "We also overexpressed acetyl-CoA synthetase ( acs ), an enzyme that "
        "converts acetate into acetyl-CoA",
        _expect(["acetate"], ["acetyl-CoA"], ["acs"]),
    ),
    (
        "PMC12657337",
        "The enzyme Idi , which converts IPP to farnesyl pyrophosphate (FPP), was "
        "crucial for further conversion by HepPPS to form heptaprenyl "
        "pyrophosphate (HepPP)",
        _expect(["IPP"], ["farnesyl pyrophosphate (FPP)"], ["Idi"]),
    ),
    (
        "PMC12657337",
        "The acs gene, responsible for converting acetate to acetyl-CoA, was "
        "overexpressed by replacing its native promoter with the strong, "
        "constitutive J23119 promoter",
        _expect(["acetate"], ["acetyl-CoA"], ["acs"]),
    ),
    (
        "PMC12657337",
        "catalyzes the methylation of demethylmenaquinone to produce MK-7",
        # A subjectless clause: the sentence fragment names no catalyst, so the
        # honest expectation is none.
        _expect(["demethylmenaquinone"], ["MK-7"], []),
    ),
    (
        "PMC12657337",
        "PMK, phosphomevalonate kinase; and MVD, mevalonate pyrophosphate "
        "decarboxylase",
        None,  # a legend listing enzymes
    ),
    # --- PMC12444477, lipid A biosynthesis ----------------------------------
    (
        "PMC12444477",
        "LpxB then catalyzes the formation of a tetra-acylated disaccharide "
        "intermediate, which is phosphorylated by LpxK to produce lipid IV A",
        # Two predicates. The one with a product is LpxK's, and its substrate is
        # behind the relative pronoun.
        _expect(
            ["tetra-acylated disaccharide intermediate"], ["lipid IV A"], ["LpxK"]
        ),
    ),
    (
        "PMC12444477",
        "the phospholipase PldA that degrades GPLs localized in the outer leaflet "
        "of the OM to produce lysophospholipids and fatty acids",
        _expect(["GPLs"], ["lysophospholipids", "fatty acids"], ["PldA"]),
    ),
    (
        "PMC12444477",
        "LpxC is a deacetylase that removes the acetyl group from the GlcNAc moiety",
        None,  # a group removal, with no product species named
    ),
    (
        "PMC12444477",
        "removal of UMP by LpxH, LpxI, or LpxG depending on the organism",
        None,  # a fragment; no product, and three alternative actors
    ),
    # --- PMC12312563, MenD structures ---------------------------------------
    (
        "PMC12312563",
        "decarboxylation of 2-oxoglutarate produces intermediate I",
        _expect(["2-oxoglutarate"], ["intermediate I"], []),
    ),
    # --- PMC12856317, ALA synthesis -----------------------------------------
    (
        "PMC12856317",
        "condensation of glycine and succinyl-CoA to produce aminolevulinic acid",
        _expect(["glycine", "succinyl-CoA"], ["aminolevulinic acid"], []),
    ),
    # --- PMC13278307, mcr / PEtN transfer -----------------------------------
    (
        "PMC13278307",
        "These enzymes catalyze the transfer of PEtN to the phosphate groups of "
        "lipid A",
        # "These enzymes" is an anaphor: the sentence names no catalyst.
        _expect(["PEtN"], ["phosphate groups of lipid A"], []),
    ),
    # --- PMC12096016, EntB isochorismatase ----------------------------------
    (
        "PMC12096016",
        "In this assay, isochorismate is converted to "
        "2,3‐dihydro‐2,3‐dihydroxybenzoate (2,3‐diDHB) and "
        "pyruvate by EntB isochorismatase activity",
        _expect(
            ["isochorismate"],
            ["2,3‐dihydro‐2,3‐dihydroxybenzoate (2,3‐diDHB)", "pyruvate"],
            ["EntB"],
        ),
    ),
)

SUPPORTED = tuple(row for row in LABELLED if row[2] is not None)


def _folded(names) -> set:
    return {_norm(n).casefold() for n in names if _norm(n)}


def _metabolites_match(relation, expected) -> bool:
    return _folded(relation.inputs) == _folded(expected["inputs"]) and _folded(
        relation.outputs
    ) == _folded(expected["outputs"])


def _catalysts_match(relation, expected) -> bool:
    """Judged exactly as :func:`validate_evidence_span` judges a claimed enzyme.

    Every expected enzyme must be spelled by one of the phrases the span attached
    to the reaction, and the span must attach no MORE phrases than expected — an
    extra one would let a candidate ship a catalyst nobody named.
    """
    if len(relation.catalysts) != len(expected["catalysts"]):
        return False
    spellings = {
        _norm(s).casefold()
        for phrase in relation.catalysts
        for s in actor_spellings(phrase)
    }
    return all(_norm(e).casefold() in spellings for e in expected["catalysts"])


def _classify():
    """Run the real parser over the labelled set.

    Returns ``(metabolites_correct, full_correct, wrong, missed, unsupported)``
    where ``full_correct`` also requires catalysts and direction.
    """
    metabolites_correct: list = []
    full_correct: list = []
    wrong: list = []
    missed: list = []
    unsupported: list = []

    for pmc, sentence, expected in LABELLED:
        relation = parse_span_relation(sentence)
        if expected is None:
            if relation is not None:
                unsupported.append((pmc, sentence, relation))
            continue
        if relation is None:
            missed.append((pmc, sentence))
            continue
        if not _metabolites_match(relation, expected):
            wrong.append((pmc, sentence, relation, "metabolites"))
            continue
        metabolites_correct.append((pmc, sentence))
        if not _catalysts_match(relation, expected):
            wrong.append((pmc, sentence, relation, "catalysts"))
            continue
        if bool(relation.reversible) != bool(expected["reversible"]):
            wrong.append((pmc, sentence, relation, "reversibility"))
            continue
        full_correct.append((pmc, sentence))
    return metabolites_correct, full_correct, wrong, missed, unsupported


# ---------------------------------------------------------------------------
# The fixture is source text, and this is what proves it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pmc,sentence,_expected", LABELLED)
def test_every_labelled_sentence_is_verbatim_source(pmc, sentence, _expected) -> None:
    """Each labelled sentence occurs, verbatim, in the artifact it names.

    Whitespace collapsing is the only transformation. If this fails, the labelled
    set has drifted into paraphrase and every number below it is meaningless.
    """
    artifact = artifact_for(pmc)
    assert artifact.exists(), artifact
    body = _norm(artifact.read_text(encoding="utf-8", errors="replace"))
    assert _norm(sentence) in body, (
        f"{pmc}: not verbatim in {artifact.relative_to(ROOT).as_posix()}\n"
        f"  {sentence!r}"
    )


def test_the_labelled_set_spans_more_than_one_paper() -> None:
    """A single paper's prose style is not a measurement of coverage."""
    assert len({pmc for pmc, _s, _e in LABELLED}) >= 6
    assert len(LABELLED) == 18
    assert len(SUPPORTED) == 14


# ---------------------------------------------------------------------------
# The numbers.
# ---------------------------------------------------------------------------
def test_no_unsupported_sentence_yields_a_relation() -> None:
    """The number that must stay at zero.

    A sentence that states no single transformation must produce no relation at
    all. This is the guarantee the whole tightening exists for: everything else
    below is a cost, this is the benefit.
    """
    _mets, _full, _wrong, _missed, unsupported = _classify()
    assert unsupported == [], [
        (pmc, s[:70], r.inputs, r.outputs, r.catalysts) for pmc, s, r in unsupported
    ]


def test_no_supported_sentence_is_parsed_with_the_wrong_roles() -> None:
    """A wrong parse is worse than no parse: it can agree with a wrong claim.

    "Wrong" now covers catalysts and direction too. The three constructions that
    forced this — a reversible reaction bound to MenF, an appositive gloss on
    ``acs``, and a relative clause on ``Idi`` — all passed the old
    metabolites-only check while reporting no catalyst at all.
    """
    _mets, _full, wrong, _missed, _unsupported = _classify()
    assert wrong == [], [
        (pmc, s[:60], why, r.inputs, r.outputs, r.catalysts, r.reversible)
        for pmc, s, r, why in wrong
    ]


def test_recall_over_real_source_sentences_is_measured_and_pinned() -> None:
    """The honest cost, in numbers, on verbatim paper text.

    Both recalls are pinned so the tradeoff is visible in the diff of any future
    change: widening a template should move them up and ``missed`` down, and must
    never move ``unsupported`` or ``wrong`` off zero.
    """
    metabolites, full, wrong, missed, unsupported = _classify()

    assert len(SUPPORTED) == 14
    # C-061 moved these deliberately, by exactly one sentence: the MenA "joins X
    # and Y to produce Z" condensation form, previously the first named false
    # negative below. 9 -> 10 on both recalls, 5 -> 4 missed, and the two numbers
    # that must not move did not: wrong and unsupported are still zero.
    assert len(metabolites) == 10  # source-grounded metabolite-role recall
    assert len(full) == 10  # source-grounded FULL-role recall
    assert len(wrong) == 0
    assert len(missed) == 4
    assert len(unsupported) == 0

    # Metabolite-role and full-role recall are equal here, which is the point of
    # the correction: every sentence whose chemistry parses now also gets its
    # catalyst and direction right. Before it, three of these nine reported no
    # catalyst for a sentence that named one — they scored as fully correct under
    # a metabolites-only rule and would still have refused the honest claim.
    assert len(full) == len(metabolites)
    assert len(full) / len(SUPPORTED) >= 0.6


def test_the_known_false_negatives_are_named_not_hidden() -> None:
    """Name the shapes no template covers, so the gap is a work item not a mystery.

    The "joins X and Y to produce Z" condensation form used to head this list. It
    was not a cosmetic gap: it is PMC12421875's MenA step, the paper's own
    ``supported_reactions[2]``, and RAG retrieved it correctly and was refused
    fifteen times. C-061 covers it (``actor_combines_to_make``), so the list is
    one shorter and the recall numbers above are one higher.

    The three that remain are real constructions from the checked-in papers:

    * ``LpxB then catalyzes the formation of ..., which is phosphorylated by LpxK
      to produce lipid IV A`` — the substrate of the second predicate is behind a
      relative pronoun; binding it would need antecedent resolution, and guessing
      it wrong attributes LpxK's chemistry to LpxB;
    * ``the phospholipase PldA that degrades GPLs ... to produce ...`` — a
      restrictive relative clause with the actor outside it;
    * ``decarboxylation of 2-oxoglutarate produces intermediate I`` and
      ``condensation of glycine and succinyl-CoA to produce ...`` — a nominalized
      reaction as the grammatical subject.
    """
    _mets, _full, _wrong, missed, _unsupported = _classify()
    assert sorted(s[:26] for _pmc, s in missed) == [
        "LpxB then catalyzes the fo",
        "condensation of glycine an",
        "decarboxylation of 2-oxogl",
        "the phospholipase PldA tha",
    ]


@pytest.mark.parametrize("pmc,sentence,expected", SUPPORTED)
def test_every_supported_sentence_either_parses_fully_or_not_at_all(
    pmc, sentence, expected
) -> None:
    """Per-sentence: all four roles right, or no relation. Never partly right."""
    relation = parse_span_relation(sentence)
    if relation is None:
        return
    assert _metabolites_match(relation, expected), (
        pmc,
        relation.inputs,
        relation.outputs,
    )
    assert _catalysts_match(relation, expected), (pmc, relation.catalysts)
    assert bool(relation.reversible) is bool(expected["reversible"]), pmc


# ---------------------------------------------------------------------------
# The order of the templates is load-bearing.
# ---------------------------------------------------------------------------
def test_the_actor_binding_construction_wins_over_the_bare_one_inside_it() -> None:
    """``converting X to Y`` must not pre-empt the clause that names the actor.

    The generic form is a substring of the wider one. Matching it first is not a
    smaller answer, it is a WRONG one: it reports "nobody catalyses this", and a
    candidate naming the enzyme the paper named is then refused as injecting an
    unsupported catalyst.
    """
    bound = parse_span_relation(
        "The enzyme MenF catalyzes a reversible reaction converting chorismate (A) "
        "to isochorismate (B)"
    )
    assert bound is not None
    assert bound.pattern == "catalyzes_converting"
    assert bound.catalysts == ["MenF"]

    # The bare form still applies where no actor-binding clause exists.
    bare = parse_span_relation("by rapidly converting isochorismate to 2,3-diDHB")
    assert bare is not None and bare.pattern == "converting_to"
    assert bare.catalysts == []


@pytest.mark.parametrize(
    "sentence,enzyme",
    [
        (
            "The enzyme MenF catalyzes a reversible reaction converting chorismate "
            "(A) to isochorismate (B)",
            "MenF",
        ),
        (
            "We also overexpressed acetyl-CoA synthetase ( acs ), an enzyme that "
            "converts acetate into acetyl-CoA",
            "acs",
        ),
        (
            "The enzyme Idi , which converts IPP to farnesyl pyrophosphate (FPP), "
            "was crucial for further conversion by HepPPS to form heptaprenyl "
            "pyrophosphate (HepPP)",
            "Idi",
        ),
    ],
)
def test_the_three_named_claims_validate_with_their_stated_enzyme(
    sentence, enzyme
) -> None:
    """End to end through the gate: the claim naming the paper's enzyme passes."""
    relation = parse_span_relation(sentence)
    assert relation is not None
    verdict = validate_evidence_span(
        sentence,
        inputs=list(relation.inputs),
        outputs=list(relation.outputs),
        enzymes=[enzyme],
        reversible=bool(relation.reversible),
    )
    assert verdict.ok, verdict.reasons


def test_an_anaphoric_actor_still_yields_no_catalyst() -> None:
    """The preserved refusal: "which is" names nobody.

    Widening the templates to bind actors through appositives and relative
    clauses must not turn a relative PRONOUN into a protein. Both of these have a
    relation cue and neither has a bindable actor inside the span.
    """
    assert parse_span_relation("which is phosphorylated by LpxK to produce lipid IVA") is None

    anaphoric = parse_span_relation(
        "These enzymes catalyze the transfer of PEtN to the phosphate groups of lipid A"
    )
    assert anaphoric is not None
    assert anaphoric.catalysts == []
    # ... and a claim that names one is therefore refused.
    refused = validate_evidence_span(
        "These enzymes catalyze the transfer of PEtN to the phosphate groups of lipid A",
        inputs=["PEtN"],
        outputs=["phosphate groups of lipid A"],
        enzymes=["MCR-1"],
    )
    assert not refused.ok
    assert any(r.startswith("unsupported_catalyst_injection") for r in refused.reasons)


def test_a_statement_listing_three_reactions_never_yields_a_stitched_one() -> None:
    """Real prose packs three steps into one sentence; the parser reports one.

    "In this pathway, HepPPS catalyzes the conversion of FPP to heptaprenyl
    pyrophosphate (HepPP), MenA converts HepPP to DMK-7, and UbiE methylates
    DMK-7 to yield MK-7" is a single statement by any sentence splitter. A comma
    splitter offered ``MenA converts HepPP to DMK-7`` as a PRODUCT of the first
    reaction. What comes back now is the leading predicate and nothing stitched.
    """
    sentence = (
        "In this pathway, HepPPS catalyzes the conversion of FPP to heptaprenyl "
        "pyrophosphate (HepPP), MenA converts HepPP to DMK-7, and UbiE methylates "
        "DMK-7 to yield MK-7"
    )
    body = _norm(artifact_for("PMC12657337").read_text(encoding="utf-8", errors="replace"))
    assert _norm(sentence) in body

    relation = parse_span_relation(sentence)
    assert relation is not None
    assert relation.inputs == ["FPP"]
    assert relation.outputs == ["heptaprenyl pyrophosphate (HepPP)"]
    assert relation.catalysts == ["HepPPS"]

    # C-061: the other two reactions are now recovered as SEPARATE relations,
    # each with the catalyst its own clause names. They were previously false
    # negatives — the sentence states them, and a claim quoting one was refused
    # for disagreeing with a DIFFERENT clause of its own evidence. Recovering
    # them is not stitching: the stitched readings below are still refused.
    readings = {
        (tuple(r.inputs), tuple(r.outputs), tuple(r.catalysts))
        for r in parse_span_relations(sentence)
    }
    assert (("FPP",), ("heptaprenyl pyrophosphate (HepPP)",), ("HepPPS",)) in readings
    assert (("HepPP",), ("DMK-7",), ("MenA",)) in readings
    assert (("DMK-7",), ("MK-7",), ("UbiE",)) in readings

    for reading in readings:
        assert validate_evidence_span(
            sentence,
            inputs=list(reading[0]),
            outputs=list(reading[1]),
            enzymes=list(reading[2]),
        ).ok

    # Nothing stitched across the clause boundaries: not the first substrate to
    # the last product, and not one clause's chemistry under another's enzyme.
    for inputs, outputs, enzymes in (
        (["FPP"], ["MK-7"], ["HepPPS"]),
        (["FPP"], ["DMK-7"], ["MenA"]),
        (["HepPP"], ["DMK-7"], ["UbiE"]),
        (["HepPP"], ["MK-7"], ["MenA"]),
    ):
        assert not validate_evidence_span(
            sentence, inputs=inputs, outputs=outputs, enzymes=enzymes
        ).ok, (inputs, outputs, enzymes)


def test_a_locant_comma_does_not_split_a_compound() -> None:
    """``2,3-oxidosqualene`` is one metabolite, not ``2`` and ``3-oxidosqualene``."""
    relation = parse_span_relation(
        "LSS (Lanosterol synthase) catalyzes one of the earliest steps in sterol "
        "biosynthesis by converting 2,3-oxidosqualene to lanosterol"
    )
    assert relation is not None
    assert relation.inputs == ["2,3-oxidosqualene"]
    assert relation.outputs == ["lanosterol"]
    assert "LSS" in actor_spellings(relation.catalysts[0])


# ---------------------------------------------------------------------------
# The corpus labels clauses. Production hands over whole sentences.
# ---------------------------------------------------------------------------
def test_the_two_menaquinone_clauses_are_read_from_the_joined_sentence() -> None:
    """The gap between this corpus and production, closed (C-061).

    ``LABELLED`` carries the MenA clause and the MenG clause as two entries,
    because that is how a labelled set is built. The paper writes them as ONE
    sentence, and that is what the retriever hands the gate. Reading only the
    first template match meant the MenG clause's reading was the only thing the
    MenA claim could be checked against, so a claim quoting the sentence verbatim
    was refused for disagreeing with its own evidence — fifteen times, once per
    gap, on the committed PMC12421875 research leg.

    The corpus entries stay clause-level on purpose: this test is the one that
    asserts the joined form, so a future change that regresses either shape is
    visible as itself.
    """
    joined = (
        "Subsequently, MenA joins DHNA and prenyl diphosphate to produce "
        "demethylmenaquinone (DMK), and MenG demethylates DMK to generate MK "
        "( Fig. 1A )."
    )
    body = _norm(artifact_for("PMC12421875").read_text(encoding="utf-8", errors="replace"))
    assert _norm(joined) in body

    readings = {
        (tuple(r.inputs), tuple(r.outputs), tuple(r.catalysts))
        for r in parse_span_relations(joined)
    }
    assert (
        ("DHNA", "prenyl diphosphate"),
        ("demethylmenaquinone (DMK)",),
        ("MenA",),
    ) in readings
    assert (("DMK",), ("MK",), ("MenG",)) in readings

    # Each clause's claim is supported by the joined sentence, exactly as it is
    # by the clause on its own.
    assert validate_evidence_span(
        joined,
        inputs=["DHNA", "prenyl diphosphate"],
        outputs=["demethylmenaquinone (DMK)"],
        enzymes=["MenA"],
    ).ok
    assert validate_evidence_span(
        joined, inputs=["DMK"], outputs=["MK"], enzymes=["MenG"]
    ).ok
