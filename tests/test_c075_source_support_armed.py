"""C-075 / F-096 -- the identifier source-support pass, ARMED, under the ruling.

C-073 (merged ``6373ad1``) shipped ``t2pw.mapping.identity_admission`` complete
and tested, and then it sat dormant: no production caller ever handed the merge
site the paper, so ``map_payload`` answered ``not_evaluated`` on every leg and
nothing was ever withheld. Measured by artifact: 0 of 20 T-105 legs carry a
source index. This file is about ARMING that pass and about the two routes the
product owner's ruling of 2026-08-23 requires it to honour once it fires.

THE RULING, and how each clause is exercised below:

    "A real identifier may survive only when the entity's identity and
     biological kind are supported by paper evidence, a proven alias, or another
     permitted provenance route. Do not require the literal database accession to
     appear in the paper. Unsupported or hallucinated entities must not acquire
     real identifiers. Legitimate aliases must retain valid mappings."

* *paper evidence*        -- section 2 (activation), section 4 (paper-relativity)
* *never the accession*   -- :func:`test_ruling_a_source_naming_only_the_accession_is_not_support`
* *a proven alias*        -- section 3 (the six measured cases) and section 6
* *another route*         -- section 5
* *legitimate mappings*   -- sections 3, 5, 6 and the corpus replay in section 8

G9 LABELLING -- READ THIS BEFORE SCORING THE FILE. The three groups are not the
same kind of claim and are not offered as the same kind of proof:

  ARM A -- ACTIVATION (section 2). A **new-capability acceptance test**. The
    capability existed at base; what did not exist is a way for production to
    supply the paper. These arms call
    ``merge_additions(..., source_text=...)``, a keyword that does not exist on
    the base SHA, so on base they fail with ``TypeError``. That is SYMBOL
    ABSENCE and it is explicitly NOT offered as a G9 clause-1 behavioural proof.
    They are labelled new capability and nothing more is claimed for them.

  ARM B -- PROVENANCE ROUTE (section 5) and
  ARM C -- SOURCE-SIDE ALIAS (section 6). These ARE base-failing behavioural
    proofs in the G9 clause-1 sense. Each is written to run UNCHANGED on the base
    SHA -- it imports no symbol this card adds, builds the source index from a
    literal shape, and fails on an assertion about the identifiers
    ``map_payload`` ships. Honest caveat, stated rather than buried: the base
    behaviour they correct was UNREACHABLE in production, because nothing armed
    the index. They correct a latent defect that arming would otherwise have
    shipped, not a defect users have seen.

  ARM D -- THE AUDIT PROMPT (section 9), added in correction round 1. The two
    prompt arms ARE base-failing behavioural proofs: they call only
    ``audit_json_llm._build_llm_prompt``, which exists on base, and on base the
    armed leg's prompt carries the whole index and both assertions fail. The
    ``payload_for_prompt`` arm beside them names a new symbol and is not offered
    as one.

  Section 10's additivity arm patches a constant this card adds, so it too is
  symbol-dependent and is offered as a PROPERTY check, not as a G9 proof.

  Everything else is a preservation or fail-open arm and claims nothing about G9.

WHY B AND C EXIST AT ALL. C-073's 1-catch / 0-collateral figure was measured on
ONE run (2026-08-21_2239, 102 eligible rows). Replayed over all 70 committed
``final_mapped.json`` artifacts against their own ``01_source_text.txt`` -- 678
eligible rows -- the C-073 predicate refuses 39, and 36 of those are legitimate.
The two clauses that rescue them are disjoint, because the names are looked for
before the route is consulted: the provenance route rescues **32** (30 naming
another document, 2 naming ``seed_paper`` -- section 11), and the source-side
alias clause rescues **4**. Section 8 replays the whole corpus and pins what is
left; section 10 pins that neither clause can ever take anything away.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping import map_ids  # noqa: E402
from t2pw.mapping.map_ids import map_payload  # noqa: E402

#: The payload key and blob shape the Stage-2 merge writes. Spelled out as
#: literals, exactly as ``test_c073_identity_admission.py`` does, so every
#: behavioural arm below runs on the base SHA without importing anything new.
SOURCE_INDEX_KEY = "source_text_index"
INDEX_SCHEMA_VERSION = 1


def _fold(text: str) -> str:
    """The normalization the index stores. Restated here, and pinned against the
    production fold by ``test_c073_identity_admission.py``."""
    lowered = str(text or "").casefold()
    folded = "".join(ch if ch.isalnum() else " " for ch in lowered)
    return " ".join(folded.split())


def source_index(text: str) -> Dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "length": len(text),
        "normalized": _fold(text),
    }


# ── the offline runner ───────────────────────────────────────────────────────


class _NoNetwork(RuntimeError):
    """Raised if anything under test reaches for the network."""


def run_offline(payload: Dict[str, Any], cache_path: Path) -> Dict[str, Any]:
    """``map_payload`` with every external door shut. What comes out is a
    function of the input alone."""
    env = {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}
    with patch.dict(os.environ, env), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(
        map_ids, "_ai_protein_synonym_lookup", return_value=[]
    ), patch.object(
        map_ids.HttpClient, "get", side_effect=_NoNetwork("network call during an offline run")
    ):
        return map_payload(payload, cache_path=cache_path, id_source="db", use_cache=False)


# ── fixtures ─────────────────────────────────────────────────────────────────

#: PMC12180156's own vocabulary, minus every trace of succinyl-CoA. "succinyl"
#: really does occur zero times in that paper's 67,553-character source.
SOURCE_WITHOUT_SUCCINYL = (
    "Heme biosynthesis in erythroid precursors. Glycine is condensed by ALAS2 in the "
    "mitochondrial matrix, and ferrochelatase inserts ferrous iron into protoporphyrin "
    "IX. SFXN4 supplies the serine transport step required for one-carbon metabolism, "
    "and loss of SFXN4 impairs heme production."
)

#: PMC12856317 names it, so PMC12856317 keeps it. The rule is paper-relative.
SOURCE_WITH_SUCCINYL = SOURCE_WITHOUT_SUCCINYL + (
    " ALAS2 condenses succinyl-CoA with glycine to form 5-aminolevulinate."
)

SUCCINYL_IDS = {
    "hmdb": "HMDB0001022", "kegg": "C00091", "chebi": "15380", "pubchem": "439161",
    "cas": "604-98-8", "biocyc": "3-METHYLBENZYLSUCCINYL-COA", "chemspider": "388307",
    "pathbank_compound_id": "808",
}
HEME_IDS = {"hmdb": "HMDB0003178", "kegg": "C00032", "chebi": "17627", "pubchem": "444098"}
GLYCINE_IDS = {"hmdb": "HMDB0000123", "kegg": "C00037", "chebi": "15428", "pubchem": "750"}

REAL_NAMESPACES = ("hmdb", "kegg", "chebi", "pubchem")


def succinyl_row() -> Dict[str, Any]:
    """The row PMC12180156/research actually shipped, names and all."""
    return {
        "name": "succinyl-CoA",
        "class": "compound",
        "short_name": "Suc-CoA",
        "synonyms": ["Succinyl-CoA", "Succinyl coenzyme A"],
        "mapped_ids": dict(SUCCINYL_IDS),
    }


def heme_payload(index: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """That leg's shape: the fabrication, two legitimate compounds the paper
    really names, and a reaction that consumes all three."""
    payload: Dict[str, Any] = {
        "entities": {
            "compounds": [
                {"name": "heme", "class": "compound", "short_name": "Heme",
                 "synonyms": ["Heme", "Haem", "Protoheme"], "mapped_ids": dict(HEME_IDS)},
                {"name": "Glycine", "class": "compound", "short_name": "Gly",
                 "raw_name": "glycine", "synonyms": ["Glycine", "Aminoacetic acid", "Gly"],
                 "mapped_ids": dict(GLYCINE_IDS)},
                succinyl_row(),
            ],
            "proteins": [],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "ALAS2 condensation",
                    "inputs": ["Glycine", "succinyl-CoA"],
                    "outputs": ["heme"],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }
    if index is not None:
        payload[SOURCE_INDEX_KEY] = index
    return payload


def _row(result: Dict[str, Any], name: str, bucket: str = "compounds") -> Dict[str, Any]:
    rows = result["payload"]["entities"][bucket]
    matches = [r for r in rows if isinstance(r, dict) and r.get("name") == name]
    assert matches, f"{name} vanished from {bucket}: {[r.get('name') for r in rows]}"
    return matches[0]


def _ids(result: Dict[str, Any], name: str, bucket: str = "compounds") -> Dict[str, Any]:
    return _row(result, name, bucket).get("mapped_ids") or {}


# ── the ARMED chain, end to end ──────────────────────────────────────────────


def armed_chain(
    payload: Dict[str, Any], cache_path: Path, *, source_text: str = ""
) -> Dict[str, Any]:
    """Stage-1 payload -> ``merge_additions`` -> ``map_payload``, the real path.

    Deliberately the PRODUCTION functions and not a hand-assembled index: the
    whole point of this card is that ``screen_additions`` writes the index,
    ``apply_post_merge_cleanup`` (which ``merge_additions`` runs afterwards)
    leaves it alone, and ``map_payload`` is what acts on it. Nothing here reaches
    inside any of the three.
    """
    from t2pw.pipeline.pipeline import merge_additions

    merged = merge_additions(payload, {}, source_text=source_text)
    return run_offline(merged, cache_path)


def armed_payload_from_merge() -> Dict[str, Any]:
    """The heme leg as the ARMED Stage-2 merge really produces it -- index and
    all -- so section 9 audits the object production hands the audit loop."""
    from t2pw.pipeline.pipeline import merge_additions

    return merge_additions(heme_payload(), {}, source_text=SOURCE_WITHOUT_SUCCINYL)


# =============================================================================
# 1. the wiring itself
# =============================================================================


def test_merge_additions_forwards_source_text_to_the_gate() -> None:
    """NEW CAPABILITY. The one line this card owns: the kwarg exists, is keyword
    only, defaults to ``""``, and reaches ``screen_additions`` unchanged."""
    import inspect

    from t2pw.pipeline import pipeline as pl

    signature = inspect.signature(pl.merge_additions)
    parameter = signature.parameters["source_text"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == ""

    seen: Dict[str, Any] = {}

    def _spy(payload, **kwargs):
        seen.update(kwargs)
        return payload, {"schema_version": 1, "removed": [], "demoted": [],
                         "counts": {"removed": 0, "demoted": 0, "admitted": 0},
                         "review_required": False, "review_reasons": []}

    with patch.object(pl, "screen_additions", _spy):
        pl.merge_additions({"entities": {}, "processes": {}}, {}, source_text="the paper")
    assert seen["source_text"] == "the paper"
    # and it is NOT coupled to seed_text, which arms a REMOVAL rule both
    # production merge sites deliberately leave off (pipeline.py:726-731).
    assert seen["seed_text"] == ""


def test_merge_additions_without_source_text_writes_no_index() -> None:
    """Every existing caller is byte-identically unaffected: no ``source_text``,
    no key, and therefore ``not_evaluated`` downstream rather than a refusal."""
    from t2pw.pipeline.pipeline import merge_additions

    merged = merge_additions(heme_payload(), {})
    assert SOURCE_INDEX_KEY not in merged


def test_the_armed_merge_writes_an_index_that_survives_to_map_payload(
    tmp_path: Path,
) -> None:
    """NEW CAPABILITY, and the whole chain in one assertion: the index is written
    inside ``screen_additions``, survives everything ``merge_additions`` does
    afterwards (``apply_post_merge_cleanup`` included), and is what
    ``map_payload`` reports having evaluated."""
    from t2pw.pipeline.pipeline import merge_additions

    merged = merge_additions(heme_payload(), {}, source_text=SOURCE_WITHOUT_SUCCINYL)
    assert merged[SOURCE_INDEX_KEY]["schema_version"] == INDEX_SCHEMA_VERSION
    assert merged[SOURCE_INDEX_KEY]["length"] == len(SOURCE_WITHOUT_SUCCINYL)

    result = run_offline(merged, tmp_path / "cache.json")
    section = result["report"]["identity_admission"]
    assert section["source_index"]["status"] == "evaluated"
    assert section["source_index"]["length"] == len(SOURCE_WITHOUT_SUCCINYL)


def test_the_second_merge_of_a_rag_leg_keeps_the_index_it_was_handed() -> None:
    """``merge_additions`` runs TWICE on a RAG leg and only the Stage-2 site is
    armed. The second call's ``base`` is the first call's output, so the index
    must travel on the payload rather than being rewritten -- otherwise a RAG leg
    would silently lose the evidence a strict leg keeps."""
    from t2pw.pipeline.pipeline import merge_additions

    first = merge_additions(heme_payload(), {}, source_text=SOURCE_WITHOUT_SUCCINYL)
    second = merge_additions(first, {})  # the RAG site supplies no source_text
    assert second[SOURCE_INDEX_KEY] == first[SOURCE_INDEX_KEY]


# =============================================================================
# 2. ARM A -- activation. NEW-CAPABILITY ACCEPTANCE, not a G9 clause-1 proof.
# =============================================================================


def test_arm_a_armed_a_row_the_paper_never_names_ships_no_real_accession(
    tmp_path: Path,
) -> None:
    """NEW CAPABILITY. Through the real ``merge_additions`` -> ``screen_additions``
    -> ``map_payload`` chain with the paper supplied, the row whose name, whose
    ``short_name`` and whose two synonyms are all absent from the source ships
    NO real external accession.

    On the base SHA this raises ``TypeError`` on the missing keyword. That is
    symbol absence and is NOT offered as a behavioural proof; the arms that ARE
    live in sections 5 and 6.
    """
    result = armed_chain(heme_payload(), tmp_path / "cache.json",
                         source_text=SOURCE_WITHOUT_SUCCINYL)
    shipped = _ids(result, "succinyl-CoA")
    for namespace in REAL_NAMESPACES:
        assert not shipped.get(namespace), f"{namespace} still shipped: {shipped}"


def test_arm_a_unarmed_the_same_chain_withholds_nothing(tmp_path: Path) -> None:
    """The other half of the same proof, and the reason arming is a DECISION: the
    identical payload through the identical chain with no ``source_text`` ships
    every one of those accessions, exactly as production does today."""
    result = armed_chain(heme_payload(), tmp_path / "cache.json")
    shipped = _ids(result, "succinyl-CoA")

    assert shipped.get("kegg") == SUCCINYL_IDS["kegg"]
    assert "identity_admission" not in (_row(result, "succinyl-CoA").get("mapping_meta") or {})
    assert "identity_admission" not in result["report"]


def test_arm_a_arming_takes_nothing_from_the_rows_the_paper_does_name(
    tmp_path: Path,
) -> None:
    """Zero collateral, at the level the card cares about: the two legitimate
    compounds on the same leg come through the armed chain untouched."""
    result = armed_chain(heme_payload(), tmp_path / "cache.json",
                         source_text=SOURCE_WITHOUT_SUCCINYL)
    assert _ids(result, "heme") == HEME_IDS
    assert _ids(result, "Glycine") == GLYCINE_IDS


# =============================================================================
# 3. preservation -- the six measured alias cases, through the ARMED chain
# =============================================================================

ALIAS_CASES = [
    pytest.param(
        {"name": "2,3-dihydroxybenzoic acid", "synonyms": ["2,3-Dihydroxybenzoate"]},
        "EntB converts isochorismate to 2,3-dihydroxybenzoate during enterobactin assembly.",
        id="dihydroxybenzoic-acid-vs-benzoate",
    ),
    pytest.param(
        {"name": "L-serine", "synonyms": ["Serine"]},
        "Serine hydroxymethyltransferase consumes serine in the one-carbon cycle.",
        id="l-serine-vs-serine",
    ),
    pytest.param(
        {"name": "Adenosine triphosphate", "synonyms": ["ATP"], "short_name": "ATP"},
        "The adenylation domain performs an ATP-dependent activation of the substrate.",
        id="adenosine-triphosphate-vs-atp",
    ),
    pytest.param(
        {"name": "Adenosine monophosphate", "synonyms": ["AMP", "Adenylate"]},
        "Activation releases AMP and pyrophosphate from the enzyme active site.",
        id="adenosine-monophosphate-vs-amp",
    ),
    pytest.param(
        {"name": "ferric-enterobactin", "synonyms": []},
        "FepA imports ferric enterobactin across the outer membrane.",
        id="ferric-enterobactin-hyphen-vs-space",
    ),
    pytest.param(
        {"name": "Fe3", "synonyms": []},
        "Enterobactin chelates Fe3+ with very high affinity.",
        id="fe3-vs-fe3-plus",
    ),
]


@pytest.mark.parametrize("row_fields,source", ALIAS_CASES)
def test_all_six_measured_alias_cases_keep_their_identifiers_when_armed(
    row_fields: Dict[str, Any], source: str, tmp_path: Path
) -> None:
    """NON-NEGOTIABLE (card section 3.3). C-073 kept these six through a
    hand-built index; arming must keep them through the REAL chain. C-073's
    rejected first version is the reason this is a per-case parametrization and
    not one aggregate assertion -- a rule that strips a legitimate accession is
    worse than the defect it fixes."""
    row = {"class": "compound", "mapped_ids": dict(HEME_IDS)}
    row.update(row_fields)
    payload = {
        "entities": {"compounds": [row], "proteins": [], "protein_complexes": []},
        "processes": {"reactions": []},
    }
    result = armed_chain(payload, tmp_path / "cache.json", source_text=source)
    shipped = _ids(result, row["name"])

    assert set(HEME_IDS) <= set(shipped), (
        f"{row['name']} lost a legitimate accession against a source that names it: {shipped}"
    )
    assert shipped["kegg"] == HEME_IDS["kegg"]


# =============================================================================
# 4. paper-relativity, and the ruling on accession strings
# =============================================================================


def test_paper_relativity_one_name_two_sources_two_answers(tmp_path: Path) -> None:
    """One test, two sources (card section 6.3). ``succinyl-CoA`` is withheld
    against a paper that never names it and kept against one that does -- which
    is the whole shape of the rule. Both legs go through the armed chain."""
    absent = armed_chain(heme_payload(), tmp_path / "a.json",
                         source_text=SOURCE_WITHOUT_SUCCINYL)
    present = armed_chain(heme_payload(), tmp_path / "b.json",
                          source_text=SOURCE_WITH_SUCCINYL)

    assert not _ids(absent, "succinyl-CoA").get("kegg")
    assert _ids(present, "succinyl-CoA").get("kegg") == SUCCINYL_IDS["kegg"]
    # and the paper that names it disturbs nothing else on the leg
    assert _ids(present, "heme") == HEME_IDS


def test_ruling_a_source_naming_only_the_accession_is_not_support(
    tmp_path: Path,
) -> None:
    """PINS THE RULING (card section 6.4): "Do not require the literal database
    accession to appear in the paper" -- and its converse, which is what can
    actually be tested here. Support is about the ENTITY being named. A source
    that spells out every one of this row's accessions, and its own name nowhere,
    is not support: an accession string is a claim ABOUT the molecule, never
    evidence that the paper discussed it.

    Were the pass ever to match on accession strings, this arm would pass and the
    hallucination would ship with all eight identifiers intact."""
    source = (
        "Identifiers referenced in this methods section: CHEBI:15380, KEGG C00091, "
        "HMDB0001022, PubChem 439161, CAS 604-98-8, ChemSpider 388307. "
        "The pathway studied here is heme biosynthesis in erythroid precursors."
    )
    result = armed_chain(heme_payload(), tmp_path / "cache.json", source_text=source)
    shipped = _ids(result, "succinyl-CoA")

    for namespace in REAL_NAMESPACES:
        assert not shipped.get(namespace), (
            f"the accession string was read as support for the entity: {shipped}"
        )


# =============================================================================
# 5. ARM B -- another permitted provenance route. BASE-FAILING BEHAVIOURAL PROOF.
# =============================================================================

#: The real ``malonyl-CoA`` row from
#: ``runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/
#: strict/final_mapped.json``, verbatim but for the fields this file does not use.
#: Its nine accessions are correct, and its own payload records that it was
#: imported from PMC12898747, a paper about cardiolipins. "malonyl" occurs zero
#: times in PMC12444477's 82,306-character source text -- as do "PlsB", "PgsA",
#: "serine" and every other one of the 22 rows that run imported from elsewhere.
IMPORTED_IDS = {
    "kegg": "C00083", "hmdb": "HMDB0001175", "chebi": "CHEBI:15531",
    "pubchem": "10663", "cas": "524-14-1", "biocyc": "MALONYL-COA",
    "chemspider": "10213", "drugbank": "DB04524", "pathbank_compound_id": "911",
}
IMPORTED_ROW = {
    "name": "malonyl-CoA",
    "class": "compound",
    "rag_provenance": {
        "source_id": "PMC12898747",
        "source_title": "Essential Role of LapD in the Absence of Cardiolipins.",
        "source_type": "paper",
        "section": "introduction",
        "organism": "Escherichia coli",
    },
    "source_papers": [{"source_id": "PMC12898747", "source_type": "paper"}],
    "source_refs": ["PMC12898747"],
    "rag_confidence": 0.921955,
    "mapped_ids": dict(IMPORTED_IDS),
}

#: PMC12444477's subject. It never writes "malonyl".
SOURCE_WITHOUT_MALONYL = (
    "The regulation of lipid A biosynthesis. LpxC controls the committed step, and "
    "FtsH turnover of LpxC sets the balance between lipid A and fatty acid synthesis."
)


def _imported_payload(index: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "entities": {
            "compounds": [copy.deepcopy(IMPORTED_ROW)],
            "proteins": [],
            "protein_complexes": [],
        },
        "processes": {"reactions": []},
    }
    if index is not None:
        payload[SOURCE_INDEX_KEY] = index
    return payload


def test_arm_b_a_row_imported_from_another_named_paper_keeps_its_identifiers(
    tmp_path: Path,
) -> None:
    """G9 BEHAVIOURAL PROOF, base-failing. Imports nothing this card adds and
    builds the index from a literal, so it runs unchanged on the base SHA -- where
    ``malonyl-CoA`` loses all nine of its accessions and this assertion fails.

    The ruling admits "another permitted provenance route" beside paper evidence.
    This row states which document it came from; the seed paper is simply not its
    evidence base, so the honest answer is ``not_evaluated``, never
    ``unsupported`` (PRODUCT_CONTRACT section 8). Measured, this clause rescues
    32 of the 39 refusals the unqualified rule makes over the committed corpus;
    30 of those name the document they were imported from, as this fixture does,
    and the other 2 are section 11's.
    """
    result = run_offline(
        _imported_payload(source_index(SOURCE_WITHOUT_MALONYL)), tmp_path / "cache.json"
    )
    shipped = _ids(result, "malonyl-CoA")

    assert set(IMPORTED_IDS) <= set(shipped), (
        f"a legitimate imported mapping was stripped: {shipped}"
    )
    assert shipped["kegg"] == IMPORTED_IDS["kegg"]
    meta = _row(result, "malonyl-CoA").get("mapping_meta") or {}
    assert "identity_admission" not in meta


def test_arm_b_the_abstention_is_recorded_as_not_evaluated_never_supported(
    tmp_path: Path,
) -> None:
    """Fail-open must be legible. The mapping report says the pass abstained on
    this row; it must never claim the paper supported it."""
    from t2pw.mapping import identity_admission

    result = run_offline(
        _imported_payload(source_index(SOURCE_WITHOUT_MALONYL)), tmp_path / "cache.json"
    )
    section = result["report"][identity_admission.REPORT_KEY]

    assert section["counts"]["rows_examined"] == 1
    assert section["counts"]["not_evaluated"] == 1
    assert section["counts"]["supported"] == 0
    assert section["counts"]["identifiers_withheld"] == 0


def test_arm_b_the_route_is_named_and_it_is_not_the_evidence_quote_field() -> None:
    """NOT A G9 PROOF -- this one arm is symbol presence, and is called out here
    rather than left to be discovered: it names :func:`provenance_route`, so on
    the base SHA it fails with ``AttributeError`` and proves nothing behavioural.
    The base-failing behavioural arms of section 5 are the two above it and the
    kind-conflict arm below it.

    ``source_refs`` is NOT a route. It holds source ids on a RAG row and
    evidence QUOTES on a Stage-1 row, and the hallucinated ``succinyl-CoA`` of
    ``runs/2026-08-02_2130/.../PMC12180156/strict`` carries a ``source_refs``
    quote its own paper does not contain. Reading that field as provenance would
    walk the hallucination out through the door built for legitimate imports."""
    from t2pw.mapping import identity_admission as ia

    assert ia.provenance_route(IMPORTED_ROW) == "rag_provenance"
    assert ia.provenance_route({"source_papers": [{"source_id": "PMC1"}]}) == "source_papers"
    assert ia.provenance_route({"source_refs": ["ALAS synthesizes dALA from succinyl-CoA"]}) == ""
    assert ia.provenance_route({"rag_confidence": 0.92}) == ""
    assert ia.provenance_route({"rag_provenance": {}, "source_papers": []}) == ""
    assert ia.provenance_route(None) == ""


def test_arm_b_a_route_does_not_rescue_a_row_whose_kind_conflicts(tmp_path: Path) -> None:
    """The route abstains on PASS A only. Pass B needs no source index at all --
    a kind conflict is visible in the payload's own rows -- so an imported row is
    still refused an accession it shares with an entity of the other kind."""
    from t2pw.mapping import identity_admission

    compound = dict(copy.deepcopy(IMPORTED_ROW))
    compound.update({"name": "Pyridoxal 5-phosphate", "class": "compound",
                     "mapped_ids": {"drugbank": "DB00114"}})
    protein = dict(copy.deepcopy(IMPORTED_ROW))
    protein.update({"name": "ALAS2", "class": "protein",
                    "mapped_ids": {"drugbank": "DB00114"}})
    payload = {
        "entities": {"compounds": [compound], "proteins": [protein], "protein_complexes": []},
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index(SOURCE_WITHOUT_MALONYL),
    }
    report = map_ids._admit_identities(payload, payload["entities"])

    assert report["counts"]["kind_conflicts"] == 1
    for row in (payload["entities"]["compounds"][0], payload["entities"]["proteins"][0]):
        assert not (row.get("mapped_ids") or {}).get("drugbank")
        record = row["mapping_meta"][identity_admission.META_KEY]
        assert record["rules"] == [identity_admission.RULE_ACCESSION_KIND_CONFLICT]


# =============================================================================
# 6. ARM C -- a proven alias, read off the SOURCE. BASE-FAILING BEHAVIOURAL PROOF.
# =============================================================================

#: Both rows are verbatim from committed artifacts, and neither carries a synonym
#: that would rescue it: ``pyridoxal phosphate`` from
#: ``runs_verify/2026-08-04_1504/papers/PMC12856317/strict/final_mapped.json``
#: (``synonyms: null``) and ``CoA-SH`` from the 1754 run of the same paper
#: (``synonyms: ["CoA-SH"]``).
SOURCE_SIDE_ALIAS_CASES = [
    pytest.param(
        {"name": "pyridoxal phosphate"},
        "ALAS2 is a pyridoxal 5'-phosphate dependent enzyme of heme biosynthesis.",
        id="locant-between-the-two-words",
    ),
    pytest.param(
        {"name": "CoA-SH", "short_name": "CoA-SH", "synonyms": ["CoA-SH"]},
        "ALAS2 catalyses the condensation of glycine and succinyl-CoA.",
        id="two-character-suffix-is-not-evidence-either-way",
    ),
]


@pytest.mark.parametrize("row_fields,source", SOURCE_SIDE_ALIAS_CASES)
def test_arm_c_an_alias_spelled_on_the_papers_side_is_still_a_proven_alias(
    row_fields: Dict[str, Any], source: str, tmp_path: Path
) -> None:
    """G9 BEHAVIOURAL PROOF, base-failing. Imports nothing this card adds; on the
    base SHA both rows lose all four accessions and this assertion fails.

    The ruling requires that "legitimate aliases must retain valid mappings". The
    six cases C-073 measured were kept because the ROW carried the variant in its
    own ``synonyms``; these two carry none, and the variation sits on the paper's
    side instead -- "pyridoxal 5'-phosphate" for ``pyridoxal phosphate``, and
    "succinyl-CoA" for ``CoA-SH``, whose ``sh`` is two characters and therefore
    already not evidence in either direction. A rule that keeps the first six and
    refuses these two is not applying the ruling; it is applying whether the
    enrichment cache happened to populate ``synonyms``.
    """
    row = {"class": "compound", "mapped_ids": dict(HEME_IDS)}
    row.update(row_fields)
    payload = {
        "entities": {"compounds": [row], "proteins": [], "protein_complexes": []},
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index(source),
    }
    result = run_offline(payload, tmp_path / "cache.json")
    shipped = _ids(result, row["name"])

    assert set(HEME_IDS) <= set(shipped), f"{row['name']} lost a legitimate accession: {shipped}"


def test_arm_c_cannot_bridge_two_real_words_from_different_places(
    tmp_path: Path,
) -> None:
    """The bound that keeps ARM C from being "every word appears somewhere". Only
    material that is ALREADY not evidence -- shorter than the three-character
    minimum, or a bare number -- may sit between two matched words. A source
    naming ``pyridoxal`` in one clause and ``phosphate`` in another, with real
    words in between, is not a source that names ``pyridoxal phosphate``."""
    source = (
        "Pyridoxal was quantified by HPLC. Separately, inorganic phosphate release "
        "was followed by a malachite green assay."
    )
    payload = {
        "entities": {
            "compounds": [{"name": "pyridoxal phosphate", "class": "compound",
                           "mapped_ids": dict(HEME_IDS)}],
            "proteins": [], "protein_complexes": [],
        },
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index(source),
    }
    result = run_offline(payload, tmp_path / "cache.json")
    assert not _ids(result, "pyridoxal phosphate").get("kegg")


def test_arm_c_a_short_name_the_source_happens_to_contain_still_rescues_nothing(
    tmp_path: Path,
) -> None:
    """C-073's adversarial arm, restated against ARM C. A two-character hit is a
    coincidence, not a citation, and the token rule must not smuggle one in: the
    row's only long name is absent, and its ``Zg`` is below the minimum."""
    payload = {
        "entities": {
            "compounds": [{"name": "Zorbaline glucoside", "short_name": "Zg",
                           "class": "compound", "mapped_ids": dict(HEME_IDS)}],
            "proteins": [], "protein_complexes": [],
        },
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index("The Zg fraction was resolved by size exclusion."),
    }
    result = run_offline(payload, tmp_path / "cache.json")
    assert not _ids(result, "Zorbaline glucoside").get("kegg")


def test_arm_c_is_strictly_additive_to_substring_containment() -> None:
    """The property that makes ARM C safe to add at all: it is consulted only
    after substring containment has already failed, so it can turn a refusal into
    a keep and never the other way round. Nothing C-073 supported can stop being
    supported."""
    from t2pw.mapping import identity_admission as ia

    index = ia.SourceIndex(_fold(SOURCE_WITH_SUCCINYL))
    for candidate in ("succinyl-CoA", "heme", "ferrochelatase", "protoporphyrin IX",
                      "glycine", "SFXN4", "5-aminolevulinate"):
        needle = ia.normalize_text(candidate)
        substring = needle in index.normalized or needle.replace(" ", "") in index.squashed
        if substring:
            assert index.contains(candidate), candidate


# =============================================================================
# 7. fail open, and withhold != delete
# =============================================================================


def test_fail_open_no_index_means_not_evaluated_never_unsupported(
    tmp_path: Path,
) -> None:
    """PRODUCT_CONTRACT section 8, and the property that keeps every unit test,
    the ``interactive_curator`` path and every legacy payload working: without the
    paper the question cannot be asked, so nothing is withheld and the report
    stays silent rather than claiming a clean bill of health."""
    from t2pw.mapping import identity_admission

    result = run_offline(heme_payload(), tmp_path / "cache.json")

    assert _ids(result, "succinyl-CoA").get("kegg") == SUCCINYL_IDS["kegg"]
    assert identity_admission.REPORT_KEY not in result["report"]


def test_fail_open_an_empty_index_reads_not_evaluated(tmp_path: Path) -> None:
    """Offered but unusable is the case that MUST speak: the section appears and
    says ``not_evaluated``, so a reader can tell fail-open from support."""
    from t2pw.mapping import identity_admission

    payload = heme_payload({"schema_version": INDEX_SCHEMA_VERSION, "length": 0,
                            "normalized": ""})
    result = run_offline(payload, tmp_path / "cache.json")
    section = result["report"][identity_admission.REPORT_KEY]

    assert section["source_index"]["status"] == identity_admission.STATUS_NOT_EVALUATED
    assert section["source_index"]["reason"] == identity_admission.NOT_EVALUATED_EMPTY_INDEX
    assert section["counts"]["identifiers_withheld"] == 0
    assert _ids(result, "succinyl-CoA").get("kegg") == SUCCINYL_IDS["kegg"]


def test_an_armed_leg_withholds_and_never_deletes(tmp_path: Path) -> None:
    """MERGE RULE 7, through the armed chain. The row keeps its name, its class,
    its synonyms and its place in the reaction; the identifiers are FILED under
    ``rejected_mapped_ids``, not dropped, so the refusal is reviewable."""
    result = armed_chain(heme_payload(), tmp_path / "cache.json",
                         source_text=SOURCE_WITHOUT_SUCCINYL)
    row = _row(result, "succinyl-CoA")

    assert row["name"] == "succinyl-CoA"
    assert row["class"] == "compound"
    assert row["synonyms"] == ["Succinyl-CoA", "Succinyl coenzyme A"]
    rejected = (row.get("mapping_meta") or {}).get("rejected_mapped_ids") or {}
    assert rejected["kegg"] == SUCCINYL_IDS["kegg"]
    assert set(REAL_NAMESPACES) <= set(rejected)

    reactions = result["payload"]["processes"]["reactions"]
    assert len(reactions) == 1
    assert "succinyl-CoA" in reactions[0]["inputs"]


def test_an_armed_leg_records_the_refusal_in_the_report_and_in_lineage(
    tmp_path: Path,
) -> None:
    """PRODUCT_CONTRACT section 3, through the armed chain rather than a
    hand-built index: the decision is attributable in both places a reviewer
    looks."""
    from t2pw.mapping import identity_admission
    from t2pw.pipeline.lineage import read as read_lineage

    result = armed_chain(heme_payload(), tmp_path / "cache.json",
                         source_text=SOURCE_WITHOUT_SUCCINYL)

    section = result["report"][identity_admission.REPORT_KEY]
    assert section["counts"]["rows_withheld"] == 1
    assert section["withheld"][0]["name"] == "succinyl-CoA"
    assert section["withheld"][0]["rule"] == identity_admission.RULE_NOT_SUPPORTED

    entries = [
        entry for entry in read_lineage(_row(result, "succinyl-CoA")).entries
        if entry.stage == "identifier_mapping"
    ]
    assert entries, "no identifier_mapping attribution on the withheld row"
    assert entries[-1].origin == "excluded"
    assert entries[-1].review_required is True
    assert identity_admission.RULE_NOT_SUPPORTED in entries[-1].reason


# =============================================================================
# 8. full-corpus replay -- the load-bearing measurement
# =============================================================================

#: Every ``NOT_SUPPORTED`` refusal the armed pass makes over the whole committed
#: corpus, as ``(paper directory, leg, entity name)``. All three are the same
#: defect class: a compound whose name occurs ZERO times in its own paper, on a
#: leg that reported PASS, carrying real accessions and no retrieval route.
#: PMC12180156's source is 67,553 characters and contains neither "succinyl" nor
#: "protoporphyrin".
EXPECTED_CATCHES = {
    ("PMC12180156", "strict", "succinyl-CoA"),
    ("PMC12180156", "research", "succinyl-CoA"),
    ("PMC12180156", "research", "protoporphyrin IX"),
}


def _committed_artifacts() -> List[Path]:
    listing = subprocess.run(
        ["git", "ls-files", "*final_mapped.json"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    return [ROOT / rel for rel in listing.stdout.split()]


def test_corpus_the_armed_pass_over_every_committed_artifact_and_its_own_paper() -> None:
    """THE LOAD-BEARING EVIDENCE (card section 6.5).

    ``test_c073_identity_admission.py::test_the_whole_committed_corpus_yields_one_conflict_and_no_collateral``
    replays the same corpus with NO source index, which exercises pass B alone.
    This is the other half and the one arming makes real: every committed
    ``final_mapped.json`` replayed against its OWN ``01_source_text.txt``.

    Measured at this tip: 70 artifacts, 678 eligible rows, 3 refusals, all three
    of them hallucinations, 0 collateral. Measured with the ruling's two routes
    removed -- which is what the C-073 predicate does on its own -- 39 refusals,
    36 of them legitimate rows. That difference is this card's whole argument.

    If a newly committed run artifact moves these numbers, TRIAGE the new row as
    a catch or as collateral before moving the pin. Do not regenerate it.
    """
    from t2pw.mapping import identity_admission

    artifacts = _committed_artifacts()
    if not artifacts:
        pytest.skip("no committed final_mapped.json artifacts")

    examined = 0
    catches: set = set()
    for path in artifacts:
        source = path.parent.parent / "01_source_text.txt"
        payload = json.loads(path.read_text(encoding="utf-8"))
        entities = payload.get("entities") or {}
        if source.exists():
            payload[identity_admission.SOURCE_INDEX_KEY] = identity_admission.build_source_index(
                source.read_text(encoding="utf-8", errors="replace")
            )
        report = map_ids._admit_identities(payload, entities)
        examined += report["counts"]["rows_examined"]
        for entry in report["withheld"]:
            if entry["rule"] == identity_admission.RULE_NOT_SUPPORTED:
                catches.add((path.parent.parent.name.split("__")[0], path.parent.name,
                             entry["name"]))

    assert len(artifacts) >= 70
    assert examined >= 678
    assert catches == EXPECTED_CATCHES, (
        f"the armed pass changed over the corpus ({len(artifacts)} artifacts). "
        f"unexpected: {sorted(catches - EXPECTED_CATCHES)}; "
        f"no longer caught: {sorted(EXPECTED_CATCHES - catches)}"
    )


def test_corpus_every_row_the_rag_route_imported_keeps_every_accession() -> None:
    """THE ANTI-COLLATERAL ARM, at corpus scale. 32 refusals disappear when the
    ruling's provenance route is honoured -- 30 of them naming another document,
    2 naming ``seed_paper`` (section 11). Not one row that carries a route may
    lose an accession, whichever document the route points at."""
    from t2pw.mapping import identity_admission as ia

    artifacts = _committed_artifacts()
    if not artifacts:
        pytest.skip("no committed final_mapped.json artifacts")

    imported = 0
    for path in artifacts:
        source = path.parent.parent / "01_source_text.txt"
        if not source.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        entities = payload.get("entities") or {}
        payload[ia.SOURCE_INDEX_KEY] = ia.build_source_index(
            source.read_text(encoding="utf-8", errors="replace")
        )
        before: Dict[Tuple[str, str], Dict[str, str]] = {}
        rows = map_ids._identity_admission_rows(entities)
        for bucket, row in rows:
            if ia.provenance_route(row):
                before[(bucket, str(row.get("name")))] = dict(ia.external_accessions(row))
        if not before:
            continue
        imported += len(before)
        map_ids._admit_identities(payload, entities)
        for bucket, row in map_ids._identity_admission_rows(entities):
            key = (bucket, str(row.get("name")))
            if key in before and before[key]:
                assert dict(ia.external_accessions(row)) == before[key], (
                    f"{key} lost accessions in {path.relative_to(ROOT)}: collateral on a row "
                    f"imported from {row.get('rag_provenance')}"
                )

    assert imported >= 100, f"the corpus no longer exercises the route: {imported} rows"


def test_corpus_the_index_is_the_paper_and_nothing_is_added_to_it() -> None:
    """PAYLOAD COST, stated rather than assumed (card section 4.5). The index is
    one normalized copy of the paper: never larger than the paper it folds, and
    it adds no second structure. Measured over the corpus the mean blob is
    ~56 KB per leg against a mean source of ~58 KB.

    It is NOT reduced further, and the reason is that every smaller form measured
    either weakens the match or is bigger. A distinct word-n-gram set is LARGER
    than the text for a non-repetitive paper; a bag of distinct words loses the
    adjacency ``succinyl coa`` needs; dropping the space-separated fold and
    keeping only the space-free one saves ~17 percent but changes the stored
    shape, which means a schema bump and a dual-version reader in a module whose
    unknown-version branch exists to fail open. Correctness outranks size.
    """
    from t2pw.mapping import identity_admission as ia

    sources = sorted({
        path.parent.parent / "01_source_text.txt" for path in _committed_artifacts()
    })
    sources = [path for path in sources if path.exists()]
    if not sources:
        pytest.skip("no committed source texts")

    for path in sources:
        raw = path.read_text(encoding="utf-8", errors="replace")
        index = ia.build_source_index(raw)
        assert set(index) == {"schema_version", "length", "normalized"}
        assert index["length"] == len(raw)
        assert len(index["normalized"]) <= len(raw)
        assert len(json.dumps(index, ensure_ascii=False)) <= len(raw) + 128


# =============================================================================
# 9. the audit prompt must not carry the paper (C-075 correction round 1)
# =============================================================================
#
# Arming the merge site puts a ~56 KB normalized copy of the paper on
# ``final_payload``, and the audit loop json.dumps THE WHOLE PAYLOAD into its
# user message (``audit_json_llm._build_llm_prompt``, called at :1409 with the
# raw payload; ``strip_payload_for_interactive_context`` is not on that path at
# all -- it is only reachable from ``interactive_curator.py:188`` and ``:255``).
# Unfixed, every audit round of every leg would carry the paper, risking prompt
# truncation and degraded audit quality across the whole run: a self-inflicted
# regression worse than the defect being fixed.
#
# The fix is at the SERIALIZATION, never at the payload. The payload must keep
# the index, because the AUTHORITATIVE mapping run happens AFTER the audit
# (``streamlit_app.py:4225``) and a payload that reached it without an index
# would fail open and withhold nothing.


def test_the_audit_prompt_does_not_carry_the_source_index() -> None:
    """(a) of the correction: the blob is gone from the prompt, and the payload
    the auditor still needs to see is all still there."""
    from t2pw.curation import audit_json_llm

    payload = heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL))
    prompt = audit_json_llm._build_llm_prompt(payload)

    assert SOURCE_INDEX_KEY not in prompt
    assert _fold(SOURCE_WITHOUT_SUCCINYL)[:40] not in prompt
    # ... and the structure the auditor actually judges is untouched
    assert "succinyl-CoA" in prompt and "ALAS2 condensation" in prompt
    assert "entities" in prompt and "processes" in prompt


def test_the_audit_prompt_shrinks_by_the_whole_index_and_no_more() -> None:
    """The saving is exactly the blob. Measured over the committed corpus the
    index is ~56 KB per leg; here the fixture's is small, but the property that
    matters is that the delta is the index's own serialized size and nothing
    else disappeared with it."""
    from t2pw.curation import audit_json_llm

    without = heme_payload()
    with_index = heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL))

    bare = audit_json_llm._build_llm_prompt(without)
    armed = audit_json_llm._build_llm_prompt(with_index)
    assert armed == bare, "the armed leg's prompt is no longer byte-identical to today's"


def test_payload_for_prompt_never_mutates_and_never_rebuilds_needlessly() -> None:
    """The half that keeps mapping working. It filters a COPY, and a payload with
    nothing to filter is returned as the very same object -- so no existing
    prompt changes by one byte and no caller's payload is edited underneath it."""
    from t2pw.curation import audit_json_llm

    payload = heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL))
    seen = audit_json_llm.payload_for_prompt(payload)

    assert SOURCE_INDEX_KEY not in seen
    assert SOURCE_INDEX_KEY in payload, "the caller's payload was edited"
    assert seen["entities"] is payload["entities"], "the payload was needlessly rebuilt"

    bare = heme_payload()
    assert audit_json_llm.payload_for_prompt(bare) is bare
    assert audit_json_llm.payload_for_prompt("not-a-dict") == "not-a-dict"


def test_the_audited_payload_still_carries_the_index_and_mapping_still_refuses(
    tmp_path: Path,
) -> None:
    """(b) of the correction, end to end. Through the real ``audit_payload`` with
    the LLM mocked: the prompt the model saw has no index, the payload that comes
    out the other side still has one, and ``map_payload`` -- which in production
    runs AFTER the audit -- still withholds ``succinyl-CoA``.

    That last assertion is the one that would catch a "fix" that stripped the key
    from the payload instead of from the serialization: mapping would fail open
    and this leg would ship all eight fabricated accessions again."""
    from t2pw.curation import audit_json_llm

    payload = armed_payload_from_merge()
    prompts: List[str] = []

    def _chat(messages, **_kwargs):
        prompts.append(messages[-1]["content"])
        return json.dumps({"issues": {}, "patch": []})

    with patch.object(audit_json_llm, "chat", _chat):
        result = audit_json_llm.audit_payload(payload, use_llm=True)

    assert prompts, "the audit never called the model"
    assert SOURCE_INDEX_KEY not in prompts[0]
    assert isinstance(result["report"], dict)

    # the payload survived the audit intact -- index and all
    assert payload[SOURCE_INDEX_KEY]["length"] == len(SOURCE_WITHOUT_SUCCINYL)

    mapped = run_offline(payload, tmp_path / "cache.json")
    assert not _ids(mapped, "succinyl-CoA").get("kegg")
    assert _ids(mapped, "heme") == HEME_IDS


# =============================================================================
# 10. the two properties a reviewer will attack
# =============================================================================


def test_both_new_clauses_are_strictly_additive_over_the_whole_corpus() -> None:
    """THE PROPERTY. Neither clause can CREATE a refusal or new collateral.

    Run over every committed artifact twice: once with both clauses neutralized
    -- which is exactly the C-073 predicate -- and once as shipped. Every row's
    surviving accession set at tip must be a SUPERSET of its set under C-073.
    A single accession present under C-073 and absent here would mean one of the
    clauses had taken something away, which is the only way either could do harm.

    Stated per row rather than per corpus on purpose: a count could stay equal
    while one row lost an accession and another gained one."""
    from t2pw.mapping import identity_admission as ia

    artifacts = _committed_artifacts()
    if not artifacts:
        pytest.skip("no committed final_mapped.json artifacts")

    def _survivors(payload: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, str]]:
        entities = payload.get("entities") or {}
        map_ids._admit_identities(payload, entities)
        return {
            (bucket, str(row.get("name"))): dict(ia.external_accessions(row))
            for bucket, row in map_ids._identity_admission_rows(entities)
        }

    compared = 0
    for path in artifacts:
        source = path.parent.parent / "01_source_text.txt"
        if not source.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw[ia.SOURCE_INDEX_KEY] = ia.build_source_index(
            source.read_text(encoding="utf-8", errors="replace")
        )
        with patch.object(ia, "PROVENANCE_ROUTE_KEYS", ()), patch.object(
            ia.SourceIndex, "names_in_one_span", lambda self, needle: False
        ):
            before = _survivors(copy.deepcopy(raw))
        after = _survivors(copy.deepcopy(raw))

        for key, kept in before.items():
            compared += 1
            assert set(kept) <= set(after.get(key, {})), (
                f"{key} in {path.relative_to(ROOT)} LOST "
                f"{sorted(set(kept) - set(after.get(key, {})))} -- a clause that was "
                f"supposed to only keep took something away"
            )
    assert compared >= 678, f"the additivity check no longer covers the corpus: {compared}"


def test_source_refs_is_not_a_route_read_off_the_real_artifact() -> None:
    """THE EVIDENCE for excluding ``source_refs``, taken from the artifact rather
    than from a literal. The hallucinated ``succinyl-CoA`` of
    ``runs/2026-08-02_2130/papers/PMC12180156/strict`` carries a ``source_refs``
    entry -- and it is an evidence QUOTE, not a source id, and its own paper does
    not contain it. Had ``source_refs`` been read as a provenance route, this row
    would have walked out through the door built for legitimate imports."""
    from t2pw.mapping import identity_admission as ia

    artifact = ROOT / "runs/2026-08-02_2130/papers/PMC12180156/strict/final_mapped.json"
    source = ROOT / "runs/2026-08-02_2130/papers/PMC12180156/01_source_text.txt"
    if not artifact.exists() or not source.exists():
        pytest.skip(f"run directory absent: {artifact}")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    row = next(
        r for r in payload["entities"]["compounds"]
        if isinstance(r, dict) and r.get("name") == "succinyl-CoA"
    )
    refs = row.get("source_refs") or []

    assert refs and not str(refs[0]).startswith("PMC"), "source_refs here is a quote, not an id"
    assert ia.provenance_route(row) == "", "the quote was read as a provenance route"
    # and the quote itself is not in the paper it claims to come from
    folded = ia.normalize_text(source.read_text(encoding="utf-8", errors="replace"))
    assert "succinyl" not in folded

    payload[ia.SOURCE_INDEX_KEY] = ia.build_source_index(
        source.read_text(encoding="utf-8", errors="replace")
    )
    report = map_ids._admit_identities(payload, payload["entities"])
    assert any(
        entry["name"] == "succinyl-CoA"
        and entry["rule"] == ia.RULE_NOT_SUPPORTED
        for entry in report["withheld"]
    ), "the hallucination survived"


# =============================================================================
# 11. the self-referential route (C-075 correction round 2)
# =============================================================================
#
# THE CASE THE REVIEWER FOUND, AND IT DESERVED A NAME. ``provenance_route`` is
# blind to the route's VALUE, so two corpus rows are excused on a route whose
# ``source_id`` is ``"seed_paper"`` -- a route that names this very paper. Read
# literally, the ruling's "another permitted provenance route" does not cover
# that, so the abstention is a DECISION and these arms are what make it one
# instead of an accident.
#
# THE DECISION IS (b): a self-referential route still abstains. The reasoning is
# on ``identity_admission.SELF_REFERENTIAL_ROUTE_SOURCE`` in full; the short form
# is that the mark establishes who WROTE the name -- the RAG synthesizer, out of
# a pathway record's vocabulary -- not which document is the row's alibi. Both
# measured rows are synonym misses, and refusing them would be exactly the
# collateral this card exists to avoid.
#
# ``seed_paper`` IS NOT A LICENCE. Nothing in the module reads the value, these
# rows acquire no identifier they did not already have, ``not_evaluated`` is not
# ``supported``, and pass B still refuses them anything a kind conflict would.
# The last two arms below pin those three limits so a future reader cannot mistake
# the abstention for permission.

#: The two rows, verbatim from committed artifacts, with the paper's own spelling
#: of each beside it. Neither row offers a synonym; each offers one name.
SEED_PAPER_ROUTE_CASES = [
    pytest.param(
        "compounds",
        {
            "name": "α-ketoglutarate",
            "class": "compound",
            "rag_provenance": {
                "source_id": "seed_paper",
                "source_title": "menaquinone biosynthesis",
                "source_type": "paper",
            },
            "mapped_ids": dict(HEME_IDS),
        },
        "MenD condenses 2-oxoglutarate with isochorismate in menaquinone biosynthesis.",
        id="alpha-ketoglutarate-vs-2-oxoglutarate",
    ),
    pytest.param(
        "proteins",
        {
            "name": "pmrCAB operon",
            "class": "protein",
            "rag_provenance": {
                "source_id": "seed_paper",
                "source_title": "lipid A modification (colistin resistance)",
                "source_type": "paper",
                "section": "introduction",
            },
            "mapped_ids": {"uniprot": "P30843"},
        },
        "Activation of pmrA drives phosphoethanolamine addition to lipid A.",
        id="pmrcab-operon-vs-pmra",
    ),
]


@pytest.mark.parametrize("bucket,row_fields,source", SEED_PAPER_ROUTE_CASES)
def test_a_self_referential_route_still_abstains(
    bucket: str, row_fields: Dict[str, Any], source: str, tmp_path: Path
) -> None:
    """DECISION (b), pinned. The route names this paper, the paper does not name
    the row, and the row keeps its identifiers anyway -- because the name is the
    synthesizer's spelling of something the paper really does discuss under
    another one. Measured in the real sources: PMC12312563 writes
    ``2-oxoglutarate`` 8 times and ``ketoglutarate`` 0 times; PMC13278307 writes
    ``pmrA`` 4 times and ``pmrCAB`` 0 times.

    Were this to flip, ``α-ketoglutarate`` would lose 9 real accessions and
    ``pmrCAB operon`` its ``uniprot`` -- collateral on a synonym miss, which is
    worse than the defect the pass exists to fix."""
    payload = {
        "entities": {"compounds": [], "proteins": [], "protein_complexes": []},
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index(source),
    }
    row = copy.deepcopy(row_fields)
    payload["entities"][bucket] = [row]
    # The two passes directly, as C-073's replay arm does it: an offline
    # ``map_payload`` re-runs the protein ladder, which has its own opinion about
    # an unresolvable UniProt accession, and this assertion is about THIS pass.
    report = map_ids._admit_identities(payload, payload["entities"])
    shipped = row.get("mapped_ids") or {}

    assert set(row_fields["mapped_ids"]) <= set(shipped), (
        f"{row_fields['name']} lost accessions on a synonym miss: {shipped}"
    )
    assert report["counts"]["identifiers_withheld"] == 0
    assert report["counts"]["not_evaluated"] == 1


def test_the_route_value_is_never_read_so_seed_paper_is_not_a_special_case() -> None:
    """The route's VALUE is not what excuses the row -- the abstention is. The
    predicate never inspects ``source_id``, so ``seed_paper`` gets no treatment a
    retrieved document does not, and the constant that names it is documentation,
    not a branch."""
    from t2pw.mapping import identity_admission as ia

    seed = {"rag_provenance": {"source_id": ia.SELF_REFERENTIAL_ROUTE_SOURCE}}
    other = {"rag_provenance": {"source_id": "PMC12898747"}}
    nameless = {"rag_provenance": {"section": "introduction"}}

    assert ia.provenance_route(seed) == "rag_provenance"
    assert ia.provenance_route(other) == "rag_provenance"
    assert ia.provenance_route(nameless) == "rag_provenance"
    # The value is documented, never dispatched on. Asked of the AST rather than
    # of the text, so the prose that explains the constant does not read as a use
    # of it: the only binding occurrence may be the assignment itself.
    import ast

    tree = ast.parse(Path(ia.__file__).read_text(encoding="utf-8"))
    loads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "SELF_REFERENTIAL_ROUTE_SOURCE"
        and isinstance(node.ctx, ast.Load)
    ]
    assert not loads, (
        f"the constant is READ at line(s) {[n.lineno for n in loads]} -- it is meant to be "
        f"documentation, not a branch, so a self-referential route gets no special case"
    )


def test_an_abstaining_row_acquires_nothing_and_is_never_called_supported(
    tmp_path: Path,
) -> None:
    """THE THREE LIMITS on the abstention, in one arm. A routed row (a) gains no
    identifier it did not arrive with, (b) is reported ``not_evaluated`` and
    never ``supported``, and (c) is still refused by pass B, which needs no
    source index and does not consult the route at all."""
    from t2pw.mapping import identity_admission as ia

    compound = {
        "name": "Pyridoxal 5-phosphate", "class": "compound",
        "rag_provenance": {"source_id": ia.SELF_REFERENTIAL_ROUTE_SOURCE},
        "mapped_ids": {"drugbank": "DB00114"},
    }
    protein = {
        "name": "ALAS2", "class": "protein",
        "rag_provenance": {"source_id": ia.SELF_REFERENTIAL_ROUTE_SOURCE},
        "mapped_ids": {"drugbank": "DB00114"},
    }
    lone = {
        "name": "α-ketoglutarate", "class": "compound",
        "rag_provenance": {"source_id": ia.SELF_REFERENTIAL_ROUTE_SOURCE},
        "mapped_ids": dict(HEME_IDS),
    }
    payload = {
        "entities": {"compounds": [compound, lone], "proteins": [protein],
                     "protein_complexes": []},
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index("This paper names none of those three."),
    }
    report = map_ids._admit_identities(payload, payload["entities"])

    # (b) the pass abstained on all three and claimed support for none
    assert report["counts"]["supported"] == 0
    assert report["counts"]["not_evaluated"] == 3
    # (c) pass B is unmoved by the route
    assert report["counts"]["kind_conflicts"] == 1
    assert not (compound.get("mapped_ids") or {}).get("drugbank")
    assert not (protein.get("mapped_ids") or {}).get("drugbank")
    # (a) and nothing was gained anywhere
    assert lone["mapped_ids"] == HEME_IDS
