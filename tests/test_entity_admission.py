"""C-060 / R-003 mechanism M1 -- the assay-reagent / assay-reporter admission gate.

Two kinds of test live here and each is labelled in its own name (PACK2-SHARED S5):

``test_g9_*``  -- **G9 behavioural base proofs.** They call the PRE-EXISTING
    ``merge_additions`` with its PRE-EXISTING two-positional-argument signature and
    assert on observed payload content. At the dispatch base SHA they fail with an
    ``AssertionError`` about rows that are present and must not be -- never on an
    import, never on a missing symbol. This is pre-existing observable behaviour
    that C-060 corrects (R-003 F1, F3).

``test_new_acceptance_*`` -- **explicitly labelled new acceptance tests.** The gate
    module, its removal ledger, its A0-C5 ordering and its advisory second phase are
    new capability. No base failure is claimed or fabricated for them.

Fixtures are transcribed from R-003 section 4 (F1.6, F3/F4.6, F7.6).
"""

import copy
import json

import pytest

from t2pw.pipeline.cofactor_policy import PathwayContext, classify_entity
from t2pw.pipeline.pipeline import merge_additions


def _gate():
    """Import the gate lazily so the G9 tests above collect and RUN at base."""
    from t2pw.pipeline import entity_admission

    return entity_admission


def _names(payload, bucket="compounds"):
    return [row.get("name") for row in payload.get("entities", {}).get(bucket, [])]


def _reaction_names(payload):
    return [row.get("name") for row in payload.get("processes", {}).get("reactions", [])]


# --------------------------------------------------------------------------- F1
# R-003 4.F1.6 -- hemin is the reagent form of heme; the payload asserts two
# distinct species doing the same thing, each with its own duplicated edge.

def _f1_payload():
    return {
        "entities": {
            "compounds": [{"name": "heme"}, {"name": "hemin"}],
            "proteins": [{"name": "ALAS2"}],
        },
        "processes": {
            "interactions": [
                {"entity_1": "heme", "entity_2": "ALAS2", "relationship": "binds_to"},
                {"entity_1": "hemin", "entity_2": "ALAS2", "relationship": "binds_to"},
            ]
        },
    }


# ------------------------------------------------------------------------ F3/F4
# R-003 4.F3/F4.6 -- the Stage-2 merge synthesized a coupled-assay composite that
# fuses EntB's hydrolase chemistry with the porcine LDH readout.

ASSAY_SPAN = (
    "LDH-catalyzed conversion of pyruvate to lactate is then monitored by loss of "
    "OD 340 following oxidation of NADH to NAD+"
)


def _f34_base():
    return {
        "entities": {
            "compounds": [
                {"name": "isochorismate"},
                {"name": "2,3-diDHB"},
                {"name": "pyruvate"},
                {"name": "NADH"},
            ],
            "proteins": [{"name": "EntB"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "EntB isochorismatase reaction",
                    "inputs": ["isochorismate"],
                    "outputs": ["2,3-diDHB", "pyruvate"],
                    "locked_reaction_id": "rxn_lock_002",
                }
            ]
        },
    }


def _f34_additions():
    return {
        "additions": {
            "entities": {"compounds": [{"name": "NAD", "provenance": "inferred"}]},
            "processes": {
                "reactions": [
                    {
                        "name": "EntB isochorismatase reaction (NADH-dependent coupled assay)",
                        "inputs": ["isochorismate", "NADH"],
                        "outputs": ["2,3-diDHB", "pyruvate", "NAD"],
                        "evidence": ASSAY_SPAN,
                    }
                ]
            },
        }
    }


ENTEROBACTIN = PathwayContext("enterobactin biosynthesis", "Escherichia coli")

# --------------------------------------------------------------------------- F7
# R-003 4.F7.6 -- the A3 ordering probe. Zero occurrences of "succinyl" in the seed.

SUCCINYL_SPAN = (
    "ALAS synthesizes the non-proteinogenic dALA from succinyl-coenzyme A "
    "(succinyl-CoA) and the proteinogenic amino acid glycine"
)
SEED_WITHOUT_SUCCINYL = (
    "SFXN4 inhibits ALAS2 translation. SFXN4 modulates ferrochelatase levels. "
    "SFXN1 supplies glycine to the mitochondrial matrix."
)
HEME = PathwayContext("heme biosynthesis", "Homo sapiens")
NO_ADMITTED_CLAIM = {"accepted": [], "counts": {"accepted": 0, "considered": 1}}


def _f7_base():
    return {"entities": {"compounds": [{"name": "glycine"}]}, "processes": {"reactions": []}}


def _f7_additions():
    return {
        "additions": {
            "entities": {
                "compounds": [
                    {
                        "name": "succinyl-CoA",
                        "provenance": "inferred",
                        "source_refs": [SUCCINYL_SPAN],
                    }
                ]
            },
            "processes": {
                "reactions": [
                    {
                        "name": "ALAS2 reaction",
                        "inputs": ["glycine", "succinyl-CoA"],
                        "outputs": ["d-aminolevulinic acid"],
                        "evidence": SUCCINYL_SPAN,
                    }
                ]
            },
        }
    }


# =========================================================== G9 behavioural base
# Both call merge_additions POSITIONALLY, exactly as the base SHA accepts it.

def test_g9_f1_hemin_reagent_duplicate_does_not_survive_the_merge():
    """A1 / R-003 F1. The reagent form of a species already present is not a
    second pathway node, and it does not bring a duplicate edge with it."""
    merged = merge_additions(_f1_payload(), {})

    assert _names(merged) == ["heme"], "hemin survived as a second heme-family node"
    edges = merged["processes"]["interactions"]
    assert len(edges) == 1, f"duplicated binds_to edge survived: {edges}"
    assert edges[0]["entity_1"] == "heme"


def test_g9_f3_assay_composite_reaction_does_not_survive_the_merge():
    """A2 (first half) / R-003 F3. The coupled-assay composite duplicates the
    genuine EntB reaction and fuses a second enzyme's chemistry into it."""
    merged = merge_additions(_f34_base(), _f34_additions())

    assert _reaction_names(merged) == ["EntB isochorismatase reaction"], (
        "the synthesized assay composite survived the Stage-2 merge"
    )


# ================================================== new acceptance (no base claim)

def test_new_acceptance_a2_nad_species_orphaned_by_the_composite_are_removed():
    """A2 (second half). nadh / nad+ do not survive as pathway members. The rule
    that fires is the CURRENCY verdict plus 'not the subject of the requested
    pathway' -- ``assay_reporter`` never fires for these, and is not relied on."""
    ea = _gate()
    assert classify_entity("NADH", ENTEROBACTIN).verdict == ea.COFACTOR_OR_CURRENCY
    assert classify_entity("NAD", ENTEROBACTIN).verdict == ea.COFACTOR_OR_CURRENCY

    merged = merge_additions(_f34_base(), _f34_additions(), pathway_context=ENTEROBACTIN)

    assert _reaction_names(merged) == ["EntB isochorismatase reaction"]
    assert _names(merged) == ["isochorismate", "2,3-diDHB", "pyruvate"]
    ledger = merged[ea.LEDGER_KEY]
    fired = {entry["entity"]: entry for entry in ledger["removed"]}
    assert fired["NADH"]["rule"] == ea.RULE_CURRENCY_NOT_SUBJECT
    assert fired["NADH"]["phase"] == ea.PHASE_ADVISORY
    assert fired["NAD"]["rule"] == ea.RULE_CURRENCY_NOT_SUBJECT
    assert all(e["rule"] != "assay_reporter_not_pathway_member" for e in ledger["removed"])


def test_new_acceptance_a3_a_participant_verdict_cannot_protect_a_hallucinated_row():
    """A3 -- THE ORDERING PROOF. cofactor_policy returns ``participant`` / ``high``
    for succinyl-CoA in heme biosynthesis (R-003 measured exactly this against the
    merged C-018 classifier). Consulting it first would PROTECT the run's worst
    fabrication. The hallucination gate runs first, so the row goes anyway."""
    ea = _gate()
    protective = classify_entity("succinyl-CoA", HEME)
    assert (protective.verdict, protective.confidence) == ("participant", "high")
    assert protective.reason == "subject_of_the_requested_pathway"
    assert "succinyl" not in SEED_WITHOUT_SUCCINYL.casefold()

    merged = merge_additions(
        _f7_base(),
        _f7_additions(),
        seed_text=SEED_WITHOUT_SUCCINYL,
        rag_admission_report=NO_ADMITTED_CLAIM,
        pathway_context=HEME,
    )

    assert _names(merged) == ["glycine"]
    assert _reaction_names(merged) == []
    removed = merged[ea.LEDGER_KEY]["removed"]
    entries = [e for e in removed if e["entity"] == "succinyl-CoA" and e["kind"] == "compound"]
    assert len(entries) == 1
    assert entries[0]["rule"] == ea.RULE_UNLOCATABLE_EVIDENCE
    assert entries[0]["phase"] == ea.PHASE_HALLUCINATION, (
        "succinyl-CoA must be removed by the hallucination gate, never reach phase 2"
    )
    assert entries[0]["evidence"] == SUCCINYL_SPAN
    # the reaction that carried the same span goes with it, in the same phase,
    # and is audited in its own right -- no removal is silent.
    assert [(e["rule"], e["phase"]) for e in removed if e["kind"] == "reaction"] == [
        (ea.RULE_REFERENCES_REMOVED_ROW, ea.PHASE_HALLUCINATION)
    ]
    assert all(e["phase"] == ea.PHASE_HALLUCINATION for e in removed)


def test_new_acceptance_a3_phase_one_never_consults_the_classifier(monkeypatch):
    """A3 by construction. Replace the classifier with one that returns
    ``participant`` / ``high`` for EVERYTHING -- the most protective verdict that
    exists -- and the row is still removed. Phase 1 never calls it at all."""
    ea = _gate()
    seen = []

    def _maximally_protective(name, context):
        seen.append(str(name))
        return classify_entity("succinyl-CoA", HEME)

    monkeypatch.setattr(ea, "classify_entity", _maximally_protective)
    kept, ledger = ea.screen_additions(
        _merged_f7_shape(),
        seed_text=SEED_WITHOUT_SUCCINYL,
        admission_report=NO_ADMITTED_CLAIM,
        context=HEME,
    )

    assert _names(kept) == ["glycine"]
    assert "succinyl-CoA" not in seen, "phase 1 consulted cofactor_policy on the removed row"
    assert {e["phase"] for e in ledger["removed"] if e["entity"] == "succinyl-CoA"} == {
        ea.PHASE_HALLUCINATION
    }
    assert seen == ["glycine"], "phase 2 saw only the survivors"


def _merged_f7_shape():
    """The F7 rows as they look AFTER the merge -- the gate's real input."""
    payload = _f7_base()
    adds = _f7_additions()["additions"]
    payload["entities"]["compounds"].extend(adds["entities"]["compounds"])
    payload["processes"]["reactions"].extend(adds["processes"]["reactions"])
    return payload


def test_new_acceptance_a4_every_removal_is_auditable_in_the_ledger():
    """A4. A silent removal is a reject: entity, rule and evidence, per row."""
    ea = _gate()
    merged = merge_additions(_f1_payload(), {})
    ledger = merged[ea.LEDGER_KEY]

    assert ledger["schema_version"] == ea.SCHEMA_VERSION
    assert {e["entity"] for e in ledger["removed"]} == {"hemin"} | {
        e["entity"] for e in ledger["removed"] if e["kind"] == "interaction"
    }
    hemin = next(e for e in ledger["removed"] if e["entity"] == "hemin")
    assert hemin["rule"] == ea.RULE_REAGENT_DUPLICATE
    assert hemin["phase"] == ea.PHASE_HALLUCINATION
    assert hemin["evidence"] == "heme"
    assert hemin["kind"] == "compound"
    assert ledger["counts"]["removed"] == len(ledger["removed"])
    for entry in ledger["removed"]:
        assert set(entry) == {"phase", "rule", "kind", "entity", "evidence"}


def test_new_acceptance_a4_below_viability_is_review_required_not_dropped():
    """A4 / merge rule 7. When the gate's own removals empty the reaction set, the
    pathway is FLAGGED, never dropped: every surviving row is still on the payload
    and the ledger states the reason for a downstream ``review_required``."""
    ea = _gate()
    merged = merge_additions(
        _f7_base(),
        _f7_additions(),
        seed_text=SEED_WITHOUT_SUCCINYL,
        rag_admission_report=NO_ADMITTED_CLAIM,
        pathway_context=HEME,
    )
    ledger = merged[ea.LEDGER_KEY]

    assert ledger["review_required"] is True
    assert ea.REASON_NO_SURVIVING_REACTION in ledger["review_reasons"]
    assert merged["entities"]["compounds"] == [{"name": "glycine"}]
    assert "entities" in merged and "processes" in merged


def test_new_acceptance_a4_a_clean_payload_is_not_flagged():
    ea = _gate()
    merged = merge_additions(_f34_base(), {}, pathway_context=ENTEROBACTIN)
    ledger = merged[ea.LEDGER_KEY]
    assert ledger["review_required"] is False and ledger["review_reasons"] == []
    assert ledger["counts"]["removed"] == 0


def test_new_acceptance_a5_the_gate_never_constructs_renames_or_repairs_a_row():
    """A5. Every surviving row is the SAME object content the gate was handed --
    byte-identical by JSON signature. Demotion is recorded advisorily in the
    ledger and mutates nothing, precisely so this property holds unconditionally."""
    ea = _gate()
    for payload, ctx in (
        (_f1_payload(), HEME),
        (_merged_f7_shape(), HEME),
        (_f34_base(), ENTEROBACTIN),
    ):
        before = {
            json.dumps(row, sort_keys=True)
            for bucket in list(payload.get("entities", {}).values())
            + list(payload.get("processes", {}).values())
            for row in bucket
        }
        kept, _ = ea.screen_additions(
            copy.deepcopy(payload),
            seed_text=SEED_WITHOUT_SUCCINYL,
            admission_report=NO_ADMITTED_CLAIM,
            context=ctx,
        )
        after = {
            json.dumps(row, sort_keys=True)
            for bucket in list(kept.get("entities", {}).values())
            + list(kept.get("processes", {}).values())
            for row in bucket
        }
        assert after <= before, "the gate emitted a row it was not given"


def test_new_acceptance_a6_the_gate_only_ever_subtracts():
    """A6 / merge rule 6. No previously-refused row becomes admitted, on any input:
    every bucket's length is non-increasing and no bucket or key is created."""
    ea = _gate()
    inputs = [
        _f1_payload(),
        _f34_base(),
        _merged_f7_shape(),
        {},
        {"entities": {"compounds": []}},
        {"entities": "not-a-dict", "processes": [1, 2]},
        {"entities": {"compounds": [{"name": "hemin"}]}},
    ]
    for payload in inputs:
        original = copy.deepcopy(payload)
        kept, ledger = ea.screen_additions(
            copy.deepcopy(payload),
            seed_text=SEED_WITHOUT_SUCCINYL,
            admission_report=NO_ADMITTED_CLAIM,
            context=HEME,
        )
        assert set(kept) <= set(original), "the gate added a top-level payload key"
        for section in ("entities", "processes"):
            src, out = original.get(section), kept.get(section)
            if not isinstance(src, dict):
                continue
            assert set(out) <= set(src)
            for bucket, rows in out.items():
                assert len(rows) <= len(src[bucket])
        assert ledger["counts"]["admitted"] == 0


def test_new_acceptance_a6_hemin_alone_is_never_removed():
    """``unknown`` is NOT a rejection (C-018's own docstring). Without heme present
    there is no duplicate identity, so the reagent rule does not fire."""
    ea = _gate()
    kept, ledger = ea.screen_additions(
        {"entities": {"compounds": [{"name": "hemin"}]}}, context=HEME
    )
    assert _names(kept) == ["hemin"]
    assert ledger["removed"] == []


def test_new_acceptance_a7_the_gate_runs_pre_freeze_at_the_stage_two_merge(monkeypatch):
    """A7 / merge rule 8. Lifecycle position: inside ``merge_additions``, BEFORE
    ``apply_post_merge_cleanup``, and therefore long before the canonical graph is
    frozen and any exporter runs. Proven by call order, not by comment."""
    from t2pw.pipeline import pipeline as pl

    order = []
    real_screen, real_cleanup = pl.screen_additions, pl.apply_post_merge_cleanup
    monkeypatch.setattr(
        pl, "screen_additions", lambda *a, **k: (order.append("gate"), real_screen(*a, **k))[1]
    )
    monkeypatch.setattr(
        pl,
        "apply_post_merge_cleanup",
        lambda *a, **k: (order.append("cleanup"), real_cleanup(*a, **k))[1],
    )
    merge_additions(_f1_payload(), {})

    assert order == ["gate", "cleanup"]


def test_new_acceptance_a8_deterministic_and_offline():
    """A8. Three runs on identical input give byte-identical payload AND ledger.
    Offline-ness is structural: the module imports nothing that can do I/O."""
    ea = _gate()
    runs = []
    for _ in range(3):
        kept, ledger = ea.screen_additions(
            _merged_f7_shape(),
            seed_text=SEED_WITHOUT_SUCCINYL,
            admission_report=NO_ADMITTED_CLAIM,
            context=HEME,
        )
        runs.append(json.dumps([kept, ledger], sort_keys=True))
    assert runs[0] == runs[1] == runs[2]

    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(ea.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported <= {"__future__", "re", "dataclasses", "typing", "t2pw"}


def test_new_acceptance_trap3_protein_rows_are_never_touched():
    """TRAP-3 / O-1. ``placeholder_backed_proteins`` is a standing open question.
    The gate removes rows only from ``entities.compounds`` and ``processes.*``;
    it can never reach protein export policy."""
    ea = _gate()
    payload = _merged_f7_shape()
    payload["entities"]["proteins"] = [
        {"name": "ALAS2", "evidence": SUCCINYL_SPAN, "identity_status": "placeholder"}
    ]
    kept, ledger = ea.screen_additions(
        payload,
        seed_text=SEED_WITHOUT_SUCCINYL,
        admission_report=NO_ADMITTED_CLAIM,
        context=HEME,
    )
    assert _names(kept, "proteins") == ["ALAS2"]
    assert all(e["kind"] != "protein" for e in ledger["removed"])


def test_new_acceptance_an_admitted_rag_record_protects_its_row():
    """The hallucination gate is a POSITIVE-grounding rule: locatable in the seed
    text, or carried by an ADMITTED RAG record. It never reads the rejected list --
    section-10 rejected-claim enforcement is ``rag/graph_delta.py`` (C-021/C-055)
    and C-060 must not re-implement it."""
    ea = _gate()
    admitted = {
        "accepted": [{"evidence": {"source_id": "PMC10886707", "span": SUCCINYL_SPAN}}],
        "rejected": [],
        "counts": {"accepted": 1, "considered": 1},
    }
    kept, ledger = ea.screen_additions(
        _merged_f7_shape(),
        seed_text=SEED_WITHOUT_SUCCINYL,
        admission_report=admitted,
        context=HEME,
    )
    assert "succinyl-CoA" in _names(kept)
    assert ledger["removed"] == []


def test_new_acceptance_without_a_seed_text_the_hallucination_rule_is_inert():
    """No evidence base means the rule cannot be evaluated, and ``not evaluated``
    is never ``false``. It fails OPEN, per PRODUCT_CONTRACT section 8."""
    ea = _gate()
    kept, ledger = ea.screen_additions(_merged_f7_shape(), context=HEME)
    assert "succinyl-CoA" in _names(kept)
    assert ledger["removed"] == []


def test_new_acceptance_demotion_is_advisory_and_mutates_no_row():
    """Phase 2 demotes a surviving currency species that is still a participant of
    a surviving reaction: recorded, never rewritten."""
    ea = _gate()
    payload = {
        "entities": {"compounds": [{"name": "isochorismate"}, {"name": "NADH"}]},
        "processes": {
            "reactions": [
                {"name": "r", "inputs": ["isochorismate", "NADH"], "outputs": ["pyruvate"]}
            ]
        },
    }
    kept, ledger = ea.screen_additions(copy.deepcopy(payload), context=ENTEROBACTIN)

    assert kept["entities"]["compounds"] == payload["entities"]["compounds"]
    assert [(e["entity"], e["rule"]) for e in ledger["demoted"]] == [
        ("NADH", ea.RULE_CURRENCY_NOT_SUBJECT)
    ]
    assert ledger["removed"] == []


@pytest.mark.parametrize(
    "entity,context,verdict,reason,confidence",
    [
        # R-003 section 7 executed the merged C-018 classifier (85fae43) against
        # these entities in their own gold contexts. Every design decision in
        # entity_admission.py rests on this table, so it is pinned here rather
        # than trusted: if C-018 ever moves, C-060's premises fail loudly.
        ("succinyl-CoA", HEME, "participant", "subject_of_the_requested_pathway", "high"),
        ("pyridoxal 5'-phosphate", HEME, "hub", "connectivity_hub_not_pathway_evidence", "moderate"),
        ("THF", HEME, "hub", "connectivity_hub_not_pathway_evidence", "moderate"),
        ("nadh", ENTEROBACTIN, "cofactor_or_currency", "ubiquitous_currency_metabolite", "moderate"),
        ("nad+", ENTEROBACTIN, "cofactor_or_currency", "ubiquitous_currency_metabolite", "moderate"),
        # unknown is NOT a rejection, so cofactor_policy cannot close F1 alone.
        ("hemin", HEME, "unknown", "not_in_curated_vocabulary", "low"),
        ("heme", HEME, "unknown", "not_in_curated_vocabulary", "low"),
    ],
)
def test_new_acceptance_r003_measured_verdicts_still_hold(
    entity, context, verdict, reason, confidence
):
    got = classify_entity(entity, context)
    assert (got.verdict, got.reason, got.confidence) == (verdict, reason, confidence)


def test_new_acceptance_no_metabolite_is_ever_an_assay_reporter():
    """A2's negative half: ``assay_reporter`` never fires for NADH/NAD+, so the
    gate must not lean on that verdict. C-018's ``_REPORTER`` holds only LacZ,
    fluorescent proteins, luciferase and dyes."""
    for name in ("nadh", "nad+", "NAD", "lactate", "pyruvate"):
        assert classify_entity(name, ENTEROBACTIN).verdict != "assay_reporter"


@pytest.mark.parametrize("payload", [None, [], "", 0, {"entities": None}])
def test_new_acceptance_gate_is_total(payload):
    ea = _gate()
    kept, ledger = ea.screen_additions(payload, context=HEME)
    assert kept is payload or isinstance(kept, dict)
    assert ledger["counts"]["removed"] == 0
