"""REAL ``clean_stage_one`` output must reach the out-of-scope filter with its label.

Why this file exists
--------------------
"Tighten default reaction scope and wire the out-of-scope reaction filter"
(2026-07-14) shipped both halves of a working feature and the feature never worked.
``pwml_system.txt`` (``:6``, ``:15-16``) still tells the model to label every
reaction and promises out-of-scope ones "are removed from the payload before
downstream stages run"; ``streamlit_app.py:3617`` still calls
``filter_out_of_scope_reactions`` between Stage 1 and Stage 2. In between,
``pipeline._clean_processes`` rebuilt every reaction from a key whitelist that
omitted ``scope_membership``, so the filter read ``rxn.get("scope_membership",
"core")`` and defaulted every reaction to core. It had never removed a reaction.

Measured before the fix, on a two-reaction Stage-1 payload: RAW labels
``['core', 'out_of_scope']`` -> filter removes ``['glucose phosphorylation']``; the
same payload through ``clean_stage_one`` -> labels ``['<ABSENT>', '<ABSENT>']``,
cleaned reaction key union ``('biological_state', 'enzymes', 'evidence', 'inputs',
'name', 'outputs')`` -> filter removes NOTHING. Confirmed on real data: all 9
Stage-1 reactions and all 27 delivered reactions of
``runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict``
carry no ``scope_membership`` key at all.

THE TESTS ARE WHY IT SURVIVED TWO ROUNDS. ``tests/test_pipeline_cleanup.py:230-284``
hands the filter a HAND-BUILT payload that already carries the label -- which the
real pipeline never produces -- and
``tests/test_streamlit_stage2_orchestration.py:807-844`` is an AST test that pins
call ORDERING only. Neither one ever put real ``clean_stage_one`` output through the
filter, so both passed for the entire time the feature was inert. Every test below
therefore starts from a RAW Stage-1-shaped payload and runs it through the real
``clean_stage_one`` (and, in one case, the real chunk-merge path) before the filter
sees it. Do not "simplify" any of them by constructing the cleaned payload by hand;
that is precisely the mistake being pinned against.

Also pinned: the payload and the lock manifest must agree. The manifest is built
from the RAW payload (``write_stage1_lock_artifacts`` runs before
``clean_stage_one`` at ``pipeline.py:2266`` / ``:2312``), and it has always refused
to lock an out-of-scope reaction -- so before this fix such a reaction shipped in
the payload with no ``locked_reaction_id``: kept by one artifact, disowned by the
other.

Offline / deterministic: no network, no live LLM. The ``openai`` stub mirrors
tests/test_pipeline_cleanup.py because importing the pipeline pulls the mapping/LLM
stack at import time.
"""

from __future__ import annotations

import copy
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

from t2pw.pipeline.pipeline import (  # noqa: E402
    clean_stage_one,
    filter_out_of_scope_reactions,
    merge_stage_one_outputs,
)
from t2pw.pipeline.reaction_lock_manifest import (  # noqa: E402
    build_locked_reaction_manifest,
    write_stage1_lock_artifacts,
)


def _raw_stage_one() -> dict:
    """A raw Stage-1 payload in the shape the extractor actually emits.

    Reaction 1 is the real first step of the reference paper's pathway (LpxA on
    UDP-GlcNAc, PMC12444477); reaction 2 is the kind of central-metabolism step the
    model is told to label ``out_of_scope``; reaction 3 uses a non-core taxonomy
    value that must be KEPT. Keys beyond the whitelist (``class``, ``confidence``)
    are present because Stage 1 emits them and the rebuild drops them -- the point
    is that ``scope_membership`` no longer goes with them.
    """
    return {
        "entities": {
            "compounds": [
                {"name": "UDP-GlcNAc"},
                {"name": "UDP-3-O-acyl-GlcNAc"},
                {"name": "R-3-hydroxymyristoyl-ACP"},
                {"name": "ACP"},
                {"name": "glucose"},
                {"name": "glucose-6-phosphate"},
                {"name": "Kdo"},
                {"name": "Kdo-lipid IVA"},
            ],
            "proteins": [{"name": "LpxA"}, {"name": "hexokinase"}, {"name": "WaaA"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "LpxA acyltransfer",
                    "inputs": ["UDP-GlcNAc", "R-3-hydroxymyristoyl-ACP"],
                    "outputs": ["UDP-3-O-acyl-GlcNAc", "ACP"],
                    "enzymes": [{"protein": "LpxA", "evidence": "LpxA catalyses the first step."}],
                    "biological_state": "cytosol",
                    "evidence": "LpxA catalyses the first step of lipid A biosynthesis.",
                    "scope_membership": "core",
                    "class": "biochemical_reaction",
                    "confidence": 1.0,
                },
                {
                    "name": "glucose phosphorylation",
                    "inputs": ["glucose", "ATP"],
                    "outputs": ["glucose-6-phosphate", "ADP"],
                    "enzymes": [{"protein": "hexokinase", "evidence": "Hexokinase phosphorylates glucose."}],
                    "evidence": "Central carbon metabolism supplies precursors.",
                    "scope_membership": "out_of_scope",
                    "class": "biochemical_reaction",
                    "confidence": 0.8,
                },
                {
                    "name": "WaaA Kdo transfer",
                    "inputs": ["lipid IVA", "Kdo"],
                    "outputs": ["Kdo-lipid IVA"],
                    "enzymes": [{"protein": "WaaA", "evidence": "WaaA adds Kdo residues."}],
                    "evidence": "WaaA adds a number of Kdo residues to form Kdo-lipid IVA.",
                    "scope_membership": "auxiliary",
                    "class": "biochemical_reaction",
                    "confidence": 0.9,
                },
            ]
        },
    }


def _names(payload: dict) -> list:
    return [rxn.get("name") for rxn in payload["processes"]["reactions"]]


def _labels(payload: dict) -> list:
    return [rxn.get("scope_membership", "<ABSENT>") for rxn in payload["processes"]["reactions"]]


def test_filter_removes_a_labelled_reaction_from_real_clean_stage_one_output() -> None:
    """The regression itself: cleaned output must still carry the label.

    Before the fix this assertion failed on the labels line -- ``clean_stage_one``
    returned ``['<ABSENT>', '<ABSENT>', '<ABSENT>']`` and the filter removed
    nothing, so ``glucose phosphorylation`` shipped to Stage 2, export and every
    gate.
    """
    raw = _raw_stage_one()

    cleaned = clean_stage_one(raw)

    assert _labels(cleaned) == ["core", "out_of_scope", "auxiliary"]

    filtered, removed = filter_out_of_scope_reactions(cleaned)

    assert removed == ["glucose phosphorylation"]
    assert _names(filtered) == ["LpxA acyltransfer", "WaaA Kdo transfer"]
    # A non-core taxonomy value is not an out-of-scope value: anaplerotic /
    # cataplerotic / auxiliary reactions are in scope by the prompt's own taxonomy
    # (pwml_system.txt:291) and only the explicit out_of_scope verdict removes.
    assert "auxiliary" in _labels(filtered)


def test_label_survives_the_chunked_merge_path_too() -> None:
    """Long papers take ``merge_stage_one_outputs`` -> ``clean_stage_one`` (:2320-2321).

    The single-chunk branch (``:2267``) and the chunked branch clean at different
    points, so the label has to survive a merge of per-chunk cleaned payloads as
    well. A paper long enough to chunk is the normal case in the overnight corpus
    run, not the exception.
    """
    raw = _raw_stage_one()
    chunk_a = clean_stage_one(
        {
            "entities": raw["entities"],
            "processes": {"reactions": raw["processes"]["reactions"][:1]},
        }
    )
    chunk_b = clean_stage_one(
        {
            "entities": raw["entities"],
            "processes": {"reactions": raw["processes"]["reactions"][1:]},
        }
    )

    merged = clean_stage_one(merge_stage_one_outputs([chunk_a, chunk_b]))

    assert "out_of_scope" in _labels(merged)

    filtered, removed = filter_out_of_scope_reactions(merged)

    assert removed == ["glucose phosphorylation"]
    assert "glucose phosphorylation" not in _names(filtered)


def test_an_unlabelled_reaction_is_kept_because_absent_is_not_out_of_scope() -> None:
    """Absent label => keep. Never drop a reaction merely for being unclassified.

    Every payload written before this fix is unlabelled -- including all 27
    delivered reactions of the reference run -- as is every Stage-2 addition and
    every RAG import (``rag/synthesize.py``'s ``_ALLOWED_ROW_KEYS`` cannot carry
    the field). Treating those as out-of-scope would silently delete evidenced
    biology on the strength of a field nobody filled in.
    """
    raw = _raw_stage_one()
    for reaction in raw["processes"]["reactions"]:
        reaction.pop("scope_membership", None)

    cleaned = clean_stage_one(raw)
    assert _labels(cleaned) == ["<ABSENT>", "<ABSENT>", "<ABSENT>"]

    filtered, removed = filter_out_of_scope_reactions(cleaned)

    assert removed == []
    assert len(filtered["processes"]["reactions"]) == 3


def test_a_non_string_label_is_dropped_and_the_reaction_is_kept() -> None:
    """A malformed label is a missing label, not a removal verdict.

    ``reaction_lock_manifest._scope_membership`` (``:59-66``) already treats a
    non-string as ``"core"``; the cleaner drops the key so the filter reaches the
    same verdict from the other side.
    """
    raw = _raw_stage_one()
    raw["processes"]["reactions"][1]["scope_membership"] = 17
    raw["processes"]["reactions"][2]["scope_membership"] = ["out_of_scope"]

    cleaned = clean_stage_one(raw)
    filtered, removed = filter_out_of_scope_reactions(cleaned)

    assert removed == []
    assert _names(filtered) == ["LpxA acyltransfer", "glucose phosphorylation", "WaaA Kdo transfer"]


def test_filter_and_lock_manifest_agree_on_case_and_whitespace_variants() -> None:
    """Both readers must exclude the same reactions, or the payload contradicts the manifest.

    The manifest case-folds (``reaction_lock_manifest.py:229``), so ``OUT_OF_SCOPE``
    and ``"  out_of_scope  "`` are refused a lock. If the filter compared exactly it
    would keep those reactions in the payload while the manifest disowned them --
    the same payload/manifest split this change exists to close, just in a different
    spelling.
    """
    raw = {
        "processes": {
            "reactions": [
                {"name": "shouty", "inputs": ["A"], "outputs": ["B"], "evidence": "e",
                 "scope_membership": "OUT_OF_SCOPE"},
                {"name": "padded", "inputs": ["B"], "outputs": ["C"], "evidence": "e",
                 "scope_membership": "  out_of_scope  "},
                {"name": "kept", "inputs": ["C"], "outputs": ["D"], "evidence": "e",
                 "scope_membership": "core"},
            ]
        }
    }

    manifest, report = build_locked_reaction_manifest(copy.deepcopy(raw))
    filtered, removed = filter_out_of_scope_reactions(clean_stage_one(copy.deepcopy(raw)))

    assert [entry["name"] for entry in manifest] == ["kept"]
    assert report["out_of_scope_excluded_count"] == 2
    assert sorted(removed) == ["padded", "shouty"]
    assert _names(filtered) == ["kept"]


def test_out_of_scope_reactions_no_longer_ship_unlocked(tmp_path) -> None:
    """The second-order defect: kept by the payload, disowned by the lock manifest.

    ``write_stage1_lock_artifacts`` runs on the RAW payload BEFORE
    ``clean_stage_one`` (``pipeline.py:2266`` / ``:2312``) and stamps
    ``locked_reaction_id`` in place, skipping out-of-scope reactions
    (``reaction_lock_manifest.py:186``). With the label erased, the out-of-scope
    reaction still reached export -- as the ONLY reaction with no lock id. This
    walks the real sequence (lock artifacts -> clean -> filter) and pins that every
    delivered Stage-1 reaction is now one the manifest actually locked.
    """
    raw = _raw_stage_one()

    write_stage1_lock_artifacts(raw, tmp_path)
    cleaned = clean_stage_one(raw)
    filtered, removed = filter_out_of_scope_reactions(cleaned)

    manifest_ids = {
        entry["locked_reaction_id"]
        for entry in build_locked_reaction_manifest(_raw_stage_one())[0]
    }
    delivered = filtered["processes"]["reactions"]

    assert removed == ["glucose phosphorylation"]
    assert delivered, "the filter must not empty the payload"
    for reaction in delivered:
        assert reaction.get("locked_reaction_id"), f"{reaction.get('name')} shipped unlocked"
        assert reaction["locked_reaction_id"] in manifest_ids


def test_unlabelled_rows_gain_no_key_and_other_process_buckets_are_untouched() -> None:
    """Blast radius: the carrier adds a key only where the model wrote one.

    A reaction with no label keeps its exact key ORDER (the same property
    ``tests/test_pipeline_reaction_rag_provenance.py:287`` pins), and transports /
    reaction-coupled transports / interactions gain nothing at all -- the prompt
    does not label them and the filter does not read them.
    """
    payload = {
        "processes": {
            "reactions": [
                {"name": "plain", "inputs": ["A"], "outputs": ["B"], "evidence": "e"},
            ],
            "transports": [
                {"name": "MsbA flip", "cargo": "lipid A", "evidence": "MsbA flips lipid A.",
                 "scope_membership": "out_of_scope"},
            ],
            "interactions": [
                {"entity_1": "LapB", "entity_2": "LpxC", "relationship": "binds",
                 "evidence": "LapB binds LpxC.", "scope_membership": "out_of_scope"},
            ],
        }
    }

    cleaned = clean_stage_one(payload)

    assert list(cleaned["processes"]["reactions"][0].keys()) == ["inputs", "outputs", "name", "evidence"]
    assert "scope_membership" not in cleaned["processes"]["transports"][0]
    assert "scope_membership" not in cleaned["processes"]["interactions"][0]

    # The filter only ever touches processes.reactions, so those rows survive.
    filtered, removed = filter_out_of_scope_reactions(cleaned)
    assert removed == []
    assert len(filtered["processes"]["transports"]) == 1
    assert len(filtered["processes"]["interactions"]) == 1
