"""C-079 / F-105 -- the interactive curator prompt filters by ALLOW-LIST.

WHAT THIS CARD CORRECTS, AND WHAT G9 IS OWED
============================================
C-075 stopped the AUDIT prompt carrying the source index by filtering at
serialization off a named constant, ``PROMPT_OMITTED_PAYLOAD_KEYS``. F-105 then
measured a SECOND serializer doing the same job the other way round:
``strip_payload_for_interactive_context`` drops keys by a blacklist -- exact names
in ``_RAW_TEXT_KEYS``, substrings in ``_BULKY_KEY_TOKENS`` -- and
``source_text_index`` matches NEITHER half. So once the source-support pass was
armed (``f12115a``), roughly 65 KB of normalized paper rode into every interactive
curator round while the audit prompt beside it had already dropped the same blob.

THE FIX IS NOT ONE MORE STRING IN THE BLACKLIST. That would close today's symptom
and leave the class open: a blacklist silently admits every FUTURE key. The
interactive path now reads the same constant object the audit path reads.

Sections 1 and 2 are BASE-FAILING BEHAVIOURAL PROOFS. They call only
``build_interactive_curator_messages`` and ``run_interactive_curator_round``, both
of which exist on base ``d50fbcd``, and they assert facts about the SERIALIZED
PROMPT TEXT, not about the presence of a symbol. On base the index is in that text
and every assertion in section 1 fails.

Section 3 is the NON-VACUITY arm, section 4 the ADDITIVITY arm, section 5 the
no-mutation arm and section 6 the anti-drift arm. Those four are property checks
about a constant this card relocates, so they are symbol-dependent by construction
and are NOT offered as G9 proofs.

The measured base-versus-tip numbers live in
``docs/pwml_recovery_sprint/evidence/c079_prompt_size_probe.py``, run on both trees
against a real committed paper: 65,777 prompt bytes on base against 825 at tip, the
same 825 bytes and the same sha256 on both trees for a payload carrying no omitted
key.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation import interactive_curator as curator  # noqa: E402
from t2pw.mapping import identity_admission as ia  # noqa: E402


SOURCE_TEXT = (
    "ALAS2 condenses succinyl-CoA and glycine to 5-aminolevulinate in the "
    "mitochondrial matrix. " * 40
)


def _payload() -> Dict[str, Any]:
    """A small, structurally real payload. No key on it is an omitted key."""
    return {
        "pathway_name": "heme biosynthesis",
        "entities": {
            "compounds": [
                {"name": "succinyl-CoA", "mapped_id": "HMDB0001022"},
                {"name": "glycine", "mapped_id": "HMDB0000123"},
            ],
            "proteins": [{"name": "ALAS2", "mapped_ids": ["P22557"]}],
        },
        "processes": [{"name": "ALAS2 condensation"}],
        "reactions": [
            {
                "id": "r1",
                "inputs": ["succinyl-CoA", "glycine"],
                "outputs": ["5-aminolevulinate"],
                "modifiers": ["ALAS2"],
            }
        ],
    }


def _payload_with_index() -> Dict[str, Any]:
    payload = _payload()
    payload[ia.SOURCE_INDEX_KEY] = ia.build_source_index(SOURCE_TEXT)
    return payload


def _prompt_text(payload: Dict[str, Any]) -> str:
    """The user message ``run_interactive_curator_round`` would send, verbatim."""
    messages = curator.build_interactive_curator_messages(
        working_json=payload,
        graph_png_bytes=b"",
        user_request="add the missing transport step",
        qa_report={"errors": [], "warnings": []},
        mapping_misses=[],
        history=[],
    )
    return messages[1]["content"][0]["text"]


# =============================================================================
# 1. the behaviour, at the production serializer  [G9 PROOF -- fails on base]
# =============================================================================


def test_the_interactive_prompt_does_not_carry_the_source_index() -> None:
    """The blob is gone from the prompt text, and the structure the curator
    actually patches is all still there.

    ON BASE d50fbcd every assertion below fails: ``source_text_index`` is a
    top-level key of the payload, the stripper's blacklist matches neither its
    exact name nor any of its substrings, and ``json.dumps`` writes the whole
    normalized paper into the message."""
    text = _prompt_text(_payload_with_index())

    assert ia.SOURCE_INDEX_KEY not in text
    assert ia.normalize_text(SOURCE_TEXT)[:60] not in text
    # ... and nothing the curator needs went with it
    assert "succinyl-CoA" in text and "ALAS2 condensation" in text
    assert "entities" in text and "reactions" in text
    assert "HMDB0001022" in text and "P22557" in text


def test_the_interactive_prompt_shrinks_by_the_whole_index_and_no_more() -> None:
    """The saving is exactly the blob. An armed leg's prompt is byte-identical to
    the prompt the same leg produced before the index existed -- so nothing else
    disappeared with it, and nothing else appeared.

    ON BASE the armed prompt is tens of KB larger and this fails."""
    bare = _prompt_text(_payload())
    armed = _prompt_text(_payload_with_index())

    assert armed == bare, "the armed leg's prompt is no longer byte-identical to today's"


def test_the_production_round_never_sends_the_index_to_the_model() -> None:
    """The same property through the real entry point, with the model mocked.

    ``run_interactive_curator_round`` is the function ``streamlit_app.py:2804``
    calls. Nothing between it and the serializer may put the payload back. No LLM
    call is made: ``chat`` is patched, so this costs nothing and asserts on the
    messages the provider WOULD have received.

    ON BASE the captured message carries the index and this fails."""
    captured: list[Any] = []

    def _fake_chat(messages: Any, **_kwargs: Any) -> str:
        captured.append(messages)
        return json.dumps(
            {"patch": [], "rationale": "", "change_summary": "", "needs_user_review": True}
        )

    with patch("t2pw.llm.client.chat", side_effect=_fake_chat) as mock_chat:
        result = curator.run_interactive_curator_round(
            working_json=_payload_with_index(),
            graph_png_bytes=b"",
            user_request="add the missing transport step",
        )

    assert mock_chat.call_count == 1, "the mocked round retried; the response was rejected"
    assert "error" not in result
    assert captured, "the production round never reached the provider seam"

    sent = captured[0][1]["content"][0]["text"]
    assert ia.SOURCE_INDEX_KEY not in sent
    assert ia.normalize_text(SOURCE_TEXT)[:60] not in sent
    assert "succinyl-CoA" in sent


# =============================================================================
# 2. the size of it -- F-105's measurement, in miniature
# =============================================================================


def test_the_index_is_the_dominant_term_in_an_armed_prompt() -> None:
    """F-105 measured 64,880 bytes of index on a real leg, and the committed
    probe measures 65,777 prompt bytes on base against 825 at tip for
    PMC12180156. The fixture here is small, but the SHAPE of the defect is the
    point: the index dwarfs the pathway it is attached to, so a filter that misses
    it does not miss a detail."""
    payload = _payload_with_index()
    index_bytes = len(
        json.dumps(payload[ia.SOURCE_INDEX_KEY], indent=2, ensure_ascii=False).encode("utf-8")
    )
    pathway_bytes = len(json.dumps(_payload(), indent=2, ensure_ascii=False).encode("utf-8"))

    assert index_bytes > pathway_bytes
    assert len(_prompt_text(payload).encode("utf-8")) < index_bytes


# =============================================================================
# 3. non-vacuity -- the clause CAN be inert, and is inert exactly when it should be
# =============================================================================


def test_a_payload_with_no_omitted_key_is_serialized_exactly_as_before() -> None:
    """THE PROPERTY the audit-side docstring claims, now true on this side too:
    "a payload carrying none of these keys is serialized by the same object it
    always was".

    Proved by NEUTRALIZING the new clause -- an empty constant is precisely the
    base code path -- and showing the output does not move by one byte. The second
    half is what stops this being vacuous: under the SAME neutralization an
    index-carrying payload DOES move, so the comparison is capable of failing and
    the clause is capable of acting."""
    inert = _payload()
    armed = _payload_with_index()

    as_shipped_inert = _prompt_text(inert)
    as_shipped_armed = _prompt_text(armed)

    with patch.object(curator, "PROMPT_OMITTED_PAYLOAD_KEYS", frozenset()):
        neutralized_inert = _prompt_text(inert)
        neutralized_armed = _prompt_text(armed)

    assert as_shipped_inert == neutralized_inert, (
        "the allow-list clause changed a prompt that carries none of its keys"
    )
    assert len(as_shipped_inert.encode("utf-8")) == len(neutralized_inert.encode("utf-8"))
    assert neutralized_armed != as_shipped_armed, (
        "neutralizing the clause changed nothing -- the comparison above proves nothing"
    )
    assert ia.SOURCE_INDEX_KEY in neutralized_armed, "the neutralized path is not the base path"


def test_the_stripper_leaves_an_ordinary_payload_untouched() -> None:
    """Same property one level down, at the stripper rather than the prompt: a
    payload with nothing to omit comes back equal to what went in."""
    payload = _payload()
    assert curator.strip_payload_for_interactive_context(payload) == payload


# =============================================================================
# 4. additivity -- every key the blacklist stripped is still stripped
# =============================================================================


def test_every_raw_text_key_is_still_stripped() -> None:
    """Enumerated from the constant, not from a copied literal, so a future
    addition to ``_RAW_TEXT_KEYS`` is covered the day it lands."""
    assert curator._RAW_TEXT_KEYS, "the exact-name blacklist is empty"
    payload = _payload()
    for key in curator._RAW_TEXT_KEYS:
        payload[key] = "a large blob of paper text"

    stripped = curator.strip_payload_for_interactive_context(payload)

    for key in curator._RAW_TEXT_KEYS:
        assert key not in stripped, f"{key} is no longer stripped"
    assert "entities" in stripped and "reactions" in stripped


def test_every_bulky_key_token_is_still_stripped() -> None:
    """The substring half. Each token is exercised as a bare key and as a key that
    merely CONTAINS it, which is what the substring rule promises."""
    assert curator._BULKY_KEY_TOKENS, "the substring blacklist is empty"
    payload = _payload()
    for token in curator._BULKY_KEY_TOKENS:
        payload[token] = {"drop": True}
        payload[f"stage2_{token}_blob"] = [{"drop": True}]

    stripped = curator.strip_payload_for_interactive_context(payload)

    for token in curator._BULKY_KEY_TOKENS:
        assert token not in stripped, f"{token} is no longer stripped"
        assert f"stage2_{token}_blob" not in stripped, f"substring match on {token} was lost"
    assert "entities" in stripped and "reactions" in stripped


def test_the_core_payload_and_the_identifiers_are_still_exempt() -> None:
    """The other direction of additivity: the new clause must not have started
    dropping something the blacklist deliberately kept."""
    payload = _payload()
    payload["entities"]["compounds"][0]["candidate_ids"] = ["HMDB_BAD_1"]

    stripped = curator.strip_payload_for_interactive_context(payload)

    assert stripped["entities"]["compounds"][0]["mapped_id"] == "HMDB0001022"
    assert stripped["entities"]["proteins"][0]["mapped_ids"] == ["P22557"]
    assert "candidate_ids" not in stripped["entities"]["compounds"][0]
    assert stripped["processes"] == payload["processes"]
    assert stripped["reactions"] == payload["reactions"]


def test_the_refused_fix_was_not_taken() -> None:
    """The card exists to refuse "add one more string to the blacklist". If
    ``source_text_index`` ever appears in either blacklist, the allow-list clause
    can be deleted without a single test going red -- and the class F-105 named
    quietly reopens."""
    assert ia.SOURCE_INDEX_KEY not in curator._RAW_TEXT_KEYS
    assert not any(token in ia.SOURCE_INDEX_KEY for token in curator._BULKY_KEY_TOKENS), (
        "a bulky token became a substring of the source index key; the allow-list "
        "clause is no longer the thing being tested"
    )
    source = (SRC / "t2pw" / "curation" / "interactive_curator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    stripper = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_should_drop_payload_key"
    )
    literals = {
        node.value for node in ast.walk(stripper)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert ia.SOURCE_INDEX_KEY not in literals, (
        "the key is hardcoded in the stripper; the filter is not keyed off the constant"
    )


# =============================================================================
# 5. the caller's payload is never mutated
# =============================================================================


def test_the_callers_payload_is_not_mutated() -> None:
    """The object travelling on to patch application and to the AUTHORITATIVE
    mapping run is the object that arrived. The mapping run reads the index off
    that payload; a stripper that edited it in place would make mapping fail open
    and withhold nothing -- which is C-073's whole defect, reintroduced."""
    payload = _payload_with_index()
    before = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    entities = payload["entities"]

    stripped = curator.strip_payload_for_interactive_context(payload)

    assert ia.SOURCE_INDEX_KEY in payload, "the caller's payload was edited"
    assert json.dumps(payload, sort_keys=True, ensure_ascii=False) == before
    assert payload["entities"] is entities
    assert stripped is not payload
    assert stripped["entities"] is not payload["entities"], "the copy aliases the caller"
    assert ia.read_source_index(payload) is not None, "mapping can still read the index"


# =============================================================================
# 6. the two sites cannot drift apart again
# =============================================================================


def test_both_prompt_paths_read_the_same_constant_object() -> None:
    """Not "the same value" -- the same OBJECT. Two frozensets that merely happen
    to be equal today are exactly the situation F-105 found."""
    from t2pw.curation import audit_json_llm
    from t2pw.curation import prompt_payload_keys

    assert curator.PROMPT_OMITTED_PAYLOAD_KEYS is prompt_payload_keys.PROMPT_OMITTED_PAYLOAD_KEYS
    assert (
        audit_json_llm.PROMPT_OMITTED_PAYLOAD_KEYS
        is prompt_payload_keys.PROMPT_OMITTED_PAYLOAD_KEYS
    )
    assert ia.SOURCE_INDEX_KEY in prompt_payload_keys.PROMPT_OMITTED_PAYLOAD_KEYS


def test_a_key_added_to_the_constant_is_dropped_from_both_prompts() -> None:
    """The consequence that matters. Both filters read MEMBERSHIP of the constant,
    so a key added to it is omitted from both prompts with no further edit. The
    audit filter is exact-match, so the interactive side is asserted to be at
    least as strict -- exact match plus a case-folded match, never less."""
    from t2pw.curation import audit_json_llm

    widened = frozenset(curator.PROMPT_OMITTED_PAYLOAD_KEYS | {"future_bulky_blob"})
    payload = _payload()
    payload["future_bulky_blob"] = {"a": "b" * 500}

    # the key survives both filters until the constant names it -- otherwise the
    # blacklist is doing this and the assertions below would prove nothing
    assert "future_bulky_blob" in _prompt_text(payload)
    assert "future_bulky_blob" in audit_json_llm.payload_for_prompt(payload)

    with patch.object(curator, "PROMPT_OMITTED_PAYLOAD_KEYS", widened), patch.object(
        audit_json_llm, "PROMPT_OMITTED_PAYLOAD_KEYS", widened
    ):
        interactive = _prompt_text(payload)
        audited = audit_json_llm.payload_for_prompt(payload)

    assert "future_bulky_blob" not in interactive
    assert "future_bulky_blob" not in audited
    assert "future_bulky_blob" in payload, "the audit filter mutated the caller's payload"


def test_a_mixed_case_spelling_of_an_omitted_key_is_dropped_too() -> None:
    """The interactive filter folds case, so a payload that spells a listed key
    differently cannot slip past it. Strictly a superset of the audit filter."""
    payload = _payload()
    payload[ia.SOURCE_INDEX_KEY.upper()] = {"normalized": "x" * 400}
    payload["Source_Text_Index"] = {"normalized": "y" * 400}

    stripped = curator.strip_payload_for_interactive_context(payload)

    assert ia.SOURCE_INDEX_KEY.upper() not in stripped
    assert "Source_Text_Index" not in stripped


def test_an_index_nested_below_the_top_level_is_dropped_too() -> None:
    """The interactive stripper is recursive and this clause runs at every depth,
    unlike the audit filter, which only ever sees the top-level payload dict.
    ``compact_mapping_misses`` feeds arbitrary records through this same function,
    so a nested index is a real shape, not a hypothetical one."""
    payload = _payload()
    payload["upstream_context"] = {
        "note": "keep me",
        ia.SOURCE_INDEX_KEY: ia.build_source_index(SOURCE_TEXT),
    }

    stripped = curator.strip_payload_for_interactive_context(payload)

    assert stripped["upstream_context"] == {"note": "keep me"}


def test_the_constant_lives_where_neither_serializer_pays_for_the_llm_sdk() -> None:
    """WHY THE CONSTANT MOVED. ``audit_json_llm`` imports ``t2pw.llm.client`` at
    module scope, and that module constructs an ``OpenAI`` client while being
    imported and raises outright when no usable provider is configured. Importing
    it from ``interactive_curator`` for one frozenset would have made a pure
    string filter unimportable without LLM credentials -- so the constant sits in
    a leaf whose only import is a pure ``re``/``typing`` module.

    Asserted statically, over the module-level imports of both files, so it holds
    regardless of what an earlier test in the same process happened to import."""

    def _top_level_imports(relpath: str) -> set[str]:
        tree = ast.parse((SRC / relpath).read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module)
        return names

    interactive = _top_level_imports("t2pw/curation/interactive_curator.py")
    leaf = _top_level_imports("t2pw/curation/prompt_payload_keys.py")

    assert "t2pw.curation.prompt_payload_keys" in interactive
    for forbidden in ("t2pw.llm.client", "t2pw.curation.audit_json_llm", "openai"):
        assert forbidden not in interactive, (
            f"{forbidden} became a module-scope import of interactive_curator"
        )
    assert leaf <= {"__future__", "t2pw.mapping"}, f"the leaf grew imports: {sorted(leaf)}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
