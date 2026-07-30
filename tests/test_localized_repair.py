"""Repair must be local: fix the broken thing, keep everything else untouched.

WHY THIS FILE EXISTS. The pre-existing recovery path answered every Stage-1
problem the same way -- re-issue the entire extraction prompt. That is expensive
(the largest prompt in the run), destructive (the second draw is a fresh sample,
not the first draw plus a fix), and it is where the wall clock on a 55-second
death goes. Three narrower recoveries replaced it, and these tests pin the
boundaries of each:

  * :func:`repair_json_text` sees only the broken text and the parser error;
  * :func:`repair_invalid_rows` sees only the failing rows, their exact contract
    errors, and their own evidence -- and can only write back to the pointers it
    was given;
  * :func:`reconstruct_registry_shells` uses no model at all.

The property that matters most is the one in
``test_one_malformed_reaction_does_not_regenerate_the_valid_reactions``: a single
bad row must not cost the good ones. Everything else here defends the ways that
could quietly stop being true -- an unrequested pointer being accepted, an
unbounded retry, a repaired row smuggling in biology its evidence never mentioned.

Offline and deterministic: every draw goes through an injected ``chat_fn``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.extraction_diagnostics import (  # noqa: E402
    BOUNDARY_REPORT_NAME,
    activate,
    deactivate,
)
from t2pw.pipeline.localized_repair import (  # noqa: E402
    MAX_ROW_REPAIR_ATTEMPTS,
    REPAIR_INCOMPLETE,
    REPAIR_NOT_ATTEMPTED,
    REPAIR_OK,
    SHELL_PROVENANCE,
    evidence_supports,
    reconstruct_registry_shells,
    repair_invalid_rows,
    repair_json_text,
    row_pointer,
)


@pytest.fixture(autouse=True)
def _clean_recorder():
    deactivate()
    yield
    deactivate()


# ---------------------------------------------------------------------------
# A canned provider. Mirrors the real ``chat_detailed`` return shape.
# ---------------------------------------------------------------------------
class _Reply:
    def __init__(self, text: str, **diagnostics: Any) -> None:
        self.text = text
        self.diagnostics = {
            "model": "test-model",
            "finish_reason": "stop",
            "attempts": 1,
            "response_status": "ok" if text.strip() else "empty",
            "terminal_reason": "",
            "request_hash": "",
            "response_hash": "",
            "raw_chars": len(text),
            "attempt_log": [],
            **diagnostics,
        }


class _Provider:
    """Hands out canned replies in order and records every prompt it was sent."""

    def __init__(self, *replies: _Reply) -> None:
        self._replies = list(replies)
        self.prompts: List[str] = []

    def __call__(self, messages: List[Dict[str, str]], **_: Any) -> _Reply:
        self.prompts.append(messages[-1]["content"])
        if not self._replies:
            raise AssertionError(
                f"the repair made {len(self.prompts)} call(s) but only "
                f"{len(self.prompts) - 1} canned replies were provided"
            )
        return self._replies.pop(0)

    @property
    def calls(self) -> int:
        return len(self.prompts)


def _content_filter_reply() -> _Reply:
    return _Reply("", finish_reason="content_filter", terminal_reason="content_filter",
                  response_status="content_filter")


# ---------------------------------------------------------------------------
# 1. Invalid JSON.
# ---------------------------------------------------------------------------
BROKEN_JSON = '{"entities": {"compounds": [{"name": "ATP"}]}, "processes": {"reactions": ['
FIXED_JSON = json.dumps(
    {"entities": {"compounds": [{"name": "ATP"}]}, "processes": {"reactions": []}}
)


def test_invalid_json_is_repaired_locally_without_re_extraction() -> None:
    """The repair prompt contains the broken TEXT, not the source paper.

    That is the whole difference from a re-draw: with no source text in the
    prompt there is nothing for the model to re-extract, so the only thing it can
    return is the same content with the syntax fixed.
    """

    provider = _Provider(_Reply(FIXED_JSON))

    result = repair_json_text(BROKEN_JSON, "JSONDecodeError: Expecting value", chat_fn=provider)

    assert result.ok
    assert result.outcome == REPAIR_OK
    assert result.payload == json.loads(FIXED_JSON)
    assert provider.calls == 1
    prompt = provider.prompts[0]
    assert BROKEN_JSON in prompt
    assert "JSONDecodeError: Expecting value" in prompt
    # The evidence of localization: the caller's source text is nowhere near it.
    assert "pathway description" not in prompt.casefold()


def test_json_repair_is_capped_and_reports_incomplete_rather_than_looping() -> None:
    provider = _Provider(_Reply("still not json"), _Reply("still not json"))

    result = repair_json_text(
        BROKEN_JSON, "JSONDecodeError", max_attempts=2, chat_fn=provider
    )

    assert not result.ok
    assert result.outcome == REPAIR_INCOMPLETE
    assert result.attempts == 2
    assert provider.calls == 2
    assert "2-attempt cap" in result.reason


def test_json_repair_stops_on_a_content_filter_instead_of_retrying(caplog) -> None:
    """A moderation stop is a property of the prompt, and the prompt here
    contains the model's own refused output. Re-sending it cannot help."""

    provider = _Provider(_content_filter_reply(), _Reply(FIXED_JSON))

    result = repair_json_text(
        BROKEN_JSON, "JSONDecodeError", max_attempts=3, chat_fn=provider
    )

    assert not result.ok
    assert result.outcome == REPAIR_INCOMPLETE
    assert provider.calls == 1  # the second canned reply was never reached
    assert "content_filter" in result.reason


def test_an_empty_reply_is_not_sent_to_the_json_repairer() -> None:
    """There is nothing to repair, and asking anyway invites invention."""

    provider = _Provider()

    result = repair_json_text("   \n ", "JSONDecodeError", chat_fn=provider)

    assert result.outcome == REPAIR_NOT_ATTEMPTED
    assert provider.calls == 0


def test_a_failing_repair_call_is_not_a_crash() -> None:
    def _explode(*_: Any, **__: Any) -> Any:
        raise RuntimeError("provider down")

    result = repair_json_text(BROKEN_JSON, "JSONDecodeError", chat_fn=_explode)

    assert result.outcome == REPAIR_INCOMPLETE
    assert "provider down" in result.reason


# ---------------------------------------------------------------------------
# 2. Contract-invalid rows.
# ---------------------------------------------------------------------------
def _payload_with_one_bad_reaction() -> Dict[str, Any]:
    return {
        "entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}, {"name": "AMP"}]},
        "processes": {
            "reactions": [
                {
                    "name": "good_one",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "evidence": "ATP is hydrolysed to ADP by the enzyme.",
                },
                {
                    "name": "broken",
                    "inputs": [],
                    "outputs": [],
                    "evidence": "ADP is further hydrolysed to AMP in the cytosol.",
                },
                {
                    "name": "good_two",
                    "inputs": ["ADP"],
                    "outputs": ["AMP"],
                    "evidence": "ADP yields AMP.",
                },
            ]
        },
    }


BROKEN_POINTER = "/processes/reactions/1"
BROKEN_ERROR = {
    "code": "reaction_missing_participants",
    "message": "Reaction must include at least one usable input or output participant.",
    "pointer": BROKEN_POINTER,
}


def test_one_malformed_reaction_does_not_regenerate_the_valid_reactions() -> None:
    """THE property this module exists for.

    Two correct reactions and one broken one. The repair must see only the broken
    one, and the two correct ones must come out byte-identical -- not
    re-extracted, not re-worded, not re-ordered.
    """

    payload = _payload_with_one_bad_reaction()
    before = json.dumps(payload["processes"]["reactions"][0], sort_keys=True)
    after_two = json.dumps(payload["processes"]["reactions"][2], sort_keys=True)

    provider = _Provider(
        _Reply(
            json.dumps(
                {
                    "repaired_rows": [
                        {
                            "pointer": BROKEN_POINTER,
                            "row": {
                                "name": "broken",
                                "inputs": ["ADP"],
                                "outputs": ["AMP"],
                                "evidence": "ADP is further hydrolysed to AMP in the cytosol.",
                            },
                        }
                    ]
                }
            )
        )
    )

    result = repair_invalid_rows(payload, [BROKEN_ERROR], chat_fn=provider)

    assert result.outcome == REPAIR_OK
    assert result.repaired_pointers == [BROKEN_POINTER]
    reactions = result.payload["processes"]["reactions"]
    assert json.dumps(reactions[0], sort_keys=True) == before
    assert json.dumps(reactions[2], sort_keys=True) == after_two
    assert reactions[1]["inputs"] == ["ADP"]
    assert len(reactions) == 3  # no row added, none removed

    # And the prompt itself never carried the valid rows.
    prompt = provider.prompts[0]
    assert "broken" in prompt
    assert "good_one" not in prompt
    assert "good_two" not in prompt


def test_the_repair_prompt_carries_the_exact_errors_and_the_rows_own_evidence() -> None:
    payload = _payload_with_one_bad_reaction()
    provider = _Provider(_Reply(json.dumps({"repaired_rows": []})))

    repair_invalid_rows(payload, [BROKEN_ERROR], chat_fn=provider)

    prompt = provider.prompts[0]
    assert "reaction_missing_participants" in prompt
    assert "at least one usable input or output participant" in prompt
    assert "ADP is further hydrolysed to AMP in the cytosol." in prompt


def test_a_pointer_that_was_not_requested_is_refused() -> None:
    """Otherwise row repair becomes a way to rewrite rows nobody complained about."""

    payload = _payload_with_one_bad_reaction()
    provider = _Provider(
        _Reply(
            json.dumps(
                {
                    "repaired_rows": [
                        {
                            "pointer": "/processes/reactions/0",
                            "row": {"name": "hijacked", "inputs": ["X"], "outputs": ["Y"]},
                        }
                    ]
                }
            )
        )
    )

    result = repair_invalid_rows(payload, [BROKEN_ERROR], chat_fn=provider)

    assert result.repaired_pointers == []
    assert result.payload["processes"]["reactions"][0]["name"] == "good_one"
    assert result.rejected[0]["reason"] == "pointer_not_requested"


def test_a_repaired_row_that_invents_biology_is_rejected() -> None:
    """"Every repaired scientific field must remain tied to supplied evidence."

    The row's evidence mentions ADP and AMP. A repair that adds NADPH is making a
    claim the passage does not support, so the original row is kept and the
    rejection is recorded with the offending value.

    The reply keeps ``evidence`` verbatim so that the GROUNDING guard is the one
    that fires; dropping it would trip the preservation guard first (see
    ``test_a_repair_that_deletes_the_evidence_is_refused``) and this test would
    stop testing what it says.
    """

    payload = _payload_with_one_bad_reaction()
    provider = _Provider(
        _Reply(
            json.dumps(
                {
                    "repaired_rows": [
                        {
                            "pointer": BROKEN_POINTER,
                            "row": {
                                "name": "broken",
                                "inputs": ["ADP", "NADPH"],
                                "outputs": ["AMP"],
                                "evidence": "ADP is further hydrolysed to AMP in the cytosol.",
                            },
                        }
                    ]
                }
            )
        )
    )

    result = repair_invalid_rows(payload, [BROKEN_ERROR], chat_fn=provider)

    assert result.repaired_pointers == []
    assert result.payload["processes"]["reactions"][1]["inputs"] == []
    assert result.rejected[0]["reason"] == "evidence_unsupported"
    assert "NADPH" in result.rejected[0]["unsupported"]


def test_a_repair_that_only_keeps_existing_participants_is_accepted() -> None:
    """The mirror of the test above: values the extractor already produced are
    not new claims, so a repair that merely re-arranges them is not blocked even
    when the evidence field is thin.

    This is also why the preservation guard checks the UNION of the fields the
    contract complained about rather than each one separately: moving ``ADP``
    from ``products`` to ``outputs`` is the normalization the repair exists to
    do, and a field-by-field rule would read it as a deletion.
    """

    payload = {
        "entities": {"compounds": [{"name": "ATP"}]},
        "processes": {
            "reactions": [
                {"name": "r", "inputs": ["ATP"], "outputs": [], "products": ["ADP"], "evidence": ""}
            ]
        },
    }
    provider = _Provider(
        _Reply(
            json.dumps(
                {
                    "repaired_rows": [
                        {
                            "pointer": "/processes/reactions/0",
                            "row": {
                                "name": "r",
                                "inputs": ["ATP"],
                                "outputs": ["ADP"],
                                # Returned verbatim even though it is empty: the
                                # shape guard requires fields outside the
                                # contract's remit to come back exactly as they
                                # went in, and an empty string is a shape.
                                "evidence": "",
                            },
                        }
                    ]
                }
            )
        )
    )

    result = repair_invalid_rows(
        payload,
        [{"code": "reaction_missing_participants", "pointer": "/processes/reactions/0"}],
        chat_fn=provider,
    )

    assert result.repaired_pointers == ["/processes/reactions/0"]
    assert result.payload["processes"]["reactions"][0]["outputs"] == ["ADP"]


def test_row_repair_is_capped_and_returns_the_payload_when_it_cannot_finish() -> None:
    """An unrecoverable row must not cost the caller the whole payload.

    ``REPAIR_INCOMPLETE`` plus the surviving payload is what lets the run report
    a clear incomplete outcome instead of choosing between a fabrication and
    nothing.
    """

    payload = _payload_with_one_bad_reaction()
    provider = _Provider(
        *[_Reply(json.dumps({"repaired_rows": []})) for _ in range(MAX_ROW_REPAIR_ATTEMPTS)]
    )

    result = repair_invalid_rows(
        payload,
        [BROKEN_ERROR],
        revalidate=lambda _candidate: [BROKEN_ERROR],
        chat_fn=provider,
    )

    assert result.outcome == REPAIR_INCOMPLETE
    assert result.attempts == MAX_ROW_REPAIR_ATTEMPTS
    assert provider.calls == MAX_ROW_REPAIR_ATTEMPTS
    assert result.unrepaired_pointers == [BROKEN_POINTER]
    assert result.payload["processes"]["reactions"][0]["name"] == "good_one"


def test_row_repair_stops_on_a_content_filter() -> None:
    payload = _payload_with_one_bad_reaction()
    provider = _Provider(_content_filter_reply(), _Reply(json.dumps({"repaired_rows": []})))

    result = repair_invalid_rows(
        payload,
        [BROKEN_ERROR],
        revalidate=lambda _candidate: [BROKEN_ERROR],
        chat_fn=provider,
    )

    assert provider.calls == 1
    assert result.outcome == REPAIR_INCOMPLETE
    assert "content_filter" in result.reason


def test_a_container_level_guard_is_never_sent_to_row_repair() -> None:
    """No row edit can fix "the payload has no processes object", and asking for
    one invites the model to supply the object -- the fabrication forbidden here."""

    provider = _Provider()

    result = repair_invalid_rows(
        {"entities": {}},
        [{"code": "processes_required", "pointer": "/processes"}],
        chat_fn=provider,
    )

    assert result.outcome == REPAIR_NOT_ATTEMPTED
    assert provider.calls == 0
    assert "container-level guard" in result.reason


@pytest.mark.parametrize(
    "pointer, expected",
    [
        ("/processes/reactions/3", "/processes/reactions/3"),
        ("/entities/proteins/2/name", "/entities/proteins/2"),
        ("/processes/reactions/0/inputs/1", "/processes/reactions/0"),
        ("/processes", ""),
        ("/", ""),
        ("", ""),
        ("/metadata/pathway_name", ""),
    ],
)
def test_row_pointer_truncates_a_field_pointer_to_its_row(pointer: str, expected: str) -> None:
    assert row_pointer(pointer) == expected


def test_evidence_supports_reports_what_was_unsupported() -> None:
    ok, unsupported = evidence_supports(
        {"inputs": ["ATP", "NADPH"]},
        {"inputs": ["ATP"]},
        "ATP is consumed by the ligase.",
    )
    assert ok is False
    assert unsupported == ["NADPH"]

    ok, unsupported = evidence_supports(
        {"inputs": ["ATP", "NADPH"]},
        {"inputs": ["ATP"]},
        "ATP and NADPH are consumed by the reductase.",
    )
    assert ok is True
    assert unsupported == []


def test_evidence_matching_tolerates_hyphenation_and_case() -> None:
    """Hyphenation is a typesetting choice. Rejecting a correct repair over one
    is worse than accepting a cosmetic difference; an unmentioned molecule still
    cannot get through."""

    ok, _ = evidence_supports(
        {"inputs": ["gamma-glutamylcysteine"]},
        {"inputs": []},
        "Gamma glutamylcysteine is formed by the ligase.",
    )
    assert ok is True


def test_row_repair_records_a_bounded_boundary(tmp_path: Path) -> None:
    activate(artifact_dir=tmp_path)
    payload = _payload_with_one_bad_reaction()
    provider = _Provider(_Reply(json.dumps({"repaired_rows": []})))

    repair_invalid_rows(payload, [BROKEN_ERROR], chat_fn=provider)

    report = json.loads((tmp_path / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8"))
    boundary = report["boundaries"][-1]
    assert boundary["boundary"] == "stage1_row_repair"
    assert boundary["repair_kind"] == "contract_rows"
    assert boundary["requested_row_count"] == 1
    # The prompt carried a row with an evidence passage; the diagnostic did not.
    assert "further hydrolysed" not in json.dumps(report)


# ---------------------------------------------------------------------------
# 3. Deterministic registry reconstruction.
# ---------------------------------------------------------------------------
def test_an_explicit_process_participant_produces_an_unresolved_registry_shell() -> None:
    """The silent-loss case.

    The reaction is correct and evidenced; the registry simply never declared
    ADP. Without a shell, ``filter_unresolvable_reactions`` deletes the reaction
    later for a bookkeeping gap. The shell makes the gap explicit instead.
    """

    payload = {
        "entities": {"compounds": [{"name": "ATP"}]},
        "processes": {
            "reactions": [
                {
                    "name": "hydrolysis",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "enzymes": [{"protein": "ATPase"}],
                    "evidence": "ATP is hydrolysed to ADP by ATPase.",
                }
            ]
        },
    }

    result = reconstruct_registry_shells(payload)

    compounds = result.payload["entities"]["compounds"]
    shell = next(row for row in compounds if row["name"] == "ADP")
    assert shell["resolution_status"] == "unresolved"
    assert shell["provenance"] == SHELL_PROVENANCE
    assert shell["reconstructed_from"]["role"] == "outputs"

    # The enzyme states its type (``protein``), so it lands in proteins. An actor
    # that states none is skipped instead -- see the entity-type group below.
    proteins = result.payload["entities"]["proteins"]
    assert [row["name"] for row in proteins] == ["ATPase"]
    assert proteins[0]["resolution_status"] == "unresolved"


def test_a_shell_never_carries_an_identifier_or_a_function() -> None:
    """"must not assign a biological ID or new function" -- enforced on the row.

    A shell is a declared gap. Every identity gate downstream must still refuse
    it; that is the point, and an ``hmdb_id`` or an ``ec_number`` would launder
    the gap into an identified entity.
    """

    payload = {
        "entities": {},
        "processes": {"reactions": [{"inputs": ["ATP"], "outputs": ["ADP"]}]},
    }

    result = reconstruct_registry_shells(payload)

    for row in result.payload["entities"]["compounds"]:
        assert set(row) == {"name", "provenance", "resolution_status", "reconstructed_from"}
        for forbidden in (
            "mapped_ids",
            "uniprot",
            "ec_number",
            "hmdb_id",
            "kegg_id",
            "chebi_id",
            "pathbank_compound_id",
            "function",
            "description",
            "organism",
        ):
            assert forbidden not in row


def test_reconstruction_is_deterministic_and_idempotent() -> None:
    payload = {
        "entities": {"compounds": [{"name": "ATP"}]},
        "processes": {"reactions": [{"inputs": ["ATP"], "outputs": ["ADP", "Pi"]}]},
    }

    once = reconstruct_registry_shells(payload)
    twice = reconstruct_registry_shells(once.payload)
    thrice = reconstruct_registry_shells(payload)

    assert twice.shells == []  # nothing left to add
    assert once.payload == thrice.payload
    assert [row["name"] for row in once.payload["entities"]["compounds"]] == ["ATP", "ADP", "Pi"]


def test_an_already_registered_name_is_not_duplicated_across_buckets() -> None:
    """A compound declared as a protein is a mapping problem, not a missing row.

    Adding a second shell for it would create the duplicate the dedupe pass then
    has to undo.
    """

    payload = {
        "entities": {"proteins": [{"name": "ATP"}]},
        "processes": {"reactions": [{"inputs": ["ATP"], "outputs": ["ADP"]}]},
    }

    result = reconstruct_registry_shells(payload)

    assert [row["name"] for row in result.payload["entities"].get("compounds", [])] == ["ADP"]
    assert len(result.payload["entities"]["proteins"]) == 1


def test_a_row_cleaning_will_discard_does_not_populate_the_registry() -> None:
    """Reconstruction reads VALID rows only.

    A reaction with no outputs is going to be discarded, so its names are not
    evidence that anything belongs in the registry -- populating from it would be
    fabrication by another route.
    """

    payload = {
        "entities": {},
        "processes": {
            "reactions": [
                {"name": "doomed", "inputs": ["Ghostane"], "outputs": []},
                {"name": "real", "inputs": ["ATP"], "outputs": ["ADP"]},
            ]
        },
    }

    result = reconstruct_registry_shells(payload)

    names = [row["name"] for row in result.payload["entities"]["compounds"]]
    assert names == ["ATP", "ADP"]
    assert "Ghostane" not in names
    assert result.skipped[0]["reason"] == "row_is_not_a_valid_process"


def test_reconstruction_never_touches_a_payload_with_no_processes() -> None:
    payload = {"entities": {"compounds": [{"name": "ATP"}]}}

    result = reconstruct_registry_shells(payload)

    assert result.changed is False
    assert result.payload == payload
