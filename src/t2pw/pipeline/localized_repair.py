"""Bounded, localized recovery for a Stage-1 payload that arrived broken.

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
When one thing is wrong, fix that one thing. The pre-existing recovery path did
the opposite: ``_run_json_stage`` reacted to a JSON parse error by re-issuing the
*entire* extraction prompt, so a single unbalanced brace cost a full re-extraction
and threw away every correct reaction the model had already produced. That is
where the wall clock on a 55-second death goes, and it is also how a good run
turns into a different, worse run -- the second draw is not the first draw plus a
fix, it is a fresh sample.

Three repair kinds live here, in escalating order, and each is strictly narrower
than a re-extraction:

1. :func:`repair_json_text` -- the reply is not parseable. The model is shown its
   own broken text and the exact parser error, and asked for *the same content*
   with the syntax corrected. No pathway content is requested, so there is
   nothing for it to invent.
2. :func:`repair_invalid_rows` -- the reply parsed and specific rows fail the
   stage contract. Only those rows are sent, each with the exact contract errors
   and the row's own evidence passage. Valid rows are never sent and never
   touched, so they come out byte-identical.
3. :func:`reconstruct_registry_shells` -- no model at all. A participant named
   explicitly by a valid process row but missing from the entity registry gets a
   deterministic unresolved shell, so the reaction stops being dropped for a
   bookkeeping gap.

WHAT REPAIR MAY NOT DO
----------------------
* It may not regenerate the pathway. Every prompt here says so, and
  :func:`repair_invalid_rows` structurally cannot: it splices returned rows back
  at their own pointers and discards anything else the model sends.
* It may not invent biology. A repaired row's scientific fields are checked
  against the evidence that was supplied with it (:func:`evidence_supports`); a
  row that introduces a claim the evidence does not contain is rejected and the
  original is kept.
* It may not loop. Attempts are capped (:data:`MAX_JSON_REPAIR_ATTEMPTS`,
  :data:`MAX_ROW_REPAIR_ATTEMPTS`) and a ``content_filter`` stop ends the attempt
  sequence immediately rather than re-sending a prompt already refused.
* It may not manufacture success. When repair cannot recover a process, the
  caller gets an explicit incomplete outcome (:data:`REPAIR_INCOMPLETE`) and the
  failure artifacts survive; nothing here fabricates a reaction to satisfy
  ``entities_required`` or ``processes_required``.
"""

from __future__ import annotations

import decimal
import difflib
import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from t2pw.pipeline.extraction_diagnostics import (
    BOUNDARY_STAGE1_JSON_REPAIR,
    BOUNDARY_STAGE1_ROW_REPAIR,
    MAX_SAMPLE,
    OUTCOME_CONTRACT_FAILED,
    OUTCOME_EMPTY_COMPLETION,
    OUTCOME_INVALID_JSON,
    OUTCOME_OK,
    OUTCOME_SEMANTIC_GUARD_FAILED,
    count_entities,
    count_processes,
    current as current_diagnostics,
    payload_hash,
)
from t2pw.pipeline.failure_detail import clip

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_JSON_REPAIR_ATTEMPTS",
    "MAX_ROW_REPAIR_ATTEMPTS",
    "REPAIR_OK",
    "REPAIR_INCOMPLETE",
    "REPAIR_NOT_ATTEMPTED",
    "REPAIR_SEMANTIC_GUARD_FAILED",
    "SCIENTIFIC_FIELDS",
    "IMMUTABLE_ROW_FIELDS",
    "SHELL_PROVENANCE",
    "JsonRepairResult",
    "RowRepairResult",
    "ReconstructionResult",
    "repair_json_text",
    "repair_invalid_rows",
    "reconstruct_registry_shells",
    "evidence_supports",
    "preserves_original_values",
    "json_repair_preserves_content",
    "row_pointer",
    "resolve_pointer",
]


# ---------------------------------------------------------------------------
# Caps. Env-overridable so an operator can tighten them, never widen them past
# what the code itself considers sane.
# ---------------------------------------------------------------------------
def _capped_env(name: str, default: int, ceiling: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(0, min(value, ceiling))


#: Draws allowed to fix unparseable text. One is almost always enough -- a model
#: that cannot close its own braces on the second try will not on the fifth --
#: and each draw costs a full round-trip on a prompt containing the broken reply.
MAX_JSON_REPAIR_ATTEMPTS = _capped_env("LLM_JSON_REPAIR_ATTEMPTS", 1, 3)

#: Draws allowed to fix contract-invalid rows. Two, because the second draw sees
#: a *different* prompt: the errors that survived the first repair.
MAX_ROW_REPAIR_ATTEMPTS = _capped_env("LLM_ROW_REPAIR_ATTEMPTS", 2, 4)

#: Repair verdicts. ``REPAIR_INCOMPLETE`` is a real, reportable outcome and not a
#: synonym for failure: it means the pipeline knows exactly what it could not
#: recover and is saying so instead of padding the payload.
REPAIR_OK = "repaired"
REPAIR_INCOMPLETE = "incomplete"
REPAIR_NOT_ATTEMPTED = "not_attempted"
#: The repair came back as valid JSON and was refused because it did not
#: preserve the content it was asked to repair. Its own outcome, not a flavour of
#: ``incomplete``: nothing about the syntax explains it, and the operator needs
#: to know a guard fired rather than that a model failed to close a brace.
REPAIR_SEMANTIC_GUARD_FAILED = "semantic_guard_failed"

#: Fields whose values are biological claims. A repaired row may only carry a new
#: value in one of these if the supplied evidence contains it; see
#: :func:`evidence_supports`. Structural/bookkeeping fields (``name``,
#: ``locked_reaction_id``, ``scope_membership``, ``inference``) are excluded on
#: purpose -- a reaction *name* is a label for participants that are themselves
#: checked, and locking/scope fields carry no claim about the biology.
SCIENTIFIC_FIELDS: Tuple[str, ...] = (
    "inputs",
    "outputs",
    "left",
    "right",
    "substrates",
    "products",
    "enzymes",
    "modifiers",
    "transporters",
    "cargo",
    "entity_1",
    "entity_2",
    "relationship",
    "biological_state",
    "from_biological_state",
    "to_biological_state",
    "elements_with_states",
    "components",
    "cofactors",
    "organism",
    "species",
)

#: Marker written on every deterministically reconstructed entity row. It is the
#: single thing a downstream reader needs to know that the row is a placeholder
#: for a name a process referenced, not an entity anybody identified.
SHELL_PROVENANCE = "reconstructed_from_process_participant"

#: Longest evidence passage sent with one row in a repair prompt. Repair needs
#: the sentence that supports the row, not the paper: a six-figure evidence field
#: would blow the prompt budget and bury the errors being fixed.
MAX_EVIDENCE_CHARS = 1200

#: Most rows sent in one row-repair request. A request that needs more rows than
#: this is not a localized repair, and sending it would be the whole-pathway
#: regeneration this module exists to prevent.
MAX_ROWS_PER_REQUEST = 12


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------
# JSON pointer helpers.
# ---------------------------------------------------------------------------
def _unescape_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _escape_token(token: str) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def _pointer_tokens(pointer: str) -> List[str]:
    text = _text(pointer)
    if not text.startswith("/"):
        return []
    return [_unescape_token(part) for part in text[1:].split("/")] if text != "/" else []


def row_pointer(pointer: str) -> str:
    """The pointer of the *row* an error points into.

    Contract errors land on fields (``/entities/proteins/2/name``) and on rows
    (``/processes/reactions/3``). Repair operates on rows, so a field pointer is
    truncated to the ``bucket/index`` prefix that owns it. Returns ``""`` for a
    pointer that names no row -- a container-level guard such as ``/processes``
    is not repairable by editing a row, and pretending otherwise would send an
    empty request.
    """

    tokens = _pointer_tokens(pointer)
    if len(tokens) < 3:
        return ""
    section, bucket, index = tokens[0], tokens[1], tokens[2]
    if section not in {"entities", "processes"} or not index.isdigit():
        return ""
    return f"/{_escape_token(section)}/{_escape_token(bucket)}/{index}"


def resolve_pointer(payload: Mapping[str, Any], pointer: str) -> Any:
    """The value at ``pointer``, or ``None`` when the path does not exist."""

    node: Any = payload
    for token in _pointer_tokens(pointer):
        if isinstance(node, Mapping):
            node = node.get(token)
        elif isinstance(node, list) and token.isdigit():
            index = int(token)
            node = node[index] if 0 <= index < len(node) else None
        else:
            return None
        if node is None:
            return None
    return node


def _assign_pointer(payload: Dict[str, Any], pointer: str, value: Any) -> bool:
    """Write ``value`` at ``pointer``, only if the path already exists.

    "Only if it exists" is the safety property: a repair may replace a row the
    contract complained about, and may not create a new row at an index nobody
    asked about. That is what keeps row repair from being a way to add reactions.
    """

    tokens = _pointer_tokens(pointer)
    if not tokens:
        return False
    node: Any = payload
    for token in tokens[:-1]:
        if isinstance(node, Mapping):
            node = node.get(token)
        elif isinstance(node, list) and token.isdigit():
            index = int(token)
            node = node[index] if 0 <= index < len(node) else None
        else:
            return False
        if node is None:
            return False
    last = tokens[-1]
    if isinstance(node, list) and last.isdigit():
        index = int(last)
        if 0 <= index < len(node):
            node[index] = value
            return True
        return False
    if isinstance(node, dict) and last in node:
        node[last] = value
        return True
    return False


# ---------------------------------------------------------------------------
# Evidence grounding.
# ---------------------------------------------------------------------------
#: Non-string JSON literals, matched only OUTSIDE string bodies by
#: :func:`_scan_literals`.
_BARE_LITERAL_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null")

#: Enough precision for any numeric literal a payload can carry. The default
#: context is 28 significant digits, which silently rounds a longer decimal and
#: would make two distinguishable numbers compare equal -- the exact defect
#: :func:`_normalize_number` exists to remove.
_DECIMAL_CONTEXT = decimal.Context(prec=200, traps=[])


def _normalize_number(value: Any) -> str:
    """A numeric literal reduced to an EXACT comparable form.

    Decimal, not float, and the difference is not academic:

    * ``float`` collapses every integer above 2**53. ``9007199254740993`` and
      ``9007199254740992`` are the same float, so a repair that changed a large
      identifier -- a taxonomy id, a PathBank row id -- would compare equal to
      the original and pass the guard.
    * ``float`` also rounds long decimals to 17 significant digits, so two
      distinguishable confidences could collide the same way.

    ``1e3`` and ``1000.0`` must still compare EQUAL: re-serializing a number is a
    formatting choice the repair is allowed to make. ``Decimal.normalize`` plus
    fixed-point formatting gives that without giving up exactness.

    Floats are routed through ``str`` rather than ``Decimal(float)`` on purpose:
    ``Decimal(0.1)`` is the binary expansion (0.1000000000000000055511151...),
    which would never match the ``0.1`` written in the source text.
    """

    if isinstance(value, decimal.Decimal):
        number = value
    elif isinstance(value, int) and not isinstance(value, bool):
        number = decimal.Decimal(value)
    else:
        try:
            number = decimal.Decimal(str(value).strip())
        except (decimal.InvalidOperation, ValueError, TypeError):
            return str(value)
    try:
        # ``normalize`` in a wide context strips the exponent difference between
        # 1e3 and 1000.0; ``'f'`` renders without re-introducing one.
        rendered = format(_DECIMAL_CONTEXT.normalize(number), "f")
    except (decimal.InvalidOperation, ValueError):  # pragma: no cover - NaN/Inf
        return str(value)
    # Decimal keeps the sign of zero; JSON does not distinguish -0.0 from 0, so
    # neither may this. Otherwise a repair that dropped a redundant minus sign
    # would be refused as a content change.
    return "0" if rendered in {"-0", "-0.0"} else rendered


def _normalize_literal(token: str) -> str:
    """A bare JSON literal reduced to a comparable form."""

    text = token.strip()
    if text in {"true", "false", "null"}:
        return text
    return _normalize_number(text)


@dataclass
class _Scan:
    """The complete literals of a malformed document, and how it ended.

    ``is_key`` marks the literals immediately followed by ``:``, and
    ``truncated`` records that scanning stopped early because something was cut
    off. Together they are what decides whether a repair is allowed to DELETE
    anything at all -- see :func:`json_repair_preserves_content`.
    """

    tokens: List[Tuple[str, str]] = field(default_factory=list)
    is_key: List[bool] = field(default_factory=list)
    truncated: bool = False

    @property
    def droppable_tail(self) -> int:
        """How many trailing literals a repair may legally delete: 0 or 1.

        One, and only when the document was cut off mid-value AND the last
        complete literal is the KEY that value belonged to. That is the single
        case ``_JSON_REPAIR_SYSTEM`` asks the model to handle by deletion, and
        widening it by any amount re-opens the hole this bound exists to close.
        """

        if not self.truncated or not self.tokens:
            return 0
        return 1 if self.is_key[-1] else 0


def _scan_literals(text: str) -> _Scan:
    """Every COMPLETE JSON literal in ``text``, in positional order.

    "Complete" is load-bearing. An unterminated string is dropped, and so is a
    bare literal running to the end of the input, where truncation cannot be
    ruled out -- ``{"n": 12`` could have been ``120``. Both set ``truncated``.

    Note what does NOT set it: a missing closing brace, a missing comma, a
    trailing comma or a code fence. Those are syntax faults with every literal
    intact, so they authorize no deletion whatsoever.

    Returns ``("s", value)`` for string literals (unescaped) and ``("n", value)``
    for numbers/booleans/null. Literals inside a string body are not scanned:
    ``"a, 1, b"`` is one string, not a string and a number.
    """

    scan = _Scan()
    index = 0
    length = len(text)

    def _emit(kind: str, value: str, after: int) -> None:
        cursor = after
        while cursor < length and text[cursor] in " \t\r\n":
            cursor += 1
        scan.tokens.append((kind, value))
        scan.is_key.append(kind == "s" and cursor < length and text[cursor] == ":")

    while index < length:
        char = text[index]
        if char == '"':
            cursor = index + 1
            closed = False
            while cursor < length:
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if text[cursor] == '"':
                    closed = True
                    break
                cursor += 1
            if not closed:
                scan.truncated = True
                break  # nothing complete can follow an unterminated string
            literal = text[index : cursor + 1]
            try:
                value = json.loads(literal)
            except (json.JSONDecodeError, ValueError):
                value = literal[1:-1]
            _emit("s", value, cursor + 1)
            index = cursor + 1
            continue
        match = _BARE_LITERAL_RE.match(text, index)
        if match and (index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_")):
            if match.end() >= length:
                scan.truncated = True
                break  # runs to EOF: possibly truncated, so not "complete"
            _emit("n", _normalize_literal(match.group(0)), match.end())
            index = match.end()
            continue
        index += 1
    return scan


def _payload_literals(value: Any, *, depth: int = 0) -> List[Tuple[str, str]]:
    """Every key and scalar of a parsed payload, in document order.

    Document order, not sorted order: ``json.loads`` preserves insertion order,
    so this walk reproduces the order the model wrote -- which is what makes a
    REORDERED array detectable at all.
    """

    if depth > 24:  # pragma: no cover - payloads are nowhere near this deep
        return []
    if isinstance(value, str):
        return [("s", value)]
    if isinstance(value, bool):
        return [("n", "true" if value else "false")]
    if value is None:
        return [("n", "null")]
    if isinstance(value, (int, float, decimal.Decimal)):
        return [("n", _normalize_number(value))]
    if isinstance(value, Mapping):
        out: List[Tuple[str, str]] = []
        for key, item in value.items():
            out.append(("s", str(key)))
            out.extend(_payload_literals(item, depth=depth + 1))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_payload_literals(item, depth=depth + 1))
        return out
    return [("s", _text(value))]


def parse_exact(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``text`` keeping every numeric literal exact, for guard use only.

    ``json.loads`` turns ``0.12345678901234567890123`` into a float and loses the
    tail, so a guard comparing against the source text would refuse an HONEST
    repair of a high-precision number. ``parse_float=Decimal`` keeps it.

    Never used for the payload the pipeline receives: ``Decimal`` is not JSON
    round-trippable and would leak an unexpected type into every downstream
    stage. The repaired payload is parsed separately, normally.
    """

    try:
        parsed = json.loads(text, parse_float=decimal.Decimal)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def json_repair_preserves_content(
    raw: str,
    repaired: Any,
    *,
    repaired_text: str = "",
) -> Tuple[bool, Dict[str, Any]]:
    """Whether a repaired document fixed the SYNTAX of ``raw`` and nothing else.

    A model instruction is not enforcement. ``_JSON_REPAIR_SYSTEM`` tells the
    model not to add, reword, reorder or drop anything, and a model that ignores
    it returns perfectly valid JSON containing a reaction nobody extracted, or
    missing one that somebody did. This is the mechanical check that makes the
    instruction binding.

    THE RULE, in two halves. The repaired literal stream must be **exactly** the
    input's complete literal stream, in order -- optionally minus the final
    literal, and only when :attr:`_Scan.droppable_tail` says the document was cut
    off mid-value and that literal is the key the value belonged to.

    * *No invention, substitution, reordering or duplication.* Any of those puts
      a literal in the repaired stream that the input cannot account for.
    * *No deletion.* Equality is what enforces this, and it is the half that was
      missing. A subsequence test alone accepts a repair that returns one of two
      reactions, or drops a middle entity row -- silently, from an extraction
      that was correct about them.

    A trailing comma, a missing comma, a code fence or a missing closing
    delimiter leaves every literal complete, so none of them authorizes deleting
    anything. Only a genuinely truncated value does, and only for its own key.

    ``repaired_text`` lets the caller supply the model's raw reply so numeric
    literals can be compared exactly; see :func:`parse_exact`.

    Returns ``(ok, finding)``; ``finding`` names what was wrong and where.
    """

    scan = _scan_literals(_text(raw))
    available = scan.tokens
    exact = parse_exact(repaired_text) if repaired_text else None
    produced = _payload_literals(exact if exact is not None else repaired)

    finding: Dict[str, Any] = {
        "input_literal_count": len(available),
        "repaired_literal_count": len(produced),
        "input_truncated": scan.truncated,
        "droppable_tail": scan.droppable_tail,
    }

    allowed = [available]
    if scan.droppable_tail:
        allowed.append(available[:-1])
    if any(produced == candidate for candidate in allowed):
        return True, finding

    # Not equal to anything allowed. Say precisely how it diverged, because
    # "added" and "removed" have completely different consequences.
    offender = _first_out_of_sequence(produced, available)
    if offender is not None:
        kind, token = offender
        finding["first_unmatched"] = clip(token, 120)
        finding["first_unmatched_kind"] = "string" if kind == "s" else "scalar"
        finding["divergence"] = "added_or_reordered"
        finding["reason"] = (
            "the repaired document contains a key or value that is not present, in "
            "that order, in the malformed input"
        )
        return False, finding

    missing = _missing_literals(available, produced, scan.droppable_tail)
    finding["divergence"] = "removed"
    finding["removed_count"] = len(missing)
    finding["removed_sample"] = [clip(token, 120) for _kind, token in missing[:MAX_SAMPLE]]
    finding["reason"] = (
        "the repaired document dropped complete content from the malformed input; "
        "only the final key of a truncated value may be deleted"
    )
    return False, finding


def _missing_literals(
    available: Sequence[Tuple[str, str]],
    produced: Sequence[Tuple[str, str]],
    droppable_tail: int,
) -> List[Tuple[str, str]]:
    """Input literals the repair failed to reproduce, ignoring a legal tail drop.

    A real diff, not a greedy forward scan, and the difference matters for the
    REPORT rather than the verdict. Payload literals repeat constantly -- ``name``
    is a key on every row, and ``ADP`` can be an entity row and a participant --
    so a greedy scan matching the dropped compound's ``name`` against the next
    reaction's ``name`` reports the wrong region as missing and buries the actual
    loss. ``SequenceMatcher`` aligns the two streams first, so the ``delete``
    opcodes are the literals genuinely gone.

    ``autojunk=False``: the heuristic treats items appearing in >1% of a long
    sequence as junk, which for these streams is exactly the repeated structural
    keys the alignment depends on.
    """

    pool = list(available[: len(available) - droppable_tail]) if droppable_tail else list(available)
    matcher = difflib.SequenceMatcher(None, pool, list(produced), autojunk=False)
    missing: List[Tuple[str, str]] = []
    for tag, start, end, _pstart, _pend in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            missing.extend(pool[start:end])
    return missing


def _first_out_of_sequence(
    needle: Sequence[Tuple[str, str]],
    haystack: Sequence[Tuple[str, str]],
) -> Optional[Tuple[str, str]]:
    """``None`` if ``needle`` is an ordered subsequence of ``haystack``.

    Otherwise the first token that could not be matched -- which is the token an
    operator needs, because it is either the invented value or the one that
    moved.
    """

    cursor = 0
    for token in needle:
        found = False
        while cursor < len(haystack):
            if haystack[cursor] == token:
                cursor += 1
                found = True
                break
            cursor += 1
        if not found:
            return token
    return None


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_for_match(value: Any) -> str:
    """Case-folded, punctuation-free text for substring comparison.

    Deliberately lossy: "gamma-glutamylcysteine" in a row must match
    "gamma glutamylcysteine" in a passage, because hyphenation is a typesetting
    choice and rejecting a correct repair over one is worse than accepting a
    cosmetic difference. What it cannot do is let an *unmentioned* molecule
    through -- the token sequence still has to be there.
    """

    return " ".join(_WORD_RE.findall(_text(value).casefold()))


def _flatten_strings(value: Any, *, depth: int = 0) -> List[str]:
    """Every string inside ``value``, for grounding checks."""

    if depth > 4:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        out: List[str] = []
        for item in value.values():
            out.extend(_flatten_strings(item, depth=depth + 1))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten_strings(item, depth=depth + 1))
        return out
    return []


def evidence_supports(
    repaired: Mapping[str, Any],
    original: Any,
    evidence: str,
) -> Tuple[bool, List[str]]:
    """Whether every *new* scientific claim in ``repaired`` is in ``evidence``.

    Returns ``(ok, unsupported)``. ``unsupported`` names the offending strings so
    the rejection is diagnosable rather than a bare "no".

    Only NEW values are checked, and that is the whole design. A repair that
    keeps a participant the extractor already produced is not making a new claim
    -- the extractor made it, from the source text, under the extraction prompt's
    own grounding rules. A repair that *introduces* "NADPH" into a reaction is
    making a new claim, and the only thing that can justify it is the passage
    supplied alongside the row. Checking everything instead would reject correct
    repairs of rows whose evidence field was itself truncated; checking nothing is
    how a repair pass becomes a hallucination pass.
    """

    haystack = _normalize_for_match(evidence)
    known = {_normalize_for_match(item) for item in _flatten_strings(original)}
    known.discard("")

    unsupported: List[str] = []
    for field_name in SCIENTIFIC_FIELDS:
        if field_name not in repaired:
            continue
        for value in _flatten_strings(repaired.get(field_name)):
            token = _normalize_for_match(value)
            if not token or token in known:
                continue
            if haystack and token in haystack:
                continue
            if clip(value, 120) not in unsupported:
                unsupported.append(clip(value, 120))
            if len(unsupported) >= MAX_SAMPLE:
                return False, unsupported
    return (not unsupported), unsupported


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return not value
    return False


def _scalar_values(value: Any) -> List[str]:
    """Every atom inside ``value``, as comparable text.

    Strings, numbers and booleans all count: a stoichiometry of ``2`` and a
    ``confidence`` of ``0.9`` are facts a repair may not quietly drop, and
    :func:`_flatten_strings` alone would not see them.
    """

    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return [repr(value)]
    if isinstance(value, Mapping):
        out: List[str] = []
        for item in value.values():
            out.extend(_scalar_values(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_scalar_values(item))
        return out
    return []


def _target_fields(pointer: str, errors: Sequence[Mapping[str, Any]]) -> set:
    """Which fields of a row the contract explicitly named as repairable.

    Three sources, in decreasing precision:

    * a FIELD-LEVEL pointer (``/entities/proteins/2/name``) names one field, and
      only that field is in play;
    * ``required_any`` on the issue -- ``_validate_reaction_structure`` emits
      ``["inputs", "outputs"]`` with ``reaction_missing_participants`` -- names
      the fields the rule is actually about, even though its pointer is the row;
    * a row-level error carrying neither falls back to every scientific field.

    This set is the ONLY licence a repair has to touch anything. Everything
    outside it must come back byte-identical, including fields that were absent
    and fields that were empty, so the narrower this set is the less a repair can
    do. That is why ``required_any`` is read rather than defaulting straight to
    :data:`SCIENTIFIC_FIELDS`: with it, a participants error cannot authorize
    writing an ``organism``.
    """

    prefix = f"{pointer}/"
    fields: set = set()
    unhinted = False
    for issue in errors:
        data = _safe_dict(issue)
        field_pointer = _text(data.get("field_pointer"))
        if field_pointer.startswith(prefix):
            token = field_pointer[len(prefix) :].split("/")[0]
            if token:
                fields.add(_unescape_token(token))
                continue
        named = [
            _text(item).strip()
            for item in _safe_list(data.get("required_any"))
            if _text(item).strip()
        ]
        if named:
            fields.update(named)
            continue
        unhinted = True
    if unhinted:
        fields.update(SCIENTIFIC_FIELDS)
    return fields


#: Fields a repair may never rewrite unless the contract error named that exact
#: field. These are the row's identity and its audit trail: which entity it is,
#: what passage supports it, where it came from, whether it is in scope, and
#: which locked reaction it belongs to. A structural contract error is never
#: about any of them, so a repair touching one is a repair doing something it was
#: not asked to do.
IMMUTABLE_ROW_FIELDS: Tuple[str, ...] = (
    "name",
    "evidence",
    "provenance",
    "source_refs",
    "source_papers",
    "rag_provenance",
    "scope_membership",
    "locked_reaction_id",
    "preservation_status",
    "confidence",
    "inference",
)


def preserves_original_values(
    repaired: Mapping[str, Any],
    original: Any,
    target_fields: Optional[set] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Whether a repaired row kept every fact the original row carried.

    THE FAILURE THIS CLOSES. Grounding alone is one-sided: it asks whether new
    values are supported and says nothing about old ones. A model handed a
    reaction with four inputs and told "this row fails a contract" can return a
    row with two inputs, and that passes a grounding check trivially -- nothing
    was added. The two deleted compounds are simply gone, from an extraction that
    was correct about them, with no error anywhere. Deletion is the *easier*
    failure mode for a repair prompt to produce, and it was the unguarded one.

    Two rules:

    * **exact shape** outside the contract-named fields. Every other key must be
      present if it was present, absent if it was absent, and equal if it was
      equal -- empty values included. An earlier version only walked the
      ORIGINAL's non-empty fields, which meant a repair could add an ``evidence``
      the row never had, fill an empty ``provenance``, delete an empty
      ``source_refs``, or introduce any brand-new top-level key, and none of it
      registered. Repair is not a rewrite, and a fabricated provenance is worse
      than a missing one because it is citable;
    * **no atom loss** across the fields it DID name. Checked over the union of
      those fields rather than field by field, because moving a participant from
      ``products`` to ``outputs`` is exactly the normalization a structural
      repair is *for*: the fact survives, only the key it sits under changed.
      Deleting it does not survive that check.

    Returns ``(ok, findings)``; each finding names what happened and a bounded
    sample of what went missing.
    """

    source = _safe_dict(original)
    targets = set(target_fields or ())
    findings: List[Dict[str, Any]] = []

    def _classify(name: str) -> str:
        return "immutable" if name in IMMUTABLE_ROW_FIELDS else "unrelated"

    # Every key on either side that the contract did not name. Union, not just
    # the original's keys: a key the repair INVENTED is exactly what a walk over
    # the original cannot see.
    frozen = (set(map(str, source)) | set(map(str, repaired))) - targets
    for name in sorted(frozen):
        present_before = name in source
        present_after = name in repaired
        if not present_before and present_after:
            findings.append(
                {
                    "field": name,
                    "reason": f"{_classify(name)}_field_added",
                    "added": clip(
                        repaired.get(name)
                        if isinstance(repaired.get(name), str)
                        else _text(repaired.get(name)),
                        120,
                    ),
                }
            )
            continue
        if present_before and not present_after:
            findings.append(
                {
                    "field": name,
                    "reason": f"{_classify(name)}_field_removed",
                    "original": clip(
                        source[name] if isinstance(source[name], str) else _text(source[name]),
                        120,
                    ),
                }
            )
            continue
        if present_before and source[name] != repaired.get(name):
            findings.append(
                {
                    "field": name,
                    "reason": f"{_classify(name)}_field_altered",
                    "original": clip(
                        source[name] if isinstance(source[name], str) else _text(source[name]),
                        120,
                    ),
                }
            )

    original_atoms: List[str] = []
    repaired_atoms: set = set()
    for name in targets:
        if name in IMMUTABLE_ROW_FIELDS:
            continue
        if name in source:
            original_atoms.extend(_scalar_values(source[name]))
        repaired_atoms.update(_scalar_values(repaired.get(name)))

    missing = [atom for atom in original_atoms if atom not in repaired_atoms]
    if missing:
        findings.insert(
            0,
            {
                "field": ", ".join(sorted(targets - set(IMMUTABLE_ROW_FIELDS))[:MAX_SAMPLE]),
                "reason": "original_value_removed",
                "missing": [clip(atom, 120) for atom in missing[:MAX_SAMPLE]],
            },
        )

    return (not findings), findings[:MAX_SAMPLE]


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------
@dataclass
class JsonRepairResult:
    """Outcome of trying to make an unparseable reply parseable."""

    payload: Optional[Dict[str, Any]]
    outcome: str
    attempts: int = 0
    reason: str = ""
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    #: The content-guard finding when ``outcome`` is
    #: :data:`REPAIR_SEMANTIC_GUARD_FAILED`: which token was unmatched, and how
    #: many literals each side carried.
    guard: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.payload is not None


@dataclass
class RowRepairResult:
    """Outcome of repairing specific contract-invalid rows.

    ``payload`` is always a payload: on every failure path it is the input with
    the valid rows untouched, so a caller can persist it and report an incomplete
    outcome rather than having to choose between a repair and nothing.
    """

    payload: Dict[str, Any]
    outcome: str
    repaired_pointers: List[str] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    unrepaired_pointers: List[str] = field(default_factory=list)
    attempts: int = 0
    reason: str = ""
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.repaired_pointers)


@dataclass
class ReconstructionResult:
    """Outcome of the deterministic registry reconstruction."""

    payload: Dict[str, Any]
    shells: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.shells)

    def to_report(self) -> Dict[str, Any]:
        return {
            "shell_count": len(self.shells),
            "shells": self.shells[: MAX_SAMPLE * 4],
            "skipped_count": len(self.skipped),
            "skipped": self.skipped[:MAX_SAMPLE],
            "provenance": SHELL_PROVENANCE,
        }


# ---------------------------------------------------------------------------
# 1. Invalid JSON.
# ---------------------------------------------------------------------------
_JSON_REPAIR_SYSTEM = """\
You are a JSON syntax repair tool. You are NOT an extraction model.

You will be given one malformed JSON document and the exact parser error it
produced. Return the SAME document with only its SYNTAX corrected.

Rules, all absolute:
- Return one JSON object and nothing else. No prose, no markdown fence.
- Do not add, remove, rename, reorder or reword any key, value, or array item.
- Do not add, remove, merge or split any row of any list.
- Do not fill in a value that is missing. If a value was cut off mid-token,
  delete that incomplete key/value pair rather than guessing what it was.
- Do not extract anything. There is no source text here and nothing to infer.
"""


def _chat_json(
    *,
    system: str,
    user: str,
    stage_name: str,
    max_tokens: int,
    temperature: float,
    model_env_var: Optional[str],
    chat_fn: Optional[Callable[..., Any]],
) -> Tuple[str, Dict[str, Any]]:
    """One repair draw. Returns ``(text, diagnostics_dict)``.

    ``chat_fn`` is injectable so every test in this module runs offline. When it
    is ``None`` the real ``chat_detailed`` is imported lazily -- module import
    time must not require a configured provider, because this module is imported
    by the pipeline on every run including the ones with no repair to do.
    """

    if chat_fn is None:
        from t2pw.llm.client import chat_detailed as chat_fn  # noqa: PLW2901

    result = chat_fn(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_json=True,
        model_env_var=model_env_var,
        stage_name=stage_name,
    )
    text = getattr(result, "text", result)
    raw_diag = getattr(result, "diagnostics", None)
    diagnostics = raw_diag.to_dict() if hasattr(raw_diag, "to_dict") else _safe_dict(raw_diag)
    return _text(text), diagnostics


def _parse_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def repair_json_text(
    raw: str,
    parse_error: str,
    *,
    stage: str = "extraction",
    max_attempts: Optional[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 12000,
    model_env_var: Optional[str] = "OPENROUTER_EXTRACTION_MODEL",
    chat_fn: Optional[Callable[..., Any]] = None,
) -> JsonRepairResult:
    """Turn unparseable reply text into a JSON object without re-extracting.

    The alternative -- what the pipeline did before -- is to re-send the whole
    extraction prompt, which costs a full round trip on the largest prompt in the
    run and returns a *different* sample rather than the same content with the
    brace closed. This sends only the broken text and the parser's own error.

    A ``content_filter`` stop ends the sequence immediately: the prompt contains
    the model's own previous output, so a moderation refusal of it will be
    repeated verbatim by every identical retry.
    """

    budget = MAX_JSON_REPAIR_ATTEMPTS if max_attempts is None else max(0, int(max_attempts))
    diagnostics_rows: List[Dict[str, Any]] = []
    if not _text(raw).strip():
        # Nothing to repair. An empty reply is an empty-completion problem and
        # belongs to the retry loop in the client, not here; sending an empty
        # document to a syntax repairer would just invite it to invent one.
        return JsonRepairResult(
            None,
            REPAIR_NOT_ATTEMPTED,
            reason="the reply was empty, so there is no JSON to repair",
        )
    if budget <= 0:
        return JsonRepairResult(
            None, REPAIR_NOT_ATTEMPTED, reason="json repair attempts are capped at 0"
        )

    recorder = current_diagnostics()
    attempts = 0
    last_error = _text(parse_error)
    working = _text(raw)

    for attempt in range(1, budget + 1):
        attempts = attempt
        user = (
            "The JSON document below could not be parsed.\n\n"
            f"Parser error: {last_error}\n\n"
            "Return the same document with its syntax corrected, following every "
            "rule in the system message.\n\n"
            "<<<MALFORMED_JSON\n"
            f"{working}\n"
            "MALFORMED_JSON>>>"
        )
        try:
            text, call_diag = _chat_json(
                system=_JSON_REPAIR_SYSTEM,
                user=user,
                stage_name=f"{stage} json repair",
                max_tokens=max_tokens,
                temperature=temperature,
                model_env_var=model_env_var,
                chat_fn=chat_fn,
            )
        except Exception as exc:  # noqa: BLE001 - a failed repair is not a crash
            row = {"attempt": attempt, "error": clip(f"{type(exc).__name__}: {exc}")}
            diagnostics_rows.append(row)
            recorder.record_boundary(
                stage=stage,
                boundary=BOUNDARY_STAGE1_JSON_REPAIR,
                outcome=OUTCOME_INVALID_JSON,
                attempts=attempt,
                response_status="error",
                error=f"{type(exc).__name__}: {exc}",
                request_hash=payload_hash(working),
            )
            return JsonRepairResult(
                None,
                REPAIR_INCOMPLETE,
                attempts=attempts,
                reason=f"json repair call failed: {type(exc).__name__}: {exc}",
                diagnostics=diagnostics_rows,
            )

        parsed = _parse_object(text)
        terminal = _text(call_diag.get("terminal_reason"))

        # The mechanical guard. A parseable reply is not yet an acceptable one:
        # it has to be provably the same content. See
        # :func:`json_repair_preserves_content`.
        guard_finding: Dict[str, Any] = {}
        if parsed is not None:
            preserved, guard_finding = json_repair_preserves_content(
                working, parsed, repaired_text=text
            )
            if not preserved:
                parsed = None
                guard_failed = True
            else:
                guard_failed = False
        else:
            guard_failed = False

        outcome = (
            OUTCOME_SEMANTIC_GUARD_FAILED
            if guard_failed
            else OUTCOME_OK
            if parsed is not None
            else (OUTCOME_EMPTY_COMPLETION if not text.strip() else OUTCOME_INVALID_JSON)
        )
        record = recorder.record_boundary(
            stage=stage,
            boundary=BOUNDARY_STAGE1_JSON_REPAIR,
            model=_text(call_diag.get("model")),
            finish_reason=_text(call_diag.get("finish_reason")),
            attempts=attempt,
            response_status=_text(call_diag.get("response_status")),
            terminal_reason=terminal,
            outcome=outcome,
            request_hash=payload_hash(working),
            response_hash=payload_hash(text),
            raw_chars=len(text),
            raw_preview=text,
            raw_entity_counts=count_entities(parsed) if parsed is not None else None,
            raw_process_counts=count_processes(parsed) if parsed is not None else None,
            repair_kind="json_syntax",
            semantic_guard=guard_finding or None,
        )
        diagnostics_rows.append(record)

        if parsed is not None:
            return JsonRepairResult(
                parsed, REPAIR_OK, attempts=attempts, diagnostics=diagnostics_rows
            )

        if guard_failed:
            # Durable, and deliberately not retried. The reply was valid JSON, so
            # the model did not fail at the task it was set -- it did something
            # else, and asking the same question again is not what changes that.
            # The caller falls back to the deterministic prefix salvage, which
            # cannot invent anything because it never asks a model.
            logger.warning(
                "%s json repair returned valid JSON that does not preserve the "
                "malformed reply's content (%s); refusing it and falling back to "
                "deterministic salvage.",
                stage,
                guard_finding.get("first_unmatched", "?"),
            )
            return JsonRepairResult(
                None,
                REPAIR_SEMANTIC_GUARD_FAILED,
                attempts=attempts,
                reason=(
                    "json repair returned valid JSON that introduced, reworded or "
                    "reordered content: "
                    f"{guard_finding.get('reason', 'content guard failed')} "
                    f"(first unmatched {guard_finding.get('first_unmatched_kind', 'token')}: "
                    f"{guard_finding.get('first_unmatched', '?')!r})"
                ),
                diagnostics=diagnostics_rows,
                guard=guard_finding,
            )

        if terminal == "content_filter":
            return JsonRepairResult(
                None,
                REPAIR_INCOMPLETE,
                attempts=attempts,
                reason=(
                    "json repair stopped on a content_filter finish_reason; the identical "
                    "prompt cannot be retried and no further attempt was made"
                ),
                diagnostics=diagnostics_rows,
            )

        if not text.strip():
            last_error = "the repair attempt returned an empty document"
        else:
            working = text
            last_error = "the repaired document still did not parse as a JSON object"

    return JsonRepairResult(
        None,
        REPAIR_INCOMPLETE,
        attempts=attempts,
        reason=f"json repair exhausted its {budget}-attempt cap: {last_error}",
        diagnostics=diagnostics_rows,
    )


# ---------------------------------------------------------------------------
# 2. Contract-invalid rows.
# ---------------------------------------------------------------------------
_ROW_REPAIR_SYSTEM = """\
You repair individual rows of an already-extracted pathway payload. You are NOT
running an extraction.

You will be given a small number of rows that failed a structural contract. For
each one you get its JSON pointer, the exact contract errors, and the evidence
passage that row was extracted from.

Return exactly this shape and nothing else:
{"repaired_rows": [{"pointer": "<the pointer you were given>", "row": {...}}]}

Rules, all absolute:
- Repair only the rows you were given. Never return a pointer you were not given.
- Never add, delete, merge, split or reorder rows. Never return the whole pathway.
- Every biological value you write must appear in the evidence passage supplied
  with that row. Do not add a participant, enzyme, compartment or organism the
  passage does not mention.
- Keep every field of the row that was already correct exactly as it was.
- If a row cannot be repaired from its own evidence, omit it from repaired_rows.
  Omitting a row is correct and expected. Inventing content to satisfy the
  contract is not.
"""


def _errors_by_row(errors: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """Group contract errors by the row pointer they belong to.

    Errors that do not belong to a row (container guards such as
    ``processes_required``) are dropped here rather than sent: no row edit can
    fix "the payload has no processes object", and including them would invite
    the model to supply one -- which is the fabrication this module forbids.
    """

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for issue in errors:
        data = _safe_dict(issue)
        pointer = row_pointer(_text(data.get("pointer") or data.get("path")))
        if not pointer:
            continue
        entry: Dict[str, Any] = {
            "code": _text(data.get("code")),
            "message": clip(_text(data.get("message") or data.get("reason"))),
            "field_pointer": _text(data.get("pointer") or data.get("path")),
        }
        # ``required_any`` is the contract naming the fields its rule is about
        # even when the pointer is the whole row. Carried through because it is
        # what stops a participants error from licensing an ``organism`` rewrite;
        # see :func:`_target_fields`. It is also genuinely useful in the prompt.
        required_any = [
            _text(item).strip()
            for item in _safe_list(data.get("required_any"))
            if _text(item).strip()
        ]
        if required_any:
            entry["required_any"] = required_any
        rows = grouped.setdefault(pointer, [])
        if entry not in rows:
            rows.append(entry)
    return grouped


def _row_evidence(row: Any) -> str:
    """The row's own evidence passage, bounded, flattened to text.

    The RAG subsystem stores evidence as a list of records under the same key
    core stages use for a sentence, so both shapes are reduced here rather than
    at the call site.
    """

    if not isinstance(row, Mapping):
        return ""
    value = row.get("evidence")
    parts: List[str] = []
    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, Mapping):
        parts.append(_text(value.get("text")))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                parts.append(_text(item.get("text")))
    joined = "\n".join(part.strip() for part in parts if part and part.strip())
    return joined[:MAX_EVIDENCE_CHARS]


def repair_invalid_rows(
    payload: Mapping[str, Any],
    errors: Sequence[Mapping[str, Any]],
    *,
    revalidate: Optional[Callable[[Dict[str, Any]], List[Dict[str, Any]]]] = None,
    stage: str = "extraction",
    max_attempts: Optional[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    model_env_var: Optional[str] = "OPENROUTER_EXTRACTION_MODEL",
    chat_fn: Optional[Callable[..., Any]] = None,
) -> RowRepairResult:
    """Repair only the rows the contract rejected, leaving every other row alone.

    ``revalidate`` re-runs the contract on the patched payload and returns its
    error list; when supplied, a second attempt is made against *the errors that
    survived*, which is a different prompt and therefore a legitimate retry
    rather than a blind one.

    Guarantees, each of which is a test in ``tests/test_localized_repair.py``:

    * a valid row is never sent to the model and is byte-identical on the way out;
    * a returned pointer that was not requested is discarded;
    * a returned row whose scientific fields are not in its supplied evidence is
      discarded and the original kept (:func:`evidence_supports`);
    * the attempt count is capped and a ``content_filter`` stop ends the sequence;
    * when rows remain broken the payload still comes back, with
      ``outcome == REPAIR_INCOMPLETE`` and the unrepaired pointers listed.
    """

    budget = MAX_ROW_REPAIR_ATTEMPTS if max_attempts is None else max(0, int(max_attempts))
    working = deepcopy(dict(payload))
    grouped = _errors_by_row(errors)
    result = RowRepairResult(working, REPAIR_NOT_ATTEMPTED)

    if not grouped:
        result.reason = (
            "no contract error names a repairable row; a container-level guard "
            "cannot be fixed by editing a row"
        )
        return result
    if budget <= 0:
        result.outcome = REPAIR_NOT_ATTEMPTED
        result.unrepaired_pointers = sorted(grouped)
        result.reason = "row repair attempts are capped at 0"
        return result

    recorder = current_diagnostics()
    pending = grouped

    for attempt in range(1, budget + 1):
        result.attempts = attempt
        requests: List[Dict[str, Any]] = []
        for pointer in sorted(pending)[:MAX_ROWS_PER_REQUEST]:
            row = resolve_pointer(working, pointer)
            if row is None:
                continue
            requests.append(
                {
                    "pointer": pointer,
                    "contract_errors": pending[pointer],
                    "row": row,
                    "evidence": _row_evidence(row),
                    # Not sent to the model -- it is the guard's input, not the
                    # prompt's. Stripped in _repair_request_payload below.
                    "_target_fields": sorted(_target_fields(pointer, pending[pointer])),
                }
            )
        dropped = max(0, len(pending) - MAX_ROWS_PER_REQUEST)
        if not requests:
            result.outcome = REPAIR_INCOMPLETE
            result.unrepaired_pointers = sorted(pending)
            result.reason = "every failing pointer is absent from the payload"
            return result

        user = (
            f"{len(requests)} row(s) of an already-extracted payload failed the "
            f"{stage} stage contract. Repair only these rows.\n\n"
            + json.dumps(
                {"rows_to_repair": [_repair_request_payload(item) for item in requests]},
                indent=2,
                ensure_ascii=False,
            )
        )
        try:
            text, call_diag = _chat_json(
                system=_ROW_REPAIR_SYSTEM,
                user=user,
                stage_name=f"{stage} row repair",
                max_tokens=max_tokens,
                temperature=temperature,
                model_env_var=model_env_var,
                chat_fn=chat_fn,
            )
        except Exception as exc:  # noqa: BLE001
            recorder.record_boundary(
                stage=stage,
                boundary=BOUNDARY_STAGE1_ROW_REPAIR,
                outcome=OUTCOME_CONTRACT_FAILED,
                attempts=attempt,
                response_status="error",
                error=f"{type(exc).__name__}: {exc}",
                requested_row_count=len(requests),
            )
            result.outcome = REPAIR_INCOMPLETE
            result.unrepaired_pointers = sorted(pending)
            result.reason = f"row repair call failed: {type(exc).__name__}: {exc}"
            return result

        parsed = _parse_object(text)
        returned = _safe_list(_safe_dict(parsed).get("repaired_rows")) if parsed else []
        applied, rejected = _apply_repaired_rows(working, requests, returned)
        result.repaired_pointers.extend(p for p in applied if p not in result.repaired_pointers)
        result.rejected.extend(rejected)

        terminal = _text(call_diag.get("terminal_reason"))
        record = recorder.record_boundary(
            stage=stage,
            boundary=BOUNDARY_STAGE1_ROW_REPAIR,
            model=_text(call_diag.get("model")),
            finish_reason=_text(call_diag.get("finish_reason")),
            attempts=attempt,
            response_status=_text(call_diag.get("response_status")),
            terminal_reason=terminal,
            outcome=OUTCOME_OK if applied else OUTCOME_CONTRACT_FAILED,
            request_hash=payload_hash(requests),
            response_hash=payload_hash(text),
            raw_chars=len(text),
            raw_preview=text if parsed is None else "",
            repair_kind="contract_rows",
            requested_row_count=len(requests),
            requested_rows_dropped=dropped or None,
            repaired_row_count=len(applied),
            rejected_row_count=len(rejected),
            rejected_sample=rejected[:MAX_SAMPLE] or None,
        )
        result.diagnostics.append(record)
        if dropped:
            logger.warning(
                "row repair sent %d of %d failing rows; %d were left for a later pass "
                "because one request is capped at %d rows.",
                len(requests),
                len(requests) + dropped,
                dropped,
                MAX_ROWS_PER_REQUEST,
            )

        if terminal == "content_filter":
            result.outcome = REPAIR_OK if not pending else REPAIR_INCOMPLETE
            result.unrepaired_pointers = sorted(pending)
            result.reason = (
                "row repair stopped on a content_filter finish_reason; the identical "
                "prompt cannot be retried and no further attempt was made"
            )
            if applied and revalidate is not None:
                pending = _errors_by_row(revalidate(working))
                result.unrepaired_pointers = sorted(pending)
                result.outcome = REPAIR_OK if not pending else REPAIR_INCOMPLETE
            return result

        if revalidate is None:
            result.outcome = REPAIR_OK if applied else REPAIR_INCOMPLETE
            result.unrepaired_pointers = [] if applied else sorted(pending)
            if not applied:
                result.reason = "the repair returned no usable row"
            return result

        pending = _errors_by_row(revalidate(working))
        if not pending:
            result.outcome = REPAIR_OK
            result.unrepaired_pointers = []
            return result

    result.outcome = REPAIR_INCOMPLETE
    result.unrepaired_pointers = sorted(pending)
    result.reason = (
        f"row repair exhausted its {budget}-attempt cap with "
        f"{len(pending)} row(s) still failing the contract"
    )
    return result


def _repair_request_payload(request: Mapping[str, Any]) -> Dict[str, Any]:
    """One request row as the MODEL sees it: no guard bookkeeping."""

    return {key: value for key, value in request.items() if not str(key).startswith("_")}


def _apply_repaired_rows(
    working: Dict[str, Any],
    requests: Sequence[Mapping[str, Any]],
    returned: Sequence[Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Splice accepted rows into ``working``; report every row refused and why.

    Four refusals, checked in this order, each a rule from the module docstring
    made executable:

    1. ``pointer_not_requested`` -- a row nobody complained about;
    2. ``row_not_an_object`` -- not a row at all;
    3. ``original_value_removed`` / ``immutable_field_altered`` /
       ``unrelated_field_altered`` -- the repair DELETED or rewrote a fact
       (:func:`preserves_original_values`);
    4. ``evidence_unsupported`` -- the repair ADDED a claim its evidence does not
       carry (:func:`evidence_supports`).

    Deletion is checked before invention deliberately. Both are refusals, but a
    repair that quietly drops two of a reaction's four inputs passes an
    invention check trivially, so running grounding first would report the less
    important finding on a row that failed both.
    """

    by_pointer = {_text(item.get("pointer")): item for item in requests}
    applied: List[str] = []
    rejected: List[Dict[str, Any]] = []

    for item in returned:
        data = _safe_dict(item)
        pointer = _text(data.get("pointer"))
        request = by_pointer.get(pointer)
        if request is None:
            rejected.append(
                {
                    "pointer": clip(pointer, 120),
                    "reason": "pointer_not_requested",
                }
            )
            continue
        row = data.get("row")
        if not isinstance(row, Mapping):
            rejected.append({"pointer": pointer, "reason": "row_not_an_object"})
            continue

        preserved, losses = preserves_original_values(
            row, request.get("row"), set(_safe_list(request.get("_target_fields")))
        )
        if not preserved:
            rejected.append(
                {
                    "pointer": pointer,
                    "reason": str(losses[0].get("reason") or "original_value_removed"),
                    "lost": losses,
                }
            )
            continue

        ok, unsupported = evidence_supports(row, request.get("row"), _text(request.get("evidence")))
        if not ok:
            rejected.append(
                {
                    "pointer": pointer,
                    "reason": "evidence_unsupported",
                    "unsupported": unsupported,
                }
            )
            continue
        if _assign_pointer(working, pointer, dict(row)):
            applied.append(pointer)
        else:
            rejected.append({"pointer": pointer, "reason": "pointer_no_longer_exists"})

    return applied, rejected


# ---------------------------------------------------------------------------
# 3. Deterministic registry reconstruction. No model involved.
# ---------------------------------------------------------------------------
#: Participant role -> the entity bucket a shell for it belongs in, or ``None``
#: when the bucket must be read off the actor row itself.
#:
#: For a reaction's ``inputs``/``outputs`` the bucket is STATIC and structural,
#: not a new biological claim: in this schema those participants ARE the
#: reaction's compounds, so ``compounds`` restates a role the payload already
#: asserted.
#:
#: For an ACTOR role it is not. "Enzyme" spans two buckets -- a single protein
#: and a protein complex are different entities with different identity gates,
#: different PWML shapes and different PathBank tables -- and picking one because
#: it is the commoner case is a guess dressed as bookkeeping. A complex demoted
#: to ``entities.proteins`` acquires a claim nobody made (that it is one gene
#: product) and then fails, or worse passes, the wrong gate. So actor roles carry
#: ``None`` and :func:`_actor_entries` reads the type the payload states; an
#: actor that states none is SKIPPED with a reason rather than assigned.
_ROLE_BUCKETS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("reactions", "inputs", "compounds"),
    ("reactions", "outputs", "compounds"),
    ("reactions", "left", "compounds"),
    ("reactions", "right", "compounds"),
    ("reactions", "substrates", "compounds"),
    ("reactions", "products", "compounds"),
    ("reactions", "enzymes", None),
    ("reactions", "modifiers", None),
    ("transports", "cargo", "compounds"),
    ("transports", "transporters", None),
    ("reaction_coupled_transports", "enzymes", None),
)

#: Roles whose members are actors (proteins or protein complexes) rather than
#: small molecules.
_ACTOR_ROLES = frozenset({"enzymes", "modifiers", "transporters"})

#: ``entity_type`` values that name an actor bucket. Anything else -- absent,
#: blank, ``compound``, a typo -- is not a type this module will act on.
_ENTITY_TYPE_BUCKETS: Dict[str, str] = {
    "protein": "proteins",
    "protein_complex": "protein_complexes",
}

#: Buckets consulted when deciding whether a name is already registered. A name
#: present in ANY of them is registered; a compound declared as a protein is a
#: mapping problem, not a missing row, and inventing a second shell for it would
#: create the duplicate the dedupe pass then has to undo.
_REGISTRY_BUCKETS: Tuple[str, ...] = (
    "compounds",
    "proteins",
    "protein_complexes",
    "nucleic_acids",
    "element_collections",
)


def _normalize_name(value: Any) -> str:
    lowered = re.sub(r"\s+", " ", _text(value).strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def _actor_entries(value: Any) -> List[Tuple[str, Optional[str]]]:
    """``(name, bucket)`` for each actor, with ``bucket=None`` when untyped.

    The type is read off the row in the order the payload states it, and every
    source is an explicit statement rather than an inference:

    * ``protein_complex`` / ``protein`` keys -- what ``_clean_enzymes`` emits, and
      the ONLY thing that survives cleaning. ``entity_type`` is deliberately
      dropped there (see its docstring), so post-cleaning these keys are the type.
    * ``entity`` plus an ``entity_type`` of ``protein`` / ``protein_complex`` --
      the typed modifier form, present on raw pre-cleaning payloads.
    * a bare ``name``, or a bare string: an actor with NO stated type. Returned
      with ``bucket=None`` so the caller skips it and says why. Guessing here is
      how a complex silently becomes a protein.
    """

    out: List[Tuple[str, Optional[str]]] = []
    for item in _safe_list(value):
        if isinstance(item, str):
            if item.strip():
                out.append((item.strip(), None))
            continue
        if not isinstance(item, Mapping):
            continue
        declared = str(item.get("entity_type") or item.get("type") or "").strip().casefold()
        complex_name = _text(item.get("protein_complex")).strip()
        if complex_name:
            out.append((complex_name, "protein_complexes"))
            continue
        protein_name = _text(item.get("protein")).strip()
        if protein_name:
            out.append((protein_name, "proteins"))
            continue
        entity_name = _text(item.get("entity")).strip()
        if entity_name:
            out.append((entity_name, _ENTITY_TYPE_BUCKETS.get(declared)))
            continue
        bare = _text(item.get("name")).strip()
        if bare:
            out.append((bare, _ENTITY_TYPE_BUCKETS.get(declared)))
    return out


def _participant_entries(
    row: Mapping[str, Any],
    role: str,
    bucket: Optional[str],
) -> List[Tuple[str, Optional[str]]]:
    """``(name, bucket)`` for every participant of ``role`` on ``row``."""

    value = row.get(role)
    if role in _ACTOR_ROLES:
        return _actor_entries(value)
    if isinstance(value, str):
        return [(value.strip(), bucket)] if value.strip() else []
    return [
        (item.strip(), bucket)
        for item in _safe_list(value)
        if isinstance(item, str) and item.strip()
    ]


def _participant_names(row: Mapping[str, Any], role: str) -> List[str]:
    """Just the names, for the row-validity test."""

    return [name for name, _bucket in _participant_entries(row, role, None) if name]


def _row_is_valid_process(bucket: str, row: Any) -> bool:
    """Whether a process row is one the payload actually keeps.

    Reconstruction reads participants from VALID rows only. A row that cleaning
    is going to discard anyway (a reaction with no outputs, say) is not evidence
    that its names belong in the registry -- adding them would populate the
    registry from rows that never ship, which is fabrication by another route.
    The tests here mirror ``pipeline._clean_processes``' own guards.
    """

    if not isinstance(row, Mapping):
        return False
    if bucket == "reactions":
        return bool(
            [n for n in _participant_names(row, "inputs") if n.strip()]
            and [n for n in _participant_names(row, "outputs") if n.strip()]
        )
    if bucket == "transports":
        return bool(
            _text(row.get("cargo")).strip()
            or _safe_list(row.get("transporters"))
            or _safe_list(row.get("elements_with_states"))
        )
    if bucket == "reaction_coupled_transports":
        return bool(
            _text(row.get("reaction")).strip()
            or _text(row.get("transport")).strip()
            or _safe_list(row.get("elements_with_states"))
        )
    return False


def reconstruct_registry_shells(
    payload: Mapping[str, Any],
    *,
    roles: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> ReconstructionResult:
    """Add an unresolved entity row for every participant the registry is missing.

    Deterministic and model-free: the payload already names these participants,
    so recovering the registry row is bookkeeping, not inference. This closes the
    failure where an extraction is structurally incomplete -- the reactions are
    right and the registry that indexes them is short -- and
    ``filter_unresolvable_reactions`` therefore deletes correct, evidenced
    reactions for a missing row rather than for a missing fact.

    What a shell carries, and nothing else::

        {"name": <the name the process used>,
         "provenance": "reconstructed_from_process_participant",
         "resolution_status": "unresolved",
         "reconstructed_from": {"pointer": ..., "role": ...}}

    No ``mapped_ids``, no ``uniprot``, no ``ec_number``, no ``hmdb_id``, no
    organism, no description. The row is a declared gap, and every identity gate
    downstream must still refuse it -- which is the point. A shell makes the
    incompleteness explicit and reviewable; it does not launder it into an
    identified entity.
    """

    working = deepcopy(dict(payload))
    result = ReconstructionResult(working)
    processes = _safe_dict(working.get("processes"))
    if not processes:
        return result

    entities = working.get("entities")
    if not isinstance(entities, dict):
        entities = {}
        working["entities"] = entities

    known: Dict[str, str] = {}
    for bucket in _REGISTRY_BUCKETS:
        for item in _safe_list(entities.get(bucket)):
            name = item.get("name") if isinstance(item, Mapping) else item
            normalized = _normalize_name(name)
            if normalized:
                known.setdefault(normalized, bucket)

    for process_bucket, role, default_bucket in roles or _ROLE_BUCKETS:
        rows = _safe_list(processes.get(process_bucket))
        for index, row in enumerate(rows):
            pointer = f"/processes/{_escape_token(process_bucket)}/{index}"
            if not _row_is_valid_process(process_bucket, row):
                if isinstance(row, Mapping):
                    _note_skip(result, pointer, role, "row_is_not_a_valid_process")
                continue
            for name, stated_bucket in _participant_entries(row, role, default_bucket):
                clean = name.strip()
                normalized = _normalize_name(clean)
                if not normalized:
                    continue
                if normalized in known:
                    continue
                target_bucket = stated_bucket
                if target_bucket is None:
                    # An actor whose type nobody stated. A shell has to live in
                    # one bucket, and choosing between proteins and
                    # protein_complexes without a statement would manufacture the
                    # identity claim this whole function exists not to make. So
                    # the gap is reported and left open.
                    _note_skip(result, pointer, role, "actor_entity_type_unknown", name=clean)
                    continue
                bucket_rows = entities.setdefault(target_bucket, [])
                if not isinstance(bucket_rows, list):
                    _note_skip(result, pointer, role, "target_bucket_is_not_a_list")
                    continue
                bucket_rows.append(
                    {
                        "name": clean,
                        "provenance": SHELL_PROVENANCE,
                        "resolution_status": "unresolved",
                        "reconstructed_from": {"pointer": f"{pointer}/{role}", "role": role},
                    }
                )
                known[normalized] = target_bucket
                result.shells.append(
                    {
                        "name": clip(clean, 120),
                        "bucket": target_bucket,
                        "role": role,
                        "pointer": f"{pointer}/{role}",
                    }
                )

    if result.shells:
        logger.info(
            "reconstruct_registry_shells: added %d unresolved entity shell(s) for "
            "participants named by valid process rows but absent from the registry.",
            len(result.shells),
        )
    return result


def _note_skip(
    result: ReconstructionResult,
    pointer: str,
    role: str,
    reason: str,
    *,
    name: str = "",
) -> None:
    """Record a participant reconstruction deliberately did not act on.

    A skip is a finding, not a silence: ``actor_entity_type_unknown`` tells a
    reviewer that a named enzyme has no registry row *and* why one was not
    invented, which is the difference between a reported gap and a lost one.
    """

    entry: Dict[str, Any] = {"pointer": pointer, "role": role, "reason": reason}
    if name:
        entry["name"] = clip(name, 120)
    if entry not in result.skipped:
        result.skipped.append(entry)
