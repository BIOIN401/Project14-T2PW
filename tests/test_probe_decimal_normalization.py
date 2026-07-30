"""PROBE 3 — numeric comparison that must be exact, not merely approximate.

The content guard compares the malformed input's literals against the repaired
document's. Numbers need normalizing before that comparison, because
re-serializing a number is a formatting choice the repair is allowed to make:
``1e3`` and ``1000.0`` are the same number and must compare equal.

The first implementation normalized through ``float``, which buys that
equivalence and gives away exactness:

* every integer above 2**53 collapses. ``9007199254740993`` and
  ``9007199254740992`` are *the same float*, so a repair that altered a large
  identifier — a taxonomy id, a PathBank row id, a PubChem CID — compared equal
  to the original and passed a guard whose entire job is to notice that;
* long decimals round to 17 significant digits, so two distinguishable values
  collide the same way.

``Decimal`` gives both properties at once: ``1e3 == 1000.0`` and
``9007199254740993 != 9007199254740992``. This file probes the equivalences that
must hold, the distinctions that must not be lost, and the parsing path that
makes the second possible — ``json.loads`` alone turns a high-precision decimal
into a float before the guard ever sees it, so an honest repair of one would have
been refused.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.localized_repair import (  # noqa: E402
    _normalize_number,
    json_repair_preserves_content,
    parse_exact,
)


def _guard(raw: str, repaired_text: str) -> Any:
    """Compare through the same path production uses, exact parsing included."""

    return json_repair_preserves_content(
        raw, json.loads(repaired_text), repaired_text=repaired_text
    )


# ---------------------------------------------------------------------------
# The equivalences that must survive.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "left, right",
    [
        ("1e3", "1000.0"),
        ("1e3", "1000"),
        ("1E+3", "1000.00"),
        ("1.5e2", "150"),
        ("0.1", "0.10"),
        ("-0.0", "0"),
        ("2", "2.0"),
        ("1e-3", "0.001"),
        ("1000000", "1e6"),
    ],
)
def test_formatting_differences_compare_equal(left: str, right: str) -> None:
    """Re-serializing a number is a formatting choice, not a change.

    A guard that fired on these would fire on honest work, which is how guards
    get switched off.
    """

    assert _normalize_number(left) == _normalize_number(right)


def test_1e3_and_1000_are_equal_through_the_full_guard() -> None:
    """Named explicitly because it is the property the float version bought."""

    ok, _ = _guard('{"confidence": 1e3,}', '{"confidence": 1000.0}')

    assert ok is True


# ---------------------------------------------------------------------------
# The distinctions that must not be lost.
# ---------------------------------------------------------------------------
#: 2**53 is where float stops being able to count. These two integers are one
#: apart and are the same float.
BELOW = "9007199254740992"
ABOVE = "9007199254740993"


def test_the_two_test_integers_really_are_indistinguishable_as_floats() -> None:
    """If this ever fails the probe below has stopped proving anything."""

    assert float(BELOW) == float(ABOVE)
    assert int(BELOW) != int(ABOVE)


def test_integers_above_2_to_the_53_stay_distinct() -> None:
    assert _normalize_number(BELOW) != _normalize_number(ABOVE)
    assert _normalize_number(ABOVE) == ABOVE


def test_a_changed_large_identifier_is_caught_by_the_guard() -> None:
    """The attack the float version let through.

    A PathBank row id, a taxonomy id or a PubChem CID altered by one digit in the
    16th place is exactly the change nobody spots by eye and the guard exists to
    spot.
    """

    ok, finding = _guard(
        '{"pathbank_protein_id": %s,}' % BELOW,
        '{"pathbank_protein_id": %s}' % ABOVE,
    )

    assert ok is False
    assert finding["divergence"] == "added_or_reordered"
    assert finding["first_unmatched"] == ABOVE


def test_the_same_large_identifier_returned_unchanged_is_accepted() -> None:
    ok, _ = _guard('{"pathbank_protein_id": %s,}' % ABOVE, '{"pathbank_protein_id": %s}' % ABOVE)

    assert ok is True


@pytest.mark.parametrize(
    "left, right",
    [
        # 20 significant digits: float keeps 17 and loses the tail.
        ("0.12345678901234567890", "0.12345678901234567891"),
        ("123456789012345678901", "123456789012345678902"),
        ("1.0000000000000000001", "1.0000000000000000002"),
    ],
)
def test_arbitrary_precision_decimals_stay_distinct(left: str, right: str) -> None:
    assert float(left) == float(right)  # the collision the float version had
    assert _normalize_number(left) != _normalize_number(right)


def test_a_changed_high_precision_decimal_is_caught_by_the_guard() -> None:
    ok, finding = _guard(
        '{"confidence": 0.12345678901234567890,}',
        '{"confidence": 0.12345678901234567891}',
    )

    assert ok is False
    assert finding["divergence"] == "added_or_reordered"


def test_an_unchanged_high_precision_decimal_is_accepted() -> None:
    """The half that needs ``parse_exact``.

    ``json.loads`` alone would turn this into a float and drop the tail, so the
    repaired stream would no longer match the source text and an HONEST repair
    would be refused. Exact parsing on the guard's side is what prevents that.
    """

    body = '{"confidence": 0.12345678901234567890}'

    ok, finding = _guard('{"confidence": 0.12345678901234567890,}', body)

    assert ok is True, finding


def test_parse_exact_keeps_decimals_and_plain_parsing_does_not() -> None:
    """The two parsers, side by side, so the reason for having both is visible."""

    body = '{"confidence": 0.12345678901234567890}'

    exact = parse_exact(body)
    plain = json.loads(body)

    assert isinstance(exact["confidence"], Decimal)
    assert isinstance(plain["confidence"], float)
    assert str(exact["confidence"]) == "0.12345678901234567890"
    assert str(plain["confidence"]) != "0.12345678901234567890"


def test_the_payload_the_pipeline_receives_never_carries_decimals() -> None:
    """``Decimal`` is for the guard only.

    It is not JSON round-trippable and would leak an unexpected type into every
    downstream stage, so the repaired payload is parsed normally and only the
    comparison uses the exact form.
    """

    from t2pw.pipeline.localized_repair import repair_json_text

    body = '{"entities": {"compounds": [{"name": "ATP", "confidence": 0.5}]}}'

    class _Reply:
        def __init__(self, text: str) -> None:
            self.text = text
            self.diagnostics = {"model": "m", "finish_reason": "stop", "attempts": 1,
                                "response_status": "ok", "terminal_reason": "", "attempt_log": []}

    result = repair_json_text(
        '{"entities": {"compounds": [{"name": "ATP", "confidence": 0.5,}]}}',
        "JSONDecodeError",
        chat_fn=lambda *_a, **_k: _Reply(body),
    )

    assert result.ok
    confidence = result.payload["entities"]["compounds"][0]["confidence"]
    assert isinstance(confidence, float)
    assert not isinstance(confidence, Decimal)
    json.dumps(result.payload)  # must round-trip


# ---------------------------------------------------------------------------
# Robustness.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["", "not a number", "NaN", "Infinity", "--3", None])
def test_normalization_never_raises_on_a_value_that_is_not_a_number(value: Any) -> None:
    """It runs on malformed input by definition; raising takes out the guard."""

    assert isinstance(_normalize_number(value), str)


def test_a_float_is_normalized_through_its_repr_not_its_binary_expansion() -> None:
    """``Decimal(0.1)`` is 0.1000000000000000055511151231257827.

    Comparing that against the ``0.1`` written in the source text would never
    match, so floats are routed through ``str`` first.
    """

    assert _normalize_number(0.1) == _normalize_number("0.1")
    assert _normalize_number(0.1) == "0.1"


def test_booleans_are_not_treated_as_numbers() -> None:
    """``True == 1`` in Python, and a guard that normalized it to ``1`` would
    accept ``true`` where the input said ``1``."""

    ok, _ = _guard('{"flag": true, "count": 1,}', '{"flag": true, "count": 1}')
    assert ok is True

    swapped, _ = _guard('{"flag": true,}', '{"flag": 1}')
    assert swapped is False


def test_a_large_integer_survives_the_full_repair_path_unchanged() -> None:
    """End to end: the value that reaches the payload is still the value sent."""

    from t2pw.pipeline.localized_repair import repair_json_text

    body = '{"entities": {"species": [{"name": "E. coli", "taxonomy_id": %s}]}}' % ABOVE

    class _Reply:
        def __init__(self, text: str) -> None:
            self.text = text
            self.diagnostics = {"model": "m", "finish_reason": "stop", "attempts": 1,
                                "response_status": "ok", "terminal_reason": "", "attempt_log": []}

    result = repair_json_text(
        '{"entities": {"species": [{"name": "E. coli", "taxonomy_id": %s,}]}}' % ABOVE,
        "JSONDecodeError",
        chat_fn=lambda *_a, **_k: _Reply(body),
    )

    assert result.ok
    assert result.payload["entities"]["species"][0]["taxonomy_id"] == int(ABOVE)
