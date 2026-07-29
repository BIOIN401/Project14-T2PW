"""Bounded, evidence-free ``detail`` payloads for gate and stage-contract errors.

An error names the offending row and the rule that rejected it, but not the
value that was checked. ``detail`` carries that value, riding the error dict
itself so the UI, ``gate_fail_report.json`` and
``stage_contract_error_report.json`` all get it from one population site.

Everything is bounded *here*, at the point of capture, not at the render site:
payload ``evidence`` fields have reached six figures of characters, and one of
those in ``st.json`` is a hung browser tab. Bulky provenance keys
(:data:`ELIDED_KEYS`) become a size census; every other scalar is clipped to
:data:`MAX_VALUE_CHARS`; containers are capped in width and depth.

Rationale, the observed sizes, and the compatibility constraints are in
``docs/regression_baseline_2026-07-28.md`` §4.
"""

from __future__ import annotations

import difflib
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence

__all__ = [
    "MAX_VALUE_CHARS",
    "MAX_LIST_ITEMS",
    "MAX_DICT_KEYS",
    "MAX_DEPTH",
    "ELIDED_KEYS",
    "HEADLINE_KEYS",
    "clip",
    "census",
    "scrub",
    "scrub_detail",
    "row_digest",
    "closest_names",
    "headline",
    "detail",
]

#: Longest string kept for any single scalar. Past this the tail is dropped and
#: the dropped count is stated inline.
MAX_VALUE_CHARS = 240

#: Longest list kept in a detail; the remainder becomes a trailing count marker.
MAX_LIST_ITEMS = 12

#: Most keys kept from any one mapping. Payload rows run ~10-20 keys, so this is
#: slack rather than a real constraint -- it only bites on unexpected shapes.
MAX_DICT_KEYS = 24

#: Nesting past this collapses to a census. Entity rows nest at most
#: row -> components -> component -> mapped_ids.
MAX_DEPTH = 4

#: Keys whose values are verbatim source text or provenance blobs. Never copied,
#: always censused. Matched case-insensitively against the exact key name.
ELIDED_KEYS = frozenset(
    {
        "abstract",
        "body",
        "chunk",
        "chunk_text",
        "chunks",
        "content",
        "evidence",
        "full_text",
        "passage",
        "passages",
        "quote",
        "quotes",
        "rag_provenance",
        "raw",
        "raw_response",
        "source_papers",
        "source_refs",
        "source_text",
        "text",
    }
)


def _json_len(value: Any) -> int:
    """Serialized size of ``value``, used only to report what was elided."""

    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):  # pragma: no cover - default=str makes this rare
        return len(str(value))


def clip(value: Any, limit: int = MAX_VALUE_CHARS) -> str:
    """``value`` as a string, cut to ``limit`` chars, stating what was dropped."""

    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars elided)"


#: Every :func:`census` output for a container or string ends with this, which
#: is what makes an already-censused value recognisable on a second pass.
_CENSUS_SUFFIX = " elided"

#: A census is a short generated sentence; the longest possible is around 45
#: chars. The bound stops a genuine payload string that happens to end in
#: "elided" from being mistaken for one -- and if a string that short ever were,
#: passing it through costs nothing, because it is already inside the clip limit.
_MAX_CENSUS_CHARS = 80


def _is_census(value: Any) -> bool:
    """Whether ``value`` is already the output of :func:`census`."""

    return (
        isinstance(value, str)
        and value.endswith(_CENSUS_SUFFIX)
        and len(value) <= _MAX_CENSUS_CHARS
    )


def census(value: Any) -> str:
    """A size description of ``value`` that contains none of its content.

    Idempotent, and that is load-bearing: a detail is censused up to three times
    on its way to an error, and without the guard each pass would measure the
    previous census string instead of the data. See the doc, §4.
    """

    if _is_census(value):
        return value
    if isinstance(value, str):
        return f"{len(value)} chars elided"
    if isinstance(value, Mapping):
        return f"{len(value)} key(s), {_json_len(value)} chars elided"
    if isinstance(value, (list, tuple, set)):
        return f"{len(value)} item(s), {_json_len(value)} chars elided"
    if value is None:
        return "null"
    return clip(value)


def scrub(value: Any, *, depth: int = 0) -> Any:
    """Recursively bound ``value`` to a size safe for ``st.json`` and disk.

    Scalars survive as themselves (strings clipped); containers are capped in
    width; anything under an :data:`ELIDED_KEYS` key or past :data:`MAX_DEPTH`
    collapses to a :func:`census` string.
    """

    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return clip(value)

    if depth >= MAX_DEPTH:
        return census(value)

    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= MAX_DICT_KEYS:
                out["…"] = f"+{len(value) - MAX_DICT_KEYS} more key(s) elided"
                break
            name = str(key)
            if name.casefold() in ELIDED_KEYS:
                out[name] = census(item)
            else:
                out[name] = scrub(item, depth=depth + 1)
        return out

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        kept = [scrub(item, depth=depth + 1) for item in items[:MAX_LIST_ITEMS]]
        if len(items) > MAX_LIST_ITEMS:
            kept.append(f"…+{len(items) - MAX_LIST_ITEMS} more item(s) elided")
        return kept

    return clip(value)


def scrub_detail(value: Any) -> Dict[str, Any]:
    """The public entry point: a bounded dict, or ``{}`` when there is nothing.

    Returning ``{}`` rather than ``None`` for empty input lets call sites write
    ``if scrubbed:`` and keeps the ``detail`` key off errors that have no detail,
    so existing consumers of a bare ``{"path", "reason"}`` error see no change.
    """

    if not isinstance(value, Mapping) or not value:
        return {}
    scrubbed = scrub(value)
    return scrubbed if isinstance(scrubbed, dict) else {}


def row_digest(row: Any, *, pointer: str = "") -> Dict[str, Any]:
    """A bounded, evidence-free projection of a payload row.

    A denylist plus universal clipping, not an allowlist: gate failures keep
    arriving as new entity shapes, and an allowlist drops exactly the unfamiliar
    field that would have explained the failure.
    """

    if isinstance(row, Mapping):
        digest = scrub_detail(row)
    elif row is None:
        digest = {}
    else:
        digest = {"value": clip(row), "type": type(row).__name__}
    if pointer:
        digest = {"pointer": pointer, **digest}
    return digest


def closest_names(
    name: str,
    candidates: Iterable[str],
    *,
    limit: int = 3,
    cutoff: float = 0.6,
) -> List[str]:
    """Registry names closest to ``name`` — the "did you mean" for a bad reference.

    An unresolved actor is usually a near-miss, not an absent entity. ``[]`` is
    itself informative: genuinely unknown rather than merely misspelled.
    """

    token = str(name or "").strip()
    if not token:
        return []
    pool = sorted({str(item).strip() for item in candidates if str(item).strip()})
    if not pool:
        return []
    return difflib.get_close_matches(token, pool, n=max(1, limit), cutoff=cutoff)


#: Detail keys worth putting on a one-line summary, in the order a reviewer asks
#: for them: what was referenced, what the payload should have said instead, what
#: the value turned out to be. Everything else stays in the full detail.
HEADLINE_KEYS: Sequence[str] = (
    "did_you_mean",
    "actor_name",
    "component_name",
    "protein",
    "token",
    "found_type",
    "found_value",
    "registry_size",
    "declared_protein_count",
)


def headline(value: Any, *, limit: int = 3) -> str:
    """The one line of a detail worth showing before it is expanded.

    Lives here rather than in the app because the batch flag rows want the same
    summary, and two implementations would drift. ``""`` when there is nothing
    headline-worthy.
    """

    if not isinstance(value, Mapping) or not value:
        return ""
    parts: List[str] = []
    for key in HEADLINE_KEYS:
        if key not in value:
            continue
        item = value[key]
        if item is None or item == "" or item == [] or item == {}:
            continue
        if isinstance(item, (list, tuple)):
            item = ", ".join(str(entry) for entry in item)
        parts.append(f"{key}={item}")
        if len(parts) >= limit:
            break
    return clip(" · ".join(parts))


def detail(**fields: Any) -> Dict[str, Any]:
    """Build a bounded detail dict, dropping keys whose value is empty.

    Empty means ``None``, ``""``, ``[]`` or ``{}`` — never ``0`` or ``False``,
    which are real findings ("degree 0", "generated: false") and must survive.
    """

    populated = {
        key: value
        for key, value in fields.items()
        if value is not None and (value or isinstance(value, (int, float)))
    }
    return scrub_detail(populated)
