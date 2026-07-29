"""Bounded, evidence-free ``detail`` payloads for gate and stage-contract errors.

Why this module exists
----------------------
A gate error used to be ``{"path", "reason"}`` and a stage-contract error
``{"code", "message", "pointer"}``. Both name the offending row and describe the
rule that rejected it, and neither carries the thing a reviewer actually needs:
the value that was checked. That value is in scope at every ``_add_error`` call
site and was thrown away there.

``logger.debug`` cannot fix this. The Streamlit app's logs are not visible from
the browser, so anything written to a logger is invisible exactly when it is
needed. Detail therefore rides the error object itself — the same dict that
already flows to the UI, to ``gate_fail_report.json`` and to
``stage_contract_error_report.json``. One population site, three destinations,
and no way for the app and the batch artifact to disagree about what happened.

The size constraint
-------------------
Payload rows carry ``evidence`` / ``source_refs`` blobs holding verbatim
passages from the source paper. One observed run reached 139,576 characters in a
single such field. Embedding a raw row in an error would push that into
``st.json`` and wedge the browser tab, so every value is bounded *here*, at the
point of capture. Bounding at the render site instead would be a truncation each
new call site could forget, and a forgotten truncation is the bug returning.

Two rules make the bound trustworthy:

* Bulky provenance keys (:data:`ELIDED_KEYS`) never have their content copied.
  They become a census — ``"3 items, 139576 chars elided"`` — because the fact
  that evidence is huge is itself diagnostic, and dropping the key outright
  would read as "this row had no evidence". This is unconditional: small
  evidence is still raw evidence, and a size-dependent rule would make the
  guarantee depend on the input.
* Every surviving scalar is clipped to :data:`MAX_VALUE_CHARS` with the elided
  count stated inline, and containers are capped in width and depth, so no
  single field can dominate the display.

The result is a hard ceiling on a detail's serialized size that does not depend
on how large the payload was.

What to put in a detail
-----------------------
The offending row (via :func:`row_digest`), the specific value compared, and the
comparison set that failed to contain it. :func:`closest_names` exists because
the dominant recurring gate failure is a name that *nearly* matches the
registry — an actor the extractor or RAG synthesis spelled differently — and
"did you mean" turns a 20-minute grep into a glance.
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

    This is what an evidence field becomes. It answers "was there evidence, and
    how much" without reproducing a word of it.

    Idempotent, which is load-bearing rather than tidiness. A detail is commonly
    built as ``build_detail(row=row_digest(row))`` and then scrubbed again by
    ``_add_error``, so the same evidence field is censused up to three times.
    Without this guard each pass measures the previous census string instead of
    the data -- ``"160000 chars elided"`` becomes ``"19 chars elided"`` becomes
    ``"15 chars elided"`` -- and the count, the one thing the reader needs, is
    silently replaced by a number about itself.
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

    Deliberately a denylist (:data:`ELIDED_KEYS`) plus universal clipping rather
    than an allowlist of interesting fields. Gate failures keep arriving as
    *new* entity shapes — a RAG-synthesized actor with a key the extractor never
    produced — and an allowlist drops exactly the unfamiliar field that would
    have explained the failure.
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

    An unresolved actor is almost always a near-miss (case, punctuation, a
    subunit suffix, a slash-joined complex) rather than an entity that is truly
    absent, so the nearest declared names are the fastest route to the cause.
    Returns ``[]`` when nothing is close enough, which is itself informative:
    the reference is genuinely unknown, not merely misspelled.
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

    A failure usually turns on a single field — ``did_you_mean`` for a bad
    reference, ``found_type`` for a shape error — and burying it in a collapsed
    blob costs the click the detail existed to save. Lives here rather than in
    the app because the batch flag rows want the same summary, and two
    implementations would drift into disagreeing about the same error.

    Returns ``""`` when nothing headline-worthy is present.
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
