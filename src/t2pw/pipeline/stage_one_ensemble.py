"""N-draw ensembling for Stage 1 extraction (ORCH-750 (dispatched as ORCH-750; renumbered because another session had already claimed 748 in the ledger for the pathway-mining lane)). OFF by default.

``docs/evaluation_v2/PAPER_FINDINGS.md`` §§ 5.3, 5.5: single-draw reaction precision is
near-ceiling, recall varies wildly between identical temperature-0 draws (9, then 7, then
4 matched reactions for one prompt on one paper), and each draw misses DIFFERENT
reactions -- so the UNION of 3 draws reached recall 1.000 at precision 1.000 in 3 of 4
measured conditions, with no false row entering any pool. This module turns that
measurement into a setting.

The knob
--------
``T2PW_EXTRACTION_DRAWS``, or the ``draws=`` parameter of
:func:`t2pw.pipeline.pipeline.run_extraction_pipeline`, which overrides it. Default 1.
At 1 nothing in this module runs: ``run_extraction_pipeline`` issues exactly one
``_run_json_stage`` call with exactly the arguments it always has. Production (D-102) is
unchanged until someone sets the variable. Cost and latency scale linearly in N.

Merge semantics
---------------
Union across draws, first occurrence wins, its ``evidence`` kept verbatim. Rows are
deduplicated by the keys the v2 evaluator already uses, so a row this merge keeps as
distinct is a row the evaluator scores as distinct (``reactions.score_leg``, R-D092-2):

* reaction: (set of DEFINING input keys, set of DEFINING output keys). A cofactor --
  ``role: cofactor`` on the participant, or a name in :data:`CURRENCY_KEYS` -- never
  defines a reaction, exactly as the evaluator scores it (``gold.ROLES``; ORCH-751 § 2.2
  measured the exact-set key inflating pooled rows). A side holding nothing but currency
  keeps every name, so ``ADP + Pi -> ATP`` is not ``GDP + Pi -> GTP``. When a later row
  collapses onto a survivor, the survivor keeps its evidence and gains any cofactor the
  later row named on that side (ORCH-752)
* interaction: (entity_1, relationship, entity_2); the names strict-keyed, the relationship
  by the type the exporter will emit for it (``ir.PWML_INTERACTION_TYPES``: ``activates``
  and ``activation`` are one ``Activation``; a word that table does not know keeps its
  strict key, so ``binds`` stays apart from ``interacts_with``)
* transport: (cargo, from_biological_state, to_biological_state)
* reaction_coupled_transport: (reaction, transport)
* ``element_locations`` rows: (located entity, biological_state), the key
  ``pipeline._dedup_element_locations`` already uses
* any other row with a ``name`` (every entity registry, biological states,
  compartments, visualizations): the strict key of its name
* a row nothing above can key (all key fields blank): its JSON serialisation, so
  distinct unkeyable rows are never collapsed into one

``strict_key`` is :func:`t2pw.bench.eval_v2.matcher.strict_key` (NFKC, Greek spelled
out, casefold, trailing charge dropped, ``-ic acid`` -> ``-ate``, pyrophosphate ==
diphosphate). It is imported inside the function because ``t2pw.pipeline`` must not
import ``t2pw.bench`` at module level (release_status.py:112; strict_quarantine.py:2704
is the precedent seam).

The payload schema is frozen: the merge adds no key of its own, so which draw(s)
produced each row cannot live in the payload. It goes to the side file
``stage1_ensemble_report.json`` beside the run's other extraction diagnostics, and a
bounded summary goes into ``extraction_boundary_report.json`` as a ``stage1_ensemble``
boundary record.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from t2pw.pipeline.extraction_diagnostics import OUTCOME_OK, current as current_diagnostics
from t2pw.pipeline.participant_schema import PARTICIPANT_NAME_KEYS

logger = logging.getLogger(__name__)

ENV_VAR = "T2PW_EXTRACTION_DRAWS"
BOUNDARY_STAGE1_ENSEMBLE = "stage1_ensemble"
REPORT_NAME = "stage1_ensemble_report.json"

#: The buckets the bounded summary totals over; the side file carries every bucket.
_SUMMARISED_PREFIXES = ("/processes/", "/entities/")
#: The fields ``_dedup_element_locations`` reads the located entity from.
_LOCATION_ENTITY_KEYS = ("compound", "element_collection", "nucleic_acid", "protein", "element")

DrawFn = Callable[[], Tuple[Dict[str, Any], List[Dict[str, Any]]]]

#: Participants that never define a reaction's identity (ORCH-752, from ORCH-751 § 2.2):
#: the evaluator's ``cofactor`` role, which ``reactions.score_leg`` lists apart and never
#: fails a match on. Spelled as ``strict_key`` output -- casefolded, trailing charge dropped,
#: so ``NADP+`` is ``nadp`` and ``Mg2+`` is ``mg2`` -- and matched whole-name only: no
#: stemming, no fuzzy matching, so what collapses is readable from this table. A payload
#: participant carrying ``role: cofactor`` counts whatever its name. Known ceiling: a
#: chelatase's metal and ATP synthase's ADP/Pi ARE the chemistry, which is why a side made
#: only of currency keeps every name (``_side``). ``pipeline.cofactor_policy._CURRENCY`` is
#: the advisory cousin of this table: unwired by design, normalised differently, no metal
#: ions, so it is not imported.
CURRENCY_KEYS = frozenset({
    "atp", "adp", "amp",
    "nad", "nadh", "nadp", "nadph",
    "fad", "fadh2",
    "coa", "coenzyme a",                      # free CoA only; acyl-CoAs are defining
    "h2o", "water",
    "pi", "ppi", "phosphate", "diphosphate",  # strict_key spells pyrophosphate as diphosphate
    "h", "proton",                            # H+
    "co2", "o2",
    "sam", "s adenosyl l methionine", "s adenosylmethionine",
    "sah", "s adenosyl l homocysteine", "s adenosylhomocysteine",
    "mg2", "mg", "fe2", "fe3", "fe", "zn2", "zn", "mn2", "mn", "ca2", "ca", "cu2", "cu",
    "k", "na", "cl",
})


def extraction_draws(override: Optional[int] = None) -> int:
    """How many Stage-1 draws to issue: ``override`` if given, else the env var, else 1.

    A value that is not an integer >= 1 raises rather than silently falling back to one
    draw: a run that believes it ensembled and did not is the § 6.2 failure mode.
    """

    raw: Any = override if override is not None else (os.environ.get(ENV_VAR, "").strip() or "1")
    try:
        count = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"{ENV_VAR} must be an integer >= 1, got {raw!r}") from None
    if count < 1:
        raise ValueError(f"{ENV_VAR} must be an integer >= 1, got {raw!r}")
    return count


def _strict_key(value: Any) -> str:
    from t2pw.bench.eval_v2.matcher import strict_key  # function-local: layering, see module doc

    return strict_key(value)


def _participants(value: Any) -> List[Tuple[Any, str]]:
    """(item, name) per participant in a slot of strings or participant dicts, production key order."""

    out: List[Tuple[Any, str]] = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, str) and item.strip():
            out.append((item, item))
        elif isinstance(item, dict):
            name = next((item[k] for k in PARTICIPANT_NAME_KEYS if item.get(k)), None)
            if name:
                out.append((item, str(name)))
    return out


def _is_cofactor(item: Any, key: str) -> bool:
    role = item.get("role") if isinstance(item, dict) else None
    return str(role or "").strip().casefold() == "cofactor" or key in CURRENCY_KEYS


def _side(value: Any) -> Tuple[frozenset, List[Tuple[Any, str, str]]]:
    """One reaction side as (defining key set, cofactors as (item, name, key)).

    A side made only of currency has nothing incidental on it: its defining set is every key."""

    defining, every, cofactors = set(), set(), []
    for item, name in _participants(value):
        key = _strict_key(name)
        if not key:
            continue
        every.add(key)
        if _is_cofactor(item, key):
            cofactors.append((item, name, key))
        else:
            defining.add(key)
    return frozenset(defining or every), cofactors


def _relationship_key(value: Any) -> str:
    """The type the exporter emits for a relationship, else its strict key (see module doc)."""

    from t2pw.pwml.ir import PWML_INTERACTION_TYPES, _interaction_type_key  # function-local: layering

    text = str(value or "")
    return PWML_INTERACTION_TYPES.get(_interaction_type_key(text)) or _strict_key(text)


def _donate_cofactors(survivor: Dict[str, Any], later: Dict[str, Any]) -> List[str]:
    """Give ``survivor`` every cofactor ``later`` names on a side it does not already carry
    there. Its evidence and its own participants are untouched. Returns the names added."""

    added: List[str] = []
    for slot in ("inputs", "outputs"):
        have = survivor.get(slot)
        if not isinstance(have, list):
            continue
        present = {_strict_key(name) for _, name in _participants(have)}
        for item, name, key in _side(later.get(slot))[1]:
            if key not in present:
                have.append(deepcopy(item))
                present.add(key)
                added.append(name)
    return added


def _signature(row: Any) -> str:
    try:
        return json.dumps(row, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(row)


def row_key(path: str, row: Any) -> Tuple[Any, ...]:
    """The dedup key of one list item under JSON-pointer ``path`` (see module doc)."""

    if isinstance(row, str):
        return ("str", _strict_key(row) or row)
    if not isinstance(row, dict):
        return ("json", _signature(row))
    key: Tuple[Any, ...]
    if path == "/processes/reactions":
        key = (_side(row.get("inputs"))[0], _side(row.get("outputs"))[0])
    elif path == "/processes/interactions":
        key = (_strict_key(row.get("entity_1")), _relationship_key(row.get("relationship")),
               _strict_key(row.get("entity_2")))
    elif path == "/processes/transports":
        key = tuple(_strict_key(row.get(f))
                    for f in ("cargo", "from_biological_state", "to_biological_state"))
    elif path == "/processes/reaction_coupled_transports":
        key = tuple(_strict_key(row.get(f)) for f in ("reaction", "transport"))
    elif path.startswith("/element_locations/"):
        entity = next((row[k] for k in _LOCATION_ENTITY_KEYS if row.get(k)), "")
        key = (_strict_key(entity), _strict_key(row.get("biological_state")))
    else:
        key = (_strict_key(row.get("name")),)
    if not any(key):
        return ("json", _signature(row))
    return key


def _label(key: Tuple[Any, ...]) -> str:
    if len(key) == 2 and isinstance(key[0], frozenset):
        return " + ".join(sorted(key[0])) + " -> " + " + ".join(sorted(key[1]))
    if key and key[0] in ("str", "json"):
        return str(key[1])[:200]
    return " | ".join(str(part) for part in key)


def _row_name(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("name") or "")
    return str(row)[:200]


def _union(dest: List[Any], items: List[Any], path: str, draw: int,
           bucket: Dict[Tuple[Any, ...], Dict[str, Any]]) -> None:
    for item in items:
        key = row_key(path, item)
        record = bucket.get(key)
        if record is None:
            dest.append(deepcopy(item))
            # "_row" is the surviving object, for cofactor donation; merge_draws strips it.
            bucket[key] = {"key": _label(key), "name": _row_name(item), "draws": [draw], "_row": dest[-1]}
            continue
        if record["draws"][-1] != draw:
            record["draws"].append(draw)
        if path == "/processes/reactions" and isinstance(item, dict):
            added = _donate_cofactors(record["_row"], item)
            if added:
                record.setdefault("cofactors_added", []).extend(added)


def _merge_into(target: Dict[str, Any], source: Dict[str, Any], path: str, draw: int,
                seen: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]],
                per_draw: Dict[str, List[int]], n: int) -> None:
    for key, value in source.items():
        child = f"{path}/{key}"
        if isinstance(value, dict):
            if key not in target:
                target[key] = {}
            if isinstance(target[key], dict):
                _merge_into(target[key], value, child, draw, seen, per_draw, n)
        elif isinstance(value, list):
            if key not in target:
                target[key] = []
            per_draw.setdefault(child, [0] * n)[draw - 1] += len(value)
            if isinstance(target[key], list):
                _union(target[key], value, child, draw, seen.setdefault(child, {}))
        elif key not in target or target[key] in ("", None):
            # First non-empty scalar wins, as _merge_dict_in_place already does for chunks.
            target[key] = value


def merge_draws(payloads: Sequence[Optional[Dict[str, Any]]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Union the draws into one payload. Position = draw number; ``None`` = a failed draw.

    Returns ``(merged, report)``. ``merged`` carries no key the draws did not themselves
    carry; ``report`` is the per-row attribution described in the module doc.
    """

    n = len(payloads)
    if not any(isinstance(p, dict) for p in payloads):
        raise ValueError("merge_draws needs at least one successful draw")
    merged: Dict[str, Any] = {}
    seen: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {}
    per_draw: Dict[str, List[int]] = {}
    for index, payload in enumerate(payloads, start=1):
        if isinstance(payload, dict):
            _merge_into(merged, payload, "", index, seen, per_draw, n)

    buckets: Dict[str, Dict[str, Any]] = {}
    for path, bucket in seen.items():
        rows = [{k: v for k, v in r.items() if k != "_row"} for r in bucket.values()]
        buckets[path] = {
            "merged": len(rows),
            "rows_per_draw": per_draw.get(path, [0] * n),
            "first_seen_in_draw": [sum(1 for r in rows if r["draws"][0] == d) for d in range(1, n + 1)],
            "unique_to_one_draw": sum(1 for r in rows if len(r["draws"]) == 1),
            "in_every_successful_draw": sum(
                1 for r in rows if len(r["draws"]) == sum(1 for p in payloads if isinstance(p, dict))),
            "rows": rows,
        }
    report = {
        "draws_requested": n,
        "draws_succeeded": sum(1 for p in payloads if isinstance(p, dict)),
        "draws_failed": sum(1 for p in payloads if not isinstance(p, dict)),
        "buckets": buckets,
    }
    return merged, report


def _summary(report: Dict[str, Any], report_path: str) -> Dict[str, Any]:
    """The bounded form for the boundary record: totals over processes + entities."""

    n = int(report["draws_requested"])
    chosen = {p: b for p, b in report["buckets"].items() if p.startswith(_SUMMARISED_PREFIXES)}
    totals = {
        key: [sum(b[key][d] for b in chosen.values()) for d in range(n)]
        for key in ("rows_per_draw", "first_seen_in_draw")
    }
    return {
        "draws_requested": n,
        "draws_succeeded": report["draws_succeeded"],
        "draws_failed": report["draws_failed"],
        "failed_draws": report.get("failed_draws", []),
        "merged_rows": sum(b["merged"] for b in chosen.values()),
        "unique_to_one_draw": sum(b["unique_to_one_draw"] for b in chosen.values()),
        "in_every_successful_draw": sum(b["in_every_successful_draw"] for b in chosen.values()),
        **totals,
        "by_bucket": {p: {"merged": b["merged"], "unique_to_one_draw": b["unique_to_one_draw"]}
                      for p, b in chosen.items()},
        "report_path": report_path,
    }


def _write_side_file(artifact_dir: Optional[Path], report: Dict[str, Any]) -> str:
    """Append ``report`` to the run's side file. Never raises: a report that can kill
    the run it describes is worse than no report. Returns the path written, or ``''``."""

    if artifact_dir is None:
        return ""
    path = Path(artifact_dir) / REPORT_NAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(report)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("stage1 ensemble report could not be written to %s: %s", path, exc)
        return ""
    return str(path)


def run_ensemble(draw: DrawFn, n: int, *, failure: type) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Issue ``draw`` ``n`` times, union the successes, raise only when every draw failed.

    ``failure`` is the exception class that means "this draw produced nothing usable"
    (``PipelineFailure`` in production); it is a parameter so this module never imports
    ``pipeline``. Anything else propagates untouched and immediately: a leg deadline or a
    cancelled run is not a bad sample. When every draw fails the last failure is
    re-raised, so callers keep seeing the type they always caught.
    """

    payloads: List[Optional[Dict[str, Any]]] = []
    attempts: List[Dict[str, Any]] = []
    failures: List[Tuple[int, BaseException]] = []
    for index in range(1, n + 1):
        try:
            payload, logs = draw()
        except failure as exc:  # type: ignore[misc]
            failures.append((index, exc))
            payloads.append(None)
            logs = getattr(exc, "attempts", None) or []
        else:
            payloads.append(payload)
        attempts.extend({**entry, "draw": index} for entry in logs if isinstance(entry, dict))
    if not any(isinstance(p, dict) for p in payloads):
        raise failures[-1][1]

    merged, report = merge_draws(payloads)
    report["failed_draws"] = [
        {"draw": index, "stage": str(getattr(exc, "stage", "") or ""), "error": str(exc)}
        for index, exc in failures
    ]
    diagnostics = current_diagnostics()
    report_path = _write_side_file(diagnostics.artifact_dir, report)
    diagnostics.record_boundary(
        stage="extraction",
        boundary=BOUNDARY_STAGE1_ENSEMBLE,
        attempts=n,
        outcome=OUTCOME_OK,
        ensemble=_summary(report, report_path),
    )
    return merged, attempts
