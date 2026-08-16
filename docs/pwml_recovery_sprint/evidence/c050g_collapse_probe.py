"""C-050g: the ambiguous-rename collapse, measured at PRODUCTION DEFAULTS.

One script, run unchanged at the base SHA and at the tip, so every clause is a
base-vs-tip *behavioural* difference and never a symbol check (G9).

``--mode all --out F`` writes one JSON with four sections:

``legs``
    Every committed leg fixture, through ``run_prefreeze_resolution`` with
    ``db_resolver=None`` and an **unset** ``name_index`` -- literally what
    ``writer.py``, ``streamlit_app.py:3587`` and ``streamlit_app.py:4091`` pass --
    in ``strict_db`` True *and* False. Records whether the stage completed or
    aborted, the abort code, the compound/reaction counts, and
    ``canonical_graph_sha256`` of the resulting payload. This is A1 for the two
    pinned legs and A4 for the whole corpus at once: a leg that exported at base
    must land on the same graph hash at the tip.

``overfire``
    **A2, the clause that matters.** Relaxing a biological gate's trigger is a
    merge-rule-6 hazard, so this constructs the adversarial cases directly on
    :func:`_reject_ambiguous_renames` and records, for each, whether
    ``AMBIGUOUS_RENAME_TARGET`` was raised. Two spellings of one molecule must
    pass; two genuinely different molecules whose targets collide must still be
    refused. A tip run in which ``distinct_*`` stops raising is worse than the
    defect this card fixes.

``idempotence``
    A7. A second ``run_prefreeze_resolution`` on an already-resolved payload
    leaves it byte-identical.

``normalizers``
    The root cause in one line per sample: ``_canonical`` collapses whitespace,
    then ``_norm``'s character-class substitution can put a double space *back*,
    and nothing collapses it again.

Runs offline end to end: no resolver is constructed, ``name_index`` is left
unset so ``default_name_index()`` reads the committed ``data/pathwhiz_id_db.json``.

**Import discipline.** ``add_src_to_path()`` inserts *this script's own
checkout's* ``src`` at ``sys.path[0]``, so a copy of this file inside a base-tree
export measures base code even if ``PYTHONPATH`` says otherwise. The written
report records ``t2pw.__file__`` so the artifact proves which tree it measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _repo_root import REPO_ROOT, add_src_to_path

add_src_to_path()

import t2pw  # noqa: E402
from t2pw.pipeline.canonical_hash import canonical_graph_sha256  # noqa: E402
from t2pw.pwml import prefreeze_resolution as pf  # noqa: E402


def _legs() -> List[str]:
    found = {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for root in ("runs", "runs_verify")
        for path in (REPO_ROOT / root).rglob("final_mapped.json")
    }
    return sorted(found)


def _counts(payload: Dict[str, Any]) -> Dict[str, int]:
    entities = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    compounds = entities.get("compounds") if isinstance(entities, dict) else None
    processes = payload.get("processes")
    return {
        "compounds": len(compounds) if isinstance(compounds, list) else -1,
        "reactions": len(processes) if isinstance(processes, list) else -1,
    }


def _one_leg(leg: str, *, strict_db: bool) -> Dict[str, Any]:
    payload = json.loads((REPO_ROOT / leg).read_text(encoding="utf-8"))
    entry: Dict[str, Any] = {"leg": leg, "strict_db": strict_db, "before": _counts(payload)}
    try:
        report = pf.run_prefreeze_resolution(payload, strict_db=strict_db)
    except pf.PrefreezeResolutionError as stop:
        entry.update(outcome="ABORT", code=stop.code, message=stop.message,
                     details={k: v for k, v in sorted(stop.details.items())
                              if isinstance(v, (str, int, float, bool, list))})
        entry["after"] = _counts(payload)
        entry["payload_untouched"] = entry["after"] == entry["before"]
        return entry
    except Exception as exc:  # pragma: no cover - a non-sprint failure is still evidence
        entry.update(outcome="ERROR", code=type(exc).__name__, message=str(exc)[:400])
        return entry
    compound = report.get("compounds") if isinstance(report.get("compounds"), dict) else {}
    entry.update(
        outcome="OK",
        after=_counts(payload),
        report_ok=report.get("ok"),
        review_required=sorted(report.get("review_required") or {}),
        failures={k: v for k, v in sorted((report.get("failures") or {}).items())},
        renamed=compound.get("renamed"),
        rename_map=dict(sorted((compound.get("rename_map") or {}).items())),
        aliases_preserved=compound.get("aliases_preserved"),
        rename_sources_collapsed=compound.get("rename_sources_collapsed"),
        canonical_graph_sha256=canonical_graph_sha256(deepcopy(payload)),
    )
    return entry


def _idempotence(leg: str, *, strict_db: bool) -> Dict[str, Any]:
    payload = json.loads((REPO_ROOT / leg).read_text(encoding="utf-8"))
    try:
        pf.run_prefreeze_resolution(payload, strict_db=strict_db)
    except pf.PrefreezeResolutionError as stop:
        return {"leg": leg, "strict_db": strict_db, "outcome": "ABORT", "code": stop.code}
    frozen = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    first_hash = canonical_graph_sha256(deepcopy(payload))
    pf.run_prefreeze_resolution(payload, strict_db=strict_db)
    again = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return {
        "leg": leg, "strict_db": strict_db, "outcome": "OK",
        "payload_equals_frozen": frozen == again,
        "graph_hash_stable": first_hash == canonical_graph_sha256(deepcopy(payload)),
    }


#: ``(label, rename_map, before_names, after_names, must_raise, why)``.
#:
#: ``before_names``/``after_names`` are the row-parallel name vectors
#: ``resolve_compounds_prefreeze`` builds; ``primary_before`` is derived from
#: ``before_names`` so the rogue-owner half of the guard is satisfied and the
#: **source-collision** half is the only thing under test.
_OVERFIRE_CASES: Tuple[Tuple[str, Dict[str, str], List[str], List[str], bool, str], ...] = (
    (
        "spelling_one_space_vs_none",
        {"sn -glycerol 3-phosphate": "sn-Glycerol 3-phosphate",
         "sn-glycerol 3-phosphate": "sn-Glycerol 3-phosphate"},
        ["sn -glycerol 3-phosphate", "sn-glycerol 3-phosphate"],
        ["sn-Glycerol 3-phosphate", "sn-Glycerol 3-phosphate"],
        False,
        "one molecule, two spellings differing by a single space; the offline index "
        "maps both to the same canonical target -- canonicalization, not a merge",
    ),
    (
        "spelling_hyphen_vs_space",
        {"beta-D-glucose": "D-Glucose", "beta D glucose": "D-Glucose"},
        ["beta-D-glucose", "beta D glucose"],
        ["D-Glucose", "D-Glucose"],
        False,
        "same molecule, punctuation-only difference; both _norm to the same token run",
    ),
    (
        "distinct_glucose_vs_fructose",
        {"D-glucose": "Hexose", "D-fructose": "Hexose"},
        ["D-glucose", "D-fructose"],
        ["Hexose", "Hexose"],
        True,
        "TWO REAL MOLECULES onto one target. Must still raise: merging them "
        "invents biology (merge rule 6, D-015 clause 6)",
    ),
    (
        "distinct_atp_vs_adp",
        {"ATP": "adenosine phosphate", "ADP": "adenosine phosphate"},
        ["ATP", "ADP"],
        ["adenosine phosphate", "adenosine phosphate"],
        True,
        "ATP and ADP are different compounds; a target collision between them is a "
        "real merge and must abort",
    ),
    (
        "distinct_glycine_vs_glycerol",
        {"glycine": "Gly", "glycerol": "Gly"},
        ["glycine", "glycerol"],
        ["Gly", "Gly"],
        True,
        "the second pinned leg's shape -- distinct names, one target",
    ),
    (
        "distinct_differs_by_one_digit",
        {"glycerol 2-phosphate": "glycerol phosphate",
         "glycerol 3-phosphate": "glycerol phosphate"},
        ["glycerol 2-phosphate", "glycerol 3-phosphate"],
        ["glycerol phosphate", "glycerol phosphate"],
        True,
        "ADVERSARIAL: the two sources are one CHARACTER apart and share the "
        "whitespace shape of the fixed case. Collapsing whitespace must not make "
        "them equal -- 2- and 3-phosphate are different molecules",
    ),
    (
        "distinct_whitespace_shaped_but_different_tokens",
        {"sn -glycerol 3-phosphate": "sn-Glycerol 3-phosphate",
         "sn -glycerol 1-phosphate": "sn-Glycerol 3-phosphate"},
        ["sn -glycerol 3-phosphate", "sn -glycerol 1-phosphate"],
        ["sn-Glycerol 3-phosphate", "sn-Glycerol 3-phosphate"],
        True,
        "ADVERSARIAL: both sources carry the SAME double-space artefact, so a fix "
        "that keyed on 'has a collapsible run' rather than on the collapsed VALUE "
        "would wave these through. They are different molecules",
    ),
)


def _overfire() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for label, rename_map, before, after, must_raise, why in _OVERFIRE_CASES:
        primary: Dict[str, Tuple[str, ...]] = {}
        for index, name in enumerate(before):
            key = pf._norm(name)
            primary[key] = primary.get(key, ()) + (f"compounds#{index}",)
        raised: str | None = None
        try:
            pf._reject_ambiguous_renames(rename_map, before, after, primary)
        except pf.PrefreezeResolutionError as stop:
            raised = stop.code
        results.append({
            "case": label, "must_raise": must_raise, "raised": raised,
            "verdict": "OK" if bool(raised == "AMBIGUOUS_RENAME_TARGET") == must_raise else "WRONG",
            "sources": sorted(rename_map), "why": why,
        })
    return results


_NORM_SAMPLES: Tuple[str, ...] = (
    "sn -glycerol 3-phosphate", "sn-glycerol 3-phosphate",
    "glycerol 2-phosphate", "glycerol 3-phosphate", "glycine", "glycerol",
)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="all", choices=("all",))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    legs = _legs()
    pinned = [leg for leg in legs if "PMC12444477__the-regulation" in leg
              or "PMC13278307__an-overview" in leg]

    report: Dict[str, Any] = {
        "task": "C-050g",
        "t2pw_file": t2pw.__file__,
        "repo_root": str(REPO_ROOT),
        "python": sys.version.split()[0],
        "pinned_legs": pinned,
        "legs": [_one_leg(leg, strict_db=strict) for leg in legs for strict in (True, False)],
        "overfire": _overfire(),
        "idempotence": [_idempotence(leg, strict_db=strict)
                        for leg in pinned for strict in (True, False)],
        "normalizers": [
            {"value": sample, "canonical": pf._canonical(sample), "norm": pf._norm(sample),
             "norm_has_double_space": "  " in pf._norm(sample)}
            for sample in _NORM_SAMPLES
        ],
    }
    exported = [e for e in report["legs"] if e["outcome"] == "OK"]
    aborted = [e for e in report["legs"] if e["outcome"] != "OK"]
    report["totals"] = {
        "leg_configurations": len(report["legs"]),
        "exported": len(exported),
        "aborted": len(aborted),
        "aborted_pairs": [(e["leg"], e["strict_db"], e.get("code")) for e in aborted],
        "overfire_wrong": [r["case"] for r in report["overfire"] if r["verdict"] == "WRONG"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(f"T2PW: {t2pw.__file__}")
    print(f"leg configurations: {len(report['legs'])}  exported: {len(exported)}  "
          f"aborted: {len(aborted)}")
    for leg, strict, code in report["totals"]["aborted_pairs"]:
        print(f"  ABORT strict_db={strict} {code} {leg}")
    for row in report["overfire"]:
        print(f"  overfire {row['verdict']:5} {row['case']:48} must_raise={row['must_raise']!s:5} "
              f"raised={row['raised']}")
    for row in report["idempotence"]:
        print(f"  idempotence strict_db={row['strict_db']!s:5} {row['outcome']:5} "
              f"equal={row.get('payload_equals_frozen')} {row['leg']}")
    print(f"wrote {out}")
    return 0 if not report["totals"]["overfire_wrong"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
