"""Are the H / I / J Stage-1 refusals legitimate, or are we discarding biology?

Read-only. Nothing re-run, no LLM, no network, no production import.

The discriminator is **cross-leg**: if a paper refused in one leg produced a
canonical pathway — or a PWML — in another leg of the same corpus, then that
refusal cost us recoverable biology rather than describing a property of the
paper. Where the paper is in the pinned gold set, gold's own ``expected_export``
and ``export_rationale`` are the authority and outrank any inference from a run.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[3]
FAMILIES = ("runs", "runs_verify", "runs_smoke", "runs_validation")
GOLD = _REPO / "src" / "t2pw" / "bench" / "gold" / "pinned_v1.json"


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _load(path: Path) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001
        return None


def _base_paper_id(name: str) -> str:
    """``PMC13231680__mechanistic-insights…`` -> ``PMC13231680``.

    Older runs suffix the slug onto the directory name. Without this the same
    paper looks like two papers and the cross-leg test silently finds nothing.
    """
    match = re.match(r"(PMC\d+)", name)
    return match.group(1) if match else name


def gold_index() -> Dict[str, Dict[str, Any]]:
    data = _load(GOLD) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for case in data.get("cases", []) or []:
        if isinstance(case, dict) and case.get("paper_id"):
            out[_base_paper_id(str(case["paper_id"]))] = case
    return out


def discover_legs() -> List[Path]:
    legs: List[Path] = []
    for family in FAMILIES:
        root = _REPO / family
        if not root.is_dir():
            continue
        for leg in root.rglob("papers/*/*"):
            if leg.is_dir() and leg.name in {"strict", "research"}:
                legs.append(leg)
    return sorted(set(legs))


def leg_outcome(leg: Path) -> Dict[str, Any]:
    """What this leg achieved, in the only terms that matter for the question."""
    text = _text(leg / "RESULT.txt")
    status = ""
    match = re.search(r"^status\s*:\s*(.+)$", text, re.MULTILINE)
    if match:
        status = match.group(1).strip()
    reactions = None
    match = re.search(r"^\s*reactions\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if match:
        reactions = int(match.group(1))
    final_mapped = _load(leg / "final_mapped.json")
    canonical = None
    if isinstance(final_mapped, dict):
        processes = final_mapped.get("processes")
        if isinstance(processes, dict):
            canonical = len(processes.get("reactions") or []) + len(processes.get("transports") or [])
    stage1 = _load(leg / "stage1_payload.json")
    s1_reactions = None
    s1_entities = None
    if isinstance(stage1, dict):
        processes = stage1.get("processes")
        if isinstance(processes, dict):
            s1_reactions = len(processes.get("reactions") or []) + len(
                processes.get("transports") or []
            )
        entities = stage1.get("entities")
        if isinstance(entities, dict):
            s1_entities = sum(
                len(v) for v in entities.values() if isinstance(v, list)
            )
    return {
        "status": status,
        "result_reactions": reactions,
        "canonical_processes": canonical,
        "stage1_processes": s1_reactions,
        "stage1_entities": s1_entities,
        "pwml": sorted(p.name for p in leg.glob("*.pwml")),
    }


def main() -> int:
    census = _load(_REPO / "docs" / "pwml_recovery_sprint" / "evidence" / "_hij_input.json")
    if not isinstance(census, dict):
        sys.stdout.write(json.dumps({"error": "missing _hij_input.json"}) + "\n")
        return 2
    targets = census.get("terminal_stage1_legs") or []

    gold = gold_index()

    # Best outcome per paper across the WHOLE corpus.
    best: Dict[str, Dict[str, Any]] = {}
    per_paper_legs: Dict[str, int] = defaultdict(int)
    for leg in discover_legs():
        paper = _base_paper_id(leg.parts[-2])
        per_paper_legs[paper] += 1
        outcome = leg_outcome(leg)
        score = (
            2 if outcome["pwml"] else 0,
            outcome["canonical_processes"] or 0,
            outcome["stage1_processes"] or 0,
        )
        if paper not in best or score > best[paper]["_score"]:
            best[paper] = {
                "_score": score,
                "leg": str(leg.relative_to(_REPO)).replace("\\", "/"),
                **outcome,
            }

    rows: List[Dict[str, Any]] = []
    for entry in targets:
        mechanism = entry.get("classification", {}).get("mechanism", "")
        if mechanism[:1] not in {"H", "I", "J"}:
            continue
        paper = _base_paper_id(str(entry.get("paper") or ""))
        gold_case = gold.get(paper)
        top = best.get(paper, {})
        rows.append(
            {
                "mechanism": mechanism,
                "paper": paper,
                "mode": entry.get("mode"),
                "leg": entry.get("path"),
                "message": str(entry.get("message") or "")[:120],
                "in_gold": bool(gold_case),
                "gold_expected_export": (gold_case or {}).get("expected_export"),
                "gold_relevance": (gold_case or {}).get("mechanistic_relevance"),
                "gold_rationale": str((gold_case or {}).get("export_rationale") or "")[:200],
                "corpus_legs_for_paper": per_paper_legs.get(paper, 0),
                "best_elsewhere": {
                    key: top.get(key)
                    for key in ("leg", "status", "canonical_processes", "stage1_processes", "pwml")
                },
            }
        )

    # Verdict per row.
    #
    # GOLD OUTRANKS THE RUNS. A first version of this scored "the same paper
    # produced a PWML elsewhere" as evidence the refusal lost biology. For the
    # two ``context_only`` papers that is exactly backwards: gold states the
    # correct outcome is an EMPTY pathway, so a PWML elsewhere is the known
    # false positive (F-100 / F-101), and the refusal is the desired behaviour.
    # The bug was that ``expected_export`` is a STRING, never boolean False, so
    # the gold branch never fired at all.
    for row in rows:
        top = row["best_elsewhere"]
        produced_pwml = bool(top.get("pwml"))
        produced_pathway = (top.get("canonical_processes") or 0) > 0
        relevance = str(row.get("gold_relevance") or "")
        expected = str(row.get("gold_expected_export") or "")
        if row["in_gold"] and relevance == "context_only":
            row["verdict"] = "CORRECT_gold_says_nothing_is_exportable"
        elif row["in_gold"] and expected == "strict_exportable":
            row["verdict"] = "LOSS_CANDIDATE_gold_calls_this_paper_strict_exportable"
        elif row["in_gold"] and relevance == "core":
            row["verdict"] = "LOSS_CANDIDATE_gold_calls_the_content_extractable"
        elif row["in_gold"]:
            row["verdict"] = "BOUNDED_gold_expects_partial_only"
        elif produced_pwml:
            row["verdict"] = "LOSS_CANDIDATE_same_paper_produced_a_PWML_elsewhere"
        elif produced_pathway:
            row["verdict"] = "PARTIAL_same_paper_produced_a_canonical_pathway_elsewhere"
        else:
            row["verdict"] = "CORRECT_paper_never_yielded_a_pathway_anywhere"

    summary: Dict[str, Any] = {}
    for mechanism in sorted({row["mechanism"] for row in rows}):
        subset = [row for row in rows if row["mechanism"] == mechanism]
        verdicts: Dict[str, int] = defaultdict(int)
        for row in subset:
            verdicts[row["verdict"]] += 1
        summary[mechanism] = {
            "legs": len(subset),
            "papers": sorted({row["paper"] for row in subset}),
            "verdicts": dict(sorted(verdicts.items())),
        }

    sys.stdout.write(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
