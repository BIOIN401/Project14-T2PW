"""ORCH-770 (D-112 § 4): write the one-page MORNING_SUMMARY.md for an overnight paper run, from its artifacts only.

Stdlib only, no network. Called by the night runner as its LAST step (process rule from the ORCH-767
incident: the runner writes its own summary). Safe to re-run by hand.

ORCH-772 (D-114 § 1): the per-paper table carries the PWML coverage of each leg, computed from the leg's directory by
``t2pw.app.reaction_evidence`` (read-only, offline) of the tree named by ``--src`` (default: the tree this script lives
in). A tree without it, or an error, shows ``n/a``; the summary is written regardless.

ORCH-774s: per paper, the omitted processes and actors with their KI ids (from the same report model), the count of
reactions the written PWML exports with no enzyme and ``spontaneous=false`` (PathWhiz drops them on import; counted in
the XML), and a "would import" coverage beside the headline and the full count.

    python orch770_morning_summary.py --task ORCH-770 --legs <worktree>/runs_papers/ORCH-770/legs \
        --out-dir <primary>/runs_papers/ORCH-770 --gold <primary>/docs/.../evidence/orch770_gold_accuracy.txt \
        --compare ORCH-767=<...>/orch767_gold_accuracy.txt --compare ORCH-759p=<...> --compare ORCH-756=<...>
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # orch772b_no_pwml_cause, beside this script
GOLD_LINE = re.compile(r"^\s*(enterobactin|menaquinone)\s+(stage1|final|pwml)\s+(.*)$")


def gold_rows(path: str) -> dict:
    """{(paper, level): 'P x R y F1 z (m/n)' or 'not written'} from an orch756_gold_accuracy.py output."""
    out = {}
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[:14]:
        m = GOLD_LINE.match(line)
        if not m:
            continue
        rest = m.group(3)
        if "NOT WRITTEN" in rest:
            val = "not written"
        else:
            g = re.search(r"P ([\d.]+)\s+R ([\d.]+)\s+F1 ([\d.]+)\s+matched (\d+/\d+)", rest)
            val = f"P {g.group(1)} R {g.group(2)} F1 {g.group(3)} ({g.group(4)})" if g else rest[:60]
        out.setdefault((m.group(1), m.group(2)), val)
    return out


def leg_sha(night_log: Path) -> dict:
    """cond -> worktree head recorded at that leg's runner start line."""
    shas, last = {}, None
    if not night_log.is_file():
        return shas
    for line in night_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r" start; tree=.* head=(\w+) .*conds=\[([^\]]*)\]", line)
        if m:
            last = m.group(1)
            for c in m.group(2).split():
                shas[c] = last
    return shas


def failure_line(leg: Path) -> str:
    res = leg / "RESULT.txt"
    if not res.is_file():
        return "no RESULT.txt"
    text = res.read_text(encoding="utf-8", errors="replace")
    msg = re.search(r"^message\s*:\s*(.*)$", text, re.M)
    codes = re.search(r"^issue codes\s*:\s*(.*)$", text, re.M)
    return f"{(msg.group(1) if msg else '?')[:200]} [{(codes.group(1) if codes else '').strip()[:80]}]"


PWML_NAMES = ("pathway.pwml", "pathway.review_required.pwml")  # what the batch driver writes (reaction_evidence)


def leg_report(leg: Path):
    """``(report, "")`` from the --src tree's ``reaction_evidence.build_report`` on the leg's own artifacts, with the
    leg's recorded no-PWML cause (ORCH-772b); ``(None, why)`` when it cannot be built. Never raises."""
    try:
        from orch772b_no_pwml_cause import recorded_no_pwml_cause
        from t2pw.app.reaction_evidence import build_report, load_run_artifacts

        if not leg.is_dir():
            return None, "n/a (no leg directory)"
        return build_report(load_run_artifacts(leg), no_pwml_cause=recorded_no_pwml_cause(leg)), ""
    except Exception as exc:  # the summary must still be written
        return None, f"n/a ({type(exc).__name__})"


def no_enzyme_reactions(leg: Path):
    """ORCH-774s: reactions in the leg's written PWML that PathWhiz drops on import (``reaction.rb`` ``check_spontaneous``,
    ORCH-773 r1 reviewer N1): a ``<reaction>`` whose ``<spontaneous>`` is not ``true`` and whose ``<reaction-enzymes>``
    holds no ``<reaction-enzyme>``. Read from the XML file itself. ``None`` when no PWML was written or it does not parse."""
    for name in PWML_NAMES:
        if (leg / name).is_file():
            try:
                root = ET.parse(leg / name).getroot()
            except ET.ParseError:
                return None
            return sum(1 for r in root.iter("reaction")
                       if (r.findtext("spontaneous") or "").strip() != "true" and r.find("reaction-enzymes/reaction-enzyme") is None)
    return None


def coverage(leg: Path) -> str:
    """From the leg's own artifacts: the headline (KI-008 left out, D-114 AMENDMENT 2), the full count beside it, for a
    leg with no PWML the known issue of its recorded cause (ORCH-772b), and for a written PWML the coverage that would
    survive import: (drawn - reactions PathWhiz drops) / the headline denominator (ORCH-774s). Never raises."""
    report, why = leg_report(leg)
    cov = (report or {}).get("pwml_coverage")
    if why or not cov:
        return why or "n/a"
    from t2pw.app.reaction_evidence import known_issue_id

    if cov["drawn"] is None:
        return f"not recorded of {cov['total']}"
    drawn, full = cov["drawn"], cov["total"]
    head = full - cov["by_design_withheld"]
    pct = lambda n, total: f"{100 * n / total:.0f}%" if total else "n/a"  # noqa: E731
    text = f"{drawn}/{head} ({pct(drawn, head)}); full count {drawn}/{full} ({pct(drawn, full)})"
    if not cov["pwml_written"]:
        return f"{text}; no PWML: {known_issue_id(cov['no_pwml_cause'])}"
    dropped = no_enzyme_reactions(leg)
    return text if dropped is None else f"{text}; would import {drawn - dropped}/{head} ({pct(drawn - dropped, head)})"


def omitted_lines(cond: str, leg: Path) -> list:
    """ORCH-774s: every omitted process (kind, name, KI) and every omitted actor of a kept process, with counts by KI."""
    report, why = leg_report(leg)
    if report is None:
        return [f"### {cond}", "", why, ""]
    rows = [e for e in report.get("omitted") or [] if e.get("counted")]
    actors = report.get("omitted_actors") or []
    by_ki: dict = {}
    for ki in [e["ki"] for e in rows] + [a["ki"] for a in actors]:
        by_ki[ki] = by_ki.get(ki, 0) + 1
    counts = ", ".join(f"{k} {n}" for k, n in sorted(by_ki.items(), key=lambda kv: (-kv[1], kv[0]))) or "none"
    return [f"### {cond}: {len(rows)} processes, {len(actors)} actors left out", "", f"By KI: {counts}", "",
            *[f"- {e['kind']} {e['name'] or '(unnamed)'}: {e['ki']}" for e in rows],
            *[f"- actor {a['actor']} of {a['kind']} {a['process']}: {a['ki']}" for a in actors], ""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--legs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--compare", action="append", default=[], help="LABEL=path to another run's gold accuracy txt")
    ap.add_argument("--extra-review", action="append", default=[], help="LABEL=review dir of an extra leg set (e.g. 770b)")
    ap.add_argument("--src", default=str(HERE.parents[2] / "src"), help="tree whose reaction_evidence computes coverage")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    out = Path(a.out_dir)
    review = out / "REVIEW"
    rows = []
    for rp in [review / "index.jsonl"] + [Path(x.split("=", 1)[1]) / "index.jsonl" for x in a.extra_review]:
        if rp.is_file():
            rows += [json.loads(line) for line in rp.read_text(encoding="utf-8").splitlines() if line.strip()]
    shas = leg_sha(out / "night.log")
    L = [f"# {a.task} — overnight run, morning summary", "",
         f"Written by the runner at the end of the run (`evidence/orch770_morning_summary.py`). Legs: `{a.legs}`.", "",
         "## Per paper", "",
         "PWML coverage = processes drawn in the PWML / processes in the text record (D-114; target 75%), "
         "from each leg's directory by `t2pw.app.reaction_evidence`. Headline first: the interactions PathWhiz cannot "
         "carry (KI-008) are left out (D-114 AMENDMENT 2); the full count follows. A leg with no PWML names the known "
         "issue of its recorded cause. \"would import\" leaves out the reactions PathWhiz drops on import: those the "
         "written PWML exports with no enzyme and `spontaneous=false` (column \"dropped on import\", counted in the XML).", "",
         "| paper | PWML | PWML coverage | dropped on import | release | reactions / interactions | min | code |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dropped = no_enzyme_reactions(Path(r.get("leg_dir", "")))
        L.append(f"| {r['cond']} | **{'yes' if str(r.get('pwml', '')).endswith('.pwml') else 'no'}** | "
                 f"{coverage(Path(r.get('leg_dir', '')))} | {'-' if dropped is None else dropped} | {r.get('release', '-')} | "
                 f"{r.get('reactions', '-')} / {r.get('interactions', '-')} | {r.get('minutes', '-')} | `{shas.get(r['cond'], '?')}` |")
    L += ["", "## Omitted from the PWML, per paper (kind, name, known issue)", ""]
    for r in rows:
        L += omitted_lines(r["cond"], Path(r.get("leg_dir", "")))
    sets = [(a.task, gold_rows(a.gold))] + [(x.split("=", 1)[0], gold_rows(x.split("=", 1)[1])) for x in a.compare]
    L += ["", "## Gold accuracy (eval-v2 score_leg)", "", "| paper / level | " + " | ".join(s for s, _ in sets) + " |",
          "|---|" + "---|" * len(sets)]
    for paper in ("enterobactin", "menaquinone"):
        for level in ("stage1", "final", "pwml"):
            L.append(f"| {paper} {level} | " + " | ".join(g.get((paper, level), "-") for _, g in sets) + " |")
    L += ["", "## Failures, one line each (from each leg's RESULT.txt)", ""]
    for r in rows:
        if not str(r.get("pwml", "")).endswith(".pwml"):
            leg = Path(r.get("leg_dir", ""))
            L.append(f"- **{r['cond']}:** {failure_line(leg) if leg.is_dir() else r.get('result', '?')}")
    L += ["", "## Paths", "", f"Review copies: `{review}/<paper>/strict/` (reaction_evidence.txt, RESULT.txt, "
          "stage1_ensemble_report.txt, the PWML where written); index `REVIEW/INDEX.txt`.", ""]
    traces = glob.glob(str(Path(a.legs) / "*" / "*" / "papers" / "*" / "*" / "LEG_TRACE.jsonl"))
    L += ["## LLM health", ""]
    if traces:
        pat = str(Path(a.legs) / "*" / "*" / "papers" / "*" / "*" / "LEG_TRACE.jsonl")
        for script in ("orch755_trace_stats.py", "orch756_leg_stats.py"):
            try:
                txt = subprocess.run([sys.executable, str(HERE / script), pat], capture_output=True, text=True, timeout=120).stdout
            except Exception as exc:  # the summary must still be written
                txt = f"{script} failed: {exc}"
            keep = [ln for ln in txt.splitlines() if ("empty" in ln and "|" in ln) or "LONG/ERROR" in ln or ln.startswith("TOTAL")]
            L += ["```", *[ln.split(" | ", 1)[-1] if "LEG_TRACE" in ln else ln for ln in keep[-12:]], "```"]
    else:
        L.append("no LEG_TRACE.jsonl found")
    (out / "MORNING_SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'MORNING_SUMMARY.md'} ({len(rows)} legs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
