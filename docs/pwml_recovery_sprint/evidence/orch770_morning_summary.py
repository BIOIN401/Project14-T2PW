"""ORCH-770 (D-112 § 4): write the one-page MORNING_SUMMARY.md for an overnight paper run, from its artifacts only.

Stdlib only, no t2pw import, no network. Called by the night runner as its LAST step (process rule from the ORCH-767
incident: the runner writes its own summary). Safe to re-run by hand.

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
from pathlib import Path

HERE = Path(__file__).resolve().parent
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--legs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--compare", action="append", default=[], help="LABEL=path to another run's gold accuracy txt")
    ap.add_argument("--extra-review", action="append", default=[], help="LABEL=review dir of an extra leg set (e.g. 770b)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    review = out / "REVIEW"
    rows = []
    for rp in [review / "index.jsonl"] + [Path(x.split("=", 1)[1]) / "index.jsonl" for x in a.extra_review]:
        if rp.is_file():
            rows += [json.loads(line) for line in rp.read_text(encoding="utf-8").splitlines() if line.strip()]
    shas = leg_sha(out / "night.log")
    L = [f"# {a.task} — overnight run, morning summary", "",
         f"Written by the runner at the end of the run (`evidence/orch770_morning_summary.py`). Legs: `{a.legs}`.", "",
         "## Per paper", "", "| paper | PWML | release | reactions / interactions | min | code |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['cond']} | **{'yes' if str(r.get('pwml', '')).endswith('.pwml') else 'no'}** | {r.get('release', '-')} | "
                 f"{r.get('reactions', '-')} / {r.get('interactions', '-')} | {r.get('minutes', '-')} | `{shas.get(r['cond'], '?')}` |")
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
