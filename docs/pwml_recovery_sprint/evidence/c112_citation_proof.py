"""C-112 -- citation work: false-at-BASE / true-at-TIP, and the drift THIS diff creates.

Three separate jobs, all against **committed blobs on both sides** (F-157's lesson: a
claim checked against a working copy is a claim about bytes that exist in no commit):

1. **§1 Live citations.** For each of the 7 live citations among REV-109 R1's 26, plus the
   two routed items and F-157, prove the address was **false at the base SHA** and the
   **anchor resolves at the tip**. Symbol/heading absence is never used as proof that
   something was fixed -- each anchor is *resolved* in the tip blob.
2. **§2 Frozen records.** Prove every citing site bucketed *frozen* is **byte-identical**
   base to tip. This is the half a reviewer cannot take on trust: it is easy to fix 26
   things and quietly rewrite a signed record among them.
3. **§3 The drift THIS diff creates -- the criterion C-109 was never given.** For every
   file the diff touches, find the last base line whose number is unchanged at the tip,
   then scan every tracked file for a committed citation below that boundary. **A C-112
   that fixes 26 stale citations and creates new ones has failed on its own terms**, so
   the required answer is 0.

Plus the `TEST_MATRIX.md:477` line-neutrality proof, mechanically, in all three parts.

Usage::  <venv-python> c112_citation_proof.py <worktree-root> <base-sha>
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import sys
from pathlib import Path

SPRINT = "docs/pwml_recovery_sprint"
FAILURES: list[str] = []
GIT = shutil.which("git")


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def blob(root: Path, ref: str, rel: str) -> list[str]:
    proc = subprocess.run([GIT, "-C", str(root), "show", f"{ref}:{rel}"],
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"cannot read {rel} at {ref}: {proc.stderr.strip()}")
    return proc.stdout.split("\n")


def line(lines: list[str], n: int) -> str:
    return lines[n - 1] if 0 < n <= len(lines) else "<out of range>"


def tracked(root: Path, ref: str) -> list[str]:
    proc = subprocess.run([GIT, "-C", str(root), "ls-tree", "-r", "--name-only", ref],
                          capture_output=True, text=True, encoding="utf-8")
    return [p for p in proc.stdout.split("\n") if p]


# ======================================================================= §1

def section1(root: Path, base: str, tip: str) -> None:
    print("\n=== 1. THE 7 LIVE CITATIONS -- false at BASE, anchor resolves at TIP ===")

    b_mp = blob(root, base, f"{SPRINT}/MASTER_PLAN.md")
    t_mp = blob(root, tip, f"{SPRINT}/MASTER_PLAN.md")
    b_f = blob(root, base, f"{SPRINT}/FINDINGS.md")
    t_f = blob(root, tip, f"{SPRINT}/FINDINGS.md")
    b_tm = blob(root, base, f"{SPRINT}/TEST_MATRIX.md")
    t_tm = blob(root, tip, f"{SPRINT}/TEST_MATRIX.md")
    b_rs = blob(root, base, f"{SPRINT}/RESUME-NEXT-SESSION.md")
    t_rs = blob(root, tip, f"{SPRINT}/RESUME-NEXT-SESSION.md")
    t_ptr = blob(root, tip, ".claude/agents/pwml-test-runner.md")

    # --- L1  FINDINGS.md:315 (F-031) -> MASTER_PLAN.md:336
    print("\n  L1  FINDINGS.md:315 (F-031)  MASTER_PLAN.md:336 -> '## 9 Canonical paths' row")
    check("BASE: F-031 cites MASTER_PLAN.md:336", "`MASTER_PLAN.md:336`" in line(b_f, 315))
    check("BASE: MASTER_PLAN.md:336 does NOT mention rag/extract.py -- the address is FALSE",
          "rag/extract" not in line(b_mp, 336), repr(line(b_mp, 336))[:70])
    check("TIP : F-031 no longer carries a MASTER_PLAN line locator",
          not re.search(r"MASTER_PLAN\.md:\d", line(t_f, 315)))
    check("TIP : F-031 cites the Canonical paths table by name",
          "Canonical paths" in line(t_f, 315) and "`extract.py`" in line(t_f, 315))
    hits = [i + 1 for i, s in enumerate(t_mp) if "src/t2pw/rag/extract.py" in s and "|" in s]
    check("TIP : that anchor RESOLVES -- the row exists and names the decoy",
          len(hits) == 1, f"MASTER_PLAN.md:{hits}")

    # --- L2  FINDINGS.md:316 (F-032) -> MASTER_PLAN.md:363
    print("\n  L2  FINDINGS.md:316 (F-032)  MASTER_PLAN.md:363 -> '## 9 branch register, C-035'")
    check("BASE: F-032 cites MASTER_PLAN.md:363", "`MASTER_PLAN.md:363`" in line(b_f, 316))
    check("BASE: MASTER_PLAN.md:363 is TRAP-5 text, not a §9 row -- the address is FALSE",
          "admission" not in line(b_mp, 363), repr(line(b_mp, 363))[:70])
    check("TIP : F-032 no longer carries a MASTER_PLAN line locator",
          not re.search(r"MASTER_PLAN\.md:\d", line(t_f, 316)))
    check("TIP : F-032 cites the branch-register row C-035 by name",
          "branch-register row `C-035`" in line(t_f, 316))
    rows = [i + 1 for i, s in enumerate(t_mp)
            if s.startswith("| C-035 ") and "rag/admission.py" in s]
    check("TIP : that anchor RESOLVES -- exactly one C-035 register row names rag/admission.py",
          len(rows) == 1, f"MASTER_PLAN.md:{rows}")

    # --- L3  FINDINGS.md:6679 (F-154 measurement, present tense)
    print("\n  L3  FINDINGS.md:6679  pwml-test-runner.md:59 -> '## Test discipline' bullet")
    check("BASE: cites pwml-test-runner.md:59", "pwml-test-runner.md:59`" in line(b_f, 6679))
    b_ptr = blob(root, base, ".claude/agents/pwml-test-runner.md")
    check("BASE: pwml-test-runner.md:59 is NOT the certification instruction -- FALSE",
          "stem-exact" not in line(b_ptr, 59), repr(line(b_ptr, 59))[:70])
    check("TIP : cites the chunk-membership bullet by name, number struck",
          "`## Test discipline` chunk-membership bullet" in line(t_f, 6679)
          and "~~`:59`~~" in line(t_f, 6679))
    bullets = [i + 1 for i, s in enumerate(t_ptr)
               if "Chunk membership covers only 28 of the 147 test files" in s]
    disc = [i + 1 for i, s in enumerate(t_ptr) if s.strip() == "## Test discipline"]
    check("TIP : that anchor RESOLVES -- one '## Test discipline' heading, one bullet under it",
          len(disc) == 1 and len(bullets) == 1 and bullets[0] > disc[0],
          f"heading:{disc} bullet:{bullets}")

    # --- L4  MASTER_PLAN.md:477 citing itself
    print("\n  L4  MASTER_PLAN.md:477 (self-citation)  :372 -> '§ 9 C-045 register row'")
    check("BASE: MASTER_PLAN.md:477 cites MASTER_PLAN.md:372",
          "`MASTER_PLAN.md:372`" in line(b_mp, 477))
    check("BASE: MASTER_PLAN.md:372 is the §8 Schedule table, not the C-045 row -- FALSE",
          "C-045" not in line(b_mp, 372), repr(line(b_mp, 372))[:70])
    check("TIP : cites the C-045 branch-register row by name",
          "`C-045` branch-register row" in line(t_mp, 477))
    c45 = [i + 1 for i, s in enumerate(t_mp)
           if s.startswith("| C-045 ") and "_canonicalize_species_offline" in s]
    check("TIP : that anchor RESOLVES -- exactly one C-045 row names the function",
          len(c45) == 1, f"MASTER_PLAN.md:{c45}")

    # --- L5/L6  TEST_MATRIX.md:533 -- TWO citations on one line
    print("\n  L5+L6  TEST_MATRIX.md:533  MASTER_PLAN.md:281 + pwml-test-runner.md:52")
    check("BASE: TEST_MATRIX.md:533 cites both by line",
          "`MASTER_PLAN.md:281`" in line(b_tm, 533)
          and "pwml-test-runner.md:52`" in line(b_tm, 533))
    check("BASE: MASTER_PLAN.md:281 is not the merge-gates heading any more -- FALSE",
          not line(b_mp, 281).startswith("## 5. Merge gates"), repr(line(b_mp, 281))[:70])
    # HONEST DISAGREEMENT WITH MY OWN FIRST CLAIM, kept rather than quietly dropped.
    # Attempt 2 of this probe asserted that `pwml-test-runner.md:52` was FALSE at base
    # and it FAILED -- the failing run is preserved as
    # `c112_citation_proof.attempt2-ptr52-not-false-at-base.log`. Measured: `:52` still
    # lands on the SMOKE bullet at the C-112 base. REV-109 counted it among the 26 under
    # its OTHER class -- "base content rewritten, no verbatim match at tip" (465 -> 503)
    # -- not under "shift +N". So this ONE sub-citation of the seven was drift-PRONE, not
    # false. **G9: converting it is HARDENING, and NO base failure is claimed for it.**
    # The other six live citations are genuine false-at-base corrections, each proved
    # above. Mislabelling this one would be exactly the kind of reject G9 names.
    check("BASE: pwml-test-runner.md:52 STILL RESOLVES to the SMOKE bullet -- NOT false "
          "at base, only drift-prone; conversion is HARDENING and claims no base failure",
          line(b_ptr, 52).startswith("- Smoke = chunks A+B+C"), repr(line(b_ptr, 52))[:70])
    check("BASE: ...and its CONTENT had been rewritten under it (465 -> 503), which is "
          "REV-109's own second class and the reason it is in the 26",
          "503" in line(b_ptr, 52) and "465" not in line(b_ptr, 52))
    check("TIP : both replaced by anchors, no line locators left",
          not re.search(r"MASTER_PLAN\.md:\d", line(t_tm, 533))
          and not re.search(r"pwml-test-runner\.md:\d", line(t_tm, 533)))
    mg = [i + 1 for i, s in enumerate(t_mp) if s.strip() == "## 5. Merge gates — all must hold"]
    smoke = [i + 1 for i, s in enumerate(t_ptr) if s.startswith("- Smoke = chunks A+B+C")]
    check("TIP : both anchors RESOLVE", len(mg) == 1 and len(smoke) == 1,
          f"merge-gates:{mg} smoke-bullet:{smoke}")

    # --- L7  RESUME-NEXT-SESSION.md F-154 row (REV-109 measured it at :94)
    print("\n  L7  RESUME-NEXT-SESSION.md F-154 row  pwml-test-runner.md:59 -> the bullet")
    b_rows = [i + 1 for i, s in enumerate(b_rs) if s.startswith("| **F-154** |")]
    t_rows = [i + 1 for i, s in enumerate(t_rs) if s.startswith("| **F-154** |")]
    check("BASE: the F-154 row exists and cites pwml-test-runner.md:59",
          len(b_rows) == 1 and "`pwml-test-runner.md:59`" in line(b_rs, b_rows[0]),
          f"at :{b_rows}")
    check("SITE DRIFT: REV-109 measured this citing site at :94; at the C-112 base it is "
          f":{b_rows[0]} -- a citation OF a citation is no more stable",
          b_rows[0] != 94, f"base site :{b_rows[0]}")
    check("TIP : the row cites the bullet by name, the number struck",
          len(t_rows) == 1
          and "`## Test discipline` chunk-membership bullet" in line(t_rs, t_rows[0])
          and "~~`:59`~~" in line(t_rs, t_rows[0]))

    # --- routed item A: FINDINGS row E; row D-qb MUST BE UNTOUCHED
    print("\n  R-A  FINDINGS.md:1126 row E  TEST_MATRIX.md:218 -> '## Chunks' anchor")
    check("BASE: row E cites TEST_MATRIX.md:218", "`TEST_MATRIX.md:218`" in line(b_f, 1126))
    check("BASE: TEST_MATRIX.md:218 is the child_env row, not Chunk E -- FALSE",
          "child_env" in line(b_tm, 218), repr(line(b_tm, 218))[:70])
    chunks_h = [i + 1 for i, s in enumerate(t_tm) if s.strip() == "## Chunks"]
    row_e = [i + 1 for i, s in enumerate(t_tm) if s.startswith("| **E** |")]
    check("TIP : row E cites the ## Chunks table row by name",
          "`## Chunks` table, row `**E**`" in line(t_f, 1126))
    check("TIP : that anchor RESOLVES, and Chunk E's row is where the note says",
          len(chunks_h) == 1 and len(row_e) == 1 and row_e[0] == 237,
          f"## Chunks:{chunks_h} row E:{row_e}")
    check("ROW D-qb LEFT ALONE -- it was already CORRECT (charter + REV-112 W8)",
          line(b_f, 1125) == line(t_f, 1125) == "| D-qb | 1 | `:70` |",
          repr(line(t_f, 1125)))
    dqb = blob(root, tip, f"{SPRINT}/evidence/chunk_d_gate.py")
    check("...and `chunk_d_gate.py:70` genuinely resolves to the QB symbol",
          "QB" in line(dqb, 70), repr(line(dqb, 70))[:60])

    # --- routed item B: TEST_MATRIX.md:568
    print("\n  R-B  TEST_MATRIX.md:568  '§ Chunks begins at line 209' -> struck")
    check("BASE: :568 claims line 209", "line 209." in line(b_tm, 568))
    check("BASE: TEST_MATRIX.md:209 is not the ## Chunks heading -- FALSE",
          line(b_tm, 209).strip() != "## Chunks", repr(line(b_tm, 209))[:70])
    check("TIP : struck, not silently restated", "~~line 209~~" in line(t_tm, 568)
          and "struck by C-112" in line(t_tm, 568))
    check("TIP : replaced by the ## Chunks heading anchor, and :228 is measured, not asserted",
          "`## Chunks`" in line(t_tm, 568) and chunks_h == [228], f"## Chunks at {chunks_h}")

    # --- F-157
    print("\n  F-157  FINDINGS.md § F-153  streamlit_app.py:5669 -> the SYMBOL")
    check("BASE: F-153's block cites streamlit_app.py:5669",
          "streamlit_app.py:5669" in line(b_f, 6624))
    app = blob(root, tip, "src/t2pw/app/streamlit_app.py")
    check("BASE: the COMMITTED blob's :5669 is not the call -- resolves for no reader",
          "run_rag_rounds(" not in line(app, 5669), repr(line(app, 5669))[:70])
    check("BASE: the committed call site is :5636", "run_rag_rounds(" in line(app, 5636),
          repr(line(app, 5636))[:70])
    check("TIP : F-153 cites the SYMBOL, and carries no line number at all",
          "streamlit_app.py :: run_rag_rounds" in line(t_f, 6624)
          and ":5669" not in line(t_f, 6624) and ":5636" not in line(t_f, 6624))
    defs = [i + 1 for i, s in enumerate(app) if s.startswith("def run_rag_rounds(")]
    calls = [i + 1 for i, s in enumerate(app) if "run_rag_rounds(" in s]
    check("TIP : the symbol RESOLVES in the committed tree", bool(defs) and bool(calls),
          f"def:{defs} refs:{calls}")
    rule = [i + 1 for i, s in enumerate(t_tm)
            if s.startswith("### Never cite a line number in a file that carries an uncommitted diff")]
    check("TIP : the standing rule landed where an agent meets it (TEST_MATRIX §, not a "
          "findings entry)", len(rule) == 1, f"TEST_MATRIX.md:{rule}")
    check("TIP : and the next session is pointed at it from RESUME's own F-157 paragraph",
          any("Never cite a line number in a file that carries an uncommitted diff" in s
              for s in t_rs))


# ======================================================================= §2

FROZEN = [
    (f"{SPRINT}/DECISIONS.md", [1361, 1919, 1950, 1956, 1959, 3206, 3619],
     "append-only, product owner only"),
    (f"{SPRINT}/FINDINGS.md", [1125, 1193, 2887, 6703],
     "row D-qb correct; C-109 repair record; probe antecedent; F-154 analysis"),
    (f"{SPRINT}/SPIKE-002-REPORT.md", [138, 143, 253],
     "signed CONTROL-PLANE-RECONCILE-001 annotations"),
    (f"{SPRINT}/prompts/C-011.md", [196, 235], "a dispatched card's charter"),
    (f"{SPRINT}/prompts/C-109.md", [66], "another card's charter"),
    (f"{SPRINT}/TEST_MATRIX.md", [726, 727, 785], "C-106's signed record; C-109's F-154 rule"),
    (f"{SPRINT}/evidence/c054_gate_counts.json", [202], "a signed gate-count artifact"),
    (f"{SPRINT}/evidence/c106_predictions.md", [190], "C-106's signed prediction record"),
    (f"{SPRINT}/evidence/c109_citation_probe.log", [50], "a probe LOG"),
]

WHOLE_FILE_UNTOUCHED = [
    f"{SPRINT}/DECISIONS.md",
    f"{SPRINT}/SPIKE-002-REPORT.md",
    f"{SPRINT}/prompts/C-011.md",
    f"{SPRINT}/prompts/C-109.md",
    f"{SPRINT}/evidence/c054_gate_counts.json",
    f"{SPRINT}/evidence/c106_predictions.md",
    f"{SPRINT}/evidence/c109_citation_probe.log",
    ".claude/agents/pwml-test-runner.md",
    "src/t2pw/app/streamlit_app.py",
]


def section2(root: Path, base: str, tip: str) -> None:
    print("\n=== 2. FROZEN RECORDS -- byte-identical base to tip ===")
    for rel, lines, why in FROZEN:
        b, t = blob(root, base, rel), blob(root, tip, rel)
        same = all(line(b, n) == line(t, n) for n in lines)
        check(f"{Path(rel).name} {lines} unchanged ({why})", same)

    print("\n  Whole files the diff must not touch at all:")
    changed = subprocess.run(
        [GIT, "-C", str(root), "diff", "--name-only", base, tip],
        capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
    changed = [c for c in changed if c]
    for rel in WHOLE_FILE_UNTOUCHED:
        check(f"{rel} not in the diff", rel not in changed)
    print(f"\n  files changed by the C-112 diff ({len(changed)}):")
    for c in changed:
        print(f"    {c}")


# ======================================================================= §3

CITE_RE = re.compile(r"([A-Za-z0-9_./\\-]+\.(?:md|py|json|log|txt|ini|toml)):(\d+)")


def unshifted_boundary(b: list[str], t: list[str]) -> int:
    """Last base line number whose number is IDENTICAL at the tip.

    A same-length in-place rewrite shifts nothing; an insert or delete shifts everything
    below it. That is the difference between C-109's edit and C-112's, and it is the
    whole reason this card exists.
    """
    sm = difflib.SequenceMatcher(None, b, t, autojunk=False)
    boundary = len(b)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "delete") or (tag == "replace" and (i2 - i1) != (j2 - j1)):
            boundary = i1          # 0-based i1 == 1-based line i1, i.e. the line above
            break
    return boundary


def citations_in_ref(root: Path, ref: str, basenames: list[str]) -> list[tuple[str, int, str]]:
    """Every ``<basename>:<n>`` citation in the whole committed tree, in ONE git call.

    ``git grep`` over the ref, not a per-file ``git show`` loop: the loop spawned one
    subprocess per tracked file and did not finish inside the wrapper's timeout, which
    stranded a heavy lock. That failed run is preserved as
    ``c112_citation_proof.attempt1-timeout.log``.
    """
    pat = "(" + "|".join(re.escape(b) for b in sorted(set(basenames))) + "):[0-9]+"
    proc = subprocess.run([GIT, "-C", str(root), "grep", "-I", "-n", "-E", pat, ref],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out: list[tuple[str, int, str]] = []
    for ln in proc.stdout.split("\n"):
        if not ln.startswith(ref + ":"):
            continue
        path, _, tail = ln[len(ref) + 1:].partition(":")
        num, _, content = tail.partition(":")
        if num.isdigit():
            out.append((path, int(num), content))
    return out


def scan(citations, rel: str, boundary: int) -> list[str]:
    """Committed citations of ``rel`` addressing a line BELOW the unshifted boundary."""
    name = Path(rel).name
    stale = []
    for src, num, content in citations:
        for m in CITE_RE.finditer(content):
            cited, target = m.group(1), int(m.group(2))
            if (cited == name or cited.endswith("/" + name) or cited == rel) and target > boundary:
                stale.append(f"{src}:{num} cites {m.group(0)}")
    return stale


def section3(root: Path, base: str, tip: str) -> None:
    print("\n=== 3. THE DRIFT **THIS** DIFF CREATES (W10) ===")
    changed = [c for c in subprocess.run(
        [GIT, "-C", str(root), "diff", "--name-only", base, tip],
        capture_output=True, text=True, encoding="utf-8").stdout.split("\n") if c]
    citations = citations_in_ref(root, tip, [Path(c).name for c in changed] + ["MASTER_PLAN.md"])
    print(f"  corpus: {len(citations)} committed lines carrying a <file>:<line> citation "
          f"of a file this diff touches")

    # ---- KNOWN-POSITIVE for the scanner itself. Before any zero is believed, the
    # scanner must be shown capable of reporting non-zero. A synthetic ONE-LINE
    # insertion at MASTER_PLAN.md:160 -- exactly where C-109's +16 landed -- must
    # produce the same class of result REV-109 measured.
    mp = f"{SPRINT}/MASTER_PLAN.md"
    t_mp = blob(root, tip, mp)
    synthetic = t_mp[:159] + ["A SYNTHETIC INSERTED LINE"] + t_mp[159:]
    syn_boundary = unshifted_boundary(t_mp, synthetic)
    syn_stale = scan(citations, mp, syn_boundary)
    check("SCANNER KNOWN-POSITIVE: a synthetic 1-line insertion at MASTER_PLAN.md:160 "
          "IS reported as creating stale citations",
          syn_boundary == 159 and len(syn_stale) > 0,
          f"boundary={syn_boundary} stale={len(syn_stale)}")
    print(f"    (the synthetic insertion strands {len(syn_stale)} committed citations -- "
          f"this is the shape of the defect C-112 exists to not repeat)")

    # ---- KNOWN-NEGATIVE: an unchanged file must report nothing.
    check("SCANNER KNOWN-NEGATIVE: an unmodified file reports zero",
          scan(citations, mp, len(t_mp)) == [])

    total_stale = 0
    for rel in changed:
        try:
            b, t = blob(root, base, rel), blob(root, tip, rel)
        except SystemExit:
            continue
        boundary = unshifted_boundary(b, t)
        net = len(t) - len(b)
        print(f"\n  {rel}")
        print(f"    base {len(b)} lines -> tip {len(t)} lines (net {net:+d})")
        print(f"    line numbers IDENTICAL up to and including base line {boundary}"
              f"{'  (= the whole file: NOTHING SHIFTED)' if boundary >= len(b) else ''}")
        stale = scan(citations, rel, boundary)
        total_stale += len(stale)
        print(f"    committed citations below that boundary: {len(stale)}")
        for s in stale:
            print(f"      STALE: {s}")

    print("\n" + "-" * 74)
    check("C-112 creates ZERO newly-stale committed citations (W10)", total_stale == 0,
          f"{total_stale} found")


# ======================================================================= §4

def section4(root: Path, base: str, tip: str) -> None:
    print("\n=== 4. TEST_MATRIX.md `:477` LINE-NEUTRALITY -- all three parts ===")
    rel = f"{SPRINT}/TEST_MATRIX.md"
    b, t = blob(root, base, rel), blob(root, tip, rel)
    check("part 1/3: identical line count at or above :477",
          len(b[:477]) == len(t[:477]) == 477, f"base {len(b[:477])} tip {len(t[:477])}")
    check("part 2/3: line :477 is BYTE-IDENTICAL",
          line(b, 477).encode("utf-8") == line(t, 477).encode("utf-8"),
          repr(line(t, 477))[:60])
    diffs = [n for n in range(1, 478) if line(b, n) != line(t, n)]
    check("part 3/3: NO differing line anywhere in the common prefix :1-477",
          diffs == [], f"differing lines {diffs}")
    print(f"    (edits made at :533, :567, :568 -- all BELOW the pin -- and {len(t) - len(b)} "
          f"lines appended at end-of-file, which nothing cites)")


def main(root: Path, base: str) -> int:
    tip = subprocess.run([GIT, "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, encoding="utf-8").stdout.strip()
    print("C-112 -- citation proof and self-drift measurement")
    print(f"worktree : {root}")
    print(f"base     : {base}")
    print(f"tip      : {tip}")
    section1(root, base, tip)
    section2(root, base, tip)
    section4(root, base, tip)
    section3(root, base, tip)
    print("\n" + "=" * 74)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print(f"  FAILED: {f}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve(), sys.argv[2]))
