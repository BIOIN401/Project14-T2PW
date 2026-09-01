"""C-109: resolve every citation this card touches, at BASE and at TIP.

Three jobs, all measurement, no opinion:

1. **F-154 proof.** Show what the retired line addresses `TEST_MATRIX.md:213-218` and
   `:242-252` ACTUALLY contain at the base SHA -- the false-text-at-base half -- and show
   that the anchors that replaced them resolve, at the tip, to the content those
   addresses claimed. This is the proof G9 asks of a correction to observable, currently
   false documentation. It is a statement about CONTENT, not about a symbol's absence.

2. **F-153 proof.** Show `controller.py`'s docstring carries `UNWIRED` at base and
   retracts it at tip, and that the call sites it now names resolve in
   `streamlit_app.py` BY SYMBOL.

3. **The `:477` line-neutrality proof for `TEST_MATRIX.md`.** The count of lines above
   `:477` must be identical at base and tip, and line `:477` byte-identical.

Usage::

    <python> c109_citation_probe.py <worktree-root> <base-sha>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASE = sys.argv[2]

TM = "docs/pwml_recovery_sprint/TEST_MATRIX.md"
RUNNER = ".claude/agents/pwml-test-runner.md"
FINDINGS = "docs/pwml_recovery_sprint/FINDINGS.md"
GATE = "docs/pwml_recovery_sprint/evidence/chunk_d_gate.py"
CONTROLLER = "src/t2pw/rag/controller.py"
APP = "src/t2pw/app/streamlit_app.py"

FAILURES: list[str] = []


def show(rel: str, sha: str = BASE) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{sha}:{rel}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        sys.exit(f"cannot read {rel} at {sha}: {proc.stderr.strip()}")
    return proc.stdout.splitlines()


def tip(rel: str) -> list[str]:
    return (ROOT / rel).read_text(encoding="utf-8").splitlines()


def head(line: str, n: int = 96) -> str:
    return line[:n]


def check(label: str, ok: bool) -> None:
    print(f"  [{'OK ' if ok else 'BAD'}] {label}")
    if not ok:
        FAILURES.append(label)


def rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


# ---------------------------------------------------------------------------
rule("1a. F-154 -- what the RETIRED addresses contain at base")

tm_base = show(TM)
print(f"  {TM} at {BASE}: {len(tm_base)} lines")
print("  TEST_MATRIX.md:213-218 (claimed: the chunk membership table) actually reads:")
for i in range(213, 219):
    print(f"    {i:>4} | {head(tm_base[i - 1])}")
print("  TEST_MATRIX.md:242-252 (claimed: the SMOKE command block) actually reads:")
for i in range(242, 253):
    print(f"    {i:>4} | {head(tm_base[i - 1])}")

stems_213_218 = [ln for ln in tm_base[212:218] if "test_" in ln]
check(
    "213-218 contains NO test-file stem, so a stem-exact match against it finds nothing",
    not stems_213_218,
)
check(
    "213-218 is in fact the bounded-runner function table",
    any("launch_child" in ln for ln in tm_base[212:218]),
)
check(
    "242-252 is in fact the Chunk-D-excluded / Chunk-E paragraphs",
    any("Chunk D is excluded" in ln for ln in tm_base[241:252]),
)
check(
    "242-252 does NOT contain the SMOKE command block",
    not any("# SMOKE (every merge)" in ln for ln in tm_base[241:252]),
)

# ---------------------------------------------------------------------------
rule("1b. F-154 -- the ANCHORS that replaced them, resolved at tip")

tm_tip = tip(TM)


def find(lines: list[str], pred) -> int:
    for i, ln in enumerate(lines, start=1):
        if pred(ln):
            return i
    return -1


chunks_h = find(tm_tip, lambda l: l.strip() == "## Chunks")
check("anchor '## Chunks' heading resolves", chunks_h > 0)
print(f"    '## Chunks' -> line {chunks_h}")

rows: dict[str, int] = {}
for name in ("**A**", "**B**", "**C**", "**D-core**", "**D-apptest**", "**E**"):
    n = find(tm_tip[chunks_h:], lambda l, nm=name: l.startswith(f"| {nm} |"))
    rows[name] = chunks_h + n if n > 0 else -1
print(f"    chunk table rows -> {rows}")
check("all six chunk rows resolve under the '## Chunks' anchor", all(v > 0 for v in rows.values()))
check(
    "the chunk table under the anchor DOES carry test-file stems (unlike :213-218)",
    "test_reference_repair" in tm_tip[rows["**A**"] - 1],
)

cmds_h = find(tm_tip, lambda l: l.strip() == "## Commands")
smoke = find(tm_tip, lambda l: l.startswith("# SMOKE (every merge)") and "503 passed" in l)
check("anchor '## Commands' heading resolves", cmds_h > 0)
check("anchor '# SMOKE (every merge) - expect 503 passed' resolves", smoke > 0)
check("the SMOKE anchor is BELOW the '## Commands' anchor", 0 < cmds_h < smoke)
print(f"    '## Commands' -> line {cmds_h};  SMOKE block -> line {smoke}")
smoke_files = [
    m for ln in tm_tip[smoke - 1: smoke + 14] for m in re.findall(r"tests/(test_[a-z0-9_]+)\.py", ln)
]
print(f"    SMOKE block names {len(smoke_files)} test files")
check("the SMOKE anchor block names 22 test files", len(smoke_files) == 22)

gate_tip = tip(GATE)
for sym in ("CORE", "S8", "QB", "MONOLITHIC"):
    n = find(gate_tip, lambda l, s=sym: re.match(rf"^{s}\s*[:=]", l))
    print(f"    chunk_d_gate.py symbol {sym} -> line {n}")
    check(f"anchor symbol chunk_d_gate.py::{sym} resolves", n > 0)

# ---------------------------------------------------------------------------
rule("1c. F-154 -- the citing sites: false at base, true at tip")

runner_base, runner_tip = show(RUNNER), tip(RUNNER)
print(f"  {RUNNER}:59 at base | {head(runner_base[58])}")
check("BASE: pwml-test-runner.md cites the drifted 'TEST_MATRIX.md:213-218'",
      any("TEST_MATRIX.md:213-218" in l for l in runner_base))
check("BASE: pwml-test-runner.md cites the drifted ':242-252'",
      any(":242-252" in l for l in runner_base))
check("TIP: no drifting locator is used as the certification address",
      not any(re.search(r"against `TEST_MATRIX\.md:\d", l) for l in runner_tip))
check("TIP: cites the '## Chunks' anchor", any("## Chunks" in l for l in runner_tip))
check("TIP: cites the SMOKE block anchor text",
      any("# SMOKE (every merge)" in l for l in runner_tip))
check("TIP: cites the chunk_d_gate.py symbols",
      any("`CORE`" in l and "`MONOLITHIC`" in l for l in runner_tip))
check("TIP: the underlying claim is UNCHANGED -- still stem-exact",
      any("**stem-exact**" in l for l in runner_tip))
check("TIP: the underlying claim is UNCHANGED -- still forbids grepping the filename",
      any("grepping the filename" in l for l in runner_tip))
check("TIP: the load-bearing provenance pin `d8de94d` is RETAINED",
      any("d8de94d" in l for l in runner_tip))
check("BASE: pwml-test-runner.md states the stale SMOKE count 465",
      any("**465** tests" in l for l in runner_base))
check("TIP: pwml-test-runner.md states the measured SMOKE count 503",
      any("**503** tests" in l for l in runner_tip))

f_base, f_tip = show(FINDINGS), tip(FINDINGS)
print(f"  {FINDINGS}:1120 at base | {head(f_base[1119])}")
print(f"  {FINDINGS}:1120 at tip  | {head(f_tip[1119])}")
check("BASE: FINDINGS.md:1120 cites `TEST_MATRIX.md:213` for Chunk A",
      "TEST_MATRIX.md:213`" in f_base[1119])
check("TIP: FINDINGS.md:1120 cites the '## Chunks' table anchor instead",
      "## Chunks" in f_tip[1119])
check("FINDINGS.md edit is LINE-NEUTRAL", len(f_base) == len(f_tip))

# ---------------------------------------------------------------------------
rule("2. F-153 -- controller.py docstring, false at base, retracted at tip")

c_base, c_tip = show(CONTROLLER), tip(CONTROLLER)
print(f"  {CONTROLLER}:11 at base | {head(c_base[10])}")
check("BASE: the docstring asserts UNWIRED / nothing in production calls it",
      any("**UNWIRED**: nothing in production calls it" in l for l in c_base))
check("TIP: that sentence is retracted, struck, not deleted",
      any("~~**UNWIRED**: nothing in production calls it" in l for l in c_tip))
check("TIP: the retraction is labelled",
      any("RETRACTION" in l for l in c_tip))
for site in ("streamlit_app.py:1270", "streamlit_app.py:1426", "streamlit_app.py:5636"):
    check(f"TIP: names call site {site}", any(site in l for l in c_tip))
check("TIP: names tests/test_c055_rag_loop_wiring.py",
      any("test_c055_rag_loop_wiring.py" in l for l in c_tip))
c_tip_flat = " ".join(c_tip)
check("TIP: graph-delta partiality is PRESERVED, not certified",
      "graph-delta\nvalidation is partial" in "\n".join(c_tip)
      or "graph-delta validation is partial" in c_tip_flat)
check("TIP: says explicitly it did not certify it",
      any("NOT certified" in l for l in c_tip))
check("TIP: conform.py is still named as the incomplete part",
      any("conform.py" in l for l in c_tip))
check("wiring test file exists", (ROOT / "tests/test_c055_rag_loop_wiring.py").is_file())

# C-109 MEASURED CORRECTION. F-153, MASTER_PLAN and the C-109 charter all cite :5669 for
# the run_rag_rounds call. At the committed integration tip it is at :5636; :5669 is where
# it sits in the UNCOMMITTED working copy in the primary checkout. Both are asserted here:
# the true address must resolve, and the cited one must NOT, so the correction is measured
# rather than asserted.
app_tip = tip(APP)
check("F-153's cited :5669 does NOT resolve to run_rag_rounds( at the committed tip",
      "run_rag_rounds(" not in app_tip[5668])
print(f"    {APP}:5669 (as cited by F-153) | {head(app_tip[5668])}")
for n, needle in ((1239, "def run_rag_rounds("), (1270, "run_rag_loop"),
                  (1426, "run_rag_loop("), (5636, "run_rag_rounds(")):
    ok = n <= len(app_tip) and needle in app_tip[n - 1]
    print(f"    {APP}:{n} | {head(app_tip[n - 1]) if n <= len(app_tip) else '<past EOF>'}")
    check(f"{APP}:{n} resolves to {needle}", ok)
by_symbol = [i for i, l in enumerate(app_tip, 1) if "run_rag_loop" in l or "run_rag_rounds" in l]
print(f"    by symbol, run_rag_loop/run_rag_rounds occur at lines: {by_symbol}")
check("re-verifiable BY SYMBOL, independent of the line numbers", len(by_symbol) >= 3)

# ---------------------------------------------------------------------------
rule("3. TEST_MATRIX.md -- the `:477` line-neutrality proof")

above_base = tm_base[:476]
above_tip = tm_tip[:476]
print(f"  lines above :477   base {len(above_base)}   tip {len(above_tip)}")
check("identical count of lines above :477", len(above_base) == len(above_tip))
check("every line above :477 byte-identical", above_base == above_tip)
print(f"  line 477 base | {head(tm_base[476])}")
print(f"  line 477 tip  | {head(tm_tip[476])}")
check("line :477 byte-identical", tm_base[476] == tm_tip[476])
check("the chunk table is still at :230 and SMOKE still at :259 (drift did not worsen)",
      chunks_h == 228 and rows["**A**"] == 232 and smoke == 259)
diff_from = next((i for i in range(min(len(tm_base), len(tm_tip))) if tm_base[i] != tm_tip[i]), None)
print(f"  first differing line: {None if diff_from is None else diff_from + 1}"
      f"   (base {len(tm_base)} lines -> tip {len(tm_tip)})")
check("every TEST_MATRIX.md change is at end-of-file, strictly below :477",
      diff_from is None or diff_from + 1 > 477)

# ---------------------------------------------------------------------------
rule("VERDICT")
if FAILURES:
    print(f"  {len(FAILURES)} FAILED:")
    for f in FAILURES:
        print(f"    - {f}")
    sys.exit(1)
print("  all citations resolve; all base/tip claims measured true")
sys.exit(0)
