"""REV-112 -- independent checks on C-112 item 5 (C-108 R4, M16 re-pointing).

Three questions the reviewer must answer on the CODE, not on the report:

1. **Was M16 genuinely dead at the C-112 base?**  The base blob of
   ``c107_mutation_attack.py`` must have an M16 anchor that occurs **ZERO** times in the
   base blob of the target, and the tip's anchor must occur **exactly once**.  That is the
   behavioural base failure G9 requires; symbol absence is not used as proof of anything,
   the anchors are counted against the actual committed target bytes.
2. **Was the mutation WEAKENED to make it match?**  Compare the base M16 replacement and
   the tip M16 replacement as regex source: the same six stems must be present, and the
   two lookaround anchors ``(?<![a-z])`` / ``(?![a-z])`` must be stripped by BOTH.  A
   mutation that stopped removing an anchor would go red for less.
3. **Was dropping the leading ``|`` right?**  C-112 says keeping it would inject an empty
   alternative at the use sites.  Demonstrate it, do not accept it: build both candidate
   replacements, splice each into the real use-site expression, compile, and show what the
   resulting pattern matches.

Usage::  <venv-python> REV112_m16_and_pipe.py <worktree-root> <base-sha>
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HARNESS = "docs/pwml_recovery_sprint/evidence/c107_mutation_attack.py"
TARGET = "src/t2pw/curation/apply_audit_patch.py"
STEMS = ["blockade", "impair", "silenc", "sequestr", "ablat", "interfer"]

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def blob(root: Path, ref: str, rel: str) -> str:
    proc = subprocess.run([shutil.which("git"), "-C", str(root), "show", f"{ref}:{rel}"],
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"cannot read {rel} at {ref}: {proc.stderr}")
    return proc.stdout


def m16_of(harness_src: str, tmpname: str) -> tuple[str, str]:
    """Return (pattern, replacement) of M16 from a harness source string."""
    ns: dict = {}
    # The harness imports c102 helpers and resolves a venv path at import time; execute
    # only the MUTATIONS literal by slicing it out, so nothing else runs.
    start = harness_src.index("MUTATIONS = [")
    end = harness_src.index("\n]\n", start) + 3
    exec(compile(harness_src[start:end], tmpname, "exec"), ns)  # noqa: S102
    for name, _desc, pat, rep in ns["MUTATIONS"]:
        if name == "M16":
            return pat, rep
    raise SystemExit("M16 not present in this harness -- deleting it is a stop condition")


def main(root: Path, base: str) -> int:
    tip = subprocess.run([shutil.which("git"), "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, encoding="utf-8").stdout.strip()
    print("REV-112 -- independent M16 / dropped-pipe checks")
    print(f"worktree : {root}\nbase     : {base}\ntip      : {tip}")

    b_h, t_h = blob(root, base, HARNESS), blob(root, tip, HARNESS)
    b_t, t_t = blob(root, base, TARGET), blob(root, tip, TARGET)

    print("\n=== 1. M16 WAS DEAD AT BASE, AND IS ALIVE AT TIP (behavioural, on bytes) ===")
    check("the TARGET is byte-identical base to tip -- C-112 changed no product code",
          b_t == t_t, f"base {len(b_t)} bytes, tip {len(t_t)} bytes")
    b_pat, b_rep = m16_of(b_h, "base_harness")
    t_pat, t_rep = m16_of(t_h, "tip_harness")
    check("M16 still EXISTS at the tip -- it was re-pointed, not deleted", bool(t_pat))
    check("BASE: M16's anchor occurs ZERO times in the target -> apply_mutation raises "
          "-> the harness printed ABORT and returned 3",
          b_t.count(b_pat) == 0, f"count={b_t.count(b_pat)}")
    check("TIP : M16's anchor occurs EXACTLY ONCE in the same target bytes",
          t_t.count(t_pat) == 1, f"count={t_t.count(t_pat)}")
    check("...and the base anchor is the 8-space nested-alternation spelling, the tip "
          "anchor the 4-space assignment-body spelling C-108 (d) created",
          b_pat.startswith("        r\"|") and t_pat.startswith("    r\"("),
          f"base={b_pat.splitlines()[0]!r} tip={t_pat.splitlines()[0]!r}")

    print("\n=== 2. THE MUTATION WAS NOT WEAKENED ===")
    for stem in STEMS:
        check(f"stem {stem!r} present in BOTH the base and the tip replacement",
              stem in b_rep and stem in t_rep)
    for anchor in ("(?<![a-z])", "(?![a-z])"):
        check(f"anchor {anchor!r} is in the PATTERN and stripped by the REPLACEMENT, "
              f"base and tip alike -- the mutation still removes it",
              anchor in b_pat and anchor not in b_rep
              and anchor in t_pat and anchor not in t_rep)
    check("the completions (?:s|ed|ing|ment|ments) etc. are dropped by BOTH replacements",
          "impair(?:s|ed" not in b_rep and "impair(?:s|ed" not in t_rep)
    check("the only textual difference between the two replacements is the indent and "
          "the leading pipe",
          b_rep.strip().replace('r"|', 'r"') == t_rep.strip(),
          f"base={b_rep.strip()!r}\n                tip ={t_rep.strip()!r}")

    print("\n=== 3. THE DROPPED LEADING `|` -- DEMONSTRATED, NOT ASSERTED ===")
    # The real use sites, read out of the target rather than retyped.
    use_sites = [n + 1 for n, ln in enumerate(t_t.split("\n"))
                 if "_C107_INHIBITION_WORDS_SRC" in ln and "=" not in ln.split("#")[0][:40]]
    print(f"    use sites of _C107_INHIBITION_WORDS_SRC in the target: {use_sites}")
    check("every use site supplies the pipe itself (r\"|\" + _C107_INHIBITION_WORDS_SRC)",
          all('r"|" + _C107_INHIBITION_WORDS_SRC' in t_t.split("\n")[n - 1]
              for n in use_sites), str(use_sites))

    def body(rep: str) -> str:
        """The regex source the replacement lines produce, as Python would build it."""
        return "".join(
            re.match(r"\s*r\"(.*)\"(?:\s+#.*)?$", ln).group(1)
            for ln in rep.rstrip("\n").split("\n") if ln.strip()
        )

    as_landed = body(t_rep)                       # C-112's replacement: no leading pipe
    with_pipe = "|" + as_landed                   # what keeping the pipe would produce
    print(f"    as landed  : {as_landed!r}")
    print(f"    with pipe  : {with_pipe!r}")

    for label, src in (("as landed", as_landed), ("with the pipe kept", with_pipe)):
        full = "somethingelse" + "|" + src        # the use site: r"|" + <SRC>
        rx = re.compile(full)
        m = rx.search("a totally unrelated sentence about kinases")
        empty = m is not None and m.group(0) == ""
        print(f"    use site {label:<20} -> {full[:60]!r}...  "
              f"matches-empty-everywhere={empty}")
        if label == "as landed":
            check("as landed: the spliced use-site pattern does NOT match the empty "
                  "string in unrelated text -- the mutation is the one M16 names",
                  not empty)
        else:
            check("with the pipe kept: the spliced use-site pattern DOES match empty "
                  "everywhere -- C-112's stated reason for dropping it is CORRECT",
                  empty)

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
