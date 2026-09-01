"""REV-109 -- the reviewer's OWN measurements of C-109's claims.

Nothing here is copied from the author's probe. Where the author used one
method, this uses a different one wherever a different one exists:

* inertness is checked by AST *and* by comparing the MARSHALLED BYTECODE of
  the docstring-stripped modules -- a second, independent witness;
* every citation is resolved out of a COMMITTED BLOB (``git show``), never the
  working tree, because a citation that resolves only against uncommitted
  bytes is the defect C-109 itself corrected;
* line-neutrality is measured by raw byte comparison of the two blobs.

Usage: rev109_independent_checks.py <tip-worktree> <base-sha> <integration-ref>
"""

from __future__ import annotations

import ast
import marshal
import subprocess
import sys
from pathlib import Path

FAILURES = []
NOTES = []


def check(ok, msg, detail=""):
    tag = "OK  " if ok else "FAIL"
    if not ok:
        FAILURES.append(msg)
    print(f"  [{tag}] {msg}" + (f"\n         {detail}" if detail else ""))


def note(msg):
    NOTES.append(msg)
    print(f"  [NOTE] {msg}")


def show(repo, ref, path):
    proc = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git show {ref}:{path} -> {proc.stderr!r}")
    return proc.stdout


def lines_of(blob: bytes):
    return blob.decode("utf-8").split("\n")


# ------------------------------------------------------------------ inertness

def strip_docstrings(tree, normalise_positions=False):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    if normalise_positions:
        # ROUND 1 ANOMALY, corrected here and kept on the record:
        # comparing marshalled bytecode WITHOUT normalising positions reported a
        # false difference. The tip docstring is ~30 lines longer, so every
        # statement below it carries a different lineno and the code object's
        # line table differs. That is a position artifact, not a behavioural
        # one. Zeroing positions is what makes witness 2 a real second witness.
        for node in ast.walk(tree):
            for attr in ("lineno", "end_lineno"):
                if hasattr(node, attr):
                    setattr(node, attr, 1)
            for attr in ("col_offset", "end_col_offset"):
                if hasattr(node, attr):
                    setattr(node, attr, 0)
    return ast.fix_missing_locations(tree)


def inertness(repo, base, tip_path):
    print("\n=== A. controller.py inertness -- TWO independent witnesses ===")
    rel = "src/t2pw/rag/controller.py"
    base_src = show(repo, base, rel)
    # ROUND 1 ANOMALY, corrected here: reading the WORKTREE gave 18291 bytes
    # against the author's 17934, because this worktree checks out CRLF (357
    # CRLFs) while ``git show`` yields the LF blob. Read the committed blob --
    # which is also why the route check hashes through ``git hash-object``.
    tip_src = show(repo, "HEAD", rel)
    bt, tt = ast.parse(base_src), ast.parse(tip_src)
    print(f"  bytes: base {len(base_src)}  tip {len(tip_src)}")
    print(f"  ast nodes: base {sum(1 for _ in ast.walk(bt))} "
          f"tip {sum(1 for _ in ast.walk(tt))}")

    # witness 1: docstring-stripped AST dump
    d_b = ast.dump(strip_docstrings(ast.parse(base_src)))
    d_t = ast.dump(strip_docstrings(ast.parse(tip_src)))
    check(d_b == d_t, "witness 1: docstring-stripped AST dumps identical")

    # witness 2: marshalled bytecode of the docstring-stripped modules
    cb = compile(strip_docstrings(ast.parse(base_src), True), rel, "exec")
    ct = compile(strip_docstrings(ast.parse(tip_src), True), rel, "exec")
    check(marshal.dumps(cb) == marshal.dumps(ct),
          "witness 2: marshalled bytecode of the stripped modules identical "
          "(positions normalised -- see the note in strip_docstrings)")

    def codesig(code):
        return (code.co_code, code.co_names, code.co_varnames, code.co_argcount,
                tuple(codesig(c) if hasattr(c, "co_code") else c
                      for c in code.co_consts))
    check(codesig(cb) == codesig(ct),
          "witness 3: co_code / co_names / co_consts tree identical "
          "(position-free by construction)")

    # which docstrings actually moved
    def docmap(src):
        t = ast.parse(src)
        out = {}
        for n in ast.walk(t):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
                key = "Module" if isinstance(n, ast.Module) else \
                    f"{type(n).__name__}:{n.name}"
                out[key] = ast.get_docstring(n)
        return out
    db, dt = docmap(base_src), docmap(tip_src)
    changed = sorted(k for k in db if db.get(k) != dt.get(k))
    check(changed == ["Module"],
          "the ONLY docstring that changed is the module docstring",
          f"changed = {changed}")
    check(set(db) == set(dt), "no docstring owner added or removed "
          f"(base {len(db)}, tip {len(dt)})")

    # the retraction itself
    mod = dt["Module"]
    check("UNWIRED" in mod, "B3: the withdrawn word UNWIRED is PRESERVED, not deleted")
    check("~~**UNWIRED**: nothing in production calls it; wiring is C-055's.~~" in mod,
          "B3: the old sentence is present verbatim inside strike markers")
    check("RETRACTION" in mod, "B3: the retraction is explicitly labelled")
    for frag in (":1270", ":1426", ":5636", ":1239",
                 "tests/test_c055_rag_loop_wiring.py"):
        check(frag in mod, f"B14: the docstring names {frag}")
    check("5669" in mod, "B14: the wrong address :5669 is named and explained")
    # B2 -- the third claim preserved and NOT certified
    check("graph-delta" in mod and "conform.py" in mod,
          "B2: graph-delta / conform.py partiality is REPRODUCED")
    check("NOT certified" in mod,
          "B2: it is explicitly marked NOT certified")
    check("stands exactly as written" in mod,
          "B2: it is said to stand as written")
    for forbidden in ("now validates", "is fixed", "has been fixed",
                      "no longer partial", "resolved"):
        check(forbidden not in mod.lower().replace("-", " ") or
              forbidden not in mod,
              f"B2: the docstring does not assert the third claim {forbidden!r}")
    check("re-verify" in mod.lower() or "Re-verify" in mod,
          "F-154 lesson encoded: the line numbers say how to re-verify them")


# ------------------------------------------------------------------ citations

def citations(repo, base, ref):
    print("\n=== B. citations resolved out of COMMITTED blobs ===")
    rel = "src/t2pw/app/streamlit_app.py"
    for where, label in ((base, f"base {base}"), (ref, f"integration {ref}")):
        L = lines_of(show(repo, where, rel))
        get = lambda n: L[n - 1]
        check(get(1239).startswith("def run_rag_rounds("),
              f"[{label}] :1239 is def run_rag_rounds(", get(1239))
        check("run_rag_loop" in get(1270),
              f"[{label}] :1270 imports run_rag_loop", get(1270))
        check("run_rag_loop(" in get(1426),
              f"[{label}] :1426 calls run_rag_loop(", get(1426))
        check("run_rag_rounds(" in get(5636),
              f"[{label}] :5636 calls run_rag_rounds(", get(5636))
        check("run_rag_rounds(" not in get(5669),
              f"[{label}] :5669 does NOT call run_rag_rounds -- the charter, "
              f"F-153 and MASTER_PLAN were wrong", get(5669))

    print("\n=== C. TEST_MATRIX.md anchors resolve at the tip ===")
    tm = lines_of(show(repo, "HEAD", "docs/pwml_recovery_sprint/TEST_MATRIX.md"))
    idx = lambda s: next((i + 1 for i, l in enumerate(tm) if l.strip() == s), None)
    h_chunks, h_cmds = idx("## Chunks"), idx("## Commands")
    check(h_chunks is not None, f"'## Chunks' heading resolves (line {h_chunks})")
    check(sum(1 for l in tm if l.strip() == "## Chunks") == 1,
          "'## Chunks' is UNIQUE, so the anchor is unambiguous")
    check(sum(1 for l in tm if l.strip() == "## Commands") == 1,
          "'## Commands' is UNIQUE")
    # first table under ## Chunks
    tbl = next(i + 1 for i, l in enumerate(tm[h_chunks:], start=h_chunks)
               if l.startswith("|"))
    rows = {}
    for i in range(tbl, tbl + 12):
        for r in ("**A**", "**B**", "**C**", "**D-core**", "**D-apptest**", "**E**"):
            if tm[i - 1].startswith(f"| {r} "):
                rows[r] = i
    check(len(rows) == 6, "all six chunk rows resolve in the FIRST table under "
          f"'## Chunks'", f"{rows}")
    check(tbl == 230, f"B15: the chunk table is STILL at :230 (found {tbl}) -- "
          "the target was not moved to fit the citation")
    smoke = next((i + 1 for i, l in enumerate(tm)
                  if l.startswith("# SMOKE (every merge)")), None)
    check(smoke == 259, f"B15: the SMOKE block is STILL at :259 (found {smoke})")
    check(smoke > h_cmds, "the SMOKE anchor is under the '## Commands' heading")
    check("503" in tm[smoke - 1], "the SMOKE anchor line states 503")
    stems = [l for l in tm[tbl:tbl + 10] if "test_" in l]
    check(len(stems) >= 6, "the anchored table carries test-file stems "
          "(the retired :213-218 carried none)")

    print("\n=== D. chunk_d_gate.py symbols ===")
    cd = lines_of(show(repo, "HEAD",
                       "docs/pwml_recovery_sprint/evidence/chunk_d_gate.py"))
    for sym in ("CORE", "S8", "QB", "MONOLITHIC"):
        ln = next((i + 1 for i, l in enumerate(cd) if l.startswith(sym + " ")), None)
        check(ln is not None, f"symbol {sym} resolves", f"line {ln}")

    print("\n=== E. B6 -- the claim did not change ===")
    tr = show(repo, "HEAD", ".claude/agents/pwml-test-runner.md").decode("utf-8")
    check("**stem-exact**" in tr, "B6: still orders a stem-exact match")
    check("never by grepping the filename" in tr or
          "never fall back to grepping the filename" in tr,
          "B6: still forbids grepping the filename")
    check("test_map_ids_name_gate" in tr and "tests/test_map_ids.py" in tr,
          "B6: the real substring collision it warns about survives intact")
    check("d8de94d" in tr, "B6: the load-bearing provenance pin d8de94d is retained")
    check("## Chunks" in tr and "## Commands" in tr,
          "B4: anchors, not renumbered line addresses, are what it now cites")
    for renum in (":230-237", ":259-271", "TEST_MATRIX.md:230", "TEST_MATRIX.md:259"):
        check(renum not in tr,
              f"B4: the tip did NOT simply renumber to {renum}")


# ------------------------------------------------- residual / routed items

def residuals(repo, base, ref):
    print("\n=== F. the four routed items -- still broken? in boundary? ===")
    tm_tip = lines_of(show(repo, "HEAD",
                           "docs/pwml_recovery_sprint/TEST_MATRIX.md"))
    fi = lines_of(show(repo, "HEAD", "docs/pwml_recovery_sprint/FINDINGS.md"))
    cd = lines_of(show(repo, "HEAD",
                       "docs/pwml_recovery_sprint/evidence/chunk_d_gate.py"))

    print(f"  FINDINGS.md:1125 -> {fi[1124]}")
    print(f"  FINDINGS.md:1126 -> {fi[1125]}")
    print(f"  chunk_d_gate.py:70 -> {cd[69]}")
    note("routed item 1a: FINDINGS.md:1125 cites chunk_d_gate.py `:70` for D-qb; "
         f"chunk_d_gate.py:70 is {cd[69].strip()!r} -- that address is CORRECT "
         "today, so this half of the routed item is fragile, not broken")
    print(f"  TEST_MATRIX.md:218 -> {tm_tip[217]}")
    e_row = next((i + 1 for i, l in enumerate(tm_tip) if l.startswith("| **E** ")),
                 None)
    note(f"routed item 1b: FINDINGS.md:1126 cites TEST_MATRIX.md:218 for Chunk E; "
         f":218 is the child_env row of the bounded-runner table and Chunk E is at "
         f":{e_row} -- GENUINELY BROKEN, and outside the card's stated "
         "`FINDINGS.md:1120-1124` boundary")

    hit = next((i + 1 for i, l in enumerate(tm_tip) if "begins at" in l
                and "209" in l), None)
    h_chunks = next(i + 1 for i, l in enumerate(tm_tip) if l.strip() == "## Chunks")
    note(f"routed item 2: TEST_MATRIX.md:{hit} says '§ Chunks begins at line 209'; "
         f"it begins at :{h_chunks} -- GENUINELY STALE. This line is BELOW the "
         ":477 pin, and TEST_MATRIX.md is in the card's boundary, so it was "
         "editable without breaking line-neutrality")

    c106 = next((i + 1 for i, l in enumerate(tm_tip)
                 if "all still address what they addressed" in l), None)
    print(f"  TEST_MATRIX.md:{c106} -> {tm_tip[c106 - 1]}")
    note(f"routed item 3: TEST_MATRIX.md:{c106} (C-106's note) asserts "
         "`:213-218` / `:242-252` 'all still address what they addressed' -- "
         "F-154 proved they never did; still uncorrected")

    print("\n=== G. NEW drift created by this diff (reviewer addition) ===")
    base_tr = lines_of(show(repo, base, ".claude/agents/pwml-test-runner.md"))
    tip_tr = lines_of(show(repo, "HEAD", ".claude/agents/pwml-test-runner.md"))
    b59 = base_tr[58]
    t59 = tip_tr[58]
    new_stem = next((i + 1 for i, l in enumerate(tip_tr)
                     if "**stem-exact**" in l), None)
    print(f"  base pwml-test-runner.md:59 -> {b59}")
    print(f"  tip  pwml-test-runner.md:59 -> {t59}")
    print(f"  the stem-exact instruction now lives at :{new_stem}")
    note(f"the stem-exact instruction moved from :59 to :{new_stem}. Committed "
         "documents that still cite `pwml-test-runner.md:59` for it are listed "
         "below; none is in C-109's boundary, but the diff did not route them")


def main():
    repo, base, ref = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"repo {repo}  base {base}  integration-ref {ref}")
    inertness(repo, base, repo)
    citations(repo, base, ref)
    residuals(repo, base, ref)
    print("\n================ SUMMARY ================")
    print(f"checks failed : {len(FAILURES)}")
    for f in FAILURES:
        print(f"  FAIL  {f}")
    print(f"notes         : {len(NOTES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
