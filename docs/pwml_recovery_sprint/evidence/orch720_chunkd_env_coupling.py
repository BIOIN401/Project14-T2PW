"""ORCH-720: is the Chunk D core failure a CODE regression or an ENVIRONMENT coupling?

The claim under test
--------------------
`tests/test_pwml_writer.py::test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance`
ends with::

    # P4-01: no worktree carries a .env, so ``PathBankDbResolver.from_env()``
    # answers ``None`` and this leg is offline. Asserted rather than assumed --
    # a reachable DB would make the preflight below silently absent instead of
    # wrong, and a vacuous assertion is not evidence.
    assert report["db_resolution"]["available"] is False

It is GREEN in every committed Chunk D run and RED in the primary checkout at
`0859fba9`. **Two explanations fit that pattern and they have opposite consequences:**

* **(A) a code regression** at some commit after the last green run, or
* **(B) an environment coupling** -- the assertion depends on `.env` being ABSENT,
  every committed green run was executed inside a **worktree** (which carries no
  `.env`), and this is simply the first time the gate has been run in the primary
  checkout, which has one.

**A SHA-based A/B cannot separate these**, because checking out an older SHA in the
primary checkout leaves `.env` in place, and exporting it to a temp tree removes
`.env` -- each arm moves BOTH variables. So this probe holds the tree and the
commit fixed and moves **only the environment**: it runs the same test twice in the
same checkout, once as-is and once with the resolution DB deconfigured for the
child process.

**How the child is deconfigured, and why DELETING the keys would not have worked.**
`config.ensure_dotenv_loaded` calls `load_dotenv(dotenv_path=PROJECT_ROOT/".env",
override=False)`, so any key ALREADY PRESENT in `os.environ` is left alone and every
absent one is filled in from `.env`. Removing `PATHBANK_DB_*` from the child would
therefore have been undone by the loader a moment later, and arm 2 would have
reproduced arm 1 for a reason having nothing to do with the hypothesis. Instead the
keys are set to the EMPTY STRING: present, so `override=False` preserves them, and
empty, so `from_env`'s `if not host or not user: return None` guard fires --
`map_ids.py:855`. **That is the same code path a worktree reaches, reached
deliberately.**

* If arm 2 is GREEN, the variable is the environment and (B) is established.
* If arm 2 is RED, the environment is exonerated and (A) must be investigated.

**This probe changes nothing on disk.** It never edits, moves or deletes `.env`,
and it writes nothing outside its own `--basetemp`. The only thing it varies is
the environment dictionary handed to a child process.

Usage:  orch720_chunkd_env_coupling.py <repo>
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: BOTH Chunk D failures at 0859fba9, not just the one whose comment names the
#: mechanism. The second is included precisely because its assertion
#: ("processes moved pre-freeze") does NOT mention .env at all -- if the same
#: lever moves both, the coupling is a property of the environment rather than of
#: one test's wording.
NODES = (
    "tests/test_pwml_writer.py::"
    "test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance",
    "tests/test_streamlit_quarantine_boundary.py::"
    "test_research_mode_keeps_the_unmapped_candidate_and_does_not_block",
)


def run_arm(tree: Path, label: str, env: dict, basetemp: str) -> dict:
    """Run each node in its OWN process, as chunk_d_gate.py does.

    One process per node, never a batch: `TEST_MATRIX` records that these
    Streamlit AppTest files stall a shared pytest process silently, and a stall
    here would be indistinguishable from the result this probe exists to read.
    """
    print()
    print("=" * 78)
    print(f"ARM: {label}")
    print("=" * 78)
    host = env.get("PATHBANK_DB_HOST")
    user = env.get("PATHBANK_DB_USER")
    print(f"  PATHBANK_DB_HOST in child : {'<empty>' if host == '' else ('<set>' if host else '(absent)')}")
    print(f"  PATHBANK_DB_USER in child : {'<empty>' if user == '' else ('<set>' if user else '(absent)')}")
    results = {}
    for index, node in enumerate(NODES, start=1):
        cmd = [
            sys.executable, "-m", "pytest", "-q", "--no-header",
            f"--basetemp={basetemp}/n{index}", node,
        ]
        proc = subprocess.run(cmd, cwd=str(tree), env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        summary = [ln for ln in proc.stdout.splitlines()
                   if ln.strip() and ("passed" in ln or "failed" in ln or "error" in ln)]
        print()
        print(f"  [{index}] {node.split('::')[-1]}")
        print(f"      {summary[-1].strip() if summary else '(no summary line)'}")
        print(f"      EXIT {proc.returncode} -> {'GREEN' if proc.returncode == 0 else 'RED'}")
        results[node] = proc.returncode
    return results


def main() -> int:
    tree = Path(sys.argv[1]).resolve()
    basetemp_root = sys.argv[2] if len(sys.argv) > 2 else "C:/t/orch720/envcouple"

    env_file = tree / ".env"
    print(f"tree            : {tree}")
    print(f".env present    : {env_file.is_file()}")
    print(f"nodes under test: {len(NODES)}")
    for node in NODES:
        print(f"  - {node}")
    print()
    print("NOTHING ON DISK IS MODIFIED. .env is neither edited, moved nor deleted.")

    base = dict(os.environ)
    base.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    base.setdefault("PYTHONIOENCODING", "utf-8")

    # ARM 1 -- exactly what the Chunk D gate ran: the primary checkout, untouched.
    arm1 = run_arm(tree, "1. primary checkout, environment UNTOUCHED (what the gate ran)",
                   dict(base), f"{basetemp_root}/arm1")

    # ARM 2 -- same tree, same commit, same test. Only the environment moves.
    # EMPTY, not absent: load_dotenv(override=False) refills an ABSENT key from
    # .env but leaves a PRESENT-and-empty one alone, so this is what actually
    # reaches from_env's `if not host or not user: return None` guard
    # (map_ids.py:855). Deleting the keys would have been silently undone.
    stripped = dict(base)
    stripped["PATHBANK_DB_HOST"] = ""
    stripped["PATHBANK_DB_USER"] = ""
    arm2 = run_arm(tree, "2. SAME tree, SAME commit -- resolution DB deconfigured (worktree-like)",
                   stripped, f"{basetemp_root}/arm2")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    for node in NODES:
        a1 = "GREEN" if arm1[node] == 0 else "RED"
        a2 = "GREEN" if arm2[node] == 0 else "RED"
        moved = "MOVED" if arm1[node] != arm2[node] else "unchanged"
        print(f"  {node.split('::')[-1][:62]:<62} arm1={a1:<5} arm2={a2:<5} {moved}")
    red_with = sum(1 for n in NODES if arm1[n] != 0)
    green_without = sum(1 for n in NODES if arm2[n] == 0)
    print()
    print(f"  RED with the DB configured      : {red_with} / {len(NODES)}")
    print(f"  GREEN with it deconfigured      : {green_without} / {len(NODES)}")
    print()
    arm1_bad = red_with == len(NODES)
    arm2_ok = green_without == len(NODES)
    if arm1_bad and arm2_ok:
        print("  ENVIRONMENT COUPLING ESTABLISHED (B).")
        print("  The tree and the commit are held fixed; only the DB config moved, and the")
        print("  test's verdict moved with it. This is NOT a code regression, and the")
        print("  committed green Chunk D runs were green because they ran in WORKTREES,")
        print("  which carry no .env. The gate has never been shown green in the primary")
        print("  checkout.")
        return 0
    if arm1_bad:
        print("  ENVIRONMENT EXONERATED for at least one node. The test is red with and without the DB config, so")
        print("  the cause is elsewhere and (A) must be investigated before any launch.")
        return 1
    if not arm1_bad:
        print("  ARM 1 DID NOT REPRODUCE every Chunk D failure here. Do not")
        print("  conclude anything: re-read the gate's own report rather than this probe.")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
