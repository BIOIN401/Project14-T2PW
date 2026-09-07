"""ORCH-724 reproducibility proof — reconstruct the exact pilot-running bytes.

READ-ONLY with respect to the primary checkout. Creates a DETACHED worktree at the
pilot production SHA in a temporary directory, applies the preserved patch, and checks
the resulting ``streamlit_app.py`` against the recorded pilot hashes.

WHY TWO HASHES, AND WHY THAT IS NOT A HEDGE
-------------------------------------------
``core.autocrlf=true`` is set globally on this machine and this path carries no ``text``
attribute, so every tracked text file is **LF in the object store and CRLF in the working
tree** (see ``.gitattributes``, which documents exactly this). The pilot therefore executed
**CRLF** bytes whose sha256 is ``47e4fafa…``. That hash is *platform-dependent*: the
identical content checked out on an LF platform hashes ``25112238…``.

So the content identity is the **LF** hash, and the **CRLF** hash is what this Windows
machine actually executed. Both are pinned. A proof that verified only one of them would
either fail spuriously on Linux or fail to describe what actually ran here.

Runs no pipeline leg, no LLM call, and no test suite. Does not touch the primary
worktree's modified file.

Usage:
  python reconstruct_proof.py <repo-root> [--json OUT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

REL = "src/t2pw/app/streamlit_app.py"

#: The pilot production SHA. ORCH-724's twenty legs ran with the tree at this commit
#: PLUS the preserved patch.
PILOT_SHA = "c4a97f6006cdfe2d3a30790072826d4f1eaa74de"

#: sha256 of the bytes the pilot actually executed (CRLF working tree, Windows).
PILOT_CRLF_SHA256 = "47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632"

#: sha256 of the same content with LF endings -- the platform-independent identity.
PILOT_LF_SHA256 = "251122389a2d29e80c157ee139837d06c6f82b7ad6e215d144ecc41b436933bf"

#: sha256 of the committed blob at PILOT_SHA (always LF; it is the object store).
COMMITTED_LF_SHA256 = "70299631b41762f7f1aa5c1cc35bcbdffd4b3711f83f69ab953f9f82f8713c50"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run(args, cwd, check=True) -> subprocess.CompletedProcess:
    p = subprocess.run(args, cwd=str(cwd), capture_output=True)
    if check and p.returncode != 0:
        raise SystemExit("command failed: %s\n%s\n%s" % (
            " ".join(args), p.stdout.decode("utf-8", "replace"),
            p.stderr.decode("utf-8", "replace")))
    return p


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    patch = root / "docs/pwml_recovery_sprint/evidence/repro/ORCH-724/streamlit_app.pilot.patch"
    patch_bytes = patch.read_bytes()

    result: Dict[str, Any] = {
        "instrument": "orch724_reconstruct_proof",
        "read_only_wrt_primary_checkout": True,
        "ran_no_pipeline_leg": True,
        "pilot_production_sha": PILOT_SHA,
        "patch_path": patch.relative_to(root).as_posix(),
        "patch_sha256": sha256(patch_bytes),
        "patch_bytes": len(patch_bytes),
        "expected": {
            "committed_lf_sha256": COMMITTED_LF_SHA256,
            "pilot_lf_sha256": PILOT_LF_SHA256,
            "pilot_crlf_sha256": PILOT_CRLF_SHA256,
        },
    }

    tmp = Path(tempfile.mkdtemp(prefix="orch724_repro_"))
    wt = tmp / "tree"
    try:
        # Detached worktree at the pilot SHA. Detached so no branch is created or moved.
        run(["git", "worktree", "add", "--detach", str(wt), PILOT_SHA], cwd=root)

        target = wt / REL
        before = target.read_bytes()
        before_lf = before.replace(b"\r\n", b"\n")
        result["checked_out"] = {
            "sha256_as_checked_out": sha256(before),
            "sha256_lf_normalized": sha256(before_lf),
            "matches_committed_after_lf_normalization":
                sha256(before_lf) == COMMITTED_LF_SHA256,
        }

        # Apply the preserved patch. git apply works on the LF-canonical form.
        run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=wt)

        after = target.read_bytes()
        after_lf = after.replace(b"\r\n", b"\n")
        after_crlf = after_lf.replace(b"\n", b"\r\n")

        result["reconstructed"] = {
            "sha256_as_written": sha256(after),
            "sha256_lf_normalized": sha256(after_lf),
            "sha256_crlf_normalized": sha256(after_crlf),
        }
        result["verdict"] = {
            "lf_identity_matches_pilot": sha256(after_lf) == PILOT_LF_SHA256,
            "crlf_bytes_match_what_the_pilot_executed":
                sha256(after_crlf) == PILOT_CRLF_SHA256,
        }
        result["reconstruction_exact"] = bool(
            result["verdict"]["lf_identity_matches_pilot"]
            and result["verdict"]["crlf_bytes_match_what_the_pilot_executed"])
    finally:
        # Always detach and remove the temporary worktree. Never prune anything else.
        run(["git", "worktree", "remove", "--force", str(wt)], cwd=root, check=False)
        shutil.rmtree(tmp, ignore_errors=True)
        result["temp_worktree_removed"] = not wt.exists()

    print("=" * 84)
    print("ORCH-724 RECONSTRUCTION PROOF")
    print("=" * 84)
    print("pilot production SHA :", PILOT_SHA)
    print("patch                :", result["patch_path"])
    print("patch sha256         :", result["patch_sha256"])
    print()
    print("checked out at SHA, LF-normalized == committed blob :",
          result["checked_out"]["matches_committed_after_lf_normalization"])
    print()
    print("AFTER APPLYING THE PATCH")
    print("  LF   sha256 :", result["reconstructed"]["sha256_lf_normalized"])
    print("       expected:", PILOT_LF_SHA256)
    print("       MATCH   :", result["verdict"]["lf_identity_matches_pilot"])
    print("  CRLF sha256 :", result["reconstructed"]["sha256_crlf_normalized"])
    print("       expected:", PILOT_CRLF_SHA256)
    print("       MATCH   :", result["verdict"]["crlf_bytes_match_what_the_pilot_executed"])
    print()
    print("RECONSTRUCTION EXACT :", result["reconstruction_exact"])
    print("temp worktree removed:", result["temp_worktree_removed"])

    if args.json_path:
        p = Path(args.json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("\nJSON:", p)

    return 0 if result["reconstruction_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
