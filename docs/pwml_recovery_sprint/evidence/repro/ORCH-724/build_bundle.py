"""ORCH-724 — build the EVALUATION REPRODUCIBILITY BUNDLE for the unseen pilot.

READ-ONLY over the repository. Writes one manifest describing exactly what produced the
twenty pilot legs, so the run can be reconstructed without committing user-owned
working-tree changes into production.

THE CLAIM THIS BUNDLE EXISTS TO MAKE TRUE
-----------------------------------------
    pilot state == <production SHA> + <streamlit pilot patch>

and nothing weaker. The previously recorded claim -- "reproducible from the production
commit alone" -- is FALSE and is corrected here.

A CORRECTION TO THE EARLIER FREEZE MANIFEST, MADE DELIBERATELY
--------------------------------------------------------------
``orch724_freeze_manifest.py`` redacted every key whose NAME contains "TOKEN". That
swept up ``OPENROUTER_*_MAX_TOKENS``, which are not secrets -- they are **generation
length limits that materially change pipeline behaviour**, and the pilot patch makes the
Stage-1/Stage-2 budgets read from exactly those variables. Redacting them hid a
load-bearing part of the run's configuration behind a rule meant for credentials. They
are recorded here in full. Genuine credentials are still recorded as presence + length
only, never by value.

Runs no pipeline leg and no LLM call.

Usage:
  python build_bundle.py <repo-root> [--json OUT]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REL_APP = "src/t2pw/app/streamlit_app.py"
PILOT_SHA = "c4a97f6006cdfe2d3a30790072826d4f1eaa74de"
PILOT_RUN_DIR = "runs_verify/2026-09-06_1425"

#: Credential-bearing names. NOT a substring match on "TOKEN" -- see the module docstring.
SECRET_NAMES = ("API_KEY", "PASSWORD", "SECRET", "ACCESS_TOKEN", "AUTH_TOKEN")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes()) if p.is_file() else ""


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(("git",) + args, cwd=str(root), capture_output=True).stdout


def is_secret(name: str) -> bool:
    up = name.upper()
    return any(s in up for s in SECRET_NAMES)


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    here = root / "docs/pwml_recovery_sprint/evidence/repro/ORCH-724"

    app_wt = (root / REL_APP).read_bytes()
    app_committed = git(root, "show", f"{PILOT_SHA}:{REL_APP}")
    patch_path = here / "streamlit_app.pilot.patch"
    patch_bytes = patch_path.read_bytes()

    # ---- environment: real config vs credentials -------------------------
    env_cfg: Dict[str, str] = {}
    creds: Dict[str, str] = {}
    env_file = root / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.split("#")[0].strip()
            if is_secret(k):
                creds[k] = ("present (len %d)" % len(v)) if v else "absent"
            else:
                env_cfg[k] = v

    # ---- prompt surface --------------------------------------------------
    prompts: Dict[str, str] = {}
    for pattern in ("src/t2pw/**/*.txt", "src/t2pw/**/prompt*.py", "src/t2pw/llm/**/*.py"):
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                prompts[p.relative_to(root).as_posix()] = sha256_file(p)[:16]

    # ---- evaluator instruments ------------------------------------------
    instruments = [
        "docs/pwml_recovery_sprint/evidence/orch724_pilot_summary.py",
        "docs/pwml_recovery_sprint/evidence/orch724_admission_audit.py",
        "docs/pwml_recovery_sprint/evidence/orch724_freeze_manifest.py",
        "docs/pwml_recovery_sprint/evidence/f179_repair_regression.py",
        "docs/pwml_recovery_sprint/evidence/rd093_rag_metrics.py",
        "docs/pwml_recovery_sprint/evidence/bounded_run.py",
        "docs/pwml_recovery_sprint/evidence/repro/ORCH-724/reconstruct_proof.py",
    ]

    run_dir = root / PILOT_RUN_DIR
    legs = sorted(p.relative_to(root).as_posix()
                  for p in run_dir.glob("papers/*/*") if p.is_dir()) if run_dir.is_dir() else []

    bundle: Dict[str, Any] = {
        "bundle": "ORCH-724 evaluation reproducibility bundle",
        "purpose": (
            "Preserve the exact state that produced the twenty unseen-pilot legs WITHOUT "
            "committing user-owned working-tree changes into production."
        ),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "read_only": True,
        "ran_no_pipeline_leg": True,

        "REPRODUCIBILITY_STATEMENT": {
            "pilot_state_equals": "<production SHA> + <streamlit pilot patch>",
            "production_sha": PILOT_SHA,
            "streamlit_pilot_patch_sha256": sha256_bytes(patch_bytes),
            "correct_claim": (
                "The unseen pilot is reproducible from production SHA %s WITH the patch at "
                "%s applied to %s. It is NOT reproducible from the production commit alone."
                % (PILOT_SHA, patch_path.relative_to(root).as_posix(), REL_APP)
            ),
            "superseded_false_claim": (
                "'the pilot is reproducible from the production commit alone' -- FALSE, "
                "corrected by this bundle."
            ),
        },

        "streamlit_app": {
            "path": REL_APP,
            "committed_at_pilot_sha_sha256": sha256_bytes(app_committed),
            "working_tree_pilot_sha256_CRLF": sha256_bytes(app_wt),
            "working_tree_pilot_sha256_LF": sha256_bytes(app_wt.replace(b"\r\n", b"\n")),
            "committed_bytes": len(app_committed),
            "working_tree_bytes": len(app_wt),
            "byte_delta_total": len(app_wt) - len(app_committed),
            "byte_delta_from_line_endings": app_wt.count(b"\r\n"),
            "byte_delta_real_content": (
                len(app_wt.replace(b"\r\n", b"\n")) - len(app_committed)),
            "line_ending_note": (
                "core.autocrlf=true is set globally and this path carries no `text` "
                "attribute, so the object store holds LF and the working tree holds CRLF "
                "(.gitattributes documents this). The CRLF hash is what THIS machine "
                "executed; the LF hash is the platform-independent content identity. "
                "7,269 of the 8,279 total byte delta is line endings, not content."
            ),
            "patch": patch_path.relative_to(root).as_posix(),
            "patch_sha256": sha256_bytes(patch_bytes),
            "patch_bytes": len(patch_bytes),
            "patch_is_binary_safe_committed": (
                "yes -- .gitattributes gives this path `-text` so checkout cannot "
                "CRLF-mangle it and change its pinned hash"
            ),
        },

        "SEMANTIC_IMPACT": {
            "ui_only": False,
            "alters_pipeline_execution_configuration": True,
            "what_changes": [
                "Stage 1 (extraction) LLM max_tokens: committed literal 24000 -> "
                "_bounded_env_int('OPENROUTER_EXTRACTION_MAX_TOKENS', 64000)",
                "Stage 2 (inference) LLM max_tokens: committed literal 20000 -> "
                "_bounded_env_int('OPENROUTER_INFERENCE_MAX_TOKENS', 64000)",
            ],
            "consumed_at": [
                f"{REL_APP}:5467  max_tokens=int(extract_tokens)",
                f"{REL_APP}:5588  max_tokens=int(infer_tokens)",
            ],
            "reaches_the_pilot_because": (
                "batch/driver.py drives the app through AppTest and sets ONLY the "
                "export-mode radio, the input-mode radio, the source text area and the "
                "buttons. It never sets the Stage 1 / Stage 2 max-token number inputs, so "
                "the widget DEFAULTS are what executed."
            ),
            "effective_values_during_the_pilot": {
                "OPENROUTER_EXTRACTION_MAX_TOKENS": env_cfg.get("OPENROUTER_EXTRACTION_MAX_TOKENS"),
                "OPENROUTER_INFERENCE_MAX_TOKENS": env_cfg.get("OPENROUTER_INFERENCE_MAX_TOKENS"),
                "committed_code_would_have_used": {"stage1": 24000, "stage2": 20000},
            },
            "direction_of_effect": (
                "The pilot ran with SMALLER generation budgets than the committed code "
                "would give (16000 vs 24000 for Stage 1; 16000 vs 20000 for Stage 2), so "
                "if this mattered it would tend to UNDER-state extraction, not flatter it."
            ),
            "evidence_consistent_but_not_proof": (
                "Two pilot legs failed with 'failed to produce valid JSON', a known "
                "symptom of a generation budget cut mid-object. This is CIRCUMSTANTIAL. "
                "Proving causation would require re-running legs, which this task forbids "
                "and which the pilot charter forbids without an infrastructure fault."
            ),
        },

        "model_and_provider": {
            "LLM_PROVIDER": env_cfg.get("LLM_PROVIDER"),
            "OPENROUTER_MODEL": env_cfg.get("OPENROUTER_MODEL"),
            "LLM_TEMPERATURE": env_cfg.get("LLM_TEMPERATURE"),
            "per_stage_models": {k: v for k, v in env_cfg.items()
                                 if k.startswith("OPENROUTER_") and k.endswith("_MODEL")},
            "token_budgets": {k: v for k, v in env_cfg.items() if "MAX_TOKENS" in k},
            "retries": env_cfg.get("LLM_MAX_RETRIES"),
        },
        "credentials_presence_only": creds,
        "retrieval_and_index": {k: v for k, v in env_cfg.items() if k.startswith("RAG_")},
        "environment_config_full_no_secrets": env_cfg,
        "prompt_surface_sha256_16": prompts,
        "evaluator_versions_sha256_16": {
            p: sha256_file(root / p)[:16] for p in instruments},

        "unseen_paper_manifest": {
            "topics_file": "topics_unseen_pilot.txt",
            "topics_sha256": sha256_file(root / "topics_unseen_pilot.txt"),
            "manifest_doc": "docs/pwml_recovery_sprint/UNSEEN-COHORT-MANIFEST.md",
            "known_manifest_defect": (
                "PMC3480714 carries the wrong organism (Salmonella enterica); the paper "
                "works in a recombinant E. coli host. Both its legs ended scope_conflict "
                "and the gate was correct. Recorded, not silently corrected."
            ),
        },
        "pilot_run": {
            "run_dir": PILOT_RUN_DIR,
            "leg_dirs": legs,
            "leg_count": len(legs),
            "summary_json": "docs/pwml_recovery_sprint/evidence/orch724_pilot_summary.json",
            "manual_review_package": "docs/pwml_recovery_sprint/PILOT-MANUAL-REVIEW.md",
            "bounded_run_report": "docs/pwml_recovery_sprint/evidence/g11/ORCH-724/21-unseen-pilot.json",
            "wall_clock_seconds": 34978.27,
            "note": (
                "Run directories are NOT tracked, following the precedent of every recent "
                "run (runs_verify/2026-09-02_2052 is likewise untracked). This bundle pins "
                "their paths and content hashes rather than committing 56 MB."
            ),
        },
        "reconstruction_proof": {
            "instrument": "docs/pwml_recovery_sprint/evidence/repro/ORCH-724/reconstruct_proof.py",
            "result": "docs/pwml_recovery_sprint/evidence/repro/ORCH-724/reconstruct_proof.json",
            "method": (
                "detached worktree at the production SHA in a temp dir -> git apply the "
                "preserved patch -> compare sha256 in both LF and CRLF normalizations -> "
                "remove the temp worktree. No pipeline leg re-run."
            ),
        },
        "scope_statement": {
            "production_semantics_changed": False,
            "gold_changed": False,
            "pilot_outputs_changed": False,
            "run_directories_changed": False,
            "protected_working_tree_bytes_changed": False,
            "main_touched": False,
            "streamlit_app_still_uncommitted": True,
        },
    }

    print("=" * 84)
    print("ORCH-724 EVALUATION REPRODUCIBILITY BUNDLE")
    print("=" * 84)
    print("production SHA        :", PILOT_SHA)
    print("patch sha256          :", bundle["streamlit_app"]["patch_sha256"])
    print("committed  sha256     :", bundle["streamlit_app"]["committed_at_pilot_sha_sha256"])
    print("pilot CRLF sha256     :", bundle["streamlit_app"]["working_tree_pilot_sha256_CRLF"])
    print("pilot LF   sha256     :", bundle["streamlit_app"]["working_tree_pilot_sha256_LF"])
    print("byte delta total      :", bundle["streamlit_app"]["byte_delta_total"])
    print("  of which line endings:", bundle["streamlit_app"]["byte_delta_from_line_endings"])
    print("  real content         :", bundle["streamlit_app"]["byte_delta_real_content"])
    print()
    print("UI ONLY               :", bundle["SEMANTIC_IMPACT"]["ui_only"])
    print("alters pipeline config:", bundle["SEMANTIC_IMPACT"]["alters_pipeline_execution_configuration"])
    for w in bundle["SEMANTIC_IMPACT"]["what_changes"]:
        print("   -", w)
    print("effective during pilot:", json.dumps(
        bundle["SEMANTIC_IMPACT"]["effective_values_during_the_pilot"]))
    print()
    print("legs pinned           :", len(legs))
    print("prompt files hashed   :", len(prompts))

    if args.json_path:
        p = Path(args.json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nJSON:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
