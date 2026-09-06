"""ORCH-724 — capture the FROZEN product configuration before the unseen pilot.

READ-ONLY. Records the exact state the twenty unseen legs run against, so the pilot's
numbers are attributable to a configuration that can be reconstructed later.

SECRETS ARE NEVER RECORDED. API keys are reported only as present/absent with a length,
never by value, and never by prefix.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

SECRET_KEYS = ("API_KEY", "PASSWORD", "SECRET", "TOKEN")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(("git",) + args, capture_output=True,
                              cwd=str(ROOT)).stdout.decode("utf-8", "replace").strip()
    except Exception:
        return ""


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

# ---------------------------------------------------------------------------
# 1. Production SHA and the protected identities.
# ---------------------------------------------------------------------------
manifest: Dict[str, Any] = {
    "captured_for": "ORCH-724 unseen-paper pilot",
    "read_only": True,
    "production_sha": git("rev-parse", "HEAD"),
    "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "origin_sha": git("rev-parse", "origin/sprint/pwml-recovery"),
    "gold_blob": git("hash-object", "src/t2pw/bench/gold/pinned_v1.json"),
    "main_local": git("rev-parse", "main"),
}

# ---------------------------------------------------------------------------
# 2. Model configuration — names and numbers only, never a secret.
# ---------------------------------------------------------------------------
env_path = ROOT / ".env"
model_cfg: Dict[str, str] = {}
secrets_presence: Dict[str, str] = {}
if env_path.is_file():
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split("#")[0].strip()
        if any(s in key.upper() for s in SECRET_KEYS):
            secrets_presence[key] = ("present (len %d)" % len(value)) if value else "absent"
            continue
        model_cfg[key] = value
manifest["model_configuration"] = model_cfg
manifest["secrets"] = secrets_presence

# ---------------------------------------------------------------------------
# 3. Prompts — every prompt the pipeline sends, by content hash.
# ---------------------------------------------------------------------------
prompt_files: Dict[str, str] = {}
for pattern in ("src/t2pw/**/*.txt", "src/t2pw/**/prompt*.py", "src/t2pw/llm/**/*.py"):
    for p in sorted(ROOT.glob(pattern)):
        if p.is_file():
            prompt_files[p.relative_to(ROOT).as_posix()] = sha256_file(p)[:16]
manifest["prompt_surface"] = prompt_files

# ---------------------------------------------------------------------------
# 4. Retrieval / index configuration.
# ---------------------------------------------------------------------------
index_dir = ROOT / "data" / "rag_index"
manifest["retrieval"] = {
    "index_dir": index_dir.as_posix(),
    "index_dir_exists": index_dir.is_dir(),
    "acquire_cache_fulltext_files": len(list((index_dir / "acquire_cache" / "fulltext").glob("*.json")))
    if (index_dir / "acquire_cache" / "fulltext").is_dir() else 0,
    "config_keys_from_env": {k: v for k, v in model_cfg.items() if k.startswith("RAG_")},
}

# ---------------------------------------------------------------------------
# 5. The frozen production surface and the evaluator/instrument versions.
# ---------------------------------------------------------------------------
prod = [
    "src/t2pw/rag/admission.py",
    "src/t2pw/pipeline/reaction_support.py",
    "src/t2pw/pipeline/stage_contracts.py",
    "src/t2pw/app/streamlit_app.py",
    "src/t2pw/pwml/writer.py",
    "src/t2pw/batch/driver.py",
    "src/t2pw/bench/acceptance.py",
    "src/t2pw/bench/semantic.py",
    "src/t2pw/bench/goldset.py",
]
manifest["production_surface_sha256"] = {
    p: sha256_file(ROOT / p)[:16] for p in prod
}
# streamlit_app.py is MODIFIED-AND-NEVER-COMMITTED; record that explicitly rather
# than letting a later reader assume the working tree matches HEAD.
manifest["uncommitted_tracked_changes"] = [
    line[3:] for line in git("status", "--porcelain").splitlines()
    if line[:2].strip() and not line.startswith("??")
]

instruments = [
    "docs/pwml_recovery_sprint/evidence/orch724_admission_audit.py",
    "docs/pwml_recovery_sprint/evidence/orch724_pilot_summary.py",
    "docs/pwml_recovery_sprint/evidence/rd093_rag_metrics.py",
    "docs/pwml_recovery_sprint/evidence/rd093_two_table_metrics.py",
    "docs/pwml_recovery_sprint/evidence/rd092_1_reaction_lineage.py",
    "docs/pwml_recovery_sprint/evidence/f179_repair_regression.py",
    "docs/pwml_recovery_sprint/evidence/bounded_run.py",
]
# THE RUNNING IDENTITY IS NOT THE COMMIT ALONE. streamlit_app.py is tracked,
# modified and never committed -- a state this sprint has carried for many waves and
# which every historical benchmark also ran against. The pilot therefore executes the
# WORKING TREE bytes, not what production_sha contains, and a reader who checks out
# production_sha alone will NOT reproduce these results. Both hashes are recorded so
# the difference is on the record rather than discovered later.
_wt_streamlit = ROOT / "src/t2pw/app/streamlit_app.py"
_head_blob = subprocess.run(("git", "show", "HEAD:src/t2pw/app/streamlit_app.py"),
                            capture_output=True, cwd=str(ROOT)).stdout
manifest["running_identity_caveat"] = {
    "file": "src/t2pw/app/streamlit_app.py",
    "working_tree_sha256": sha256_file(_wt_streamlit),
    "head_blob_sha256": hashlib.sha256(_head_blob).hexdigest(),
    "differs_from_head": sha256_file(_wt_streamlit) != hashlib.sha256(_head_blob).hexdigest(),
    "note": (
        "The twenty pilot legs run against the WORKING TREE file. The reproducible "
        "identity of this pilot is (production_sha + this working_tree_sha256), not "
        "production_sha alone. Committing it is a production change and is NOT "
        "authorized under D-090; it must be resolved before publication."
    ),
}

manifest["evaluator_versions_sha256"] = {
    p: sha256_file(ROOT / p)[:16] for p in instruments
}

# ---------------------------------------------------------------------------
# 6. Environment.
# ---------------------------------------------------------------------------
manifest["environment"] = {
    "python": sys.version.split()[0],
    "platform": sys.platform,
    "cohort_topics_file": "topics_unseen_pilot.txt",
    "cohort_topics_sha256": sha256_file(ROOT / "topics_unseen_pilot.txt")[:16],
}

out = ROOT / "docs/pwml_recovery_sprint/evidence/orch724_freeze_manifest.json"
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("=" * 84)
print("ORCH-724 FROZEN CONFIGURATION")
print("=" * 84)
print("production SHA :", manifest["production_sha"])
print("origin agrees  :", manifest["production_sha"] == manifest["origin_sha"])
print("gold blob      :", manifest["gold_blob"])
print("provider       :", model_cfg.get("LLM_PROVIDER"))
print("model          :", model_cfg.get("OPENROUTER_MODEL"))
print("temperature    :", model_cfg.get("LLM_TEMPERATURE"))
print("secrets        :", secrets_presence)
print("prompt files   :", len(prompt_files))
print("index fulltext :", manifest["retrieval"]["acquire_cache_fulltext_files"])
print("uncommitted tracked changes:", len(manifest["uncommitted_tracked_changes"]))
for p in manifest["uncommitted_tracked_changes"]:
    print("    ", p)
print()
print("JSON:", out)
