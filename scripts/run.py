from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from apply_audit_patch import run_apply  # noqa: E402
from audit_json_llm import run_audit  # noqa: E402
from json_to_sbml import build_sbml  # noqa: E402
from map_ids import run_mapping  # noqa: E402
from sbml_overwatch import run_sbml_overwatch  # noqa: E402


def _resolve_output_path(out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-pipeline converter: final.json -> audit -> mapped IDs -> SBML."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default="pwml_pipeline_output.json",
        help="Input final JSON path from the existing extraction/inference pipeline",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=".",
        help="Directory for generated artifacts",
    )
    parser.add_argument("--no-llm-audit", action="store_true", help="Disable LLM stage for audit and use deterministic checks only")
    parser.add_argument("--default-compartment", default="cell", help="Default compartment name if missing")
    parser.add_argument(
        "--mapping-cache",
        default="id_mapping_cache.json",
        help="Path to mapping cache JSON (relative to out-dir if not absolute)",
    )
    parser.add_argument("--no-sbml-overwatch", action="store_true", help="Disable semantic SBML overwatch stage")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    audit_report = _resolve_output_path(out_dir, "audit_report.json")
    audit_patch = _resolve_output_path(out_dir, "audit_patch.json")
    apply_report = _resolve_output_path(out_dir, "audit_apply_report.json")
    audited_json = _resolve_output_path(out_dir, "final.audited.json")
    mapped_json = _resolve_output_path(out_dir, "final.mapped.json")
    mapping_report = _resolve_output_path(out_dir, "mapping_report.json")
    sbml_file = _resolve_output_path(out_dir, "pathway.sbml")
    sbml_report_json = _resolve_output_path(out_dir, "sbml_validation_report.json")
    sbml_report_txt = _resolve_output_path(out_dir, "sbml_validation_report.txt")
    sbml_overwatch_report = _resolve_output_path(out_dir, "sbml_overwatch_report.json")

    mapping_cache = Path(args.mapping_cache)
    if not mapping_cache.is_absolute():
        mapping_cache = out_dir / mapping_cache

    # Stage 1: audit -> patch proposal
    run_audit(
        input_path,
        audit_report,
        audit_patch,
        use_llm=not args.no_llm_audit,
        llm_temperature=0.0,
        llm_max_tokens=2000,
    )

    # Stage 1b: deterministic patch acceptance
    run_apply(
        input_path,
        audit_patch,
        audited_json,
        audit_report_path=audit_report,
        apply_report_path=apply_report,
    )

    # Stage 2: deterministic ID mapping with API lookups
    run_mapping(
        audited_json,
        mapped_json,
        mapping_report,
        cache_path=mapping_cache,
    )

    # Stage 3: deterministic SBML build + validation
    sbml_result = build_sbml(
        mapped_json,
        sbml_file,
        sbml_report_json,
        sbml_report_txt,
        default_compartment_name=str(args.default_compartment),
    )

    overwatch_result: Dict[str, Any] = {}
    if not args.no_sbml_overwatch:
        overwatch_result = run_sbml_overwatch(
            mapped_json,
            sbml_file,
            sbml_report_json,
            sbml_overwatch_report,
            use_llm=True,
            llm_max_tokens=1800,
        )

    summary = {
        "audit_report": str(audit_report),
        "audit_patch": str(audit_patch),
        "audited_json": str(audited_json),
        "mapped_json": str(mapped_json),
        "mapping_report": str(mapping_report),
        "sbml_file": str(sbml_file),
        "sbml_validation_report_json": str(sbml_report_json),
        "sbml_validation_report_txt": str(sbml_report_txt),
        "sbml_overwatch_report_json": str(sbml_overwatch_report) if not args.no_sbml_overwatch else "",
        "sbml_overwatch_summary": overwatch_result.get("summary", {}),
        "sbml_validation_has_errors": bool(sbml_result.get("validation", {}).get("has_errors")),
    }
    print(json.dumps(summary, indent=2))

    if summary["sbml_validation_has_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
