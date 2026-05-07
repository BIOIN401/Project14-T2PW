from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from lxml import etree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pwml_ir import build_pwml_ir, validate_pwml_ir  # noqa: E402
from pwml_qa import run_pwml_qa  # noqa: E402
from pwml_validate import discover_structure_signature, repair_tree, validate_generated_tree  # noqa: E402
from pwml_writer import DeterministicPwmlBuilder  # noqa: E402


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="PWML-first converter: mapped final JSON -> PWML IR -> PWML.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input final mapped JSON path")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for PWML artifacts")
    parser.add_argument("--ref", default=str(PROJECT_ROOT / "reference" / "PW000001.pwml"), help="Reference PWML file")
    parser.add_argument("--name", default="Generated Pathway", help="Pathway name")
    parser.add_argument("--subject", default="Metabolic", help="Pathway subject")
    parser.add_argument("--description", default="", help="Pathway description")
    parser.add_argument("--width", type=int, default=3200, help="PWML canvas width")
    parser.add_argument("--height", type=int, default=1400, help="PWML canvas height")
    parser.add_argument("--background-color", default="#FFFFFF", help="PWML background color")
    parser.add_argument("--non-strict-db", action="store_true", help="Warn instead of erroring on missing DB identities")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    ir, ir_report = build_pwml_ir(
        payload,
        pathway_name=args.name,
        pathway_subject=args.subject,
        strict_db=not args.non_strict_db,
        width=args.width,
        height=args.height,
    )
    ir.setdefault("pathway", {})["description"] = args.description
    ir_validation = validate_pwml_ir(ir)

    ir_path = out_dir / "final.pwml_ir.json"
    ir_report_path = out_dir / "pwml_ir_report.json"
    ir_validation_path = out_dir / "pwml_ir_validation_report.json"
    pwml_path = out_dir / "pathway.pwml"
    validation_report_path = out_dir / "pwml_validation_report.json"
    qa_report_path = out_dir / "pwml_qa_report.json"

    _write_json(ir_path, ir)
    _write_json(ir_report_path, ir_report)
    _write_json(ir_validation_path, ir_validation)

    if ir_report.get("errors") or ir_validation.get("errors"):
        print(
            json.dumps(
                {
                    "ok": False,
                    "pwml_ir": str(ir_path),
                    "pwml_ir_report": str(ir_report_path),
                    "pwml_ir_validation_report": str(ir_validation_path),
                    "error": "PWML IR validation failed.",
                },
                indent=2,
            )
        )
        raise SystemExit(1)

    ref_path = Path(args.ref)
    signature = discover_structure_signature(ref_path)
    writer_args = SimpleNamespace(
        name=args.name,
        description=args.description,
        subject=args.subject,
        pw_id="PW000000",
        height=args.height,
        width=args.width,
        background_color=args.background_color,
        ref=str(ref_path),
    )
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=writer_args)
    build_result = builder.build()
    repaired = repair_tree(etree.ElementTree(build_result.root), signature)
    validation_report = validate_generated_tree(repaired, signature)
    xml_bytes = etree.tostring(repaired.getroot(), encoding="utf-8", xml_declaration=True, pretty_print=True)
    qa_report = run_pwml_qa(xml_bytes)

    pwml_path.write_bytes(xml_bytes)
    _write_json(validation_report_path, validation_report)
    _write_json(qa_report_path, qa_report)

    summary = {
        "ok": bool(validation_report.get("ok")) and bool(qa_report.get("ok")),
        "pwml_ir": str(ir_path),
        "pwml_ir_report": str(ir_report_path),
        "pwml_ir_validation_report": str(ir_validation_path),
        "pwml_file": str(pwml_path),
        "pwml_validation_report": str(validation_report_path),
        "pwml_qa_report": str(qa_report_path),
        "counts": build_result.counts,
        "pwml_validation_issue_count": validation_report.get("issue_count", 0),
        "pwml_qa_ok": qa_report.get("ok"),
    }
    print(json.dumps(summary, indent=2))

    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
