from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

from llm_client import PROVIDER, chat


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _deterministic_semantic_scan(
    mapped_payload: Dict[str, Any],
    sbml_text: str,
    validation_report: Dict[str, Any],
) -> Dict[str, Any]:
    issues: Dict[str, List[Dict[str, Any]]] = {"errors": [], "warnings": [], "suggestions": []}
    metrics: Dict[str, Any] = {}

    root = ElementTree.fromstring(sbml_text)
    species = root.findall(".//{*}listOfSpecies/{*}species")
    reactions = root.findall(".//{*}listOfReactions/{*}reaction")
    metrics["species_count"] = len(species)
    metrics["reaction_count"] = len(reactions)

    unmapped_species = [sp.get("id", "") for sp in species if "unmapped_" in (sp.get("id") or "")]
    metrics["unmapped_species_count"] = len(unmapped_species)
    if species and len(unmapped_species) / len(species) > 0.45:
        issues["warnings"].append(
            {
                "path": "/sbml/listOfSpecies",
                "reason": f"High unmapped species fraction ({len(unmapped_species)}/{len(species)}).",
                "evidence": ", ".join(unmapped_species[:8]),
            }
        )

    for reaction in reactions:
        rid = reaction.get("id", "")
        reactants = sorted(
            [
                ref.get("species", "")
                for ref in reaction.findall("./{*}listOfReactants/{*}speciesReference")
                if (ref.get("species") or "").strip()
            ]
        )
        products = sorted(
            [
                ref.get("species", "")
                for ref in reaction.findall("./{*}listOfProducts/{*}speciesReference")
                if (ref.get("species") or "").strip()
            ]
        )
        if reactants == products and reactants:
            issues["warnings"].append(
                {
                    "path": f"/sbml/listOfReactions/{rid}",
                    "reason": "Reaction has identical reactants and products.",
                    "evidence": f"reactants=products={reactants}",
                }
            )

    validation_messages = _safe_list(_safe_dict(validation_report.get("validation")).get("messages"))
    severe = [m for m in validation_messages if int(_safe_dict(m).get("severity", 0)) >= 2]
    if severe:
        issues["errors"].append(
            {
                "path": "/validation/messages",
                "reason": f"Validation contains {len(severe)} severe messages (severity >= 2).",
                "evidence": _safe_dict(severe[0]).get("message", ""),
            }
        )

    entities = _safe_dict(mapped_payload.get("entities"))
    compounds = {
        str(_safe_dict(item).get("name", "")).strip()
        for item in _safe_list(entities.get("compounds"))
        if str(_safe_dict(item).get("name", "")).strip()
    }
    proteins = {
        str(_safe_dict(item).get("name", "")).strip()
        for item in _safe_list(entities.get("proteins"))
        if str(_safe_dict(item).get("name", "")).strip()
    }
    overlap = sorted({c for c in compounds if c and c in proteins})
    if overlap:
        issues["warnings"].append(
            {
                "path": "/entities",
                "reason": "Some names appear in both compounds and proteins.",
                "evidence": ", ".join(overlap[:8]),
            }
        )

    return {"issues": issues, "metrics": metrics}


def _build_llm_prompt(
    mapped_payload: Dict[str, Any],
    sbml_text: str,
    validation_report: Dict[str, Any],
    deterministic_issues: Dict[str, Any],
) -> str:
    mapped_preview = json.dumps(mapped_payload, ensure_ascii=False)[:12000]
    validation_preview = json.dumps(validation_report, ensure_ascii=False)[:10000]
    deterministic_preview = json.dumps(deterministic_issues, ensure_ascii=False)
    sbml_preview = sbml_text[:18000]
    return "\n".join(
        [
            "Review SBML output semantically (do not rewrite files).",
            "Focus on biological/process coherence, not syntax-only checks.",
            "Return JSON only in shape:",
            '{"errors":[{"path":"","reason":"","confidence":0.0,"evidence":""}],'
            '"warnings":[{"path":"","reason":"","confidence":0.0,"evidence":""}],'
            '"suggestions":[{"path":"","reason":"","confidence":0.0,"evidence":""}]}',
            "",
            "Mapped JSON preview:",
            mapped_preview,
            "",
            "Validation preview:",
            validation_preview,
            "",
            "Deterministic issue preview:",
            deterministic_preview,
            "",
            "SBML preview:",
            sbml_preview,
        ]
    )


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.replace("```json", "```").replace("```", "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right > left:
        try:
            parsed = json.loads(text[left : right + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _has_pointer_path(item: Dict[str, Any]) -> bool:
    path = item.get("path")
    return isinstance(path, str) and path.strip().startswith("/")


def run_sbml_overwatch(
    mapped_json_path: Path,
    sbml_path: Path,
    validation_report_path: Path,
    output_path: Path,
    *,
    use_llm: bool = True,
    llm_max_tokens: int = 2600,
) -> Dict[str, Any]:
    mapped_payload = json.loads(mapped_json_path.read_text(encoding="utf-8"))
    if not isinstance(mapped_payload, dict):
        raise ValueError("Mapped JSON must be an object.")
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if not isinstance(validation_report, dict):
        validation_report = {}
    sbml_text = sbml_path.read_text(encoding="utf-8")

    deterministic = _deterministic_semantic_scan(mapped_payload, sbml_text, validation_report)

    llm_result: Dict[str, Any] = {"enabled": use_llm, "provider": PROVIDER, "ok": False, "error": "", "issues": {}}
    if use_llm:
        try:
            raw = chat(
                [
                    {
                        "role": "system",
                        "content": "You are an SBML semantic reviewer. Output JSON only and do not fabricate evidence.",
                    },
                    {
                        "role": "user",
                        "content": _build_llm_prompt(
                            mapped_payload,
                            sbml_text,
                            validation_report,
                            deterministic,
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=llm_max_tokens,
                response_json=True,
            )
            parsed = _parse_llm_json(raw)
            if parsed is None:
                llm_result["error"] = "LLM output was not valid JSON."
                llm_result["raw_preview"] = raw[:800]
            else:
                llm_result["ok"] = True
                llm_result["issues"] = parsed
        except Exception as exc:  # noqa: BLE001
            llm_result["error"] = str(exc)

    merged = {
        "errors": list(_safe_list(_safe_dict(deterministic.get("issues")).get("errors"))),
        "warnings": list(_safe_list(_safe_dict(deterministic.get("issues")).get("warnings"))),
        "suggestions": list(_safe_list(_safe_dict(deterministic.get("issues")).get("suggestions"))),
    }
    if llm_result.get("ok"):
        llm_issues = _safe_dict(llm_result.get("issues"))
        for item in _safe_list(llm_issues.get("errors")):
            if not isinstance(item, dict):
                continue
            enriched = dict(item)
            enriched.setdefault("source", "llm")
            if _has_pointer_path(enriched):
                merged["errors"].append(enriched)
            else:
                enriched["source"] = "llm_downgraded"
                enriched["reason"] = f"(Downgraded from error) {enriched.get('reason', '')}".strip()
                merged["warnings"].append(enriched)

        for key in ["warnings", "suggestions"]:
            for item in _safe_list(llm_issues.get(key)):
                if isinstance(item, dict):
                    enriched = dict(item)
                    enriched.setdefault("source", "llm")
                    merged[key].append(enriched)

    report = {
        "summary": {
            "errors": len(merged["errors"]),
            "warnings": len(merged["warnings"]),
            "suggestions": len(merged["suggestions"]),
        },
        "deterministic": deterministic,
        "llm": llm_result,
        "issues": merged,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic overwatch on generated SBML.")
    parser.add_argument("--mapped", required=True, help="Mapped JSON path")
    parser.add_argument("--sbml", required=True, help="SBML path")
    parser.add_argument("--validation", required=True, help="SBML validation report JSON path")
    parser.add_argument("--out", default="sbml_overwatch_report.json", help="Output report JSON path")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM semantic review")
    parser.add_argument("--max-tokens", type=int, default=2600, help="LLM max tokens")
    args = parser.parse_args()

    report = run_sbml_overwatch(
        Path(args.mapped),
        Path(args.sbml),
        Path(args.validation),
        Path(args.out),
        use_llm=not args.no_llm,
        llm_max_tokens=int(args.max_tokens),
    )
    print(f"Wrote SBML overwatch report: {args.out}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
