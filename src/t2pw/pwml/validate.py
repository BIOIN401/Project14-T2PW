from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from lxml import etree


PV_PATH = "./pathway-visualization-contexts/pathway-visualization-context/pathway-visualization"


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _singularize(tag: str) -> str:
    overrides = {
        "species": "species",
        "cell-types": "cell-type",
        "subcellular-locations": "subcellular-location",
        "biological-states": "biological-state",
        "element-collections": "element-collection",
        "nucleic-acids": "nucleic-acid",
        "protein-complexes": "protein-complex",
        "reaction-coupled-transports": "reaction-coupled-transport",
        "bound-visualizations": "bound-visualization",
        "compound-locations": "compound-location",
        "element-collection-locations": "element-collection-location",
        "nucleic-acid-locations": "nucleic-acid-location",
        "protein-locations": "protein-location",
        "protein-complex-visualizations": "protein-complex-visualization",
        "reaction-visualizations": "reaction-visualization",
        "reaction-coupled-transport_visualizations": "reaction-coupled-transport-visualization",
        "transport-visualizations": "transport-visualization",
        "interaction-visualizations": "interaction-visualization",
        "sub-pathway-visualizations": "sub-pathway-visualization",
        "vacuous-compound-visualizations": "vacuous-compound-visualization",
        "vacuous-edge-visualizations": "vacuous-edge-visualization",
        "vacuous-nucleic-acid-visualizations": "vacuous-nucleic-acid-visualization",
        "vacuous-element-collection-visualizations": "vacuous-element-collection-visualization",
        "vacuous-protein-visualizations": "vacuous-protein-visualization",
        "drawable-element-locations": "drawable-element-location",
        "membrane-visualizations": "membrane-visualization",
        "label-locations": "label-location",
        "zoom-visualizations": "zoom-visualization",
        "reaction-left-elements": "reaction-left-element",
        "reaction-right-elements": "reaction-right-element",
        "reaction-enzymes": "reaction-enzyme",
        "transport-elements": "transport-element",
        "transport-transporters": "transport-transporter",
        "reaction_compound_visualizations": "reaction-compound-visualization",
        "reaction_element_collection_visualizations": "reaction-element-collection-visualization",
        "reaction_enzyme_visualizations": "reaction-enzyme-visualization",
        "transport_compound_visualizations": "transport-compound-visualization",
        "transport_transporter_visualizations": "transport-transporter-visualization",
        "protein_complex_protein_visualizations": "protein-complex-protein-visualization",
        "protein_complex_compound_visualizations": "protein-complex-compound-visualization",
        "sub_pathway_element_collection_visualizations": "sub-pathway-element-collection-visualization",
        "element-states": "element-state",
        "ec-numbers": "ec-number",
        "synonyms": "synonym",
        "sub-pathways": "sub-pathway",
        "references": "reference",
    }
    if tag in overrides:
        return overrides[tag]
    if tag.endswith("ies") and len(tag) > 3:
        return f"{tag[:-3]}y"
    if tag.endswith("s") and len(tag) > 1:
        return tag[:-1]
    return tag


def _is_boolean_field(tag: str) -> bool:
    return tag in {
        "hidden",
        "spontaneous",
        "currency",
        "complete-membrane",
        "option:end_arrow",
        "option:end_flat_arrow",
        "option:start_arrow",
        "option:start_flat_arrow",
    }


@dataclass
class SectionSignature:
    container_tag: str
    item_tag: str
    required_fields: List[str] = field(default_factory=list)
    integer_fields: List[str] = field(default_factory=list)
    boolean_fields: List[str] = field(default_factory=list)
    nil_fields: List[str] = field(default_factory=list)


@dataclass
class StructureSignature:
    root_tag: str
    root_children: List[str]
    pv_path: str
    pv_children: List[str]
    pathway_children: List[str]
    sections: Dict[str, SectionSignature]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_tag": self.root_tag,
            "root_children": list(self.root_children),
            "pv_path": self.pv_path,
            "pv_children": list(self.pv_children),
            "pathway_children": list(self.pathway_children),
            "sections": {k: asdict(v) for k, v in self.sections.items()},
        }


def parse_xml(path: Path | str) -> etree._ElementTree:
    parser = etree.XMLParser(remove_blank_text=True, huge_tree=True, recover=True)
    return etree.parse(str(path), parser)


def discover_structure_signature(reference_path: Path | str, sample_items: int = 3) -> StructureSignature:
    tree = parse_xml(reference_path)
    root = tree.getroot()

    root_children = [child.tag for child in root]
    pv = root.find(PV_PATH)
    if pv is None:
        raise ValueError(f"Missing pathway visualization node at {PV_PATH}")

    pv_children = [child.tag for child in pv]
    pathway_node = pv.find("pathway")
    pathway_children = _dedupe_preserve_order(
        [child.tag for child in pathway_node] if pathway_node is not None else []
    )

    sections: Dict[str, SectionSignature] = {}
    for section in pv:
        if section.tag == "pathway":
            continue
        children = list(section)
        item_tag = _singularize(section.tag)
        if children:
            counts: Dict[str, int] = {}
            for child in children:
                counts[child.tag] = counts.get(child.tag, 0) + 1
            item_tag = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        sample_nodes = [child for child in children if child.tag == item_tag][:sample_items]
        if not sample_nodes:
            sample_nodes = children[:sample_items]

        required_fields: List[str] = []
        seen_fields = set()
        integer_fields = set()
        boolean_fields = set()
        nil_fields = set()

        for sample in sample_nodes:
            for field_node in sample:
                tag = field_node.tag
                if ":" in tag:
                    # Reference files may contain non-namespace-safe tags like option:end_arrow.
                    # Skip these so generation remains valid XML.
                    continue
                if tag not in seen_fields:
                    seen_fields.add(tag)
                    required_fields.append(tag)
                type_attr = field_node.get("type")
                if type_attr == "integer":
                    integer_fields.add(tag)
                if type_attr == "boolean":
                    boolean_fields.add(tag)
                if field_node.get("nil") == "true":
                    nil_fields.add(tag)

        sections[section.tag] = SectionSignature(
            container_tag=section.tag,
            item_tag=item_tag,
            required_fields=required_fields,
            integer_fields=sorted(integer_fields),
            boolean_fields=sorted(boolean_fields),
            nil_fields=sorted(nil_fields),
        )

    return StructureSignature(
        root_tag=root.tag,
        root_children=root_children,
        pv_path=PV_PATH,
        pv_children=pv_children,
        pathway_children=pathway_children,
        sections=sections,
    )


@dataclass
class ValidationIssue:
    kind: str
    path: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "path": self.path, "message": self.message}


def _compare_order(expected: List[str], actual: List[str], path: str, issues: List[ValidationIssue]) -> None:
    if expected == actual:
        return
    issues.append(
        ValidationIssue(
            kind="order",
            path=path,
            message=f"Expected order {expected}, got {actual}",
        )
    )


def _ensure_child(parent: etree._Element, tag: str) -> etree._Element:
    existing = parent.find(tag)
    if existing is not None:
        return existing
    return etree.SubElement(parent, tag)


def _default_node_value(node: etree._Element, tag: str) -> None:
    if tag == "named-for-id":
        node.set("type", "integer")
        if node.text is None:
            node.text = "0"
    elif tag == "named-for-type":
        node.text = node.text or "Pathway"
    elif tag == "cached-name":
        node.text = node.text or "Generated Pathway"
    elif tag == "cached-description":
        node.text = node.text or ""
    elif tag == "cached-subject":
        node.text = node.text or "Metabolic"
    elif tag == "pw-id":
        node.text = node.text or "PW000000"


def _reorder_children(parent: etree._Element, expected_order: List[str]) -> None:
    grouped: Dict[str, List[etree._Element]] = {}
    for child in list(parent):
        grouped.setdefault(child.tag, []).append(child)

    ordered: List[etree._Element] = []
    for tag in expected_order:
        if tag in grouped and grouped[tag]:
            ordered.append(grouped[tag].pop(0))

    leftovers: List[etree._Element] = []
    for nodes in grouped.values():
        leftovers.extend(nodes)

    parent[:] = ordered + leftovers


def repair_tree(tree: etree._ElementTree, signature: StructureSignature) -> etree._ElementTree:
    root = tree.getroot()

    for tag in signature.root_children:
        node = _ensure_child(root, tag)
        _default_node_value(node, tag)
    _reorder_children(root, signature.root_children)

    contexts = _ensure_child(root, "pathway-visualization-contexts")
    context = _ensure_child(contexts, "pathway-visualization-context")
    position = _ensure_child(context, "position")
    if not (position.text or "").strip():
        position.text = "Center"
    context_id = _ensure_child(context, "id")
    if not (context_id.text or "").strip():
        context_id.text = "PathwayVisualizationContext1"
    pv = _ensure_child(context, "pathway-visualization")

    for tag in signature.pv_children:
        child = _ensure_child(pv, tag)
        if tag in {"height", "width"}:
            child.set("type", "integer")
            if not (child.text or "").strip():
                child.text = "1200"
        elif tag == "background-color" and not (child.text or "").strip():
            child.text = "#FFFFFF"
        elif tag == "id" and not (child.text or "").strip():
            child.text = "PathwayVisualization1"
    _reorder_children(pv, signature.pv_children)

    pathway = _ensure_child(pv, "pathway")
    for tag in signature.pathway_children:
        _ensure_child(pathway, tag)
    _reorder_children(pathway, signature.pathway_children)
    p_id = pathway.find("id")
    if p_id is not None:
        p_id.set("type", "integer")
        if not (p_id.text or "").strip():
            p_id.text = "1"
    p_species_id = pathway.find("species-id")
    if p_species_id is not None and not (p_species_id.text or "").strip():
        p_species_id.set("nil", "true")
        p_species_id.set("type", "integer")

    for section_tag, section_sig in signature.sections.items():
        section = pv.find(section_tag)
        if section is None:
            section = etree.SubElement(pv, section_tag)

        items = [node for node in section if node.tag == section_sig.item_tag]
        if not items:
            continue
        for item in items:
            for field_tag in section_sig.required_fields:
                child = item.find(field_tag)
                if child is None:
                    child = etree.SubElement(item, field_tag)
                    if field_tag in section_sig.nil_fields:
                        child.set("nil", "true")
                        if field_tag in section_sig.integer_fields:
                            child.set("type", "integer")
                        if field_tag in section_sig.boolean_fields:
                            child.set("type", "boolean")
            _reorder_children(item, section_sig.required_fields)

            for int_tag in section_sig.integer_fields:
                child = item.find(int_tag)
                if child is not None:
                    child.set("type", "integer")
            for bool_tag in section_sig.boolean_fields:
                child = item.find(bool_tag)
                if child is not None and child.get("type") is None:
                    child.set("type", "boolean")
            for nil_tag in section_sig.nil_fields:
                child = item.find(nil_tag)
                if child is not None and not (child.text or "").strip():
                    child.set("nil", "true")

    _reorder_children(pv, signature.pv_children)
    return tree


def validate_generated_tree(
    generated_tree: etree._ElementTree, signature: StructureSignature
) -> Dict[str, Any]:
    issues: List[ValidationIssue] = []
    root = generated_tree.getroot()

    if root.tag != signature.root_tag:
        issues.append(
            ValidationIssue(
                kind="root",
                path="/",
                message=f"Expected root <{signature.root_tag}> but got <{root.tag}>",
            )
        )

    actual_root_children = [child.tag for child in root]
    _compare_order(signature.root_children, actual_root_children, "/", issues)

    pv = root.find(signature.pv_path)
    if pv is None:
        issues.append(
            ValidationIssue(
                kind="missing-node",
                path=signature.pv_path,
                message="Missing pathway visualization subtree",
            )
        )
    else:
        actual_pv_children = [child.tag for child in pv]
        _compare_order(signature.pv_children, actual_pv_children, signature.pv_path, issues)

        pathway = pv.find("pathway")
        if pathway is None:
            issues.append(
                ValidationIssue(
                    kind="missing-node",
                    path=f"{signature.pv_path}/pathway",
                    message="Missing <pathway> node",
                )
            )
        else:
            actual_pathway_order = _dedupe_preserve_order([child.tag for child in pathway])
            _compare_order(
                signature.pathway_children,
                actual_pathway_order,
                f"{signature.pv_path}/pathway",
                issues,
            )

        for section_tag, section_sig in signature.sections.items():
            section = pv.find(section_tag)
            if section is None:
                issues.append(
                    ValidationIssue(
                        kind="missing-node",
                        path=f"{signature.pv_path}/{section_tag}",
                        message=f"Missing section <{section_tag}>",
                    )
                )
                continue

            items = [node for node in section if node.tag == section_sig.item_tag]
            for idx, item in enumerate(items):
                base_path = f"{signature.pv_path}/{section_tag}/{section_sig.item_tag}[{idx + 1}]"
                for field_tag in section_sig.required_fields:
                    child = item.find(field_tag)
                    if child is None:
                        issues.append(
                            ValidationIssue(
                                kind="missing-field",
                                path=base_path,
                                message=f"Missing required field <{field_tag}>",
                            )
                        )
                        continue
                    if field_tag in section_sig.integer_fields and child.get("type") != "integer":
                        issues.append(
                            ValidationIssue(
                                kind="field-attribute",
                                path=f"{base_path}/{field_tag}",
                                message='Expected attribute type="integer"',
                            )
                        )

    return {
        "ok": not issues,
        "issue_count": len(issues),
        "issues": [issue.to_dict() for issue in issues],
        "signature": signature.to_dict(),
    }


def validate_pwml(
    reference_path: Path | str, generated_path: Path | str, signature: Optional[StructureSignature] = None
) -> Dict[str, Any]:
    sig = signature or discover_structure_signature(reference_path)
    generated_tree = parse_xml(generated_path)
    return validate_generated_tree(generated_tree, sig)


def write_json_report(report: Dict[str, Any], out_path: Path | str) -> None:
    path = Path(out_path)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated PWML against a reference PWML structure.")
    parser.add_argument("--ref", required=True, help="Reference PWML path")
    parser.add_argument("--in", dest="in_path", required=True, help="Generated PWML path")
    parser.add_argument("--report", default="pwml_validation_report.json", help="Output report JSON path")
    parser.add_argument("--repair-out", default="", help="Optional output path for repaired PWML")
    args = parser.parse_args()

    signature = discover_structure_signature(args.ref)
    generated_tree = parse_xml(args.in_path)

    report = validate_generated_tree(generated_tree, signature)
    if not report["ok"] and args.repair_out:
        repaired = repair_tree(generated_tree, signature)
        repaired.write(args.repair_out, encoding="utf-8", xml_declaration=True, pretty_print=True)
        repaired_report = validate_generated_tree(repaired, signature)
        report["repair_out"] = args.repair_out
        report["repair_ok"] = repaired_report["ok"]
        report["repair_issue_count"] = repaired_report["issue_count"]
        report["repair_issues"] = repaired_report["issues"]

    write_json_report(report, args.report)
    status = "PASS" if report["ok"] else "FAIL"
    print(f"Validation: {status} ({report['issue_count']} issues)")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
