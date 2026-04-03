from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from lxml import etree


_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "in", "to", "is", "are", "was",
    "were", "be", "been", "being", "by", "for", "with", "as", "at", "from",
    "on", "into", "that", "this", "it", "its", "also", "not", "no", "can",
    "may", "has", "have", "had", "do", "does", "did", "via", "per",
}


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [t for t in re.findall(r"[a-z0-9]+", text.casefold()) if t not in _STOP_WORDS]


def _norm(text: str) -> str:
    return " ".join(_tokenize(text))


def _dedupe_names(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        raw = value.strip()
        key = _norm(raw)
        if not raw or not key or key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def _is_protein_like(species_id: str, species_name: str) -> bool:
    sid = (species_id or "").strip().casefold()
    sname = (species_name or "").strip().casefold()
    if sid.startswith("p_"):
        return True
    protein_hints = [
        "enzyme",
        "peroxidase",
        "deiodinase",
        "symporter",
        "transporter",
        "atpase",
        "kinase",
        "phosphatase",
        "protein",
    ]
    return any(hint in sname for hint in protein_hints)


def _get_sbml_ns(root: etree._Element) -> Dict[str, str]:
    uri = root.nsmap.get(None) or "http://www.sbml.org/sbml/level3/version1/core"
    return {"sbml": uri}


def parse_sbml(path: Path) -> Dict[str, Any]:
    tree = etree.parse(str(path))
    root = tree.getroot()
    ns = _get_sbml_ns(root)

    model_node = root.xpath("./sbml:model", namespaces=ns)
    if not model_node:
        raise ValueError("Missing SBML model element.")
    model = model_node[0]

    model_id = (model.get("id") or "").strip()
    model_name = (model.get("name") or "").strip() or model_id

    compartments: List[Dict[str, str]] = []
    compartment_name_by_id: Dict[str, str] = {}
    for c in model.xpath("./sbml:listOfCompartments/sbml:compartment", namespaces=ns):
        cid = (c.get("id") or "").strip()
        cname = (c.get("name") or "").strip() or cid
        if not cid:
            continue
        compartments.append({"id": cid, "name": cname})
        compartment_name_by_id[cid] = cname

    species_rows: List[Dict[str, str]] = []
    species_name_by_id: Dict[str, str] = {}
    species_compartment_by_id: Dict[str, str] = {}
    for s in model.xpath("./sbml:listOfSpecies/sbml:species", namespaces=ns):
        sid = (s.get("id") or "").strip()
        sname = (s.get("name") or "").strip() or sid
        comp = (s.get("compartment") or "").strip()
        if not sid:
            continue
        species_rows.append({"id": sid, "name": sname, "compartment_id": comp})
        species_name_by_id[sid] = sname
        species_compartment_by_id[sid] = comp

    reactions: List[Dict[str, Any]] = []
    for r in model.xpath("./sbml:listOfReactions/sbml:reaction", namespaces=ns):
        rid = (r.get("id") or "").strip()
        rname = (r.get("name") or "").strip() or rid
        rcomp = (r.get("compartment") or "").strip()

        reactant_ids = [
            (n.get("species") or "").strip()
            for n in r.xpath("./sbml:listOfReactants/sbml:speciesReference", namespaces=ns)
            if (n.get("species") or "").strip()
        ]
        product_ids = [
            (n.get("species") or "").strip()
            for n in r.xpath("./sbml:listOfProducts/sbml:speciesReference", namespaces=ns)
            if (n.get("species") or "").strip()
        ]
        modifier_ids = [
            (n.get("species") or "").strip()
            for n in r.xpath("./sbml:listOfModifiers/sbml:modifierSpeciesReference", namespaces=ns)
            if (n.get("species") or "").strip()
        ]

        reactions.append(
            {
                "id": rid,
                "name": rname,
                "compartment_id": rcomp,
                "reactant_ids": reactant_ids,
                "product_ids": product_ids,
                "modifier_ids": modifier_ids,
                "reactants": [species_name_by_id.get(sid, sid) for sid in reactant_ids],
                "products": [species_name_by_id.get(sid, sid) for sid in product_ids],
                "modifiers": [species_name_by_id.get(sid, sid) for sid in modifier_ids],
            }
        )

    return {
        "source_path": str(path),
        "model": {"id": model_id, "name": model_name},
        "compartments": compartments,
        "species": species_rows,
        "reactions": reactions,
        "maps": {
            "species_name_by_id": species_name_by_id,
            "species_compartment_by_id": species_compartment_by_id,
            "compartment_name_by_id": compartment_name_by_id,
        },
    }


def sbml_to_silver_payload(sbml_doc: Dict[str, Any]) -> Dict[str, Any]:
    compartments = sbml_doc.get("compartments", [])
    species = sbml_doc.get("species", [])
    reactions = sbml_doc.get("reactions", [])
    maps = sbml_doc.get("maps", {})
    species_name_by_id = maps.get("species_name_by_id", {})
    species_compartment_by_id = maps.get("species_compartment_by_id", {})
    compartment_name_by_id = maps.get("compartment_name_by_id", {})

    compound_names: List[str] = []
    protein_names: List[str] = []
    subcellular_locations = _dedupe_names([c.get("name", "") for c in compartments if isinstance(c, dict)])

    for row in species:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id", ""))
        sname = str(row.get("name", ""))
        if _is_protein_like(sid, sname):
            protein_names.append(sname)
        else:
            compound_names.append(sname)

    compound_names = _dedupe_names(compound_names)
    protein_names = _dedupe_names(protein_names)
    protein_name_set = {_norm(x) for x in protein_names}

    biological_states: List[Dict[str, Any]] = []
    for c in compartments:
        if not isinstance(c, dict):
            continue
        cname = str(c.get("name", "")).strip()
        if not cname:
            continue
        biological_states.append({"name": cname, "subcellular_location": cname})

    state_by_compartment: Dict[str, str] = {}
    for c in compartments:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id", "")).strip()
        cname = str(c.get("name", "")).strip() or cid
        if cid and cname:
            state_by_compartment[cid] = cname

    compound_locations: List[Dict[str, str]] = []
    protein_locations: List[Dict[str, str]] = []
    for row in species:
        if not isinstance(row, dict):
            continue
        sname = str(row.get("name", "")).strip()
        sid = str(row.get("id", "")).strip()
        comp_id = str(row.get("compartment_id", "")).strip()
        state_name = state_by_compartment.get(comp_id, compartment_name_by_id.get(comp_id, comp_id))
        if not sname or not state_name:
            continue
        if _is_protein_like(sid, sname):
            protein_locations.append({"protein": sname, "biological_state": state_name})
        else:
            compound_locations.append({"compound": sname, "biological_state": state_name})

    silver_reactions: List[Dict[str, Any]] = []
    silver_transports: List[Dict[str, Any]] = []
    for r in reactions:
        if not isinstance(r, dict):
            continue
        reactant_ids = [x for x in r.get("reactant_ids", []) if isinstance(x, str) and x.strip()]
        product_ids = [x for x in r.get("product_ids", []) if isinstance(x, str) and x.strip()]
        reactants = _dedupe_names(r.get("reactants", []))
        products = _dedupe_names(r.get("products", []))
        modifiers = _dedupe_names(r.get("modifiers", []))
        if not reactants and not products:
            continue

        is_transport = False
        if len(reactant_ids) == 1 and len(product_ids) == 1:
            r_id = reactant_ids[0]
            p_id = product_ids[0]
            r_name = species_name_by_id.get(r_id, r_id)
            p_name = species_name_by_id.get(p_id, p_id)
            if _norm(r_name) == _norm(p_name):
                src_state = state_by_compartment.get(species_compartment_by_id.get(r_id, ""), "")
                dst_state = state_by_compartment.get(species_compartment_by_id.get(p_id, ""), "")
                if src_state and dst_state and src_state != dst_state:
                    is_transport = True
                    transporters = [{"protein": name} for name in modifiers if _norm(name) in protein_name_set]
                    silver_transports.append(
                        {
                            "name": str(r.get("name", "")).strip(),
                            "cargo": r_name,
                            "from_biological_state": src_state,
                            "to_biological_state": dst_state,
                            "transporters": transporters,
                        }
                    )

        if is_transport:
            continue

        enzymes = [{"protein": name} for name in modifiers if _norm(name) in protein_name_set]
        silver_reactions.append(
            {
                "name": str(r.get("name", "")).strip(),
                "inputs": reactants,
                "outputs": products,
                "enzymes": enzymes,
            }
        )

    return {
        "metadata": {
            "source_sbml_path": sbml_doc.get("source_path", ""),
            "source_model_name": sbml_doc.get("model", {}).get("name", ""),
            "conversion": "silver_from_sbml",
        },
        "entities": {
            "compounds": [{"name": name} for name in compound_names],
            "proteins": [{"name": name} for name in protein_names],
            "protein_complexes": [],
            "element_collections": [],
            "nucleic_acids": [],
            "species": [],
            "subcellular_locations": [{"name": name} for name in subcellular_locations],
        },
        "biological_states": biological_states,
        "element_locations": {
            "compound_locations": compound_locations,
            "protein_locations": protein_locations,
            "nucleic_acid_locations": [],
            "element_collection_locations": [],
        },
        "processes": {
            "reactions": silver_reactions,
            "transports": silver_transports,
            "reaction_coupled_transports": [],
            "interactions": [],
        },
    }


def _reaction_pattern(reaction: Dict[str, Any]) -> str:
    inputs = [x for x in reaction.get("reactants", []) if isinstance(x, str) and x.strip()]
    outputs = [x for x in reaction.get("products", []) if isinstance(x, str) and x.strip()]
    left = " + ".join(inputs[:3]) if inputs else "?"
    right = " + ".join(outputs[:3]) if outputs else "?"
    return f"{left} -> {right}"


def _entry_tokens(model_name: str, compartments: Sequence[str], species: Sequence[str], reactions: Sequence[str]) -> Set[str]:
    bag = " ".join([model_name, " ".join(compartments), " ".join(species), " ".join(reactions)])
    return set(_tokenize(bag))


def build_motif_entry(sbml_doc: Dict[str, Any]) -> Dict[str, Any]:
    source_path = str(sbml_doc.get("source_path", ""))
    model_name = str(sbml_doc.get("model", {}).get("name", "")).strip()
    compartments = _dedupe_names([c.get("name", "") for c in sbml_doc.get("compartments", []) if isinstance(c, dict)])
    species = _dedupe_names([s.get("name", "") for s in sbml_doc.get("species", []) if isinstance(s, dict)])
    reactions = [_reaction_pattern(r) for r in sbml_doc.get("reactions", []) if isinstance(r, dict)]
    tokens = _entry_tokens(model_name, compartments, species, reactions)

    species_tokens = set(_tokenize(" ".join(species)))
    reaction_tokens = set(_tokenize(" ".join(reactions)))
    compartment_tokens = set(_tokenize(" ".join(compartments)))

    return {
        "source_path": source_path,
        "source_name": Path(source_path).name if source_path else "",
        "model_name": model_name,
        "compartments": compartments[:24],
        "species": species[:40],
        "reaction_patterns": reactions[:30],
        "tokens": sorted(tokens)[:800],
        "species_tokens": sorted(species_tokens)[:800],
        "reaction_tokens": sorted(reaction_tokens)[:800],
        "compartment_tokens": sorted(compartment_tokens)[:300],
    }


def build_motif_index(sbml_dir: Path, index_path: Path, *, max_files: int = 1000) -> Dict[str, Any]:
    sbml_paths = sorted(
        [p for p in sbml_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".sbml", ".xml"}]
    )[: max(1, int(max_files))]

    entries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for path in sbml_paths:
        try:
            doc = parse_sbml(path)
            entries.append(build_motif_entry(doc))
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_path": str(path), "error": str(exc)})

    index = {
        "version": 1,
        "created_at_unix": int(time.time()),
        "source_dir": str(sbml_dir),
        "entry_count": len(entries),
        "failure_count": len(failures),
        "entries": entries,
        "failures": failures[:100],
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def load_motif_index(index_path: Path) -> Dict[str, Any]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Motif index must be a JSON object.")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Motif index missing 'entries' list.")
    return data


def _overlap(query: Set[str], candidate: Set[str]) -> float:
    if not query:
        return 0.0
    return len(query & candidate) / max(1, len(query))


def _score_entry(query_tokens: Set[str], entry: Dict[str, Any]) -> float:
    entry_tokens = set(x for x in entry.get("tokens", []) if isinstance(x, str))
    species_tokens = set(x for x in entry.get("species_tokens", []) if isinstance(x, str))
    reaction_tokens = set(x for x in entry.get("reaction_tokens", []) if isinstance(x, str))
    compartment_tokens = set(x for x in entry.get("compartment_tokens", []) if isinstance(x, str))
    return (
        2.6 * _overlap(query_tokens, reaction_tokens)
        + 1.8 * _overlap(query_tokens, species_tokens)
        + 1.0 * _overlap(query_tokens, compartment_tokens)
        + 0.6 * _overlap(query_tokens, entry_tokens)
    )


def retrieve_motif_examples(query_text: str, index_data: Dict[str, Any], *, top_k: int = 3) -> List[Dict[str, Any]]:
    entries = index_data.get("entries", [])
    if not isinstance(entries, list):
        return []
    query_tokens = set(_tokenize(query_text))
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        score = _score_entry(query_tokens, entry)
        if score <= 0:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, entry in scored[: max(1, int(top_k))]:
        row = dict(entry)
        row["score"] = round(float(score), 6)
        out.append(row)
    return out


def build_retrieval_context(
    query_text: str,
    index_data: Dict[str, Any],
    *,
    top_k: int = 3,
    max_chars: int = 3600,
) -> Tuple[str, Dict[str, Any]]:
    selected = retrieve_motif_examples(query_text, index_data, top_k=top_k)
    if not selected:
        return "", {"selected_count": 0, "top_k": int(top_k)}

    lines: List[str] = []
    for idx, entry in enumerate(selected, start=1):
        lines.append(f"[Example {idx}] score={entry.get('score', 0)}")
        lines.append(f"Source: {entry.get('source_name', '')}")
        lines.append(f"Model: {entry.get('model_name', '')}")
        compartments = ", ".join([x for x in entry.get("compartments", [])[:8] if isinstance(x, str) and x.strip()])
        species = ", ".join([x for x in entry.get("species", [])[:10] if isinstance(x, str) and x.strip()])
        lines.append(f"Compartments: {compartments if compartments else 'n/a'}")
        lines.append(f"Species: {species if species else 'n/a'}")
        lines.append("Reactions:")
        for pattern in [x for x in entry.get("reaction_patterns", [])[:6] if isinstance(x, str) and x.strip()]:
            lines.append(f"- {pattern}")
        lines.append("")
        if len("\n".join(lines)) > max_chars:
            break

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text, {"selected_count": len(selected), "top_k": int(top_k)}


def payload_to_query_text(payload: Dict[str, Any], *, extra: str = "") -> str:
    pieces: List[str] = []
    entities = payload.get("entities", {})
    if isinstance(entities, dict):
        for section in [
            "compounds",
            "proteins",
            "protein_complexes",
            "element_collections",
            "nucleic_acids",
            "subcellular_locations",
        ]:
            for row in entities.get(section, []):
                if not isinstance(row, dict):
                    continue
                name = row.get("name")
                if isinstance(name, str) and name.strip():
                    pieces.append(name.strip())

    processes = payload.get("processes", {})
    if isinstance(processes, dict):
        for reaction in processes.get("reactions", []):
            if not isinstance(reaction, dict):
                continue
            for key in ["name"]:
                value = reaction.get(key)
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())
            for key in ["inputs", "outputs"]:
                vals = reaction.get(key, [])
                if isinstance(vals, list):
                    pieces.extend([str(x).strip() for x in vals if isinstance(x, str) and str(x).strip()])
        for transport in processes.get("transports", []):
            if not isinstance(transport, dict):
                continue
            for key in ["name", "cargo", "from_biological_state", "to_biological_state"]:
                value = transport.get(key)
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())

    if extra.strip():
        pieces.append(extra.strip())
    return "\n".join(_dedupe_names(pieces))


def _cmd_index(args: argparse.Namespace) -> None:
    index = build_motif_index(Path(args.sbml_dir), Path(args.out), max_files=int(args.max_files))
    print(json.dumps({"entry_count": index.get("entry_count"), "failure_count": index.get("failure_count")}, indent=2))


def _cmd_silver(args: argparse.Namespace) -> None:
    doc = parse_sbml(Path(args.input))
    payload = sbml_to_silver_payload(doc)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote silver JSON: {out_path}")


def _cmd_retrieve(args: argparse.Namespace) -> None:
    index = load_motif_index(Path(args.index))
    if args.query_file:
        query_text = Path(args.query_file).read_text(encoding="utf-8")
    else:
        query_text = str(args.query or "")
    context, meta = build_retrieval_context(query_text, index, top_k=int(args.top_k), max_chars=int(args.max_chars))
    print(json.dumps(meta, indent=2))
    print(context)


def main() -> None:
    parser = argparse.ArgumentParser(description="SBML-only bootstrap utilities: silver conversion + motif retrieval.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build motif index from a directory of SBML/XML files.")
    p_index.add_argument("--sbml-dir", required=True, help="Directory with SBML/XML files.")
    p_index.add_argument("--out", required=True, help="Output motif index JSON path.")
    p_index.add_argument("--max-files", type=int, default=1000, help="Maximum number of files to index.")
    p_index.set_defaults(func=_cmd_index)

    p_silver = sub.add_parser("silver", help="Convert one SBML file to silver intermediate JSON.")
    p_silver.add_argument("--input", required=True, help="Input SBML file path.")
    p_silver.add_argument("--out", required=True, help="Output silver JSON path.")
    p_silver.set_defaults(func=_cmd_silver)

    p_retrieve = sub.add_parser("retrieve", help="Retrieve top motif examples for a query.")
    p_retrieve.add_argument("--index", required=True, help="Motif index JSON path.")
    p_retrieve.add_argument("--query", default="", help="Inline query string.")
    p_retrieve.add_argument("--query-file", default="", help="Optional query text file path.")
    p_retrieve.add_argument("--top-k", type=int, default=3, help="Number of examples to retrieve.")
    p_retrieve.add_argument("--max-chars", type=int, default=3600, help="Maximum context characters.")
    p_retrieve.set_defaults(func=_cmd_retrieve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
