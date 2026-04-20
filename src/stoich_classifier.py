import json
from typing import Any

from llm_client import chat
from stoich_templates import REACTION_TEMPLATES, REACTION_CLASS_NAMES

_MOLECULE_HINTS: dict[str, list[str]] = {
    t: tmpl.get("required_inputs", []) + tmpl.get("required_outputs", [])
    for t, tmpl in REACTION_TEMPLATES.items()
}

_COFACTORS = {"ATP", "ADP", "NAD+", "NADH", "FAD", "FADH2", "CO2", "H2O", "CoA", "Pi"}


def _molecule_names(reaction: dict) -> set[str]:
    names: set[str] = set()
    for side in ("inputs", "outputs"):
        for item in reaction.get(side, []):
            if isinstance(item, dict):
                names.add(item.get("name", ""))
            elif isinstance(item, str):
                names.add(item)
    return names


def _keyword_scores(text: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for tname, tmpl in REACTION_TEMPLATES.items():
        hits = sum(kw in text for kw in tmpl["match_keywords"])
        if hits:
            scores[tname] = hits
    return scores


def classify_reaction(reaction: dict) -> dict:
    """Returns {"class": str|None, "confidence": "high"|"medium"|"low", "evidence": list[str]}"""
    evidence: list[str] = []

    # --- Pass 1: keyword match ---
    name_text = reaction.get("name", "").lower()
    modifier_text = " ".join(
        m.get("name", "").lower() if isinstance(m, dict) else str(m).lower()
        for m in reaction.get("modifiers", [])
    )
    search_text = name_text + " " + modifier_text

    scores = _keyword_scores(search_text)

    top_class: str | None = None
    confidence = "low"
    candidates: list[str] = []

    if scores:
        max_score = max(scores.values())
        candidates = [t for t, s in scores.items() if s == max_score]
        if len(candidates) == 1:
            top_class = candidates[0]
            confidence = "high"
            evidence.append(f"keyword match: {top_class} (score {max_score})")
        else:
            confidence = "medium"
            evidence.append(f"keyword tie between: {candidates} (score {max_score})")

    if confidence == "high":
        return {"class": top_class, "confidence": confidence, "evidence": evidence}

    # --- Pass 2: molecule presence ---
    molecules = _molecule_names(reaction)
    cofactors_present = molecules & _COFACTORS

    if cofactors_present:
        evidence.append(f"molecules present: {sorted(cofactors_present)}")

    if confidence == "medium" and len(candidates) > 1:
        # try to break the tie
        scored = {}
        for t in candidates:
            mol_hits = sum(m in molecules for m in _MOLECULE_HINTS.get(t, []))
            scored[t] = mol_hits
        best_mol = max(scored.values(), default=0)
        mol_winners = [t for t, s in scored.items() if s == best_mol]
        if len(mol_winners) == 1:
            top_class = mol_winners[0]
            confidence = "high"
            evidence.append(f"molecule tiebreak: {top_class}")
        else:
            top_class = mol_winners[0]
            confidence = "medium"
    elif confidence == "low" and cofactors_present:
        # try to infer class from cofactors alone
        for tname, hints in _MOLECULE_HINTS.items():
            if hints and all(m in molecules for m in hints):
                top_class = tname
                confidence = "medium"
                evidence.append(f"cofactor inference: {tname}")
                break

    if confidence in ("high", "medium") and top_class:
        return {"class": top_class, "confidence": confidence, "evidence": evidence}

    # --- Pass 3: LLM fallback ---
    inputs_summary = [
        (i.get("name") if isinstance(i, dict) else i)
        for i in reaction.get("inputs", [])
    ]
    outputs_summary = [
        (o.get("name") if isinstance(o, dict) else o)
        for o in reaction.get("outputs", [])
    ]
    modifier_names = [
        (m.get("name") if isinstance(m, dict) else m)
        for m in reaction.get("modifiers", [])
    ]

    system = (
        "You are a biochemistry expert. Classify the following reaction into exactly one "
        f"of these classes: {REACTION_CLASS_NAMES}. "
        "Respond with a JSON object: {\"class\": str, \"confidence\": \"high\"|\"medium\"|\"low\", \"reasoning\": str}."
    )
    user_msg = (
        f"Reaction name: {reaction.get('name', 'unknown')}\n"
        f"Inputs: {inputs_summary}\n"
        f"Outputs: {outputs_summary}\n"
        f"Modifiers: {modifier_names}"
    )

    raw = chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=300,
        response_json=True,
    )

    try:
        parsed: dict[str, Any] = json.loads(raw)
        llm_class = parsed.get("class")
        llm_conf = parsed.get("confidence", "low")
        llm_reason = parsed.get("reasoning", "")
        if llm_class not in REACTION_CLASS_NAMES:
            llm_class = None
            llm_conf = "low"
        if llm_conf not in ("high", "medium", "low"):
            llm_conf = "low"
        evidence.append(f"LLM: {llm_reason[:120]}" if llm_reason else "LLM fallback used")
        return {"class": llm_class, "confidence": llm_conf, "evidence": evidence}
    except (json.JSONDecodeError, AttributeError):
        evidence.append("LLM fallback used (parse error)")
        return {"class": None, "confidence": "low", "evidence": evidence}
