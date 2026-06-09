from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple


COMPOUND_ROW_COLUMNS = (
    "id, name, short_name, hmdb_id, kegg_id, chebi_id, pubchem_cid, cas, "
    "biocyc_id, chemspider_id, drugbank_id, pwc_id, description"
)


def normalize_empty(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def normalize_chebi_id(value: Any) -> Optional[str]:
    text = normalize_empty(value)
    if not text:
        return None
    text = re.sub(r"^CHEBI:", "", text, flags=re.IGNORECASE).strip()
    # Some DB rows store multiple space-separated ChEBI IDs (e.g. "15378 24636").
    # Take only the first numeric token so we emit a single valid ID.
    first = text.split()[0] if text else text
    return first or None


def normalize_hmdb_id(value: Any) -> Optional[str]:
    text = normalize_empty(value)
    if not text:
        return None
    return text.upper()


def normalize_kegg_id(value: Any) -> Optional[str]:
    text = normalize_empty(value)
    if not text:
        return None
    return re.sub(r"^cpd:", "", text, flags=re.IGNORECASE).strip().upper() or None


def normalize_pubchem_cid(value: Any) -> Optional[str]:
    return normalize_empty(value)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _alias_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _first_alias_value(row: Dict[str, Any], mapped_ids: Dict[str, Any], aliases: Sequence[str]) -> Optional[str]:
    sources = [row, mapped_ids]
    normalized_sources = [
        {_alias_key(str(key)): value for key, value in source.items()}
        for source in sources
    ]
    for alias in aliases:
        for source in sources:
            value = source.get(alias)
            text = normalize_empty(value)
            if text:
                return text
        alias_key = _alias_key(alias)
        for source in normalized_sources:
            text = normalize_empty(source.get(alias_key))
            if text:
                return text
    return None


def _canonical_name(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _canonical_name(value).casefold()).strip()


def _split_synonyms(value: Any) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    out: List[str] = []
    seen: set[str] = set()
    for part in re.split(r"[;|]", value):
        cleaned = _canonical_name(part)
        key = _name_key(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def _row_id(row: Dict[str, Any]) -> Optional[int]:
    try:
        value = int(float(str(row.get("id") or row.get("pathbank_compound_id") or "").strip()))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def compound_db_row(row: Dict[str, Any]) -> Dict[str, Any]:
    chebi = normalize_chebi_id(row.get("chebi_id"))
    db_id = _row_id(row)
    return {
        "id": db_id,
        "name": normalize_empty(row.get("name")) or "",
        "hmdb_id": normalize_empty(row.get("hmdb_id")),
        "description": normalize_empty(row.get("description")),
        "cas": normalize_empty(row.get("cas")),
        "kegg_id": normalize_kegg_id(row.get("kegg_id")),
        "pubchem_cid": normalize_pubchem_cid(row.get("pubchem_cid") or row.get("pubchem_id")),
        "chebi_id": chebi,
        "biocyc_id": normalize_empty(row.get("biocyc_id")),
        "chemspider_id": normalize_empty(row.get("chemspider_id")),
        "drugbank_id": normalize_empty(row.get("drugbank_id")),
        "pwc_id": normalize_empty(row.get("pwc_id")) or (f"PW_C{db_id:06d}" if db_id else None),
        "short_name": normalize_empty(row.get("short_name")) or normalize_empty(row.get("name")) or "",
    }


def _candidate_from_row(row: Dict[str, Any], *, rule: str, confidence: float) -> Dict[str, Any]:
    db_row = compound_db_row(row)
    return {
        "id": db_row.get("id"),
        "name": db_row.get("name"),
        "hmdb_id": db_row.get("hmdb_id"),
        "kegg_id": db_row.get("kegg_id"),
        "pubchem_cid": db_row.get("pubchem_cid"),
        "chebi_id": db_row.get("chebi_id"),
        "pwc_id": db_row.get("pwc_id"),
        "short_name": db_row.get("short_name"),
        "chosen_rule": rule,
        "confidence": confidence,
        "db_row": db_row,
    }


def _mapped_ids_from_row(row: Dict[str, Any]) -> Dict[str, str]:
    db_row = compound_db_row(row)
    out = {
        "hmdb": db_row.get("hmdb_id"),
        "kegg": db_row.get("kegg_id"),
        "chebi": db_row.get("chebi_id"),
        "pubchem": db_row.get("pubchem_cid"),
        "cas": db_row.get("cas"),
        "biocyc": db_row.get("biocyc_id"),
        "chemspider": db_row.get("chemspider_id"),
        "drugbank": db_row.get("drugbank_id"),
        "pwc_id": db_row.get("pwc_id"),
        "pathbank_compound_id": str(db_row["id"]) if db_row.get("id") is not None else None,
    }
    return {key: str(value) for key, value in out.items() if value not in (None, "")}


def _identifier_candidate_score(rules: Sequence[Tuple[str, float]]) -> Tuple[str, float]:
    unique = {rule: confidence for rule, confidence in rules}
    unique_rules = list(unique)
    if "pathbank_compound_id" in unique_rules:
        return "pathbank_compound_id", 1.0
    if "pwc_id_exact" in unique_rules:
        return "pwc_id_exact", min(0.999, 0.99 + max(0, len(unique_rules) - 1) * 0.001)
    if len(unique_rules) >= 2:
        strength_bonus = sum(unique.values()) / 10000.0
        return "multiple_strong_ids_exact", min(0.989, 0.98 + (len(unique_rules) - 2) * 0.001 + strength_bonus)
    if not rules:
        return "strong_id_exact", 0.85
    return max(rules, key=lambda item: item[1])


class PathWhizCompoundResolver:
    """Resolve extracted compound rows to exact PathWhiz compound table rows."""

    def __init__(self, db: Any) -> None:
        self.db = db

    def _query(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        query = getattr(self.db, "_query", None)
        if not callable(query):
            return []
        return [dict(row) for row in query(sql, params) if isinstance(row, dict)]

    def _find_by_column(self, column: str, value: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        return self._query(
            f"SELECT {COMPOUND_ROW_COLUMNS} FROM compounds WHERE LOWER({column})=LOWER(%s) LIMIT {int(limit)}",
            (value,),
        )

    def _find_by_id(self, value: str) -> List[Dict[str, Any]]:
        return self._query(
            f"SELECT {COMPOUND_ROW_COLUMNS} FROM compounds WHERE id=%s LIMIT 2",
            (value,),
        )

    def _exact_name_rows(self, name: str) -> List[Dict[str, Any]]:
        if not name:
            return []
        rows = self._query(
            (
                f"SELECT {COMPOUND_ROW_COLUMNS}, synonyms FROM compounds "
                "WHERE LOWER(name)=LOWER(%s) LIMIT 40"
            ),
            (name,),
        )
        key = _name_key(name)
        return [row for row in rows if _name_key(row.get("name")) == key]

    def _exact_short_or_synonym_rows(self, name: str) -> List[Dict[str, Any]]:
        if not name:
            return []
        rows = self._query(
            (
                f"SELECT {COMPOUND_ROW_COLUMNS}, synonyms FROM compounds "
                "WHERE LOWER(short_name)=LOWER(%s) OR LOWER(synonyms) LIKE LOWER(%s) LIMIT 120"
            ),
            (name, f"%{name}%"),
        )
        key = _name_key(name)
        return [
            row
            for row in rows
            if _name_key(row.get("short_name")) == key
            or any(_name_key(synonym) == key for synonym in _split_synonyms(row.get("synonyms")))
        ]

    def _fuzzy_rows(self, name: str) -> List[Tuple[float, Dict[str, Any]]]:
        if not name:
            return []
        rows = self._query(
            (
                f"SELECT {COMPOUND_ROW_COLUMNS}, synonyms FROM compounds "
                "WHERE LOWER(name) LIKE LOWER(%s) "
                "   OR LOWER(short_name) LIKE LOWER(%s) "
                "   OR LOWER(synonyms) LIKE LOWER(%s) "
                "LIMIT 120"
            ),
            (f"%{name}%", f"%{name}%", f"%{name}%"),
        )
        query_key = _name_key(name)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            candidates = [row.get("name"), row.get("short_name"), *_split_synonyms(row.get("synonyms"))]
            best = max(
                (SequenceMatcher(None, query_key, _name_key(candidate)).ratio() for candidate in candidates if candidate),
                default=0.0,
            )
            if query_key and any(query_key in _name_key(candidate) for candidate in candidates if candidate):
                best = max(best, 0.65)
            if best >= 0.50:
                scored.append((best, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _result(
        self,
        *,
        status: str,
        raw_name: str,
        chosen: Optional[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        chosen_rule: str,
        confidence: float,
        reason: str = "",
    ) -> Dict[str, Any]:
        result = {
            "status": status,
            "raw_name": raw_name,
            "chosen": chosen,
            "candidates": list(candidates),
            "chosen_rule": chosen_rule,
            "confidence": float(confidence),
        }
        if reason:
            result["reason"] = reason
        return result

    def resolve(self, row: Dict[str, Any]) -> Dict[str, Any]:
        name = _canonical_name(row.get("name"))
        mapped_ids = _safe_dict(row.get("mapped_ids"))

        id_values = [
            (
                "pathbank_compound_id",
                "id",
                normalize_empty(
                    _first_alias_value(
                        row,
                        mapped_ids,
                        ["pathbank_compound_id", "pw_compound_id", "pathwhiz_id", "pathbank id", "pathbank_id", "db_id", "id"],
                    )
                ),
                1.0,
            ),
            ("pwc_id_exact", "pwc_id", normalize_empty(_first_alias_value(row, mapped_ids, ["pwc_id"])), 0.99),
            ("hmdb_id_exact", "hmdb_id", normalize_hmdb_id(_first_alias_value(row, mapped_ids, ["hmdb_id", "hmdb"])), 0.95),
            ("kegg_id_exact", "kegg_id", normalize_kegg_id(_first_alias_value(row, mapped_ids, ["kegg_id", "kegg"])), 0.95),
            ("chebi_id_exact", "chebi_id", normalize_chebi_id(_first_alias_value(row, mapped_ids, ["chebi_id", "chebi"])), 0.90),
            (
                "pubchem_cid_exact",
                "pubchem_cid",
                normalize_pubchem_cid(_first_alias_value(row, mapped_ids, ["pubchem_cid", "pubchem_id", "pubchem"])),
                0.85,
            ),
        ]

        strong_hits: Dict[int, Dict[str, Any]] = {}
        hit_rules: Dict[int, List[Tuple[str, float]]] = {}
        for rule, column, value, confidence in id_values:
            if not value:
                continue
            rows = self._find_by_id(value) if column == "id" else self._find_by_column(column, value)
            if not rows:
                continue
            for hit in rows:
                db_id = _row_id(hit)
                if db_id is None:
                    continue
                strong_hits.setdefault(db_id, hit)
                rules = hit_rules.setdefault(db_id, [])
                if rule not in {existing_rule for existing_rule, _ in rules}:
                    rules.append((rule, confidence))

        if strong_hits:
            candidates = []
            for db_id, hit in strong_hits.items():
                rules = hit_rules.get(db_id) or [("strong_id_exact", 0.85)]
                top_rule, top_confidence = _identifier_candidate_score(rules)
                candidate = _candidate_from_row(hit, rule=top_rule, confidence=top_confidence)
                candidate["matched_rules"] = [rule for rule, _ in rules]
                candidates.append(candidate)
            candidates.sort(
                key=lambda candidate: (
                    -float(candidate.get("confidence") or 0.0),
                    -len(candidate.get("matched_rules") or []),
                    int(candidate.get("id") or 0),
                )
            )
            top_confidence = float(candidates[0]["confidence"])
            top_candidates = [candidate for candidate in candidates if float(candidate.get("confidence") or 0.0) == top_confidence]
            if len(top_candidates) > 1:
                matched_rules = sorted({rule for candidate in top_candidates for rule in candidate.get("matched_rules", [])})
                return self._result(
                    status="ambiguous",
                    raw_name=name,
                    chosen=None,
                    candidates=candidates,
                    chosen_rule="conflicting_strong_ids",
                    confidence=top_confidence,
                    reason=f"multiple_candidates_for_{'_'.join(matched_rules) or 'strong_ids'}",
                )
            chosen = compound_db_row(strong_hits[int(candidates[0]["id"])])
            return self._result(
                status="matched",
                raw_name=name,
                chosen=chosen,
                candidates=candidates[1:],
                chosen_rule=str(candidates[0]["chosen_rule"]),
                confidence=top_confidence,
            )

        for rule, confidence, finder in [
            ("exact_name", 0.70, self._exact_name_rows),
            ("exact_short_name_or_synonym", 0.70, self._exact_short_or_synonym_rows),
        ]:
            rows = finder(name)
            if len(rows) > 1:
                return self._result(
                    status="ambiguous",
                    raw_name=name,
                    chosen=None,
                    candidates=[_candidate_from_row(candidate, rule=rule, confidence=confidence) for candidate in rows],
                    chosen_rule=rule,
                    confidence=confidence,
                    reason=f"multiple_rows_for_{rule}",
                )
            if rows:
                return self._result(
                    status="matched",
                    raw_name=name,
                    chosen=compound_db_row(rows[0]),
                    candidates=[],
                    chosen_rule=rule,
                    confidence=confidence,
                )

        fuzzy = self._fuzzy_rows(name)
        if fuzzy:
            top_score, top_row = fuzzy[0]
            confidence = max(0.50, min(0.65, top_score))
            second = fuzzy[1][0] if len(fuzzy) > 1 else 0.0
            candidates = [
                _candidate_from_row(candidate, rule="fuzzy_name", confidence=max(0.50, min(0.65, score)))
                for score, candidate in fuzzy[:10]
            ]
            if top_score >= 0.62 and top_score >= second + 0.06:
                return self._result(
                    status="matched",
                    raw_name=name,
                    chosen=compound_db_row(top_row),
                    candidates=candidates[1:],
                    chosen_rule="fuzzy_name",
                    confidence=confidence,
                )
            return self._result(
                status="ambiguous",
                raw_name=name,
                chosen=None,
                candidates=candidates,
                chosen_rule="fuzzy_name",
                confidence=confidence,
                reason="unsafe_fuzzy_candidates",
            )

        return self._result(
            status="unmatched",
            raw_name=name,
            chosen=None,
            candidates=[],
            chosen_rule="",
            confidence=0.0,
            reason="No PathWhiz DB match by IDs or name",
        )


def apply_compound_db_resolution(row: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    chosen = _safe_dict(match.get("chosen"))
    out["raw_name"] = str(match.get("raw_name") or row.get("raw_name") or row.get("name") or "")
    out["db_status"] = match.get("status")
    out["db_match"] = match
    out["chosen_rule"] = match.get("chosen_rule")
    out["confidence"] = match.get("confidence", row.get("confidence"))
    if match.get("status") != "matched" or not chosen:
        return out

    out["db_id"] = chosen.get("id")
    out["db_row"] = chosen
    out["pathwhiz_id"] = chosen.get("id")
    out["pathbank_compound_id"] = chosen.get("id")
    out["pw_compound_id"] = chosen.get("id")
    out["name"] = chosen.get("name") or out.get("name")
    out["hmdb_id"] = chosen.get("hmdb_id")
    out["kegg_id"] = chosen.get("kegg_id")
    out["pubchem_cid"] = chosen.get("pubchem_cid")
    out["chebi_id"] = chosen.get("chebi_id")
    out["pwc_id"] = chosen.get("pwc_id")
    out["short_name"] = chosen.get("short_name")
    for key in ["description", "cas", "biocyc_id", "chemspider_id", "drugbank_id"]:
        out[key] = chosen.get(key)
    out["mapped_ids"] = _mapped_ids_from_row(chosen)
    return out
