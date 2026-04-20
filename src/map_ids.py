from __future__ import annotations

import argparse
import json
import os
import re
import time
from copy import deepcopy
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_name(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def _canonical_name(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def _name_variants(value: str, *, max_variants: int = 4) -> List[str]:
    base = _canonical_name(value)
    variants: List[str] = []
    candidates = [
        base,
        re.sub(r"\([^)]*\)", " ", base),
        re.sub(r"[/,;:_-]", " ", base),
        re.sub(r"\b(protein|enzyme)\b", " ", base, flags=re.IGNORECASE),
    ]
    seen_norm: set = set()
    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", candidate).strip()
        norm = _normalize_name(cleaned)
        if not cleaned or not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        variants.append(cleaned)
        if len(variants) >= max_variants:
            break
    return variants or ([base] if base else [])


def _search_terms(value: str, *, max_terms: int = 6) -> List[str]:
    base = _canonical_name(value)
    if not base:
        return []
    candidates = [
        base,
        re.sub(r"[+/]", " ", base),
        re.sub(r"[-_]", " ", base),
        re.sub(r"[^A-Za-z0-9 ]+", " ", base),
        _normalize_name(base),
        _normalize_name(base).replace(" ", ""),
    ]
    out: List[str] = []
    seen: set = set()
    for cand in candidates:
        cleaned = re.sub(r"\s+", " ", str(cand)).strip()
        norm = _normalize_name(cleaned)
        if not cleaned or len(cleaned) < 2 or not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(cleaned)
        if len(out) >= max_terms:
            break
    return out or [base]


def _token_set(value: str) -> set:
    return {tok for tok in _normalize_name(value).split(" ") if tok}


def _punct_token_set(value: str) -> set:
    """Token set replacing all punctuation/hyphens with spaces before splitting.
    Fixes cases like 'fructose-1,6-bisphosphate' vs 'Fructose 1,6-bisphosphate'
    where _normalize_name produces different token counts."""
    return {tok for tok in re.sub(r"[^a-z0-9]+", " ", value.strip().casefold()).split() if tok}


def _punct_jaccard(a: str, b: str) -> float:
    """Jaccard similarity using punctuation-replaced token sets."""
    sa = _punct_token_set(a)
    sb = _punct_token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _jaccard(a: str, b: str) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _split_synonyms(value: str, *, max_items: int = 64) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    parts = re.split(r"[;|]", value)
    out: List[str] = []
    seen: set = set()
    for part in parts:
        cleaned = _canonical_name(part)
        norm = _normalize_name(cleaned)
        if not cleaned or not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def _merge_mapped_ids(*mapped_dicts: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in mapped_dicts:
        for key, value in _safe_dict(item).items():
            sval = str(value or "").strip()
            if sval and key not in out:
                out[key] = sval
    return out


class HttpClient:
    def __init__(self, timeout: int = 15, max_retries: int = 3, backoff_seconds: float = 0.6) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Project14-T2PW-IDMapper/1.0"})

    def get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"Server error {resp.status_code}")
                return resp
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)
        raise RuntimeError(f"HTTP request failed after retries: {url}; last error: {last_exc}")


class MappingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {"proteins": {}, "compounds": {}}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.data["proteins"] = _safe_dict(raw.get("proteins"))
                self.data["compounds"] = _safe_dict(raw.get("compounds"))

    def get(self, section: str, key: str) -> Optional[Dict[str, Any]]:
        value = _safe_dict(self.data.get(section)).get(key)
        return value if isinstance(value, dict) else None

    def set(self, section: str, key: str, value: Dict[str, Any]) -> None:
        self.data.setdefault(section, {})
        self.data[section][key] = value

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")


class PathBankDbResolver:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        schema: str,
        connect_timeout: int = 6,
        read_timeout: int = 20,
        write_timeout: int = 20,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.schema = schema
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.write_timeout = write_timeout
        self._driver = None
        self._conn = None
        self.last_error = ""
        try:
            import pymysql  # type: ignore[import-not-found]

            self._driver = pymysql
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"pymysql_unavailable:{exc}"

    @classmethod
    def from_env(cls, overrides: Optional[Dict[str, Any]] = None) -> Optional["PathBankDbResolver"]:
        cfg = _safe_dict(overrides)

        def _pick(key: str, env_key: str, default: str = "") -> str:
            value = cfg.get(key)
            if value is None or str(value).strip() == "":
                return str(os.getenv(env_key, default) or "").strip()
            return str(value).strip()

        host = _pick("host", "PATHBANK_DB_HOST")
        user = _pick("user", "PATHBANK_DB_USER")
        password = _pick("password", "PATHBANK_DB_PASSWORD")
        schema = _pick("schema", "PATHBANK_DB_SCHEMA", "pathbank")

        if not host or not user:
            return None

        try:
            port = int(_pick("port", "PATHBANK_DB_PORT", "3306") or "3306")
        except ValueError:
            port = 3306
        try:
            connect_timeout = int(_pick("connect_timeout", "PATHBANK_DB_CONNECT_TIMEOUT", "6") or "6")
        except ValueError:
            connect_timeout = 6
        try:
            read_timeout = int(_pick("read_timeout", "PATHBANK_DB_READ_TIMEOUT", "20") or "20")
        except ValueError:
            read_timeout = 20
        try:
            write_timeout = int(_pick("write_timeout", "PATHBANK_DB_WRITE_TIMEOUT", "20") or "20")
        except ValueError:
            write_timeout = 20

        return cls(
            host=host,
            port=port,
            user=user,
            password=password,
            schema=schema,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )

    def available(self) -> bool:
        return self._driver is not None

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        self._conn = None

    def _ensure_connection(self) -> bool:
        if self._conn is not None:
            return True
        if self._driver is None:
            return False
        try:
            self._conn = self._driver.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.schema,
                charset="utf8mb4",
                connect_timeout=self.connect_timeout,
                read_timeout=self.read_timeout,
                write_timeout=self.write_timeout,
                cursorclass=self._driver.cursors.DictCursor,
                autocommit=True,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"db_connect_failed:{exc}"
            self._conn = None
            return False

    def _query(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        if not self._ensure_connection():
            return []
        try:
            assert self._conn is not None
            with self._conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"db_query_failed:{exc}"
            return []

    def _find_species_ids(self, organism: str) -> List[int]:
        text = _canonical_name(organism)
        if not text:
            return []
        rows = self._query(
            (
                "SELECT id, name, common_name, taxonomy_id "
                "FROM species "
                "WHERE LOWER(name)=LOWER(%s) "
                "   OR LOWER(common_name)=LOWER(%s) "
                "   OR taxonomy_id=%s "
                "   OR LOWER(name) LIKE LOWER(%s) "
                "   OR LOWER(common_name) LIKE LOWER(%s) "
                "LIMIT 40"
            ),
            (text, text, text, f"%{text}%", f"%{text}%"),
        )
        if not rows:
            return []
        scored: List[Tuple[float, int]] = []
        norm_text = _normalize_name(text)
        for row in rows:
            sid = int(row.get("id") or 0)
            if sid <= 0:
                continue
            name = str(row.get("name") or "")
            common_name = str(row.get("common_name") or "")
            taxonomy_id = str(row.get("taxonomy_id") or "")
            score = 0.0
            if norm_text and norm_text == _normalize_name(name):
                score = max(score, 1.0)
            if norm_text and norm_text == _normalize_name(common_name):
                score = max(score, 0.95)
            if text and text == taxonomy_id:
                score = max(score, 0.98)
            score = max(score, 0.45 + 0.5 * _jaccard(text, name), 0.42 + 0.5 * _jaccard(text, common_name))
            scored.append((score, sid))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            return []
        top = scored[0][0]
        chosen = [sid for score, sid in scored if score >= max(0.7, top - 0.08)]
        return chosen[:6]

    def map_compound(self, name: str) -> Dict[str, Any]:
        variants = _name_variants(name, max_variants=4)
        by_id: Dict[int, Dict[str, Any]] = {}
        for variant_idx, variant in enumerate(variants):
            for term_idx, term in enumerate(_search_terms(variant, max_terms=5)):
                rows = self._query(
                    (
                        "SELECT id, name, short_name, hmdb_id, kegg_id, chebi_id, pubchem_cid, cas, biocyc_id, chemspider_id, drugbank_id, synonyms "
                        "FROM compounds "
                        "WHERE LOWER(name)=LOWER(%s) "
                        "   OR LOWER(short_name)=LOWER(%s) "
                        "   OR LOWER(name) LIKE LOWER(%s) "
                        "   OR LOWER(short_name) LIKE LOWER(%s) "
                        "   OR LOWER(synonyms) LIKE LOWER(%s) "
                        "LIMIT 120"
                    ),
                    (term, term, f"%{term}%", f"%{term}%", f"%{term}%"),
                )
                variant_penalty = 0.06 * variant_idx
                term_penalty = 0.03 * term_idx
                for row in rows:
                    cid = int(row.get("id") or 0)
                    if cid <= 0:
                        continue
                    db_name = str(row.get("name") or "")
                    short_name = str(row.get("short_name") or "")
                    synonyms = _split_synonyms(str(row.get("synonyms") or ""), max_items=60)
                    norm_name = _normalize_name(name)
                    exact = norm_name in {
                        _normalize_name(db_name),
                        _normalize_name(short_name),
                    }
                    syn_exact = any(norm_name == _normalize_name(s) for s in synonyms)
                    contains_bonus = 0.0
                    if norm_name and (norm_name in _normalize_name(db_name) or norm_name in _normalize_name(short_name)):
                        contains_bonus = 0.08
                    jaccard = max(
                        _jaccard(name, db_name),
                        _jaccard(name, short_name),
                        _punct_jaccard(name, db_name),
                        _punct_jaccard(name, short_name),
                        max((_jaccard(name, s) for s in synonyms), default=0.0),
                    )
                    score = (0.9 if exact else 0.0) + (0.84 if syn_exact else 0.0) + contains_bonus + (0.35 + 0.55 * jaccard)
                    score = max(0.0, min(1.0, score - variant_penalty - term_penalty))
                    mapped_ids = {
                        "hmdb": str(row.get("hmdb_id") or "").strip(),
                        "kegg": str(row.get("kegg_id") or "").strip(),
                        "chebi": (lambda v: f"CHEBI:{v}" if v and not v.upper().startswith("CHEBI:") else v)(str(row.get("chebi_id") or "").strip()),
                        "pubchem": str(row.get("pubchem_cid") or "").strip(),
                        "cas": str(row.get("cas") or "").strip(),
                        "biocyc": str(row.get("biocyc_id") or "").strip(),
                        "chemspider": str(row.get("chemspider_id") or "").strip(),
                        "drugbank": str(row.get("drugbank_id") or "").strip(),
                    }
                    mapped_ids = {k: v for k, v in mapped_ids.items() if v}
                    candidate = {
                        "pathbank_compound_id": cid,
                        "name": db_name,
                        "short_name": short_name,
                        "score": round(score, 4),
                        "mapped_ids": mapped_ids,
                    }
                    existing = by_id.get(cid)
                    if not existing or float(candidate["score"]) > float(existing.get("score", 0.0)):
                        by_id[cid] = candidate

        candidates = sorted(by_id.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if not candidates:
            return {"status": "unmapped", "reason": "no_db_match", "provider": "PathBankDB", "source": "db", "candidates": []}

        best = candidates[0]
        second = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        mapped_ids = _safe_dict(best.get("mapped_ids"))
        best_score = float(best.get("score", 0.0))
        if mapped_ids and (best_score >= 0.9 or (best_score >= 0.74 and best_score >= second + 0.03)):
            merged_ids = dict(mapped_ids)
            for candidate in candidates[1:]:
                if float(candidate.get("score", 0.0)) >= 0.9:
                    merged_ids = _merge_mapped_ids(merged_ids, _safe_dict(candidate.get("mapped_ids")))
            # Carry the PathWhiz internal compound ID so json_to_sbml can use it
            best_pw_cid = best.get("pathbank_compound_id")
            if best_pw_cid:
                merged_ids["pathbank_compound_id"] = str(best_pw_cid)
            return {
                "status": "mapped",
                "provider": "PathBankDB",
                "source": "db",
                "mapped_ids": merged_ids,
                "pathbank_compound_id": best_pw_cid,
                "confidence": best_score,
                "chosen_rule": "db_top_candidate_relaxed",
                "candidates": candidates[:10],
            }

        reason = "ambiguous" if mapped_ids else "no_external_ids"
        return {
            "status": "unmapped",
            "reason": reason,
            "provider": "PathBankDB",
            "source": "db",
            "confidence": float(best.get("score", 0.0)),
            "candidates": candidates[:10],
        }

    def map_protein(self, name: str, organism: str) -> Dict[str, Any]:
        variants = _name_variants(name, max_variants=4)
        species_ids = self._find_species_ids(organism)
        by_id: Dict[int, Dict[str, Any]] = {}

        for variant_idx, variant in enumerate(variants):
            for term_idx, term in enumerate(_search_terms(variant, max_terms=5)):
                pass_modes = [True, False] if species_ids else [False]
                for pass_idx, use_species_filter in enumerate(pass_modes):
                    params: List[Any] = [term, term, term, f"%{term}%", f"%{term}%", f"%{term}%"]
                    species_sql = ""
                    if use_species_filter and species_ids:
                        marks = ", ".join(["%s"] * len(species_ids))
                        species_sql = f" AND species_id IN ({marks})"
                        params.extend(species_ids)
                    rows = self._query(
                        (
                            "SELECT id, name, uniprot_id, gene_name, species_id, synonyms "
                            "FROM proteins "
                            "WHERE (LOWER(name)=LOWER(%s) "
                            "   OR LOWER(gene_name)=LOWER(%s) "
                            "   OR LOWER(uniprot_id)=LOWER(%s) "
                            "   OR LOWER(name) LIKE LOWER(%s) "
                            "   OR LOWER(gene_name) LIKE LOWER(%s) "
                            "   OR LOWER(synonyms) LIKE LOWER(%s))"
                            f"{species_sql} "
                            "LIMIT 120"
                        ),
                        tuple(params),
                    )
                    variant_penalty = 0.06 * variant_idx
                    term_penalty = 0.03 * term_idx
                    relaxed_penalty = 0.02 if (not use_species_filter and pass_idx > 0) else 0.0
                    for row in rows:
                        pid = int(row.get("id") or 0)
                        if pid <= 0:
                            continue
                        db_name = str(row.get("name") or "")
                        gene_name = str(row.get("gene_name") or "")
                        uniprot_id = str(row.get("uniprot_id") or "").strip()
                        row_species_id = int(row.get("species_id") or 0)
                        synonyms = _split_synonyms(str(row.get("synonyms") or ""), max_items=60)
                        norm_name = _normalize_name(name)
                        exact = norm_name in {_normalize_name(db_name), _normalize_name(gene_name)}
                        syn_exact = any(norm_name == _normalize_name(s) for s in synonyms)
                        contains_bonus = 0.0
                        if norm_name and (norm_name in _normalize_name(db_name) or norm_name in _normalize_name(gene_name)):
                            contains_bonus = 0.08
                        jaccard = max(
                            _jaccard(name, db_name),
                            _jaccard(name, gene_name),
                            _punct_jaccard(name, db_name),
                            _punct_jaccard(name, gene_name),
                            max((_jaccard(name, s) for s in synonyms), default=0.0),
                        )
                        species_bonus = 0.14 if species_ids and row_species_id in species_ids else 0.0
                        uniprot_bonus = 0.08 if uniprot_id else 0.0
                        score = (
                            (0.9 if exact else 0.0)
                            + (0.83 if syn_exact else 0.0)
                            + contains_bonus
                            + (0.35 + 0.52 * jaccard)
                            + species_bonus
                            + uniprot_bonus
                        )
                        score = max(0.0, min(1.0, score - variant_penalty - term_penalty - relaxed_penalty))
                        candidate = {
                            "pathbank_protein_id": pid,
                            "name": db_name,
                            "gene_name": gene_name,
                            "uniprot": uniprot_id,
                            "species_id": row_species_id,
                            "score": round(score, 4),
                        }
                        existing = by_id.get(pid)
                        if not existing or float(candidate["score"]) > float(existing.get("score", 0.0)):
                            by_id[pid] = candidate

        candidates = sorted(by_id.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if not candidates:
            return {"status": "unmapped", "reason": "no_db_match", "provider": "PathBankDB", "source": "db", "candidates": []}

        best = candidates[0]
        second = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        uniprot_id = str(best.get("uniprot") or "").strip()
        best_score = float(best.get("score", 0.0))
        if uniprot_id and (
            best_score >= 0.88
            or (best_score >= 0.72 and best_score >= second + 0.03)
            or (len(candidates) == 1 and best_score >= 0.68)
        ):
            # Carry the PathWhiz internal protein ID so json_to_sbml can use it
            best_pw_pid = best.get("pathbank_protein_id")
            protein_mapped_ids: Dict[str, str] = {"uniprot": uniprot_id}
            if best_pw_pid:
                protein_mapped_ids["pathbank_protein_id"] = str(best_pw_pid)
            return {
                "status": "mapped",
                "provider": "PathBankDB",
                "source": "db",
                "mapped_ids": protein_mapped_ids,
                "pathbank_protein_id": best_pw_pid,
                "confidence": best_score,
                "chosen_rule": "db_top_candidate_relaxed",
                "candidates": candidates[:10],
            }
        reason = "ambiguous" if uniprot_id else "no_external_ids"
        return {
            "status": "unmapped",
            "reason": reason,
            "provider": "PathBankDB",
            "source": "db",
            "confidence": float(best.get("score", 0.0)),
            "candidates": candidates[:10],
        }


def _extract_global_organism(payload: Dict[str, Any]) -> str:
    entities = _safe_dict(payload.get("entities"))
    species_names = [
        (item.get("name") or "").strip()
        for item in _safe_list(entities.get("species"))
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip()
    ]
    if len(species_names) == 1:
        return species_names[0]
    biological_states = _safe_list(payload.get("biological_states"))
    state_species = {
        (state.get("species") or "").strip()
        for state in biological_states
        if isinstance(state, dict) and isinstance(state.get("species"), str) and state.get("species").strip()
    }
    if len(state_species) == 1:
        return sorted(state_species)[0]
    return ""


def _entity_locations(payload: Dict[str, Any], location_key: str, name_key: str) -> Dict[str, List[str]]:
    states = {
        (item.get("name") or "").strip(): (item.get("subcellular_location") or "").strip()
        for item in _safe_list(payload.get("biological_states"))
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item.get("name").strip()
        and isinstance(item.get("subcellular_location"), str)
    }
    out: Dict[str, List[str]] = {}
    rows = _safe_list(_safe_dict(payload.get("element_locations")).get(location_key))
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = (row.get(name_key) or "").strip() if isinstance(row.get(name_key), str) else ""
        if not name:
            continue
        state = (row.get("biological_state") or "").strip() if isinstance(row.get("biological_state"), str) else ""
        loc = states.get(state, "")
        out.setdefault(name, [])
        if loc and loc not in out[name]:
            out[name].append(loc)
    return out


def _extract_uniprot_candidates(payload: Dict[str, Any], query_name: str, organism: str) -> List[Dict[str, Any]]:
    results = _safe_list(payload.get("results"))
    out: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        accession = item.get("primaryAccession")
        if not isinstance(accession, str) or not accession:
            continue
        protein_desc = _safe_dict(item.get("proteinDescription"))
        recommended = _safe_dict(protein_desc.get("recommendedName"))
        fullname = _safe_dict(recommended.get("fullName")).get("value")
        alt_names = _safe_list(protein_desc.get("alternativeNames"))
        alt_values: List[str] = []
        for alt in alt_names:
            if not isinstance(alt, dict):
                continue
            alt_full = _safe_dict(alt.get("fullName")).get("value")
            if isinstance(alt_full, str) and alt_full.strip():
                alt_values.append(alt_full.strip())
        gene_names: List[str] = []
        for gene_obj in _safe_list(item.get("genes")):
            if not isinstance(gene_obj, dict):
                continue
            primary = _safe_dict(gene_obj.get("geneName")).get("value")
            if isinstance(primary, str) and primary.strip():
                gene_names.append(primary.strip())
            for synonym in _safe_list(gene_obj.get("synonyms")):
                syn = _safe_dict(synonym).get("value")
                if isinstance(syn, str) and syn.strip():
                    gene_names.append(syn.strip())
        organism_name = _safe_dict(item.get("organism")).get("scientificName", "")
        reviewed = str(item.get("entryType", "")).lower().find("reviewed") != -1

        candidate_names = [v for v in [fullname] if isinstance(v, str)] + [v for v in alt_values if isinstance(v, str)] + gene_names
        best_name_score = max((_jaccard(query_name, c) for c in candidate_names), default=0.0)
        exact_name_match = any(_normalize_name(query_name) == _normalize_name(c) for c in candidate_names)
        organism_score = 0.0
        if organism and isinstance(organism_name, str):
            if _normalize_name(organism) == _normalize_name(organism_name):
                organism_score = 0.25
            elif _normalize_name(organism) in _normalize_name(organism_name):
                organism_score = 0.15
        reviewed_score = 0.05 if reviewed else 0.0
        score = min(1.0, (0.55 if exact_name_match else 0.35 * best_name_score) + organism_score + reviewed_score)

        out.append(
            {
                "accession": accession,
                "protein_name": fullname if isinstance(fullname, str) else "",
                "gene_names": sorted(set(gene_names))[:8],
                "organism": organism_name if isinstance(organism_name, str) else "",
                "reviewed": reviewed,
                "score": round(score, 4),
            }
        )
    out.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return out


def map_protein_uniprot(client: HttpClient, name: str, organism: str) -> Dict[str, Any]:
    variants = _name_variants(name, max_variants=4)
    query_plan: List[Tuple[str, str, bool]] = []
    for variant in variants:
        query_parts = [f'(protein_name:"{variant}" OR gene:"{variant}")']
        if organism:
            query_parts.append(f'organism_name:"{organism}"')
        query_plan.append((variant, " AND ".join(query_parts), True))
    if organism:
        for variant in variants[:2]:
            query_plan.append((variant, f'(protein_name:"{variant}" OR gene:"{variant}")', False))

    aggregated: Dict[str, Dict[str, Any]] = {}
    queries_tried: List[str] = []
    network_errors: List[str] = []

    for variant, query, used_organism in query_plan:
        params = {
            "query": query,
            "format": "json",
            "size": 10,
            "fields": "accession,protein_name,gene_names,organism_name,reviewed",
        }
        try:
            resp = client.get("https://rest.uniprot.org/uniprotkb/search", params=params)
        except Exception as exc:  # noqa: BLE001
            network_errors.append(str(exc))
            continue
        queries_tried.append(query)
        if resp.status_code != 200:
            continue
        payload = resp.json()
        candidates = _extract_uniprot_candidates(payload, query_name=variant, organism=organism if used_organism else "")
        for candidate in candidates:
            accession = str(candidate.get("accession") or "").strip()
            if not accession:
                continue
            adjusted = dict(candidate)
            if not used_organism:
                adjusted["score"] = round(float(adjusted.get("score", 0.0)) - 0.04, 4)
            existing = aggregated.get(accession)
            if not existing or float(adjusted.get("score", 0.0)) > float(existing.get("score", 0.0)):
                aggregated[accession] = adjusted

        ranked = sorted(aggregated.values(), key=lambda item: item.get("score", 0.0), reverse=True)
        if ranked:
            best_score = float(ranked[0].get("score", 0.0))
            second_score = float(ranked[1].get("score", 0.0)) if len(ranked) > 1 else 0.0
            if best_score >= 0.9 and best_score >= second_score + 0.12:
                break

    candidates = sorted(aggregated.values(), key=lambda item: item.get("score", 0.0), reverse=True)
    if not candidates:
        reason = f"network_error:{network_errors[0]}" if network_errors else "no_match"
        return {"status": "unmapped", "reason": reason, "query": " | ".join(queries_tried), "candidates": []}

    best = candidates[0]
    second_score = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
    strong_unique = best["score"] >= 0.78 and best["score"] >= second_score + 0.08
    reviewed_unique = bool(best.get("reviewed")) and best["score"] >= 0.74 and best["score"] >= second_score + 0.06
    if strong_unique or reviewed_unique:
        return {
            "status": "mapped",
            "query": " | ".join(queries_tried),
            "mapped_ids": {"uniprot": best["accession"]},
            "confidence": best["score"],
            "chosen_rule": "top_unique_candidate",
            "candidates": candidates[:8],
            "reviewed": bool(best.get("reviewed")),
            "queries_tried": queries_tried,
        }
    return {
        "status": "unmapped",
        "reason": "ambiguous",
        "query": " | ".join(queries_tried),
        "confidence": best["score"],
        "candidates": candidates[:8],
        "queries_tried": queries_tried,
    }


def _score_compound_candidate(query: str, candidate_name: str) -> float:
    norm_q = _normalize_name(query)
    norm_c = _normalize_name(candidate_name)
    if norm_q == norm_c:
        return 0.95
    jac = _jaccard(query, candidate_name)
    return round(0.35 + 0.6 * jac, 4)


def _query_chebi(client: HttpClient, name: str) -> List[Dict[str, Any]]:
    url = "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getLiteEntity"
    params = {"search": name, "searchCategory": "ALL NAMES", "maximumResults": 10, "stars": "ALL"}
    try:
        resp = client.get(url, params=params)
    except Exception:  # noqa: BLE001
        return []
    if resp.status_code != 200:
        return []
    try:
        root = ElementTree.fromstring(resp.text)
    except ElementTree.ParseError:
        return []

    # Namespace agnostic parsing
    results: List[Dict[str, Any]] = []
    for node in root.iter():
        if node.tag.lower().endswith("liteentity"):
            chebi_id = ""
            chebi_name = ""
            for child in node:
                tag = child.tag.split("}")[-1]
                text = (child.text or "").strip()
                if tag == "chebiId":
                    chebi_id = f"CHEBI:{text}"
                elif tag == "chebiAsciiName":
                    chebi_name = text
            if chebi_id:
                results.append(
                    {
                        "database": "chebi",
                        "id": chebi_id,
                        "name": chebi_name,
                        "score": _score_compound_candidate(name, chebi_name or chebi_id),
                    }
                )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:10]


def _query_kegg(client: HttpClient, name: str) -> List[Dict[str, Any]]:
    encoded = quote_plus(name)
    url = f"https://rest.kegg.jp/find/compound/{encoded}"
    try:
        resp = client.get(url)
    except Exception:  # noqa: BLE001
        return []
    if resp.status_code != 200 or not resp.text.strip():
        return []
    out: List[Dict[str, Any]] = []
    for line in resp.text.splitlines():
        if "\t" not in line:
            continue
        left, right = line.split("\t", 1)
        kid = left.replace("cpd:", "").strip()
        names = [n.strip() for n in right.split(";") if n.strip()]
        best_name = names[0] if names else right.strip()
        score = max((_score_compound_candidate(name, n) for n in names), default=_score_compound_candidate(name, best_name))
        out.append({"database": "kegg", "id": kid, "name": best_name, "score": score})
    out.sort(key=lambda item: item["score"], reverse=True)
    return out[:10]


def _query_hmdb(client: HttpClient, name: str) -> List[Dict[str, Any]]:
    # HMDB has no guaranteed public search API; prefer configured API endpoint, fallback to HTML extraction.
    api_url = str(os.getenv("HMDB_API_URL", "")).strip()
    api_key = str(os.getenv("HMDB_API_KEY", "")).strip()
    if api_url:
        api_params = {
            "query": name,
            "q": name,
            "term": name,
            "search": name,
            "limit": int(os.getenv("HMDB_API_LIMIT", "12") or "12"),
        }
        api_headers: Dict[str, str] = {"Accept": "application/json"}
        if api_key:
            auth_header = str(os.getenv("HMDB_API_AUTH_HEADER", "X-API-Key") or "X-API-Key").strip()
            if auth_header:
                api_headers[auth_header] = api_key
        try:
            api_resp = client.get(api_url, params=api_params, headers=api_headers)
            if api_resp.status_code == 200:
                payload = api_resp.json()
                rows: List[Dict[str, Any]] = []
                if isinstance(payload, dict):
                    for key in ["results", "data", "items", "metabolites"]:
                        value = payload.get(key)
                        if isinstance(value, list):
                            rows = [item for item in value if isinstance(item, dict)]
                            break
                    if not rows:
                        rows = [payload]
                elif isinstance(payload, list):
                    rows = [item for item in payload if isinstance(item, dict)]

                out_api: List[Dict[str, Any]] = []
                seen_api: set = set()
                for row in rows:
                    hid = _canonical_name(
                        str(
                            row.get("hmdb_id")
                            or row.get("accession")
                            or row.get("id")
                            or row.get("identifier")
                            or ""
                        )
                    ).upper()
                    if not hid.startswith("HMDB"):
                        continue
                    if hid in seen_api:
                        continue
                    seen_api.add(hid)
                    cname = _canonical_name(str(row.get("name") or row.get("metabolite_name") or ""))
                    out_api.append(
                        {
                            "database": "hmdb",
                            "id": hid,
                            "name": cname,
                            "score": _score_compound_candidate(name, cname or hid),
                        }
                    )
                    if len(out_api) >= 12:
                        break
                if out_api:
                    out_api.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
                    return out_api[:10]
        except Exception:  # noqa: BLE001
            pass

    url = "https://hmdb.ca/unearth/q"
    params = {"query": name, "searcher": "metabolites"}
    try:
        resp = client.get(url, params=params)
    except Exception:  # noqa: BLE001
        return []
    if resp.status_code != 200:
        return []
    text = resp.text
    ids = re.findall(r"/metabolites/(HMDB\d{5,})", text, flags=re.IGNORECASE)
    seen = set()
    out: List[Dict[str, Any]] = []
    for hid in ids:
        hid_norm = hid.upper()
        if hid_norm in seen:
            continue
        seen.add(hid_norm)
        out.append({"database": "hmdb", "id": hid_norm, "name": "", "score": 0.6})
        if len(out) >= 10:
            break
    return out


def lookup_hmdb_background(client: HttpClient, name: str, *, max_results: int = 6) -> Dict[str, Any]:
    rows = _query_hmdb(client, name)
    limit = max(1, min(20, int(max_results)))
    candidates: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        hid = _canonical_name(str(row.get("id", ""))).upper()
        if not hid:
            continue
        candidates.append(
            {
                "hmdb_id": hid,
                "name": _canonical_name(str(row.get("name", ""))),
                "score": float(row.get("score", 0.0)),
            }
        )
    return {
        "query": _canonical_name(name),
        "provider": "hmdb",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def lookup_compound_api_background(client: HttpClient, name: str, *, max_results: int = 8) -> Dict[str, Any]:
    result = map_compound_all(client, name)
    limit = max(1, min(20, int(max_results)))
    rows = _safe_list(result.get("candidates"))[:limit]
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        db = _canonical_name(str(row.get("database", ""))).lower()
        cid = _canonical_name(str(row.get("id", "")))
        if not db or not cid:
            continue
        candidates.append(
            {
                "database": db,
                "id": cid,
                "name": _canonical_name(str(row.get("name", ""))),
                "score": float(row.get("score", 0.0) or 0.0),
            }
        )
    return {
        "query": _canonical_name(name),
        "provider": "compound_api_bundle",
        "status": str(result.get("status", "")).strip().lower(),
        "reason": str(result.get("reason", "")).strip(),
        "candidate_count": len(candidates),
        "mapped_ids": _safe_dict(result.get("mapped_ids")),
        "candidates": candidates,
    }


def lookup_protein_api_background(
    client: HttpClient,
    name: str,
    organism: str,
    *,
    max_results: int = 8,
) -> Dict[str, Any]:
    result = map_protein_uniprot(client, name, organism)
    limit = max(1, min(20, int(max_results)))
    rows = _safe_list(result.get("candidates"))[:limit]
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        accession = _canonical_name(str(row.get("accession", "")))
        if not accession:
            continue
        candidates.append(
            {
                "accession": accession,
                "protein_name": _canonical_name(str(row.get("protein_name", ""))),
                "organism": _canonical_name(str(row.get("organism", ""))),
                "reviewed": bool(row.get("reviewed", False)),
                "score": float(row.get("score", 0.0) or 0.0),
            }
        )
    return {
        "query": _canonical_name(name),
        "organism": _canonical_name(organism),
        "provider": "uniprot",
        "status": str(result.get("status", "")).strip().lower(),
        "reason": str(result.get("reason", "")).strip(),
        "candidate_count": len(candidates),
        "mapped_ids": _safe_dict(result.get("mapped_ids")),
        "queries_tried": _safe_list(result.get("queries_tried")),
        "candidates": candidates,
    }


def map_compound_all(client: HttpClient, name: str) -> Dict[str, Any]:
    variants = _name_variants(name, max_variants=3)
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for variant_index, variant in enumerate(variants):
        variant_weight = max(0.8, 1.0 - (0.08 * variant_index))
        chebi_candidates = _query_chebi(client, variant)
        kegg_candidates = _query_kegg(client, variant)
        hmdb_candidates = _query_hmdb(client, variant)
        for candidate in chebi_candidates + kegg_candidates + hmdb_candidates:
            db = str(candidate.get("database") or "").strip()
            cid = str(candidate.get("id") or "").strip()
            if not db or not cid:
                continue
            adjusted = dict(candidate)
            adjusted["score"] = round(float(candidate.get("score", 0.0)) * variant_weight, 4)
            key = (db, cid)
            existing = by_key.get(key)
            if not existing or float(adjusted.get("score", 0.0)) > float(existing.get("score", 0.0)):
                by_key[key] = adjusted

        ranked = sorted(by_key.values(), key=lambda item: item["score"], reverse=True)
        if ranked:
            best = ranked[0]
            second = ranked[1]["score"] if len(ranked) > 1 else 0.0
            if best["score"] >= 0.92 and best["score"] >= second + 0.12:
                break

    all_candidates = sorted(by_key.values(), key=lambda item: item["score"], reverse=True)
    if not all_candidates:
        return {"status": "unmapped", "reason": "no_match", "candidates": []}

    best = all_candidates[0]
    second = all_candidates[1]["score"] if len(all_candidates) > 1 else 0.0
    if best["score"] >= 0.78 and best["score"] >= second + 0.08:
        mapped_ids = {best["database"]: best["id"]}
        # Keep additional high-confidence IDs from other databases.
        for cand in all_candidates[1:]:
            if cand["database"] in mapped_ids:
                continue
            if cand["score"] >= 0.9:
                mapped_ids[cand["database"]] = cand["id"]
        return {
            "status": "mapped",
            "mapped_ids": mapped_ids,
            "confidence": best["score"],
            "chosen_rule": "top_unique_candidate",
            "candidates": all_candidates[:12],
        }

    return {
        "status": "unmapped",
        "reason": "ambiguous",
        "confidence": best["score"],
        "candidates": all_candidates[:12],
    }


def _looks_protein_like_name(name: str) -> bool:
    norm = _normalize_name(name)
    if not norm:
        return False
    return bool(
        re.search(
            r"(protein|globulin|peroxidase|deiodinase|kinase|phosphatase|atpase|receptor|transporter|enzyme)",
            norm,
            flags=re.IGNORECASE,
        )
    )


def _collect_protein_like_names(payload: Dict[str, Any]) -> Set[str]:
    entities = _safe_dict(payload.get("entities"))
    processes = _safe_dict(payload.get("processes"))
    element_locations = _safe_dict(payload.get("element_locations"))

    out: Set[str] = set()
    for row in _safe_list(entities.get("proteins")):
        if isinstance(row, dict) and isinstance(row.get("name"), str) and row.get("name").strip():
            out.add(_normalize_name(row["name"]))
    for row in _safe_list(entities.get("protein_complexes")):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name:
            out.add(_normalize_name(name))
        for component in _safe_list(row.get("components")):
            if isinstance(component, str) and component.strip():
                out.add(_normalize_name(component))
    for row in _safe_list(element_locations.get("protein_locations")):
        if isinstance(row, dict) and isinstance(row.get("protein"), str) and row.get("protein").strip():
            out.add(_normalize_name(row["protein"]))
    for reaction in _safe_list(processes.get("reactions")):
        if not isinstance(reaction, dict):
            continue
        for enzyme in _safe_list(reaction.get("enzymes")):
            if not isinstance(enzyme, dict):
                continue
            for key in ["protein", "protein_complex", "name"]:
                value = str(enzyme.get(key) or "").strip()
                if value:
                    out.add(_normalize_name(value))
                    break
    for transport in _safe_list(processes.get("transports")):
        if not isinstance(transport, dict):
            continue
        for transporter in _safe_list(transport.get("transporters")):
            if not isinstance(transporter, dict):
                continue
            for key in ["protein", "protein_complex", "name"]:
                value = str(transporter.get(key) or "").strip()
                if value:
                    out.add(_normalize_name(value))
                    break
    return {value for value in out if value}


def route_entity_for_mapping(
    entity_name: str,
    entity_type_hint: str,
    *,
    protein_like_names: Optional[Set[str]] = None,
) -> Dict[str, str]:
    name = _canonical_name(entity_name)
    hint = _canonical_name(entity_type_hint).lower()
    norm = _normalize_name(name)
    protein_like_set = {v for v in (protein_like_names or set()) if v}

    if ":" in name or hint in {"complex", "protein_complex"}:
        return {"route": "complex", "reason": "complex_entity"}
    if hint in {"protein", "enzyme", "modifier"}:
        return {"route": "protein", "reason": "type_hint"}
    if norm in protein_like_set:
        return {"route": "protein", "reason": "known_protein_like"}
    if _looks_protein_like_name(name):
        return {"route": "protein", "reason": "name_pattern"}
    return {"route": "compound", "reason": "default_compound_route"}


def _map_protein_with_strategy(
    *,
    id_source: str,
    db: Optional[PathBankDbResolver],
    client: HttpClient,
    cache: MappingCache,
    name: str,
    organism: str,
) -> Dict[str, Any]:
    base_key = f"{_normalize_name(name)}::{_normalize_name(organism)}"
    db_key = f"db::{base_key}"
    api_key = f"api::{base_key}"

    if id_source in {"db", "hybrid"}:
        db_result = cache.get("proteins", db_key)
        if db_result is None:
            if db and db.available():
                db_result = db.map_protein(name, organism)
            else:
                db_reason = db.last_error if db else "db_not_configured"
                db_result = {
                    "status": "unmapped",
                    "reason": f"db_unavailable:{db_reason}",
                    "provider": "PathBankDB",
                    "source": "db",
                    "candidates": [],
                }
            cache.set("proteins", db_key, db_result)
        if db_result.get("status") == "mapped" or id_source == "db":
            return db_result

    if id_source in {"api", "hybrid"}:
        api_result = cache.get("proteins", api_key)
        if api_result is None:
            # Backward-compatible cache key from pre-strategy versions.
            legacy = cache.get("proteins", base_key)
            if legacy is not None and id_source == "api":
                api_result = legacy
            else:
                api_result = map_protein_uniprot(client, name, organism)
            api_result.setdefault("provider", "UniProt")
            api_result.setdefault("source", "api")
            cache.set("proteins", api_key, api_result)
        return api_result

    return {"status": "unmapped", "reason": "invalid_id_source", "provider": "none", "source": "none", "candidates": []}


def _map_compound_with_strategy(
    *,
    id_source: str,
    db: Optional[PathBankDbResolver],
    client: HttpClient,
    cache: MappingCache,
    name: str,
) -> Dict[str, Any]:
    base_key = _normalize_name(name)
    db_key = f"db::{base_key}"
    api_key = f"api::{base_key}"

    if id_source in {"db", "hybrid"}:
        db_result = cache.get("compounds", db_key)
        if db_result is None:
            if db and db.available():
                db_result = db.map_compound(name)
            else:
                db_reason = db.last_error if db else "db_not_configured"
                db_result = {
                    "status": "unmapped",
                    "reason": f"db_unavailable:{db_reason}",
                    "provider": "PathBankDB",
                    "source": "db",
                    "candidates": [],
                }
            cache.set("compounds", db_key, db_result)
        if db_result.get("status") == "mapped" or id_source == "db":
            return db_result

    if id_source in {"api", "hybrid"}:
        api_result = cache.get("compounds", api_key)
        if api_result is None:
            legacy = cache.get("compounds", base_key)
            if legacy is not None and id_source == "api":
                api_result = legacy
            else:
                api_result = map_compound_all(client, name)
            api_result.setdefault("provider", "ChEBI/KEGG/HMDB")
            api_result.setdefault("source", "api")
            cache.set("compounds", api_key, api_result)
        return api_result

    return {"status": "unmapped", "reason": "invalid_id_source", "provider": "none", "source": "none", "candidates": []}


def run_mapping(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    cache_path: Path,
    id_source: str = "hybrid",
    db_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    mapped = deepcopy(payload)

    source_mode = (id_source or os.getenv("PATHBANK_ID_SOURCE", "hybrid")).strip().lower()
    if source_mode not in {"api", "db", "hybrid"}:
        source_mode = "hybrid"

    client = HttpClient()
    cache = MappingCache(cache_path)
    db: Optional[PathBankDbResolver] = None
    if source_mode in {"db", "hybrid"}:
        db = PathBankDbResolver.from_env(db_config)

    entities = _safe_dict(mapped.get("entities"))
    proteins = _safe_list(entities.get("proteins"))
    compounds = _safe_list(entities.get("compounds"))
    protein_complexes = _safe_list(entities.get("protein_complexes"))
    protein_like_names = _collect_protein_like_names(mapped)

    global_organism = _extract_global_organism(mapped)
    protein_locations = _entity_locations(mapped, "protein_locations", "protein")
    compound_locations = _entity_locations(mapped, "compound_locations", "compound")

    logs: List[Dict[str, Any]] = []
    proteins_mapped = 0
    compounds_mapped = 0
    protein_ambiguous = 0
    compound_ambiguous = 0
    proteins_mapped_by_db = 0
    proteins_mapped_by_api = 0
    compounds_mapped_by_db = 0
    compounds_mapped_by_api = 0
    compounds_rerouted_to_protein = 0
    compounds_skipped_as_complex = 0
    protein_complexes_skipped = 0

    for idx, protein in enumerate(proteins):
        if not isinstance(protein, dict):
            continue
        name = (protein.get("name") or "").strip() if isinstance(protein.get("name"), str) else ""
        if not name:
            logs.append(
                {
                    "entity_type": "protein",
                    "name": "",
                    "json_pointer": f"/entities/proteins/{idx}",
                    "status": "unmapped",
                    "reason": "missing_name",
                    "location": "",
                }
            )
            continue
        organism = (protein.get("organism") or "").strip() if isinstance(protein.get("organism"), str) else ""
        if not organism and global_organism:
            organism = global_organism
            protein["organism"] = global_organism

        result = _map_protein_with_strategy(
            id_source=source_mode,
            db=db,
            client=client,
            cache=cache,
            name=name,
            organism=organism,
        )
        provider = str(result.get("provider") or ("PathBankDB" if result.get("source") == "db" else "UniProt"))
        source = str(result.get("source") or ("db" if provider == "PathBankDB" else "api"))

        protein.setdefault("mapping_meta", {})
        protein["mapping_meta"]["query"] = {"name": name, "organism": organism}
        protein["mapping_meta"]["provider"] = provider
        protein["mapping_meta"]["source"] = source
        protein["mapping_meta"]["candidates"] = result.get("candidates", [])
        protein["mapping_meta"]["chosen_rule"] = result.get("chosen_rule", "")
        protein["mapping_meta"]["confidence"] = float(result.get("confidence", 0.0))
        protein["mapping_meta"]["reviewed"] = bool(result.get("reviewed", False))

        if result.get("status") == "mapped":
            proteins_mapped += 1
            if source == "db":
                proteins_mapped_by_db += 1
            else:
                proteins_mapped_by_api += 1
            protein["mapped_ids"] = _merge_mapped_ids(_safe_dict(protein.get("mapped_ids")), _safe_dict(result.get("mapped_ids")))
            # Stamp PathWhiz internal protein ID directly on entity for json_to_sbml
            if result.get("pathbank_protein_id"):
                protein["pathbank_protein_id"] = int(result["pathbank_protein_id"])
                protein["mapping_meta"]["pathbank_protein_id"] = int(result["pathbank_protein_id"])
            status = "mapped"
            reason = ""
        else:
            status = "unmapped"
            reason = str(result.get("reason", "unknown"))
            if reason == "ambiguous":
                protein_ambiguous += 1

        logs.append(
            {
                "entity_type": "protein",
                "name": name,
                "json_pointer": f"/entities/proteins/{idx}",
                "status": status,
                "reason": reason,
                "location": ", ".join(protein_locations.get(name, [])),
                "organism": organism,
                "candidate_count": len(_safe_list(result.get("candidates"))),
                "source": source,
                "provider": provider,
            }
        )

    for idx, complex_row in enumerate(protein_complexes):
        if not isinstance(complex_row, dict):
            continue
        name = (complex_row.get("name") or "").strip() if isinstance(complex_row.get("name"), str) else ""
        if not name:
            continue
        complex_row.setdefault("mapping_meta", {})
        complex_row["mapping_meta"]["route"] = "complex"
        complex_row["mapping_meta"]["provider"] = "none"
        complex_row["mapping_meta"]["source"] = "none"
        complex_row["mapping_meta"]["chosen_rule"] = "skip_external_mapping_for_complex"
        complex_row["mapping_meta"]["confidence"] = 0.0
        complex_row["mapping_meta"]["candidates"] = []
        protein_complexes_skipped += 1
        logs.append(
            {
                "entity_type": "protein_complex",
                "name": name,
                "json_pointer": f"/entities/protein_complexes/{idx}",
                "status": "unmapped",
                "reason": "complex_external_mapping_skipped",
                "location": ", ".join(protein_locations.get(name, [])),
                "candidate_count": 0,
                "source": "none",
                "provider": "none",
            }
        )

    for idx, compound in enumerate(compounds):
        if not isinstance(compound, dict):
            continue
        name = (compound.get("name") or "").strip() if isinstance(compound.get("name"), str) else ""
        if not name:
            logs.append(
                {
                    "entity_type": "compound",
                    "name": "",
                    "json_pointer": f"/entities/compounds/{idx}",
                    "status": "unmapped",
                    "reason": "missing_name",
                    "location": "",
                }
            )
            continue

        route = route_entity_for_mapping(name, "compound", protein_like_names=protein_like_names)
        result: Dict[str, Any]
        provider = "none"
        source = "none"
        if route["route"] == "compound":
            result = _map_compound_with_strategy(
                id_source=source_mode,
                db=db,
                client=client,
                cache=cache,
                name=name,
            )
            provider = str(result.get("provider") or ("PathBankDB" if result.get("source") == "db" else "ChEBI/KEGG/HMDB"))
            source = str(result.get("source") or ("db" if provider == "PathBankDB" else "api"))
        elif route["route"] == "protein":
            compounds_rerouted_to_protein += 1
            result = _map_protein_with_strategy(
                id_source=source_mode,
                db=db,
                client=client,
                cache=cache,
                name=name,
                organism=global_organism,
            )
            provider = str(result.get("provider") or ("PathBankDB" if result.get("source") == "db" else "UniProt"))
            source = str(result.get("source") or ("db" if provider == "PathBankDB" else "api"))
        else:
            compounds_skipped_as_complex += 1
            result = {
                "status": "unmapped",
                "reason": "complex_external_mapping_skipped",
                "provider": "none",
                "source": "none",
                "candidates": [],
            }

        compound.setdefault("mapping_meta", {})
        compound["mapping_meta"]["query"] = {"name": name}
        compound["mapping_meta"]["route"] = route["route"]
        compound["mapping_meta"]["route_reason"] = route["reason"]
        compound["mapping_meta"]["providers"] = [provider]
        compound["mapping_meta"]["source"] = source
        compound["mapping_meta"]["candidates"] = result.get("candidates", [])
        compound["mapping_meta"]["chosen_rule"] = result.get("chosen_rule", "")
        compound["mapping_meta"]["confidence"] = float(result.get("confidence", 0.0))

        if result.get("status") == "mapped":
            if route["route"] == "compound":
                compounds_mapped += 1
                if source == "db":
                    compounds_mapped_by_db += 1
                else:
                    compounds_mapped_by_api += 1
            compound["mapped_ids"] = _merge_mapped_ids(_safe_dict(compound.get("mapped_ids")), _safe_dict(result.get("mapped_ids")))
            # Stamp PathWhiz internal compound ID directly on entity for json_to_sbml
            if result.get("pathbank_compound_id"):
                compound["pathbank_compound_id"] = int(result["pathbank_compound_id"])
                compound["mapping_meta"]["pathbank_compound_id"] = int(result["pathbank_compound_id"])
            status = "mapped"
            reason = ""
        else:
            status = "unmapped"
            reason = str(result.get("reason", "unknown"))
            if route["route"] == "compound" and reason == "ambiguous":
                compound_ambiguous += 1

        logs.append(
            {
                "entity_type": "compound",
                "name": name,
                "json_pointer": f"/entities/compounds/{idx}",
                "status": status,
                "reason": reason,
                "route": route["route"],
                "route_reason": route["reason"],
                "location": ", ".join(compound_locations.get(name, [])),
                "candidate_count": len(_safe_list(result.get("candidates"))),
                "source": source,
                "provider": provider,
            }
        )

    cache.save()
    if db is not None:
        db.close()
    output_path.write_text(json.dumps(mapped, indent=2, ensure_ascii=False), encoding="utf-8")

    proteins_total = len([p for p in proteins if isinstance(p, dict) and isinstance(p.get("name"), str) and p.get("name").strip()])
    compounds_total = len([c for c in compounds if isinstance(c, dict) and isinstance(c.get("name"), str) and c.get("name").strip()])
    protein_complexes_total = len(
        [c for c in protein_complexes if isinstance(c, dict) and isinstance(c.get("name"), str) and c.get("name").strip()]
    )
    summary = {
        "proteins_total": proteins_total,
        "proteins_mapped": proteins_mapped,
        "proteins_mapped_pct": round((100.0 * proteins_mapped / proteins_total), 2) if proteins_total else 0.0,
        "proteins_ambiguous": protein_ambiguous,
        "compounds_total": compounds_total,
        "compounds_mapped": compounds_mapped,
        "compounds_mapped_pct": round((100.0 * compounds_mapped / compounds_total), 2) if compounds_total else 0.0,
        "compounds_ambiguous": compound_ambiguous,
        "id_source_mode": source_mode,
        "db_available": bool(db and db.available()),
        "db_last_error": db.last_error if db else "",
        "proteins_mapped_by_db": proteins_mapped_by_db,
        "proteins_mapped_by_api": proteins_mapped_by_api,
        "compounds_mapped_by_db": compounds_mapped_by_db,
        "compounds_mapped_by_api": compounds_mapped_by_api,
        "compounds_rerouted_to_protein": compounds_rerouted_to_protein,
        "compounds_skipped_as_complex": compounds_skipped_as_complex,
        "protein_complexes_total": protein_complexes_total,
        "protein_complexes_skipped": protein_complexes_skipped,
        "complexes_skipped": compounds_skipped_as_complex + protein_complexes_skipped,
    }

    report = {"summary": summary, "entities": logs}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic ID mapping for proteins and compounds.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input audited JSON path")
    parser.add_argument("--out", dest="output_path", default="final.mapped.json", help="Output mapped JSON path")
    parser.add_argument(
        "--report",
        dest="report_path",
        default="mapping_report.json",
        help="Output mapping report JSON path",
    )
    parser.add_argument(
        "--cache",
        dest="cache_path",
        default="id_mapping_cache.json",
        help="Cache file path for deterministic mapping reuse",
    )
    parser.add_argument(
        "--id-source",
        dest="id_source",
        choices=["api", "db", "hybrid"],
        default=os.getenv("PATHBANK_ID_SOURCE", "hybrid"),
        help="ID resolver mode: api, db, or hybrid (db first then api fallback).",
    )
    parser.add_argument("--db-host", dest="db_host", default=os.getenv("PATHBANK_DB_HOST", ""))
    parser.add_argument("--db-port", dest="db_port", type=int, default=int(os.getenv("PATHBANK_DB_PORT", "3306")))
    parser.add_argument("--db-user", dest="db_user", default=os.getenv("PATHBANK_DB_USER", ""))
    parser.add_argument("--db-password", dest="db_password", default=os.getenv("PATHBANK_DB_PASSWORD", ""))
    parser.add_argument("--db-schema", dest="db_schema", default=os.getenv("PATHBANK_DB_SCHEMA", "pathbank"))
    args = parser.parse_args()

    report = run_mapping(
        Path(args.input_path),
        Path(args.output_path),
        Path(args.report_path),
        cache_path=Path(args.cache_path),
        id_source=args.id_source,
        db_config={
            "host": args.db_host,
            "port": args.db_port,
            "user": args.db_user,
            "password": args.db_password,
            "schema": args.db_schema,
        },
    )
    print(f"Wrote mapped JSON: {args.output_path}")
    print(
        "Protein mapped: "
        f"{report['summary']['proteins_mapped']}/{report['summary']['proteins_total']} | "
        "Compound mapped: "
        f"{report['summary']['compounds_mapped']}/{report['summary']['compounds_total']}"
    )


if __name__ == "__main__":
    main()
