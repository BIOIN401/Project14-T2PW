from __future__ import annotations

import argparse
import json
import os
import re
import time
from copy import deepcopy
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


def _first_row_value(row: Dict[str, Any], *keys: str) -> str:
    mapped_ids = _safe_dict(row.get("mapped_ids"))
    for key in keys:
        value = row.get(key)
        if value is None or str(value).strip() == "":
            value = mapped_ids.get(key)
        if value is None or str(value).strip() == "":
            value = mapped_ids.get(key.replace("_id", ""))
        sval = str(value or "").strip()
        if sval:
            return sval
    return ""


def _to_positive_int(value: Any) -> Optional[int]:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _row_external_ids(row: Dict[str, Any], id_keys: List[Tuple[str, Tuple[str, ...]]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for canonical_key, aliases in id_keys:
        value = _first_row_value(row, canonical_key, *aliases)
        if value:
            out[canonical_key] = value
    return out


def _component_name(value: Any) -> str:
    if isinstance(value, str):
        return _canonical_name(value)
    if not isinstance(value, dict):
        return ""
    return _canonical_name(
        str(
            value.get("name")
            or value.get("protein")
            or value.get("component")
            or value.get("entity")
            or value.get("gene")
            or value.get("gene_name")
            or ""
        )
    )


def _component_stoichiometry(value: Any) -> int:
    if isinstance(value, dict):
        parsed = _to_positive_int(value.get("stoichiometry") or value.get("coefficient"))
        if parsed:
            return parsed
    return 1


def _component_mapped_ids(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return _merge_mapped_ids(
        _safe_dict(value.get("mapped_ids")),
        _row_external_ids(value, [("uniprot", ("uniprot_id",)), ("gene_name", ("gene",))]),
    )


def _with_resolution(
    result: Dict[str, Any],
    resolution_status: str,
    *,
    issue: str = "",
    order_step: str = "",
) -> Dict[str, Any]:
    result.setdefault("resolution", {})
    resolution = _safe_dict(result.get("resolution"))
    resolution["status"] = resolution_status
    if issue:
        resolution["issue"] = issue
    if order_step:
        resolution["order_step"] = order_step
    result["resolution"] = resolution
    return result


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
        self.data: Dict[str, Dict[str, Any]] = {"proteins": {}, "compounds": {}, "complexes": {}}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.data["proteins"] = _safe_dict(raw.get("proteins"))
                self.data["compounds"] = _safe_dict(raw.get("compounds"))
                self.data["complexes"] = _safe_dict(raw.get("complexes"))

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

    # ------------------------------------------------------------------
    # Public DB lookup primitives
    # ------------------------------------------------------------------

    def _compound_result_from_row(
        self,
        row: Dict[str, Any],
        *,
        confidence: float,
        chosen_rule: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        cid = int(row.get("id") or row.get("pathbank_compound_id") or 0)
        mapped_ids = {
            "hmdb": str(row.get("hmdb_id") or "").strip(),
            "kegg": str(row.get("kegg_id") or "").strip(),
            "chebi": (lambda v: f"CHEBI:{v}" if v and not v.upper().startswith("CHEBI:") else v)(
                str(row.get("chebi_id") or "").strip()
            ),
            "pubchem": str(row.get("pubchem_cid") or row.get("pubchem_id") or "").strip(),
            "cas": str(row.get("cas") or "").strip(),
            "biocyc": str(row.get("biocyc_id") or "").strip(),
            "chemspider": str(row.get("chemspider_id") or "").strip(),
            "drugbank": str(row.get("drugbank_id") or "").strip(),
            "pwc_id": str(row.get("pwc_id") or "").strip(),
        }
        mapped_ids = {k: v for k, v in mapped_ids.items() if v}
        if cid:
            mapped_ids["pathbank_compound_id"] = str(cid)
        candidate = {
            "pathbank_compound_id": cid,
            "name": str(row.get("name") or ""),
            "short_name": str(row.get("short_name") or ""),
            "score": round(confidence, 4),
            "mapped_ids": mapped_ids,
        }
        return {
            "status": "mapped",
            "provider": "PathBankDB",
            "source": "db",
            "mapped_ids": mapped_ids,
            "pathbank_compound_id": cid,
            "confidence": float(confidence),
            "chosen_rule": chosen_rule,
            "candidates": candidates or [candidate],
        }

    def _protein_result_from_row(
        self,
        row: Dict[str, Any],
        *,
        confidence: float,
        chosen_rule: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        pid = int(row.get("id") or row.get("pathbank_protein_id") or 0)
        uniprot = str(row.get("uniprot_id") or row.get("uniprot") or "").strip()
        mapped_ids: Dict[str, str] = {}
        if uniprot:
            mapped_ids["uniprot"] = uniprot
        if pid:
            mapped_ids["pathbank_protein_id"] = str(pid)
        gene_name = str(row.get("gene_name") or "").strip()
        if gene_name:
            mapped_ids["gene_name"] = gene_name
        candidate = {
            "pathbank_protein_id": pid,
            "name": str(row.get("name") or ""),
            "uniprot": uniprot,
            "gene_name": gene_name,
            "species_id": int(row.get("species_id") or 0),
            "score": round(confidence, 4),
        }
        return {
            "status": "mapped",
            "provider": "PathBankDB",
            "source": "db",
            "mapped_ids": mapped_ids,
            "pathbank_protein_id": pid,
            "confidence": float(confidence),
            "chosen_rule": chosen_rule,
            "candidates": candidates or [candidate],
        }

    def _complex_component_rows(self, complex_id: int) -> List[Dict[str, Any]]:
        if complex_id <= 0:
            return []
        rows = self._query(
            (
                "SELECT pcp.protein_id, p.name AS protein_name, p.uniprot_id, p.gene_name, p.species_id "
                "FROM protein_complex_proteins pcp "
                "JOIN proteins p ON p.id = pcp.protein_id "
                "WHERE pcp.complex_id=%s"
            ),
            (complex_id,),
        )
        out: List[Dict[str, Any]] = []
        for row in rows:
            protein_id = int(row.get("protein_id") or 0)
            if protein_id <= 0:
                continue
            component: Dict[str, Any] = {
                "name": str(row.get("protein_name") or ""),
                "pathbank_protein_id": protein_id,
                "stoichiometry": 1,
            }
            uniprot = str(row.get("uniprot_id") or row.get("uniprot") or "").strip()
            if uniprot:
                component["uniprot"] = uniprot
                component["mapped_ids"] = {"uniprot": uniprot, "pathbank_protein_id": str(protein_id)}
            else:
                component["mapped_ids"] = {"pathbank_protein_id": str(protein_id)}
            gene_name = str(row.get("gene_name") or "").strip()
            if gene_name:
                component["gene_name"] = gene_name
            species_id = _to_positive_int(row.get("species_id"))
            if species_id:
                component["species_id"] = species_id
            out.append(component)
        return out

    def _complex_result_from_row(
        self,
        row: Dict[str, Any],
        *,
        confidence: float,
        chosen_rule: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        cid = int(row.get("id") or row.get("pathbank_complex_id") or row.get("pathbank_protein_complex_id") or 0)
        species_id = _to_positive_int(row.get("species_id"))
        hydrated_components = components if components is not None else self._complex_component_rows(cid)
        candidate = {
            "pathbank_complex_id": cid,
            "pathbank_protein_complex_id": cid,
            "name": str(row.get("name") or ""),
            "species_id": species_id,
            "score": round(confidence, 4),
            "component_count": len(hydrated_components),
        }
        issues: List[Dict[str, Any]] = []
        if not hydrated_components:
            issues.append(
                {
                    "issue": "protein_complex_missing_components",
                    "reason": "mapped_complex_has_no_components",
                    "pathbank_protein_complex_id": cid,
                }
            )
        return {
            "status": "mapped",
            "provider": "PathBankDB",
            "source": "db",
            "pathbank_complex_id": cid,
            "pathbank_protein_complex_id": cid,
            "species_id": species_id,
            "components": hydrated_components,
            "confidence": float(confidence),
            "chosen_rule": chosen_rule,
            "candidates": candidates or [candidate],
            "issues": issues,
        }

    def _map_compound_by_pathbank_id(self, pathbank_id: str) -> Dict[str, Any]:
        cid_text = str(pathbank_id or "").strip()
        if not cid_text:
            return {"status": "unmapped", "reason": "no_id_provided", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        rows = self._query(
            (
                "SELECT id, name, short_name, hmdb_id, kegg_id, chebi_id, pubchem_cid, cas, "
                "biocyc_id, chemspider_id, drugbank_id "
                "FROM compounds WHERE id=%s LIMIT 2"
            ),
            (cid_text,),
        )
        if not rows:
            return {"status": "unmapped", "reason": "no_db_match", "provider": "PathBankDB", "source": "db", "candidates": []}
        return self._compound_result_from_row(rows[0], confidence=1.0, chosen_rule="pathbank_compound_id")

    def _map_compound_by_pwc_id(self, pwc_id: str) -> Dict[str, Any]:
        text = str(pwc_id or "").strip()
        if not text:
            return {"status": "unmapped", "reason": "no_id_provided", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        rows = self._query(
            (
                "SELECT id, name, short_name, hmdb_id, kegg_id, chebi_id, pubchem_cid, cas, "
                "biocyc_id, chemspider_id, drugbank_id, pwc_id "
                "FROM compounds WHERE LOWER(pwc_id)=LOWER(%s) LIMIT 20"
            ),
            (text,),
        )
        if not rows:
            return {"status": "unmapped", "reason": "no_db_match", "provider": "PathBankDB", "source": "db", "candidates": []}
        if len(rows) > 1:
            candidates = [
                self._compound_result_from_row(row, confidence=1.0, chosen_rule="pwc_id")["candidates"][0]
                for row in rows
            ]
            return {
                "status": "unmapped",
                "reason": "ambiguous",
                "provider": "PathBankDB",
                "source": "db",
                "confidence": 1.0,
                "chosen_rule": "pwc_id",
                "candidates": candidates[:10],
            }
        return self._compound_result_from_row(rows[0], confidence=1.0, chosen_rule="pwc_id")

    def _map_protein_by_pathbank_id(self, pathbank_id: str) -> Dict[str, Any]:
        pid_text = str(pathbank_id or "").strip()
        if not pid_text:
            return {"status": "unmapped", "reason": "no_id_provided", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        rows = self._query(
            (
                "SELECT id, name, uniprot_id, gene_name, species_id "
                "FROM proteins WHERE id=%s LIMIT 2"
            ),
            (pid_text,),
        )
        if not rows:
            return {"status": "unmapped", "reason": "no_db_match", "provider": "PathBankDB", "source": "db", "candidates": []}
        return self._protein_result_from_row(rows[0], confidence=1.0, chosen_rule="pathbank_protein_id")

    def _map_complex_by_pathbank_id(self, pathbank_id: str) -> Dict[str, Any]:
        cid_text = str(pathbank_id or "").strip()
        if not cid_text:
            return {"status": "unmapped", "reason": "no_id_provided", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        rows = self._query(
            "SELECT id, name, species_id FROM protein_complexes WHERE id=%s LIMIT 2",
            (cid_text,),
        )
        if not rows:
            return {
                "status": "unmapped",
                "reason": "no_db_match",
                "provider": "PathBankDB",
                "source": "db",
                "candidates": [],
                "chosen_rule": "pathbank_protein_complex_id",
                "confidence": 0.0,
            }
        return self._complex_result_from_row(rows[0], confidence=1.0, chosen_rule="pathbank_protein_complex_id")

    def find_species(
        self,
        organism: str,
        *,
        taxonomy_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve a species name/taxonomy ID to PathBank species candidates."""
        text = _canonical_name(organism)
        tid = (taxonomy_id or "").strip()
        if not text and not tid:
            return {"status": "unmapped", "reason": "empty_query", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        params: tuple
        if tid:
            rows = self._query(
                (
                    "SELECT id, name, common_name, taxonomy_id "
                    "FROM species "
                    "WHERE taxonomy_id=%s "
                    "   OR LOWER(name)=LOWER(%s) "
                    "   OR LOWER(common_name)=LOWER(%s) "
                    "   OR LOWER(name) LIKE LOWER(%s) "
                    "   OR LOWER(common_name) LIKE LOWER(%s) "
                    "LIMIT 40"
                ),
                (tid, text, text, f"%{text}%", f"%{text}%"),
            )
        else:
            rows = self._query(
                (
                    "SELECT id, name, common_name, taxonomy_id "
                    "FROM species "
                    "WHERE LOWER(name)=LOWER(%s) "
                    "   OR LOWER(common_name)=LOWER(%s) "
                    "   OR LOWER(name) LIKE LOWER(%s) "
                    "   OR LOWER(common_name) LIKE LOWER(%s) "
                    "LIMIT 40"
                ),
                (text, text, f"%{text}%", f"%{text}%"),
            )
        if not rows:
            return {"status": "unmapped", "reason": "no_db_match", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        norm_text = _normalize_name(text)
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            sid = int(row.get("id") or 0)
            if sid <= 0:
                continue
            name_db = str(row.get("name") or "")
            common_name = str(row.get("common_name") or "")
            row_tid = str(row.get("taxonomy_id") or "")
            score = 0.0
            if tid and row_tid == tid:
                score = max(score, 0.98)
            if norm_text and norm_text == _normalize_name(name_db):
                score = max(score, 1.0)
            if norm_text and norm_text == _normalize_name(common_name):
                score = max(score, 0.95)
            score = max(score, 0.45 + 0.5 * _jaccard(text, name_db), 0.42 + 0.5 * _jaccard(text, common_name))
            candidates.append({
                "pathbank_species_id": sid,
                "name": name_db,
                "common_name": common_name,
                "taxonomy_id": row_tid,
                "confidence": round(score, 4),
            })
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        if not candidates:
            return {"status": "unmapped", "reason": "no_db_match", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        best = candidates[0]
        threshold = max(0.7, best["confidence"] - 0.08)
        top = [c for c in candidates if c["confidence"] >= threshold]
        chosen_rule = "exact_match" if best["confidence"] >= 0.95 else "fuzzy_match"
        return {
            "status": "mapped" if best["confidence"] >= 0.7 else "unmapped",
            "reason": "" if best["confidence"] >= 0.7 else "low_confidence",
            "candidates": top[:6],
            "chosen_rule": chosen_rule,
            "confidence": best["confidence"],
        }

    def find_species_by_pathbank_id(self, pathbank_species_id: Any) -> Dict[str, Any]:
        """Resolve a PathBank species ID to a species candidate."""
        sid = _to_positive_int(pathbank_species_id)
        if sid is None:
            return {"status": "unmapped", "reason": "empty_query", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        rows = self._query(
            (
                "SELECT id, name, common_name, taxonomy_id "
                "FROM species "
                "WHERE id=%s "
                "LIMIT 2"
            ),
            (sid,),
        )
        if not rows:
            return {"status": "unmapped", "reason": "no_db_match", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        row = rows[0]
        candidate = {
            "pathbank_species_id": int(row.get("id") or sid),
            "name": str(row.get("name") or ""),
            "common_name": str(row.get("common_name") or ""),
            "taxonomy_id": str(row.get("taxonomy_id") or ""),
            "confidence": 1.0,
        }
        return {
            "status": "mapped",
            "reason": "",
            "candidates": [candidate],
            "chosen_rule": "pathbank_species_id",
            "confidence": 1.0,
        }

    def find_subcellular_location(self, name: str) -> Dict[str, Any]:
        """Find a subcellular location by name."""
        text = _canonical_name(name)
        if not text:
            return {"status": "unmapped", "reason": "empty_query", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        rows = self._query(
            (
                "SELECT id, name "
                "FROM subcellular_locations "
                "WHERE LOWER(name)=LOWER(%s) "
                "   OR LOWER(name) LIKE LOWER(%s) "
                "LIMIT 40"
            ),
            (text, f"%{text}%"),
        )
        if not rows:
            return {"status": "unmapped", "reason": "no_db_match", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        norm_text = _normalize_name(text)
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            lid = int(row.get("id") or 0)
            if lid <= 0:
                continue
            loc_name = str(row.get("name") or "")
            score = 1.0 if norm_text == _normalize_name(loc_name) else max(0.4, 0.4 + 0.55 * _jaccard(text, loc_name))
            candidates.append({"pathbank_subcellular_location_id": lid, "name": loc_name, "confidence": round(score, 4)})
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        best = candidates[0]
        return {
            "status": "mapped" if best["confidence"] >= 0.6 else "unmapped",
            "reason": "" if best["confidence"] >= 0.6 else "low_confidence",
            "candidates": candidates[:8],
            "chosen_rule": "exact_match" if best["confidence"] >= 0.95 else "fuzzy_match",
            "confidence": best["confidence"],
        }

    def find_cell_type(self, name: str) -> Dict[str, Any]:
        """Find a cell type by name."""
        text = _canonical_name(name)
        if not text:
            return {"status": "unmapped", "reason": "empty_query", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        rows = self._query(
            (
                "SELECT id, name "
                "FROM cell_types "
                "WHERE LOWER(name)=LOWER(%s) "
                "   OR LOWER(name) LIKE LOWER(%s) "
                "LIMIT 40"
            ),
            (text, f"%{text}%"),
        )
        if not rows:
            return {"status": "unmapped", "reason": "no_db_match", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        norm_text = _normalize_name(text)
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            cid = int(row.get("id") or 0)
            if cid <= 0:
                continue
            ct_name = str(row.get("name") or "")
            score = 1.0 if norm_text == _normalize_name(ct_name) else max(0.4, 0.4 + 0.55 * _jaccard(text, ct_name))
            candidates.append({"pathbank_cell_type_id": cid, "name": ct_name, "confidence": round(score, 4)})
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        best = candidates[0]
        return {
            "status": "mapped" if best["confidence"] >= 0.6 else "unmapped",
            "reason": "" if best["confidence"] >= 0.6 else "low_confidence",
            "candidates": candidates[:8],
            "chosen_rule": "exact_match" if best["confidence"] >= 0.95 else "fuzzy_match",
            "confidence": best["confidence"],
        }

    def find_tissue(self, name: str) -> Dict[str, Any]:
        """Find a tissue by name."""
        text = _canonical_name(name)
        if not text:
            return {"status": "unmapped", "reason": "empty_query", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        rows = self._query(
            (
                "SELECT id, name "
                "FROM tissues "
                "WHERE LOWER(name)=LOWER(%s) "
                "   OR LOWER(name) LIKE LOWER(%s) "
                "LIMIT 40"
            ),
            (text, f"%{text}%"),
        )
        if not rows:
            return {"status": "unmapped", "reason": "no_db_match", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        norm_text = _normalize_name(text)
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            tid = int(row.get("id") or 0)
            if tid <= 0:
                continue
            tissue_name = str(row.get("name") or "")
            score = 1.0 if norm_text == _normalize_name(tissue_name) else max(0.4, 0.4 + 0.55 * _jaccard(text, tissue_name))
            candidates.append({"pathbank_tissue_id": tid, "name": tissue_name, "confidence": round(score, 4)})
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        best = candidates[0]
        return {
            "status": "mapped" if best["confidence"] >= 0.6 else "unmapped",
            "reason": "" if best["confidence"] >= 0.6 else "low_confidence",
            "candidates": candidates[:8],
            "chosen_rule": "exact_match" if best["confidence"] >= 0.95 else "fuzzy_match",
            "confidence": best["confidence"],
        }

    def find_biological_state(
        self,
        species: str,
        subcellular_location: str,
        *,
        cell_type: Optional[str] = None,
        tissue: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find a biological state matching species + subcellular location + optional cell type/tissue."""
        species_result = self.find_species(species)
        loc_result = self.find_subcellular_location(subcellular_location)
        if species_result["status"] != "mapped" or not species_result["candidates"]:
            return {
                "status": "unmapped",
                "reason": f"species_not_found:{species}",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }
        if loc_result["status"] != "mapped" or not loc_result["candidates"]:
            return {
                "status": "unmapped",
                "reason": f"subcellular_location_not_found:{subcellular_location}",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }
        species_ids = [c["pathbank_species_id"] for c in species_result["candidates"][:3]]
        loc_ids = [c["pathbank_subcellular_location_id"] for c in loc_result["candidates"][:3]]

        # Build parameterised IN clauses
        sp_marks = ", ".join(["%s"] * len(species_ids))
        loc_marks = ", ".join(["%s"] * len(loc_ids))

        extra_sql = ""
        extra_params: List[Any] = []
        cell_type_id: Optional[int] = None
        tissue_id: Optional[int] = None

        if cell_type:
            ct_result = self.find_cell_type(cell_type)
            if ct_result["status"] == "mapped" and ct_result["candidates"]:
                cell_type_id = ct_result["candidates"][0]["pathbank_cell_type_id"]
                extra_sql += " AND (cell_type_id=%s OR cell_type_id IS NULL)"
                extra_params.append(cell_type_id)

        if tissue:
            tissue_result = self.find_tissue(tissue)
            if tissue_result["status"] == "mapped" and tissue_result["candidates"]:
                tissue_id = tissue_result["candidates"][0]["pathbank_tissue_id"]
                extra_sql += " AND (tissue_id=%s OR tissue_id IS NULL)"
                extra_params.append(tissue_id)

        rows = self._query(
            (
                "SELECT id, species_id, subcellular_location_id, cell_type_id, tissue_id "
                "FROM biological_states "
                f"WHERE species_id IN ({sp_marks}) "
                f"  AND subcellular_location_id IN ({loc_marks})"
                f"{extra_sql} "
                "LIMIT 40"
            ),
            tuple(species_ids + loc_ids + extra_params),
        )
        if not rows:
            return {
                "status": "unmapped",
                "reason": "no_db_match",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }

        sp_conf = {c["pathbank_species_id"]: c["confidence"] for c in species_result["candidates"][:3]}
        loc_conf = {c["pathbank_subcellular_location_id"]: c["confidence"] for c in loc_result["candidates"][:3]}

        candidates: List[Dict[str, Any]] = []
        for row in rows:
            bs_id = int(row.get("id") or 0)
            if bs_id <= 0:
                continue
            row_sp = int(row.get("species_id") or 0)
            row_loc = int(row.get("subcellular_location_id") or 0)
            row_ct = row.get("cell_type_id")
            row_ti = row.get("tissue_id")
            sp_score = sp_conf.get(row_sp, 0.5)
            loc_score = loc_conf.get(row_loc, 0.5)
            ct_bonus = 0.05 if (cell_type_id and row_ct == cell_type_id) else 0.0
            ti_bonus = 0.05 if (tissue_id and row_ti == tissue_id) else 0.0
            score = round((sp_score + loc_score) / 2.0 + ct_bonus + ti_bonus, 4)
            candidates.append({
                "pathbank_biological_state_id": bs_id,
                "species_id": row_sp,
                "subcellular_location_id": row_loc,
                "cell_type_id": row_ct,
                "tissue_id": row_ti,
                "confidence": score,
            })
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        best = candidates[0]
        return {
            "status": "mapped" if best["confidence"] >= 0.6 else "unmapped",
            "reason": "" if best["confidence"] >= 0.6 else "low_confidence",
            "candidates": candidates[:8],
            "chosen_rule": "species_and_location_match",
            "confidence": best["confidence"],
        }

    def map_compound_by_ids(self, ids: Dict[str, str]) -> Dict[str, Any]:
        """Direct compound lookup by external IDs (hmdb, kegg, chebi, pubchem, cas, drugbank, biocyc, chemspider).

        Tries IDs in priority order; first match wins.
        """
        _id_cols = [
            ("hmdb", "hmdb_id"),
            ("kegg", "kegg_id"),
            ("chebi", "chebi_id"),
            ("pubchem", "pubchem_cid"),
            ("cas", "cas"),
            ("drugbank", "drugbank_id"),
            ("biocyc", "biocyc_id"),
            ("chemspider", "chemspider_id"),
        ]
        by_id: Dict[int, Dict[str, Any]] = {}
        for id_key, col in _id_cols:
            val = str(ids.get(id_key) or "").strip()
            if not val:
                continue
            # Also try stripping CHEBI: prefix for the chebi_id column
            search_vals = [val]
            if id_key == "chebi" and val.upper().startswith("CHEBI:"):
                search_vals.append(val[6:].strip())
            for sval in search_vals:
                rows = self._query(
                    (
                        f"SELECT id, name, short_name, hmdb_id, kegg_id, chebi_id, pubchem_cid, cas, biocyc_id, chemspider_id, drugbank_id "
                        f"FROM compounds WHERE LOWER({col})=LOWER(%s) LIMIT 20"
                    ),
                    (sval,),
                )
                for row in rows:
                    cid = int(row.get("id") or 0)
                    if cid <= 0 or cid in by_id:
                        continue
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
                    by_id[cid] = {
                        "pathbank_compound_id": cid,
                        "name": str(row.get("name") or ""),
                        "short_name": str(row.get("short_name") or ""),
                        "matched_on": id_key,
                        "score": 1.0,
                        "mapped_ids": mapped_ids,
                    }
        if not by_id:
            return {"status": "unmapped", "reason": "no_db_match", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        candidates = list(by_id.values())
        best = candidates[0]
        merged = dict(best["mapped_ids"])
        merged["pathbank_compound_id"] = str(best["pathbank_compound_id"])
        return {
            "status": "mapped",
            "provider": "PathBankDB",
            "source": "db",
            "mapped_ids": merged,
            "pathbank_compound_id": best["pathbank_compound_id"],
            "confidence": 1.0,
            "chosen_rule": f"direct_id_match:{best['matched_on']}",
            "candidates": candidates[:10],
        }

    def map_compound_by_name(self, name: str) -> Dict[str, Any]:
        """Compound lookup by name/synonym fuzzy matching. Delegates to map_compound."""
        return self.map_compound(name)

    def _map_compound_exact_name(self, name: str) -> Dict[str, Any]:
        text = _canonical_name(name)
        if not text:
            return {"status": "unmapped", "reason": "empty_query", "provider": "PathBankDB", "source": "db", "candidates": []}
        rows = self._query(
            (
                "SELECT id, name, short_name, hmdb_id, kegg_id, chebi_id, pubchem_cid, cas, "
                "biocyc_id, chemspider_id, drugbank_id "
                "FROM compounds "
                "WHERE LOWER(name)=LOWER(%s) OR LOWER(short_name)=LOWER(%s) "
                "LIMIT 40"
            ),
            (text, text),
        )
        norm = _normalize_name(text)
        exact_rows = [
            row for row in rows
            if norm in {_normalize_name(str(row.get("name") or "")), _normalize_name(str(row.get("short_name") or ""))}
        ]
        if not exact_rows:
            return {"status": "unmapped", "reason": "no_db_match", "provider": "PathBankDB", "source": "db", "candidates": []}
        candidates = [
            self._compound_result_from_row(row, confidence=1.0, chosen_rule="exact_normalized_name")["candidates"][0]
            for row in exact_rows
        ]
        if len({c["pathbank_compound_id"] for c in candidates}) > 1:
            return {
                "status": "unmapped",
                "reason": "ambiguous",
                "provider": "PathBankDB",
                "source": "db",
                "confidence": 1.0,
                "chosen_rule": "exact_normalized_name",
                "candidates": candidates[:10],
            }
        return self._compound_result_from_row(exact_rows[0], confidence=1.0, chosen_rule="exact_normalized_name", candidates=candidates[:10])

    def _map_compound_synonym(self, name: str) -> Dict[str, Any]:
        text = _canonical_name(name)
        if not text:
            return {"status": "unmapped", "reason": "empty_query", "provider": "PathBankDB", "source": "db", "candidates": []}
        rows = self._query(
            (
                "SELECT id, name, short_name, hmdb_id, kegg_id, chebi_id, pubchem_cid, cas, "
                "biocyc_id, chemspider_id, drugbank_id, synonyms "
                "FROM compounds "
                "WHERE LOWER(synonyms) LIKE LOWER(%s) "
                "LIMIT 120"
            ),
            (f"%{text}%",),
        )
        norm = _normalize_name(text)
        synonym_rows = [
            row for row in rows
            if any(norm == _normalize_name(s) for s in _split_synonyms(str(row.get("synonyms") or ""), max_items=80))
        ]
        if not synonym_rows:
            return {"status": "unmapped", "reason": "no_db_match", "provider": "PathBankDB", "source": "db", "candidates": []}
        candidates = [
            self._compound_result_from_row(row, confidence=0.95, chosen_rule="synonym")["candidates"][0]
            for row in synonym_rows
        ]
        if len({c["pathbank_compound_id"] for c in candidates}) > 1:
            return {
                "status": "unmapped",
                "reason": "ambiguous",
                "provider": "PathBankDB",
                "source": "db",
                "confidence": 0.95,
                "chosen_rule": "synonym",
                "candidates": candidates[:10],
            }
        return self._compound_result_from_row(synonym_rows[0], confidence=0.95, chosen_rule="synonym", candidates=candidates[:10])

    def map_compound_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Compound resolution order: internal IDs, external IDs, exact name, synonym, fuzzy, novel."""
        name = _canonical_name(str(row.get("name") or ""))
        pathbank_id = _first_row_value(row, "pathbank_compound_id", "pw_compound_id", "pathwhiz_id")
        if pathbank_id:
            result = self._map_compound_by_pathbank_id(pathbank_id)
            if result.get("status") == "mapped":
                return _with_resolution(result, "matched", order_step="pathbank_compound_id")

        pwc_id = _first_row_value(row, "pwc_id")
        if pwc_id:
            result = self._map_compound_by_pwc_id(pwc_id)
            if result.get("status") == "mapped":
                return _with_resolution(result, "matched", order_step="pwc_id")
            if result.get("reason") == "ambiguous":
                return _with_resolution(result, "ambiguous", issue="ambiguous_pwc_id", order_step="pwc_id")

        external_ids = _row_external_ids(
            row,
            [
                ("hmdb", ("hmdb_id",)),
                ("kegg", ("kegg_id",)),
                ("chebi", ("chebi_id",)),
                ("pubchem", ("pubchem_cid", "pubchem_id")),
                ("cas", ("cas_id", "cas_number")),
            ],
        )
        if external_ids:
            result = self.map_compound_by_ids(external_ids)
            if result.get("status") == "mapped":
                return _with_resolution(result, "matched", order_step=f"external_id:{result.get('chosen_rule', '')}")
            if result.get("reason") == "ambiguous":
                return _with_resolution(result, "ambiguous", issue="ambiguous_external_id", order_step="external_id")

        exact = self._map_compound_exact_name(name)
        if exact.get("status") == "mapped":
            return _with_resolution(exact, "matched", order_step="exact_normalized_name")
        if exact.get("reason") == "ambiguous":
            return _with_resolution(exact, "ambiguous", issue="ambiguous_exact_name", order_step="exact_normalized_name")

        synonym = self._map_compound_synonym(name)
        if synonym.get("status") == "mapped":
            return _with_resolution(synonym, "matched", order_step="synonym")
        if synonym.get("reason") == "ambiguous":
            return _with_resolution(synonym, "ambiguous", issue="ambiguous_synonym", order_step="synonym")

        fuzzy = self.map_compound(name)
        if fuzzy.get("status") == "mapped":
            return _with_resolution(fuzzy, "matched", order_step="high_confidence_fuzzy")
        candidates = _safe_list(fuzzy.get("candidates"))
        if candidates:
            return _with_resolution(fuzzy, "ambiguous", issue=str(fuzzy.get("reason") or "unsafe_fuzzy_candidates"), order_step="high_confidence_fuzzy")

        novel = {
            "status": "unmapped",
            "reason": "novel_compound",
            "provider": "PathBankDB",
            "source": "db",
            "confidence": 0.0,
            "chosen_rule": "novel_compound",
            "candidates": [],
        }
        return _with_resolution(novel, "novel", issue="no_db_candidates", order_step="novel_compound")

    def map_protein_by_ids(
        self,
        ids: Dict[str, str],
        *,
        species: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Direct protein lookup by external IDs (uniprot, gene_name).

        Species is optional but narrows results when supplied.
        """
        uniprot = str(ids.get("uniprot") or ids.get("uniprot_id") or "").strip()
        gene = str(ids.get("gene") or ids.get("gene_name") or "").strip()
        if not uniprot and not gene:
            return {"status": "unmapped", "reason": "no_ids_provided", "candidates": [], "chosen_rule": "", "confidence": 0.0}

        species_ids: List[int] = []
        if species:
            species_ids = self._find_species_ids(species)
        species_lookup_failed = bool(species) and not species_ids
        if gene and not uniprot and species_lookup_failed:
            return {
                "status": "unmapped",
                "reason": f"species_not_found:{species}",
                "provider": "PathBankDB",
                "source": "db",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }

        by_id: Dict[int, Dict[str, Any]] = {}
        for query_val, col in [(uniprot, "uniprot_id"), (gene, "gene_name")]:
            if not query_val:
                continue
            if col == "gene_name" and species_lookup_failed:
                continue
            params_list: List[Any] = [query_val]
            sp_sql = ""
            if species_ids:
                sp_marks = ", ".join(["%s"] * len(species_ids))
                sp_sql = f" AND species_id IN ({sp_marks})"
                params_list.extend(species_ids)
            rows = self._query(
                (
                    f"SELECT id, name, uniprot_id, gene_name, species_id "
                    f"FROM proteins WHERE LOWER({col})=LOWER(%s){sp_sql} LIMIT 20"
                ),
                tuple(params_list),
            )
            for row in rows:
                pid = int(row.get("id") or 0)
                if pid <= 0 or pid in by_id:
                    continue
                row_sp = int(row.get("species_id") or 0)
                sp_bonus = 0.08 if species_ids and row_sp in species_ids else 0.0
                score = round(1.0 + sp_bonus, 4)
                by_id[pid] = {
                    "pathbank_protein_id": pid,
                    "name": str(row.get("name") or ""),
                    "uniprot": str(row.get("uniprot_id") or "").strip(),
                    "gene_name": str(row.get("gene_name") or "").strip(),
                    "species_id": row_sp,
                    "matched_on": col,
                    "score": score,
                }
        if not by_id:
            return {"status": "unmapped", "reason": "no_db_match", "candidates": [], "chosen_rule": "", "confidence": 0.0}
        candidates = sorted(by_id.values(), key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        uid = best["uniprot"]
        if not uid:
            return {
                "status": "unmapped",
                "reason": "no_uniprot_id",
                "candidates": candidates[:10],
                "chosen_rule": "",
                "confidence": float(best["score"]),
            }
        protein_mapped_ids: Dict[str, str] = {"uniprot": uid, "pathbank_protein_id": str(best["pathbank_protein_id"])}
        return {
            "status": "mapped",
            "provider": "PathBankDB",
            "source": "db",
            "mapped_ids": protein_mapped_ids,
            "pathbank_protein_id": best["pathbank_protein_id"],
            "confidence": float(best["score"]),
            "chosen_rule": f"direct_id_match:{best['matched_on']}",
            "candidates": candidates[:10],
        }

    def map_protein_by_name_species(self, name: str, species: str) -> Dict[str, Any]:
        """Protein lookup by name/gene with required species filter. Delegates to map_protein."""
        if not species:
            return _with_resolution({
                "status": "unmapped",
                "reason": "species_required",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }, "unresolved", issue="needs_species")
        return self.map_protein(name, species)

    def _map_protein_exact_name_species(self, name: str, species: str) -> Dict[str, Any]:
        text = _canonical_name(name)
        if not text:
            return {"status": "unmapped", "reason": "empty_query", "provider": "PathBankDB", "source": "db", "candidates": []}
        species_ids = self._find_species_ids(species)
        if not species_ids:
            return {
                "status": "unmapped",
                "reason": f"species_not_found:{species}",
                "provider": "PathBankDB",
                "source": "db",
                "candidates": [],
            }
        marks = ", ".join(["%s"] * len(species_ids))
        rows = self._query(
            (
                "SELECT id, name, uniprot_id, gene_name, species_id "
                "FROM proteins "
                f"WHERE LOWER(name)=LOWER(%s) AND species_id IN ({marks}) "
                "LIMIT 40"
            ),
            (text,) + tuple(species_ids),
        )
        norm = _normalize_name(text)
        exact_rows = [row for row in rows if norm == _normalize_name(str(row.get("name") or ""))]
        if not exact_rows:
            return {"status": "unmapped", "reason": "no_db_match", "provider": "PathBankDB", "source": "db", "candidates": []}
        candidates = [
            self._protein_result_from_row(row, confidence=1.0, chosen_rule="exact_protein_name_species")["candidates"][0]
            for row in exact_rows
        ]
        if len({c["pathbank_protein_id"] for c in candidates}) > 1:
            return {
                "status": "unmapped",
                "reason": "ambiguous",
                "provider": "PathBankDB",
                "source": "db",
                "confidence": 1.0,
                "chosen_rule": "exact_protein_name_species",
                "candidates": candidates[:10],
            }
        return self._protein_result_from_row(
            exact_rows[0],
            confidence=1.0,
            chosen_rule="exact_protein_name_species",
            candidates=candidates[:10],
        )

    def map_protein_row(self, row: Dict[str, Any], species: str) -> Dict[str, Any]:
        """Protein resolution order: internal ID, UniProt, gene/species, exact name/species, fuzzy/species, novel."""
        name = _canonical_name(str(row.get("name") or ""))
        pathbank_id = _first_row_value(row, "pathbank_protein_id", "pw_protein_id", "pathwhiz_id")
        if pathbank_id:
            result = self._map_protein_by_pathbank_id(pathbank_id)
            if result.get("status") == "mapped":
                return _with_resolution(result, "matched", order_step="pathbank_protein_id")

        protein_ids = _row_external_ids(row, [("uniprot", ("uniprot_id",))])
        if protein_ids.get("uniprot"):
            result = self.map_protein_by_ids(protein_ids, species=species or None)
            if result.get("status") == "mapped":
                return _with_resolution(result, "matched", order_step="uniprot")
            if result.get("reason") == "ambiguous":
                return _with_resolution(result, "ambiguous", issue="ambiguous_uniprot", order_step="uniprot")

        if not species:
            return _with_resolution(
                {
                    "status": "unmapped",
                    "reason": "needs_species",
                    "provider": "PathBankDB",
                    "source": "db",
                    "confidence": 0.0,
                    "chosen_rule": "",
                    "candidates": [],
                },
                "unresolved",
                issue="needs_species",
            )

        if not self._find_species_ids(species):
            return _with_resolution(
                {
                    "status": "unmapped",
                    "reason": f"species_not_found:{species}",
                    "provider": "PathBankDB",
                    "source": "db",
                    "confidence": 0.0,
                    "chosen_rule": "",
                    "candidates": [],
                },
                "unresolved",
                issue="species_not_found",
            )

        gene = _first_row_value(row, "gene", "gene_name")
        if gene:
            result = self.map_protein_by_ids({"gene_name": gene}, species=species)
            if result.get("status") == "mapped":
                return _with_resolution(result, "matched", order_step="gene_name_species")
            if result.get("reason") == "ambiguous":
                return _with_resolution(result, "ambiguous", issue="ambiguous_gene_name_species", order_step="gene_name_species")

        exact = self._map_protein_exact_name_species(name, species)
        if exact.get("status") == "mapped":
            return _with_resolution(exact, "matched", order_step="exact_protein_name_species")
        if exact.get("reason") == "ambiguous":
            return _with_resolution(exact, "ambiguous", issue="ambiguous_exact_protein_name_species", order_step="exact_protein_name_species")

        fuzzy = self.map_protein(name, species)
        if fuzzy.get("status") == "mapped":
            return _with_resolution(fuzzy, "matched", order_step="synonym_or_fuzzy_species")
        candidates = _safe_list(fuzzy.get("candidates"))
        if candidates:
            return _with_resolution(fuzzy, "ambiguous", issue=str(fuzzy.get("reason") or "unsafe_fuzzy_candidates"), order_step="synonym_or_fuzzy_species")

        novel = {
            "status": "unmapped",
            "reason": "novel_protein",
            "provider": "PathBankDB",
            "source": "db",
            "confidence": 0.0,
            "chosen_rule": "novel_protein",
            "candidates": [],
        }
        return _with_resolution(novel, "novel", issue="no_db_candidates", order_step="novel_protein")

    def map_protein_complex(self, name: str, species: str) -> Dict[str, Any]:
        """Find a protein complex by name with required species."""
        if not species:
            return {
                "status": "unmapped",
                "reason": "species_required",
                "provider": "PathBankDB",
                "source": "db",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }
        species_ids = self._find_species_ids(species)
        if not species_ids:
            return {
                "status": "unmapped",
                "reason": f"species_not_found:{species}",
                "provider": "PathBankDB",
                "source": "db",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }
        variants = _name_variants(name, max_variants=3)
        by_id: Dict[int, Dict[str, Any]] = {}
        sp_marks = ", ".join(["%s"] * len(species_ids))
        for variant_idx, variant in enumerate(variants):
            for term in _search_terms(variant, max_terms=4):
                rows = self._query(
                    (
                        "SELECT id, name, species_id "
                        "FROM protein_complexes "
                        f"WHERE (LOWER(name)=LOWER(%s) OR LOWER(name) LIKE LOWER(%s)) "
                        f"  AND species_id IN ({sp_marks}) "
                        "LIMIT 60"
                    ),
                    (term, f"%{term}%") + tuple(species_ids),
                )
                variant_penalty = 0.06 * variant_idx
                for row in rows:
                    cid = int(row.get("id") or 0)
                    if cid <= 0:
                        continue
                    db_name = str(row.get("name") or "")
                    row_sp = int(row.get("species_id") or 0)
                    norm_name = _normalize_name(name)
                    exact = norm_name == _normalize_name(db_name)
                    sp_bonus = 0.1 if row_sp in species_ids else 0.0
                    score = round(
                        max(0.0, min(1.0, (0.9 if exact else 0.35 + 0.55 * _jaccard(name, db_name)) + sp_bonus - variant_penalty)),
                        4,
                    )
                    existing = by_id.get(cid)
                    if not existing or score > float(existing.get("score", 0.0)):
                        by_id[cid] = {
                            "pathbank_complex_id": cid,
                            "pathbank_protein_complex_id": cid,
                            "name": db_name,
                            "species_id": row_sp,
                            "score": score,
                        }
        candidates = sorted(by_id.values(), key=lambda c: c["score"], reverse=True)
        if not candidates:
            return {
                "status": "unmapped",
                "reason": "no_db_match",
                "provider": "PathBankDB",
                "source": "db",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }
        best = candidates[0]
        second = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        best_score = float(best["score"])
        if best_score >= 0.8 and best_score >= second + 0.05:
            return self._complex_result_from_row(
                {
                    "id": best["pathbank_complex_id"],
                    "name": best.get("name", ""),
                    "species_id": best.get("species_id"),
                },
                confidence=best_score,
                chosen_rule="top_unique_complex_candidate",
                candidates=candidates[:10],
            )
        return {
            "status": "unmapped",
            "reason": "ambiguous" if candidates else "no_db_match",
            "provider": "PathBankDB",
            "source": "db",
            "confidence": best_score,
            "chosen_rule": "",
            "candidates": candidates[:10],
        }

    def find_complex_by_component(self, component_name: str, species: str) -> Dict[str, Any]:
        """Find all protein complexes that contain a given protein component, filtered by species."""
        if not species:
            return {
                "status": "unmapped",
                "reason": "species_required",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }
        species_ids = self._find_species_ids(species)
        if not species_ids:
            return {
                "status": "unmapped",
                "reason": f"species_not_found:{species}",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }
        # First find the protein
        protein_result = self.map_protein_by_name_species(component_name, species)
        protein_id: Optional[int] = None
        if protein_result.get("status") == "mapped":
            protein_id = protein_result.get("pathbank_protein_id")
        if protein_id is None:
            # Fallback: fuzzy protein search to get candidate IDs
            for term in _search_terms(_canonical_name(component_name), max_terms=3):
                sp_marks = ", ".join(["%s"] * len(species_ids))
                rows = self._query(
                    (
                        "SELECT id FROM proteins "
                        f"WHERE (LOWER(name) LIKE LOWER(%s) OR LOWER(gene_name) LIKE LOWER(%s)) "
                        f"  AND species_id IN ({sp_marks}) LIMIT 5"
                    ),
                    (f"%{term}%", f"%{term}%") + tuple(species_ids),
                )
                if rows:
                    protein_id = int(rows[0].get("id") or 0) or None
                    break
        if not protein_id:
            return {
                "status": "unmapped",
                "reason": f"component_protein_not_found:{component_name}",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }
        sp_marks = ", ".join(["%s"] * len(species_ids))
        rows = self._query(
            (
                "SELECT pc.id, pc.name, pc.species_id "
                "FROM protein_complexes pc "
                "JOIN protein_complex_proteins pcp ON pcp.complex_id = pc.id "
                f"WHERE pcp.protein_id=%s AND pc.species_id IN ({sp_marks}) "
                "LIMIT 40"
            ),
            (protein_id,) + tuple(species_ids),
        )
        if not rows:
            return {
                "status": "unmapped",
                "reason": "no_complex_found_for_component",
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
            }
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            cid = int(row.get("id") or 0)
            if cid <= 0:
                continue
            candidates.append({
                "pathbank_complex_id": cid,
                "name": str(row.get("name") or ""),
                "species_id": int(row.get("species_id") or 0),
                "component_protein_id": protein_id,
                "confidence": 0.9,
            })
        return {
            "status": "mapped" if candidates else "unmapped",
            "reason": "" if candidates else "no_complex_found_for_component",
            "candidates": candidates[:10],
            "chosen_rule": "component_join_lookup",
            "confidence": 0.9 if candidates else 0.0,
        }

    def _find_complexes_by_component_protein_id(self, protein_id: int, species_ids: List[int]) -> List[Dict[str, Any]]:
        if protein_id <= 0 or not species_ids:
            return []
        sp_marks = ", ".join(["%s"] * len(species_ids))
        rows = self._query(
            (
                "SELECT pc.id, pc.name, pc.species_id "
                "FROM protein_complexes pc "
                "JOIN protein_complex_proteins pcp ON pcp.complex_id = pc.id "
                f"WHERE pcp.protein_id=%s AND pc.species_id IN ({sp_marks}) "
                "LIMIT 80"
            ),
            (protein_id,) + tuple(species_ids),
        )
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            cid = int(row.get("id") or 0)
            if cid <= 0:
                continue
            candidates.append(
                {
                    "pathbank_complex_id": cid,
                    "pathbank_protein_complex_id": cid,
                    "name": str(row.get("name") or ""),
                    "species_id": int(row.get("species_id") or 0),
                    "component_protein_id": protein_id,
                    "score": 0.9,
                }
            )
        return candidates

    def _resolve_complex_components(self, row: Dict[str, Any], species: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        resolved: List[Dict[str, Any]] = []
        issues: List[Dict[str, Any]] = []
        for idx, raw_component in enumerate(_safe_list(row.get("components"))):
            name = _component_name(raw_component)
            if not name:
                issues.append({"issue": "component_missing_name", "component_index": idx})
                continue
            component_row = dict(raw_component) if isinstance(raw_component, dict) else {"name": name}
            component_row["name"] = name
            component_ids = _component_mapped_ids(component_row)
            if component_ids:
                component_row["mapped_ids"] = component_ids
            protein_result: Dict[str, Any]
            pathbank_id = _first_row_value(component_row, "pathbank_protein_id", "pw_protein_id", "pathwhiz_id")
            if pathbank_id:
                protein_result = self._map_protein_by_pathbank_id(pathbank_id)
            elif component_ids.get("uniprot"):
                protein_result = self.map_protein_by_ids(component_ids, species=species or None)
            elif species:
                protein_result = self.map_protein_row(component_row, species)
            else:
                protein_result = {
                    "status": "unmapped",
                    "reason": "needs_species",
                    "provider": "PathBankDB",
                    "source": "db",
                    "candidates": [],
                    "confidence": 0.0,
                    "chosen_rule": "",
                }

            hydrated: Dict[str, Any] = {
                "name": name,
                "stoichiometry": _component_stoichiometry(raw_component),
                "mapping_status": str(protein_result.get("status") or "unmapped"),
                "mapping_rule": str(protein_result.get("chosen_rule") or ""),
            }
            if protein_result.get("pathbank_protein_id"):
                hydrated["pathbank_protein_id"] = int(protein_result["pathbank_protein_id"])
            mapped_ids = _merge_mapped_ids(component_ids, _safe_dict(protein_result.get("mapped_ids")))
            if mapped_ids:
                hydrated["mapped_ids"] = mapped_ids
            if protein_result.get("status") == "mapped":
                resolved.append(hydrated)
            else:
                hydrated["reason"] = str(protein_result.get("reason") or "unmapped_component")
                issues.append(
                    {
                        "issue": "component_protein_unresolved",
                        "component": name,
                        "component_index": idx,
                        "reason": hydrated["reason"],
                    }
                )
                resolved.append(hydrated)
        return resolved, issues

    def map_protein_complex_row(self, row: Dict[str, Any], species: str) -> Dict[str, Any]:
        """Complex resolution order: direct ID, name/species, component/species, novel from resolved components."""
        name = _canonical_name(str(row.get("name") or ""))
        pathbank_id = _first_row_value(
            row,
            "pathbank_protein_complex_id",
            "pathbank_complex_id",
            "pw_complex_id",
            "pathwhiz_id",
        )
        input_components, component_issues = self._resolve_complex_components(row, species)

        if pathbank_id:
            result = self._map_complex_by_pathbank_id(pathbank_id)
            if result.get("status") == "mapped":
                if not _safe_list(result.get("components")) and input_components:
                    result["components"] = input_components
                    result["issues"] = [i for i in _safe_list(result.get("issues")) if i.get("issue") != "protein_complex_missing_components"]
                return _with_resolution(result, "matched", order_step="pathbank_protein_complex_id")

        if not species:
            return _with_resolution(
                {
                    "status": "unmapped",
                    "reason": "needs_species",
                    "provider": "PathBankDB",
                    "source": "db",
                    "confidence": 0.0,
                    "chosen_rule": "",
                    "candidates": [],
                    "components": input_components,
                    "issues": component_issues,
                },
                "unresolved",
                issue="needs_species",
            )

        species_ids = self._find_species_ids(species)
        if not species_ids:
            return _with_resolution(
                {
                    "status": "unmapped",
                    "reason": f"species_not_found:{species}",
                    "provider": "PathBankDB",
                    "source": "db",
                    "confidence": 0.0,
                    "chosen_rule": "",
                    "candidates": [],
                    "components": input_components,
                    "issues": component_issues,
                },
                "unresolved",
                issue="species_not_found",
            )

        if name:
            by_name = self.map_protein_complex(name, species)
            if by_name.get("status") == "mapped":
                if not _safe_list(by_name.get("components")) and input_components:
                    by_name["components"] = input_components
                    by_name["issues"] = [
                        i for i in _safe_list(by_name.get("issues")) if i.get("issue") != "protein_complex_missing_components"
                    ]
                by_name["issues"] = _safe_list(by_name.get("issues")) + component_issues
                return _with_resolution(by_name, "matched", order_step="complex_name_species")
            if by_name.get("reason") == "ambiguous":
                by_name["components"] = input_components
                by_name["issues"] = component_issues
                return _with_resolution(by_name, "ambiguous", issue="ambiguous_complex_name_species", order_step="complex_name_species")

        resolved_component_ids = [
            parsed
            for parsed in (_to_positive_int(component.get("pathbank_protein_id")) for component in input_components)
            if parsed
        ]
        by_complex_id: Dict[int, Dict[str, Any]] = {}
        for protein_id in resolved_component_ids:
            for candidate in self._find_complexes_by_component_protein_id(protein_id, species_ids):
                cid = int(candidate.get("pathbank_complex_id") or 0)
                if cid <= 0:
                    continue
                score = 0.82
                if name:
                    score = max(0.55, min(0.98, 0.35 + 0.55 * _jaccard(name, str(candidate.get("name") or ""))))
                    if _normalize_name(name) == _normalize_name(str(candidate.get("name") or "")):
                        score = 0.98
                candidate["score"] = round(score, 4)
                existing = by_complex_id.get(cid)
                if not existing or float(candidate["score"]) > float(existing.get("score", 0.0)):
                    by_complex_id[cid] = candidate
        component_candidates = sorted(by_complex_id.values(), key=lambda c: c["score"], reverse=True)
        if component_candidates:
            best = component_candidates[0]
            second = float(component_candidates[1].get("score", 0.0)) if len(component_candidates) > 1 else 0.0
            best_score = float(best.get("score", 0.0))
            if len(component_candidates) == 1 or (best_score >= 0.72 and best_score >= second + 0.08):
                result = self._complex_result_from_row(
                    {
                        "id": best["pathbank_complex_id"],
                        "name": best.get("name", ""),
                        "species_id": best.get("species_id"),
                    },
                    confidence=best_score,
                    chosen_rule="resolved_component_species",
                    candidates=component_candidates[:10],
                )
                if not _safe_list(result.get("components")) and input_components:
                    result["components"] = input_components
                    result["issues"] = [
                        i for i in _safe_list(result.get("issues")) if i.get("issue") != "protein_complex_missing_components"
                    ]
                result["issues"] = _safe_list(result.get("issues")) + component_issues
                return _with_resolution(result, "matched", order_step="resolved_component_species")
            return _with_resolution(
                {
                    "status": "unmapped",
                    "reason": "ambiguous",
                    "provider": "PathBankDB",
                    "source": "db",
                    "confidence": float(component_candidates[0].get("score", 0.0)),
                    "chosen_rule": "resolved_component_species",
                    "candidates": component_candidates[:10],
                    "components": input_components,
                    "issues": component_issues,
                },
                "ambiguous",
                issue="ambiguous_component_complex_species",
                order_step="resolved_component_species",
            )

        if resolved_component_ids:
            species_id = species_ids[0] if species_ids else None
            return _with_resolution(
                {
                    "status": "unmapped",
                    "reason": "novel_complex",
                    "provider": "PathBankDB",
                    "source": "db",
                    "confidence": 0.0,
                    "chosen_rule": "novel_complex_from_resolved_components",
                    "candidates": [],
                    "species_id": species_id,
                    "components": input_components,
                    "issues": component_issues,
                },
                "novel",
                issue="no_db_candidates",
                order_step="novel_complex_from_resolved_components",
            )

        issues = component_issues[:]
        if not _safe_list(row.get("components")):
            issues.append({"issue": "protein_complex_missing_components", "reason": "no_components_provided"})
        return _with_resolution(
            {
                "status": "unmapped",
                "reason": "no_components" if not _safe_list(row.get("components")) else "component_proteins_unresolved",
                "provider": "PathBankDB",
                "source": "db",
                "confidence": 0.0,
                "chosen_rule": "",
                "candidates": [],
                "components": input_components,
                "issues": issues,
            },
            "unresolved",
            issue="no_components" if not _safe_list(row.get("components")) else "component_proteins_unresolved",
        )

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
        if not organism:
            return _with_resolution(
                {
                    "status": "unmapped",
                    "reason": "needs_species",
                    "provider": "PathBankDB",
                    "source": "db",
                    "confidence": 0.0,
                    "chosen_rule": "",
                    "candidates": [],
                },
                "unresolved",
                issue="needs_species",
            )
        variants = _name_variants(name, max_variants=4)
        species_ids = self._find_species_ids(organism)
        if not species_ids:
            return {
                "status": "unmapped",
                "reason": f"species_not_found:{organism}",
                "provider": "PathBankDB",
                "source": "db",
                "confidence": 0.0,
                "chosen_rule": "",
                "candidates": [],
            }
        by_id: Dict[int, Dict[str, Any]] = {}

        for variant_idx, variant in enumerate(variants):
            for term_idx, term in enumerate(_search_terms(variant, max_terms=5)):
                pass_modes = [True]
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


def _species_id_from_row(row: Dict[str, Any]) -> Optional[int]:
    ref = _safe_dict(row.get("species_ref") or row.get("species_reference"))
    meta = _safe_dict(row.get("mapping_meta"))
    mapped = _safe_dict(row.get("mapped_ids"))
    for container in [row, ref, meta, mapped]:
        sid = _to_positive_int(
            container.get("pathbank_species_id")
            or container.get("pw_species_id")
            or container.get("species_id")
        )
        if sid is not None:
            return sid
    return None


def _species_hint_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    ref = _safe_dict(row.get("species_ref") or row.get("species_reference"))
    species_value = row.get("species")
    if isinstance(species_value, dict):
        ref = {**species_value, **ref}
        species_value = ref.get("name")
    name = _canonical_name(
        str(
            ref.get("name")
            or ref.get("species")
            or row.get("species_name")
            or species_value
            or row.get("organism")
            or row.get("organism_name")
            or ""
        )
    )
    taxonomy_id = _canonical_name(
        str(
            ref.get("taxonomy_id")
            or ref.get("taxonomy-id")
            or row.get("taxonomy_id")
            or row.get("taxonomy-id")
            or row.get("taxon_id")
            or row.get("ncbi_taxonomy_id")
            or ""
        )
    )
    sid = _species_id_from_row({**row, "species_ref": ref})
    return {"name": name, "taxonomy_id": taxonomy_id, "pathbank_species_id": sid}


def _compact_species_ref(ref: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in [
        "name",
        "pathbank_species_id",
        "species_id",
        "taxonomy_id",
        "common_name",
        "source",
        "status",
        "confidence",
        "reason",
    ]:
        value = ref.get(key)
        if value not in (None, ""):
            out[key] = value
    return out


def _species_ref_from_candidate(candidate: Dict[str, Any], *, source: str, chosen_rule: str = "") -> Dict[str, Any]:
    sid = _to_positive_int(candidate.get("pathbank_species_id") or candidate.get("species_id"))
    name = _canonical_name(str(candidate.get("name") or candidate.get("common_name") or ""))
    ref: Dict[str, Any] = {
        "name": name,
        "source": source,
        "status": "matched",
        "confidence": float(candidate.get("confidence", 1.0) or 1.0),
        "chosen_rule": chosen_rule,
    }
    if sid is not None:
        ref["pathbank_species_id"] = sid
        ref["species_id"] = sid
    taxonomy_id = _canonical_name(str(candidate.get("taxonomy_id") or ""))
    if taxonomy_id:
        ref["taxonomy_id"] = taxonomy_id
    common_name = _canonical_name(str(candidate.get("common_name") or ""))
    if common_name:
        ref["common_name"] = common_name
    return ref


def _novel_species_ref(hint: Dict[str, Any], *, source: str, reason: str = "no_db_match") -> Dict[str, Any]:
    name = _canonical_name(str(hint.get("name") or "Unknown species")) or "Unknown species"
    ref: Dict[str, Any] = {
        "name": name,
        "source": source,
        "status": "novel",
        "reason": reason,
        "confidence": float(hint.get("confidence", 0.0) or 0.0),
    }
    sid = _to_positive_int(hint.get("pathbank_species_id") or hint.get("species_id"))
    if sid is not None:
        ref["pathbank_species_id"] = sid
        ref["species_id"] = sid
    taxonomy_id = _canonical_name(str(hint.get("taxonomy_id") or ""))
    if taxonomy_id:
        ref["taxonomy_id"] = taxonomy_id
    return ref


def _resolve_species_hint(
    hint: Dict[str, Any],
    *,
    source: str,
    db: Optional[PathBankDbResolver],
) -> Optional[Dict[str, Any]]:
    name = _canonical_name(str(hint.get("name") or ""))
    taxonomy_id = _canonical_name(str(hint.get("taxonomy_id") or ""))
    pathbank_species_id = _to_positive_int(hint.get("pathbank_species_id") or hint.get("species_id"))
    if not name and not taxonomy_id and pathbank_species_id is None:
        return None

    if pathbank_species_id is not None:
        if db and db.available():
            result = db.find_species_by_pathbank_id(pathbank_species_id)
            if result.get("status") == "mapped" and _safe_list(result.get("candidates")):
                return _species_ref_from_candidate(
                    _safe_list(result.get("candidates"))[0],
                    source=source,
                    chosen_rule=str(result.get("chosen_rule") or "pathbank_species_id"),
                )
        if name:
            ref = _novel_species_ref({**hint, "pathbank_species_id": pathbank_species_id}, source=source, reason="explicit_species_id_unverified")
            ref["status"] = "matched"
            ref["confidence"] = max(float(ref.get("confidence", 0.0)), 0.9)
            return ref

    if db and db.available() and (name or taxonomy_id):
        result = db.find_species(name, taxonomy_id=taxonomy_id or None)
        if result.get("status") == "mapped" and _safe_list(result.get("candidates")):
            return _species_ref_from_candidate(
                _safe_list(result.get("candidates"))[0],
                source=source,
                chosen_rule=str(result.get("chosen_rule") or "species_lookup"),
            )
        return _novel_species_ref(hint, source=source, reason=str(result.get("reason") or "no_db_match"))

    return _novel_species_ref(hint, source=source, reason="db_unavailable" if db is None or not db.available() else "no_db_match")


def _merge_species_record(entities: Dict[str, Any], ref: Dict[str, Any]) -> None:
    species_rows = entities.setdefault("species", [])
    if not isinstance(species_rows, list):
        species_rows = []
        entities["species"] = species_rows
    name = _canonical_name(str(ref.get("name") or ""))
    sid = _to_positive_int(ref.get("pathbank_species_id") or ref.get("species_id"))
    taxonomy_id = _canonical_name(str(ref.get("taxonomy_id") or ""))
    if not name and sid is None:
        return

    target: Optional[Dict[str, Any]] = None
    for row in species_rows:
        if not isinstance(row, dict):
            continue
        row_sid = _species_id_from_row(row)
        row_name = _canonical_name(str(row.get("name") or ""))
        if (sid is not None and row_sid == sid) or (name and _normalize_name(row_name) == _normalize_name(name)):
            target = row
            break

    if target is None:
        target = {"name": name or f"Species {sid}"}
        species_rows.append(target)
    elif name and not _canonical_name(str(target.get("name") or "")):
        target["name"] = name

    if sid is not None:
        target["pathbank_species_id"] = sid
        target["species_id"] = sid
    if taxonomy_id:
        target["taxonomy_id"] = taxonomy_id
    if ref.get("common_name") and not target.get("common_name"):
        target["common_name"] = ref.get("common_name")
    target.setdefault("mapping_meta", {})
    target["mapping_meta"]["species_resolution"] = _compact_species_ref(ref)


def _stamp_entity_species(row: Dict[str, Any], ref: Dict[str, Any]) -> None:
    compact = _compact_species_ref(ref)
    name = str(compact.get("name") or "").strip()
    if name:
        row["species"] = name
        row["species_name"] = name
        row["organism"] = name
    sid = _to_positive_int(compact.get("pathbank_species_id") or compact.get("species_id"))
    if sid is not None:
        row["pathbank_species_id"] = sid
        row["species_id"] = sid
    if compact.get("taxonomy_id"):
        row["taxonomy_id"] = compact["taxonomy_id"]
    row["species_ref"] = compact
    row.setdefault("mapping_meta", {})
    row["mapping_meta"]["species_resolution"] = compact


def _state_species_hints_for_entity(payload: Dict[str, Any], *, entity_type: str, name: str) -> List[Dict[str, Any]]:
    state_by_name = {
        _normalize_name(str(state.get("name") or "")): state
        for state in _safe_list(payload.get("biological_states"))
        if isinstance(state, dict) and str(state.get("name") or "").strip()
    }
    state_names: List[str] = []
    name_norm = _normalize_name(name)
    element_locations = _safe_dict(payload.get("element_locations"))

    if entity_type == "protein":
        for row in _safe_list(element_locations.get("protein_locations")):
            if not isinstance(row, dict):
                continue
            row_name = _canonical_name(str(row.get("protein") or row.get("name") or ""))
            if _normalize_name(row_name) == name_norm:
                state = _canonical_name(str(row.get("biological_state") or ""))
                if state:
                    state_names.append(state)

    processes = _safe_dict(payload.get("processes"))
    for reaction in _safe_list(processes.get("reactions")):
        if not isinstance(reaction, dict):
            continue
        for enzyme in _safe_list(reaction.get("enzymes")):
            if not isinstance(enzyme, dict):
                continue
            key = "protein_complex" if entity_type == "protein_complex" else "protein"
            actor_name = _canonical_name(str(enzyme.get(key) or enzyme.get("name") or ""))
            if _normalize_name(actor_name) == name_norm:
                state = _canonical_name(str(enzyme.get("biological_state") or reaction.get("biological_state") or ""))
                if state:
                    state_names.append(state)
    for transport in _safe_list(processes.get("transports")):
        if not isinstance(transport, dict):
            continue
        for transporter in _safe_list(transport.get("transporters")):
            if not isinstance(transporter, dict):
                continue
            key = "protein_complex" if entity_type == "protein_complex" else "protein"
            actor_name = _canonical_name(str(transporter.get(key) or transporter.get("name") or ""))
            if _normalize_name(actor_name) == name_norm:
                state = _canonical_name(
                    str(
                        transporter.get("biological_state")
                        or transport.get("to_biological_state")
                        or transport.get("from_biological_state")
                        or ""
                    )
                )
                if state:
                    state_names.append(state)

    hints: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for state_name in state_names:
        state = state_by_name.get(_normalize_name(state_name))
        if not state:
            continue
        hint = _species_hint_from_row(state)
        key = f"{_normalize_name(str(hint.get('name') or ''))}:{hint.get('pathbank_species_id') or ''}:{hint.get('taxonomy_id') or ''}"
        if key.strip(":") and key not in seen:
            seen.add(key)
            hints.append(hint)
    if hints:
        return hints

    all_state_hints = []
    for state in state_by_name.values():
        hint = _species_hint_from_row(state)
        if hint.get("name") or hint.get("pathbank_species_id") or hint.get("taxonomy_id"):
            all_state_hints.append(hint)
    unique = {
        f"{_normalize_name(str(h.get('name') or ''))}:{h.get('pathbank_species_id') or ''}:{h.get('taxonomy_id') or ''}": h
        for h in all_state_hints
    }
    return list(unique.values()) if len(unique) == 1 else []


def _single_pathway_species_hint(entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    species_rows = [row for row in _safe_list(entities.get("species")) if isinstance(row, dict)]
    meaningful = [
        row for row in species_rows
        if _species_hint_from_row(row).get("name")
        or _species_hint_from_row(row).get("taxonomy_id")
        or _species_hint_from_row(row).get("pathbank_species_id")
    ]
    if len(meaningful) != 1:
        return None
    return _species_hint_from_row(meaningful[0])


def _infer_species_with_gap_resolver(
    payload: Dict[str, Any],
    *,
    entity_type: str,
    name: str,
    use_llm: bool,
) -> Optional[Dict[str, Any]]:
    if not use_llm:
        return None
    try:
        from t2pw.curation.gap_resolver import infer_entity_species  # pylint: disable=import-outside-toplevel

        result = infer_entity_species(
            payload,
            entity_type=entity_type,
            entity_name=name,
            use_llm=True,
            temperature=0.0,
            max_tokens=450,
        )
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    species_name = _canonical_name(str(result.get("name") or result.get("species") or ""))
    if not species_name:
        return None
    return {
        "name": species_name,
        "taxonomy_id": _canonical_name(str(result.get("taxonomy_id") or "")),
        "confidence": float(result.get("confidence", 0.65) or 0.65),
    }


def hydrate_species_references(
    payload: Dict[str, Any],
    *,
    db: Optional[PathBankDbResolver] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Hydrate protein/protein-complex species before ID mapping.

    Resolution order: explicit entity species, single pathway species,
    biological-state species, LLM-inferred species, then a novel species record.
    """
    entities = _safe_dict(payload.setdefault("entities", {}))
    protein_like_sections = [("proteins", "protein"), ("protein_complexes", "protein_complex")]
    single_pathway_hint = _single_pathway_species_hint(entities)
    report: Dict[str, Any] = {
        "hydrated": 0,
        "matched": 0,
        "novel": 0,
        "unresolved": 0,
        "rows": [],
    }

    for list_key, entity_type in protein_like_sections:
        for idx, row in enumerate(_safe_list(entities.get(list_key))):
            if not isinstance(row, dict):
                continue
            name = _canonical_name(str(row.get("name") or ""))
            if not name:
                continue

            selected_ref: Optional[Dict[str, Any]] = None
            selected_source = ""
            explicit_hint = _species_hint_from_row(row)
            sources: List[Tuple[str, Optional[Dict[str, Any]]]] = [
                ("explicit_entity_species", explicit_hint if any(explicit_hint.values()) else None),
                ("single_pathway_species", single_pathway_hint),
            ]
            state_hints = _state_species_hints_for_entity(payload, entity_type=entity_type, name=name)
            if state_hints:
                sources.append(("biological_state_species", state_hints[0] if len(state_hints) == 1 else None))

            for source, hint in sources:
                if not hint:
                    continue
                selected_ref = _resolve_species_hint(hint, source=source, db=db)
                selected_source = source
                if selected_ref is not None:
                    break

            if selected_ref is None:
                inferred_hint = _infer_species_with_gap_resolver(
                    payload,
                    entity_type=entity_type,
                    name=name,
                    use_llm=use_llm,
                )
                if inferred_hint:
                    selected_ref = _resolve_species_hint(inferred_hint, source="gap_resolver_llm", db=db)
                    selected_source = "gap_resolver_llm"

            if selected_ref is None:
                selected_ref = _novel_species_ref(
                    {"name": "Unknown species", "confidence": 0.0},
                    source="novel_species",
                    reason="no_species_source",
                )
                selected_source = "novel_species"

            _merge_species_record(entities, selected_ref)
            _stamp_entity_species(row, selected_ref)
            status = str(selected_ref.get("status") or "unresolved")
            report["hydrated"] += 1
            if status == "matched":
                report["matched"] += 1
            elif status == "novel":
                report["novel"] += 1
            else:
                report["unresolved"] += 1
            report["rows"].append(
                {
                    "entity_type": entity_type,
                    "name": name,
                    "json_pointer": f"/entities/{list_key}/{idx}",
                    "source": selected_source,
                    "status": status,
                    "species": selected_ref.get("name", ""),
                    "pathbank_species_id": selected_ref.get("pathbank_species_id"),
                    "taxonomy_id": selected_ref.get("taxonomy_id", ""),
                }
            )

    return report


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
    protein_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row_ids = _row_external_ids(
        _safe_dict(protein_row),
        [("uniprot", ("uniprot_id",)), ("gene_name", ("gene",))],
    )
    pathbank_id = _first_row_value(_safe_dict(protein_row), "pathbank_protein_id", "pw_protein_id", "pathwhiz_id")
    base_key = f"{_normalize_name(name)}::{_normalize_name(organism)}::{pathbank_id}::{json.dumps(row_ids, sort_keys=True)}"
    db_key = f"db::{base_key}"
    api_key = f"api::{base_key}"

    if not organism and not row_ids.get("uniprot") and not pathbank_id:
        return _with_resolution(
            {
                "status": "unmapped",
                "reason": "needs_species",
                "provider": "PathBankDB",
                "source": "db",
                "confidence": 0.0,
                "chosen_rule": "",
                "candidates": [],
            },
            "unresolved",
            issue="needs_species",
        )

    if id_source in {"db", "hybrid"}:
        db_result = cache.get("proteins", db_key)
        if db_result is None:
            if db and db.available():
                db_result = db.map_protein_row(_safe_dict(protein_row) or {"name": name}, organism)
            else:
                db_reason = db.last_error if db else "db_not_configured"
                db_result = {
                    "status": "unmapped",
                    "reason": f"db_unavailable:{db_reason}",
                    "provider": "PathBankDB",
                    "source": "db",
                    "candidates": [],
                }
                _with_resolution(db_result, "unresolved", issue="db_unavailable")
            cache.set("proteins", db_key, db_result)
        db_resolution = _safe_dict(db_result.get("resolution")).get("status")
        if db_result.get("status") == "mapped" or id_source == "db" or db_resolution in {"ambiguous", "novel", "unresolved"}:
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
            if api_result.get("status") == "mapped":
                _with_resolution(api_result, "matched", order_step="api_uniprot")
            elif api_result.get("reason") == "ambiguous":
                _with_resolution(api_result, "ambiguous", issue="api_ambiguous", order_step="api_uniprot")
            else:
                _with_resolution(api_result, "unresolved", issue=str(api_result.get("reason") or "api_unmapped"), order_step="api_uniprot")
            cache.set("proteins", api_key, api_result)
        return api_result

    return _with_resolution(
        {"status": "unmapped", "reason": "invalid_id_source", "provider": "none", "source": "none", "candidates": []},
        "unresolved",
        issue="invalid_id_source",
    )


def _map_compound_with_strategy(
    *,
    id_source: str,
    db: Optional[PathBankDbResolver],
    client: HttpClient,
    cache: MappingCache,
    name: str,
    compound_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row_ids = _row_external_ids(
        _safe_dict(compound_row),
        [
            ("hmdb", ("hmdb_id",)),
            ("kegg", ("kegg_id",)),
            ("chebi", ("chebi_id",)),
            ("pubchem", ("pubchem_cid", "pubchem_id")),
            ("cas", ("cas_id", "cas_number")),
            ("pwc_id", ()),
        ],
    )
    pathbank_id = _first_row_value(_safe_dict(compound_row), "pathbank_compound_id", "pw_compound_id", "pathwhiz_id")
    base_key = f"{_normalize_name(name)}::{pathbank_id}::{json.dumps(row_ids, sort_keys=True)}"
    db_key = f"db::{base_key}"
    api_key = f"api::{base_key}"

    if id_source in {"db", "hybrid"}:
        db_result = cache.get("compounds", db_key)
        if db_result is None:
            if db and db.available():
                db_result = db.map_compound_row(_safe_dict(compound_row) or {"name": name})
            else:
                db_reason = db.last_error if db else "db_not_configured"
                db_result = {
                    "status": "unmapped",
                    "reason": f"db_unavailable:{db_reason}",
                    "provider": "PathBankDB",
                    "source": "db",
                    "candidates": [],
                }
                _with_resolution(db_result, "unresolved", issue="db_unavailable")
            cache.set("compounds", db_key, db_result)
        db_resolution = _safe_dict(db_result.get("resolution")).get("status")
        if db_result.get("status") == "mapped" or id_source == "db" or db_resolution in {"ambiguous", "novel"}:
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
            if api_result.get("status") == "mapped":
                _with_resolution(api_result, "matched", order_step="api_external_id")
            elif api_result.get("reason") == "ambiguous":
                _with_resolution(api_result, "ambiguous", issue="api_ambiguous", order_step="api_external_id")
            else:
                _with_resolution(api_result, "unresolved", issue=str(api_result.get("reason") or "api_unmapped"), order_step="api_external_id")
            cache.set("compounds", api_key, api_result)
        return api_result

    return _with_resolution(
        {"status": "unmapped", "reason": "invalid_id_source", "provider": "none", "source": "none", "candidates": []},
        "unresolved",
        issue="invalid_id_source",
    )


def _map_complex_with_strategy(
    *,
    id_source: str,
    db: Optional[PathBankDbResolver],
    cache: MappingCache,
    name: str,
    organism: str,
    complex_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = _safe_dict(complex_row)
    pathbank_id = _first_row_value(row, "pathbank_protein_complex_id", "pathbank_complex_id", "pw_complex_id", "pathwhiz_id")
    component_key = json.dumps(
        [
            {
                "name": _component_name(component),
                "stoichiometry": _component_stoichiometry(component),
                "ids": _component_mapped_ids(component),
            }
            for component in _safe_list(row.get("components"))
        ],
        sort_keys=True,
    )
    base_key = f"{_normalize_name(name)}::{_normalize_name(organism)}::{pathbank_id}::{component_key}"
    db_key = f"db::{base_key}"

    if id_source in {"db", "hybrid"}:
        db_result = cache.get("complexes", db_key)
        if db_result is None:
            if db and db.available():
                db_result = db.map_protein_complex_row(row or {"name": name}, organism)
            else:
                db_reason = db.last_error if db else "db_not_configured"
                db_result = {
                    "status": "unmapped",
                    "reason": f"db_unavailable:{db_reason}",
                    "provider": "PathBankDB",
                    "source": "db",
                    "confidence": 0.0,
                    "chosen_rule": "",
                    "candidates": [],
                    "components": _safe_list(row.get("components")),
                    "issues": [],
                }
                _with_resolution(db_result, "unresolved", issue="db_unavailable")
            cache.set("complexes", db_key, db_result)
        return db_result

    return _with_resolution(
        {
            "status": "unmapped",
            "reason": "complex_db_mapping_disabled",
            "provider": "none",
            "source": "none",
            "confidence": 0.0,
            "chosen_rule": "",
            "candidates": [],
            "components": _safe_list(row.get("components")),
            "issues": [],
        },
        "unresolved",
        issue="complex_db_mapping_disabled",
    )


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

    species_llm_enabled = str(os.getenv("T2PW_SPECIES_LLM", "1")).strip().lower() not in {"0", "false", "no", "off"}
    species_report = hydrate_species_references(mapped, db=db, use_llm=species_llm_enabled)

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
    protein_complexes_mapped = 0
    protein_complexes_ambiguous = 0
    protein_complexes_novel = 0
    protein_complexes_gap_issues = 0

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
        organism = (
            protein.get("organism")
            or protein.get("species")
            or _safe_dict(protein.get("species_ref")).get("name")
            or ""
        ).strip() if isinstance(
            protein.get("organism") or protein.get("species") or _safe_dict(protein.get("species_ref")).get("name"),
            str,
        ) else ""
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
            protein_row=protein,
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
        protein["mapping_meta"]["resolution"] = _safe_dict(result.get("resolution"))

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
            if reason == "ambiguous" or _safe_dict(result.get("resolution")).get("status") == "ambiguous":
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
                "resolution_status": _safe_dict(result.get("resolution")).get("status", ""),
                "resolution_issue": _safe_dict(result.get("resolution")).get("issue", ""),
            }
        )

    for idx, complex_row in enumerate(protein_complexes):
        if not isinstance(complex_row, dict):
            continue
        name = (complex_row.get("name") or "").strip() if isinstance(complex_row.get("name"), str) else ""
        if not name:
            continue
        organism = (
            complex_row.get("organism")
            or complex_row.get("species")
            or _safe_dict(complex_row.get("species_ref")).get("name")
            or global_organism
            or ""
        )
        organism = organism.strip() if isinstance(organism, str) else ""
        result = _map_complex_with_strategy(
            id_source=source_mode,
            db=db,
            cache=cache,
            name=name,
            organism=organism,
            complex_row=complex_row,
        )
        result.setdefault("provider", "PathBankDB" if result.get("source") == "db" else "none")
        result.setdefault("source", "db" if result.get("provider") == "PathBankDB" else "none")
        complex_row.setdefault("mapping_meta", {})
        complex_row["mapping_meta"]["route"] = "complex"
        complex_row["mapping_meta"]["query"] = {"name": name, "organism": organism}
        complex_row["mapping_meta"]["provider"] = str(result.get("provider") or "none")
        complex_row["mapping_meta"]["source"] = str(result.get("source") or "none")
        complex_row["mapping_meta"]["chosen_rule"] = str(result.get("chosen_rule") or "")
        complex_row["mapping_meta"]["confidence"] = float(result.get("confidence", 0.0) or 0.0)
        complex_row["mapping_meta"]["candidates"] = result.get("candidates", [])
        complex_row["mapping_meta"]["resolution"] = _safe_dict(result.get("resolution"))
        complex_row["mapping_meta"]["issues"] = _safe_list(result.get("issues"))
        if _safe_list(result.get("issues")):
            protein_complexes_gap_issues += len(_safe_list(result.get("issues")))
        if result.get("species_id"):
            complex_row["species_id"] = int(result["species_id"])
            complex_row["mapping_meta"]["species_id"] = int(result["species_id"])
        if _safe_list(result.get("components")):
            complex_row["components"] = result["components"]

        if result.get("status") == "mapped":
            protein_complexes_mapped += 1
            if result.get("pathbank_complex_id"):
                complex_row["pathbank_complex_id"] = int(result["pathbank_complex_id"])
                complex_row["mapping_meta"]["pathbank_complex_id"] = int(result["pathbank_complex_id"])
            if result.get("pathbank_protein_complex_id"):
                complex_row["pathbank_protein_complex_id"] = int(result["pathbank_protein_complex_id"])
                complex_row["mapping_meta"]["pathbank_protein_complex_id"] = int(result["pathbank_protein_complex_id"])
            complex_status = "mapped"
            complex_reason = ""
        else:
            resolution_status = str(_safe_dict(result.get("resolution")).get("status") or "")
            if str(result.get("reason") or "") == "ambiguous" or resolution_status == "ambiguous":
                protein_complexes_ambiguous += 1
            if resolution_status == "novel":
                protein_complexes_novel += 1
                if result.get("species_id"):
                    complex_row["species_id"] = int(result["species_id"])
            protein_complexes_skipped += 1
            complex_status = "unmapped"
            complex_reason = str(result.get("reason") or _safe_dict(result.get("resolution")).get("issue") or "complex_unresolved")
        logs.append(
            {
                "entity_type": "protein_complex",
                "name": name,
                "json_pointer": f"/entities/protein_complexes/{idx}",
                "status": complex_status,
                "reason": complex_reason,
                "location": ", ".join(protein_locations.get(name, [])),
                "organism": organism,
                "candidate_count": len(_safe_list(result.get("candidates"))),
                "source": str(result.get("source") or "none"),
                "provider": str(result.get("provider") or "none"),
                "resolution_status": _safe_dict(result.get("resolution")).get("status", ""),
                "resolution_issue": _safe_dict(result.get("resolution")).get("issue", ""),
                "gap_issues": _safe_list(result.get("issues")),
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
                compound_row=compound,
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
                protein_row={"name": name},
            )
            provider = str(result.get("provider") or ("PathBankDB" if result.get("source") == "db" else "UniProt"))
            source = str(result.get("source") or ("db" if provider == "PathBankDB" else "api"))
        else:
            compounds_skipped_as_complex += 1
            result = {
                "status": "unmapped",
                "reason": "routed_to_complex_entity",
                "provider": "PathBankDB",
                "source": "db",
                "candidates": [],
            }
            _with_resolution(result, "unresolved", issue="compound_row_is_complex")

        compound.setdefault("mapping_meta", {})
        compound["mapping_meta"]["query"] = {"name": name}
        compound["mapping_meta"]["route"] = route["route"]
        compound["mapping_meta"]["route_reason"] = route["reason"]
        compound["mapping_meta"]["providers"] = [provider]
        compound["mapping_meta"]["source"] = source
        compound["mapping_meta"]["candidates"] = result.get("candidates", [])
        compound["mapping_meta"]["chosen_rule"] = result.get("chosen_rule", "")
        compound["mapping_meta"]["confidence"] = float(result.get("confidence", 0.0))
        compound["mapping_meta"]["resolution"] = _safe_dict(result.get("resolution"))

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
            if route["route"] == "compound" and (
                reason == "ambiguous" or _safe_dict(result.get("resolution")).get("status") == "ambiguous"
            ):
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
                "resolution_status": _safe_dict(result.get("resolution")).get("status", ""),
                "resolution_issue": _safe_dict(result.get("resolution")).get("issue", ""),
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
        "protein_complexes_mapped": protein_complexes_mapped,
        "protein_complexes_novel": protein_complexes_novel,
        "protein_complexes_ambiguous": protein_complexes_ambiguous,
        "protein_complexes_gap_issues": protein_complexes_gap_issues,
        "protein_complexes_skipped": protein_complexes_skipped,
        "complexes_skipped": compounds_skipped_as_complex + protein_complexes_skipped,
        "species_hydrated": int(species_report.get("hydrated", 0)),
        "species_matched": int(species_report.get("matched", 0)),
        "species_novel": int(species_report.get("novel", 0)),
    }

    report = {"summary": summary, "species": species_report, "entities": logs}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def resolve_mapping_gaps(
    payload: Dict[str, Any],
    mapping_report: Dict[str, Any],
    db: "PathBankDbResolver",
    *,
    global_organism: str = "",
) -> Dict[str, Any]:
    """Re-attempt resolution of every unmapped/ambiguous entity using DB primitives.

    Returns a dict with:
        resolved_count  - how many newly resolved
        total           - how many were attempted
        rows            - per-entity result rows
        patched_payload - deepcopy of payload with mapped_ids stamped in for resolved entities
    """
    from copy import deepcopy

    entity_logs = _safe_list(mapping_report.get("entities"))
    targets = [
        e for e in entity_logs
        if isinstance(e, dict) and (e.get("status") == "unmapped" or e.get("reason") == "ambiguous")
    ]

    patched = deepcopy(payload)
    p_entities = _safe_dict(patched.get("entities"))
    proteins_list = _safe_list(p_entities.get("proteins"))
    compounds_list = _safe_list(p_entities.get("compounds"))
    protein_complexes_list = _safe_list(p_entities.get("protein_complexes"))

    for entry in targets:
        if not isinstance(entry, dict):
            continue
        entry_name = str(entry.get("name") or "").strip()
        entry_type = str(entry.get("entity_type") or "").strip()
        entry_organism = str(entry.get("organism") or global_organism or "").strip()
        if not entry_name or not entry_organism:
            continue
        if entry_type == "protein":
            for ep in proteins_list:
                if isinstance(ep, dict) and ep.get("name") == entry_name and not _species_hint_from_row(ep).get("name"):
                    ep["organism"] = entry_organism
        elif entry_type == "protein_complex":
            for pc in protein_complexes_list:
                if isinstance(pc, dict) and pc.get("name") == entry_name and not _species_hint_from_row(pc).get("name"):
                    pc["organism"] = entry_organism

    species_hydration = hydrate_species_references(patched, db=db, use_llm=False)

    rows: List[Dict[str, Any]] = []
    resolved_count = 0

    for entry in targets:
        name = str(entry.get("name") or "").strip()
        etype = str(entry.get("entity_type") or "").strip()
        organism = str(entry.get("organism") or global_organism or "").strip()
        if not name:
            continue
        if not organism and etype == "protein":
            for ep in proteins_list:
                if isinstance(ep, dict) and ep.get("name") == name:
                    organism = str(ep.get("organism") or ep.get("species") or _safe_dict(ep.get("species_ref")).get("name") or "").strip()
                    break
        if not organism and etype == "protein_complex":
            for pc in protein_complexes_list:
                if isinstance(pc, dict) and pc.get("name") == name:
                    organism = str(pc.get("organism") or pc.get("species") or _safe_dict(pc.get("species_ref")).get("name") or "").strip()
                    break

        result: Dict[str, Any] = {}
        try:
            if etype == "protein":
                result = db.map_protein_by_name_species(name, organism)
            elif etype == "compound":
                result = db.map_compound_by_name(name)
            elif etype == "protein_complex":
                complex_row = next((pc for pc in protein_complexes_list if isinstance(pc, dict) and pc.get("name") == name), None)
                result = db.map_protein_complex_row(_safe_dict(complex_row) or {"name": name}, organism)
            else:
                result = db.map_compound_by_name(name)
        except Exception as exc:  # noqa: BLE001
            result = {"status": "error", "reason": str(exc), "candidates": [], "confidence": 0.0, "chosen_rule": ""}

        resolved = result.get("status") == "mapped"
        if resolved:
            resolved_count += 1

        top_cand = (_safe_list(result.get("candidates")) or [{}])[0]
        rows.append({
            "type": etype,
            "name": name,
            "organism": organism,
            "was_unmapped_reason": entry.get("reason", ""),
            "resolved": resolved,
            "confidence": round(float(result.get("confidence") or 0.0), 4),
            "chosen_rule": result.get("chosen_rule", ""),
            "top_candidate_name": str(top_cand.get("name") or top_cand.get("uniprot") or ""),
            "mapped_ids": _safe_dict(result.get("mapped_ids")) if resolved else {},
        })

        if not resolved:
            continue

        new_ids = _safe_dict(result.get("mapped_ids"))
        if etype == "protein":
            for ep in proteins_list:
                if isinstance(ep, dict) and ep.get("name") == name:
                    ep["mapped_ids"] = _merge_mapped_ids(_safe_dict(ep.get("mapped_ids")), new_ids)
                    ep.setdefault("mapping_meta", {})["db_gap_resolved"] = True
                    ep["mapping_meta"]["confidence"] = result.get("confidence")
                    if result.get("pathbank_protein_id"):
                        ep["pathbank_protein_id"] = int(result["pathbank_protein_id"])
        elif etype == "compound":
            for ec in compounds_list:
                if isinstance(ec, dict) and ec.get("name") == name:
                    ec["mapped_ids"] = _merge_mapped_ids(_safe_dict(ec.get("mapped_ids")), new_ids)
                    ec.setdefault("mapping_meta", {})["db_gap_resolved"] = True
                    ec["mapping_meta"]["confidence"] = result.get("confidence")
                    if result.get("pathbank_compound_id"):
                        ec["pathbank_compound_id"] = int(result["pathbank_compound_id"])
        elif etype == "protein_complex":
            for pc in protein_complexes_list:
                if isinstance(pc, dict) and pc.get("name") == name:
                    pc.setdefault("mapping_meta", {})["db_gap_resolved"] = True
                    pc["mapping_meta"]["confidence"] = result.get("confidence")
                    pc["mapping_meta"]["resolution"] = _safe_dict(result.get("resolution"))
                    pc["mapping_meta"]["issues"] = _safe_list(result.get("issues"))
                    if result.get("species_id"):
                        pc["species_id"] = int(result["species_id"])
                    if result.get("pathbank_complex_id"):
                        pc["pathbank_complex_id"] = int(result["pathbank_complex_id"])
                        pc["mapping_meta"]["pathbank_complex_id"] = int(result["pathbank_complex_id"])
                    if result.get("pathbank_protein_complex_id"):
                        pc["pathbank_protein_complex_id"] = int(result["pathbank_protein_complex_id"])
                        pc["mapping_meta"]["pathbank_protein_complex_id"] = int(result["pathbank_protein_complex_id"])
                    if _safe_list(result.get("components")):
                        pc["components"] = result["components"]

    return {
        "resolved_count": resolved_count,
        "total": len(rows),
        "rows": rows,
        "species_hydration": species_hydration,
        "patched_payload": patched,
    }


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
