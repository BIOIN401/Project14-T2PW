"""Stage 7 enrichment: best-effort external metadata for mapped proteins and compounds.

Every mapped protein is looked up in UniProt and every mapped compound in up to
three of ChEBI / HMDB / KEGG. Results are memoized in ``data/enrichment_cache.json``
(25.9 MB, 366 protein entries and 742 compound entries as of 2026-07-28) so a
re-run of the same paper -- or a different paper sharing a metabolite -- costs no
network at all.

WHY THIS MODULE IS THREADED (2026-07-28)
----------------------------------------
Until 2026-07-28 those lookups ran on the main thread, strictly one after another.
Reconstructing the timeline of run ``runs/2026-07-28_0919``, leg ``PMC12444477``
(strict), from cache mtimes and the LM Studio log: the leg took 49m48s of wall
clock (09:19:08 -> 10:08:57); RAG finished its last embedding at ~09:27:48;
``id_mapping_cache.json`` was written at 09:44:44 and the audit candidates at
09:45:13; and from 09:45 until 10:07:37 the only file the process wrote was
``enrichment_cache.json``. That is roughly 22 of the leg's 50 minutes attributable
to this module.

That leg's ``final_mapped.json`` carries 44 compounds and 22 proteins, which reduce
to 68 *distinct* upstream lookups: 21 UniProt accessions, 17 ChEBI ids, 17 KEGG ids
and 13 HMDB ids. (The frequently quoted "56 compounds / 31 proteins" is the count in
``merged_payload.json``, i.e. before mapping dedupes; enrichment only ever sees the
44/22 in ``final_mapped.json``.) Nothing orders those 68 lookups against each other:
each is an independent read of a different public database and the merge that
consumes them is a pure function of the fetched blobs. The work is embarrassingly
parallel; it was serial only because nobody had made it otherwise.

Measured single-request latency on 2026-07-28 against the exact URLs this module
uses (one probe each, from the dev box):

    ChEBI    https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity
             -> HTTP 500 in 0.99s. ``HttpClient`` treats 5xx as retryable, so every
                ChEBI id burns 3 attempts plus 0.6s + 1.2s of backoff ~= 4.8s and
                yields nothing. All 230 cached ChEBI entries have status "error"
                with ``request_failed`` -- see the note in ``_fetch_chebi_enrichment``.
    KEGG     https://rest.kegg.jp/get/cpd:C00002        -> HTTP 200 in 1.79s
    UniProt  https://rest.uniprot.org/uniprotkb/P19367.json -> HTTP 200 in 1.10s
    HMDB     https://hmdb.ca/metabolites/HMDB0000538    -> HTTP 403 in 0.35s
                (197 of the 201 cached HMDB entries are HTTP 403; hmdb.ca refuses
                the ``Project14-T2PW-IDMapper/1.0`` User-Agent.)

HOW THE CONCURRENCY IS STRUCTURED
---------------------------------
``enrich_payload`` now runs in three phases instead of one:

  1. PLAN (serial, no network). ``_plan_enrichment_fetches`` walks the same
     protein/compound rows with the same predicates the merge pass uses and returns
     the de-duplicated, first-encounter-ordered list of cache *misses*.
  2. PREFETCH (concurrent). ``_prefetch_enrichment_sources`` runs those misses
     through one bounded ``ThreadPoolExecutor`` per service (see ``_SERVICE_LANES``)
     and returns a plain dict keyed by ``(cache section, cache key)``.
  3. MERGE (serial, unchanged). The original two loops run exactly as before, except
     that on a cache miss they take the already-fetched blob out of the phase-2 dict
     instead of calling the network. If a key is somehow absent from that dict they
     fall through to the original inline fetch, so a planner/merge divergence can
     only ever cost a request -- never change a result.

Four invariants make the concurrent output byte-identical to the serial output:

  * The cache is read and written ONLY from the main thread, in phase 3, in the same
    order as before. Worker threads never touch ``EnrichmentCache``. This is what
    keeps ``summary.cache_hits``, ``summary.api_calls`` and ``calls.*`` identical:
    the *first* row using a key still counts as an API call and later rows using the
    same key still count as cache hits, because the ``cache.set`` that makes them
    hits still happens at exactly the same point in the same serial walk.
  * ``report["entities"]`` is appended to only in phase 3, so its order is unchanged.
  * Cache-key construction (``uniprot:<ACC>``, ``chebi:<CHEBI:n>``, ``hmdb:<HMDBn>``,
    ``kegg:<Cn>``) is untouched; phase 1 and phase 3 both go through
    ``_protein_uniprot_accession`` and ``_compound_candidate_ids``, so the two
    cannot drift apart. This was originally only an intention -- review on
    2026-07-28 found phase 3 re-deriving both predicates inline -- so
    ``test_the_merge_pass_really_calls_the_shared_predicates`` now stubs the two
    helpers to refuse everything and asserts phase 3 fetches nothing, which fails
    the moment either loop grows its own copy again. The drift that matters is
    OVER-planning, because it is the silent one: phase 2 would issue real requests
    that ``summary.api_calls`` (incremented only in phase 3) never counts, so the
    enrichment report would understate the traffic actually sent upstream.
  * Each worker thread gets its own ``HttpClient``/``requests.Session`` via
    ``_ClientPool``. ``requests.Session`` is not documented thread-safe and sharing
    one across a pool is the classic way to get interleaved-response corruption.

``max_workers=1`` restores the literal pre-2026-07-28 behaviour (phases 1 and 2 are
skipped entirely and every fetch happens inline in phase 3). ``tests/test_enrich_entities_concurrency.py``
pins the two paths to identical output.

MEASURED EFFECT
---------------
Replaying that exact leg's ``final_mapped.json`` offline -- same 44 compounds and 22
proteins, same 68 distinct lookups, fetchers stubbed to sleep for the latencies
measured above -- gives:

    serial (max_workers=1) : 139.7s   api_calls=68  {uniprot:21, chebi:17, hmdb:13, kegg:17}
    concurrent (default)   :  29.1s   api_calls=68  {uniprot:21, chebi:17, hmdb:13, kegg:17}
    -> 4.81x, 110.7s saved on one leg

and the enriched payload, the enrichment report and the written cache file compare
byte-for-byte equal between the two runs (no timestamp scrubbing needed, because the
counts, the ordering and the provenance are all produced by the serial phase 3).
``cache_hits`` is 9 in both. The ceiling is ChEBI: at ~4.8s per dead id and three
workers, 17 ids take ~29s and nothing else in the module takes that long, so fixing
the ChEBI endpoint would be worth more than widening any lane.

The lane objects are built fresh per ``enrich_payload`` call rather than being module
singletons, so the pacing floor applies within one paper's ~68 lookups. That is the
unit of work the pipeline actually issues (``streamlit_app`` calls ``run_enrichment``
once per leg), and per-call state keeps the tests free of the cross-test sleep
accumulation a global pacer would cause.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from xml.etree import ElementTree

from t2pw.mapping.map_ids import HttpClient
from t2pw.pipeline.entity_identity import is_pathbank_unknown_protein


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dedupe_preserve(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for value in values:
        text = _canonical(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _clean_list_of_strings(values: Any) -> List[str]:
    return _dedupe_preserve([_canonical(v) for v in _safe_list(values)])


def _split_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return _clean_list_of_strings(value)
    text = _canonical(value)
    if not text:
        return []
    chunks = re.split(r"[;,|]", text)
    return _dedupe_preserve([_canonical(chunk) for chunk in chunks if _canonical(chunk)])


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(str(value).strip())
    except Exception:  # noqa: BLE001
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip())
    except Exception:  # noqa: BLE001
        return None


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _canonical(value)
        if text:
            return text
    return ""


def _first_list_item(values: Any) -> str:
    for value in _safe_list(values):
        text = _canonical(value)
        if text:
            return text
    return ""


class EnrichmentCache:
    """The on-disk memo of every external lookup this module has ever made.

    ``data/enrichment_cache.json`` was 25.9 MB on 2026-07-28 and is shared by every
    run: the overnight batch runner even snapshots it per run
    (``batch/runner.py:SNAPSHOT_FILES``). Losing or truncating it does not lose
    scientific data, but it does cost every subsequent paper a full round of
    UniProt/ChEBI/KEGG/HMDB traffic, which is precisely the traffic this module was
    made concurrent to avoid. Two pieces of hardening therefore live here, both
    added alongside the 2026-07-28 concurrency work:

    * ``_lock`` guards the mutating accessors. Under the shipped design phase 2
      (prefetch) never touches the cache -- all ``get``/``set`` happen on the main
      thread in phase 3 -- so the lock is not load-bearing today. It exists because
      ``set`` is a ``setdefault``-then-assign pair that is *not* atomic under the
      GIL, so the first future caller who does decide to fill the cache from a
      worker thread would otherwise lose entries silently rather than loudly.
    * ``save`` writes to a sibling temp file and ``os.replace``s it into position.
      The previous ``write_text`` truncated the real 25.9 MB file first and then
      streamed 25.9 MB into it; a crash, a Ctrl-C, or two legs of a batch finishing
      at once left a half-written JSON document that the ``except Exception: pass``
      in ``__init__`` silently swallows as "no cache at all".
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.data: Dict[str, Dict[str, Any]] = {
            "proteins": {},
            "compounds": {},
        }
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data["proteins"] = _safe_dict(loaded.get("proteins"))
                    self.data["compounds"] = _safe_dict(loaded.get("compounds"))
            except Exception:  # noqa: BLE001
                pass

    def get(self, section: str, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = _safe_dict(self.data.get(section)).get(key)
        return item if isinstance(item, dict) else None

    def set(self, section: str, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self.data.setdefault(section, {})
            self.data[section][key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            serialized = json.dumps(self.data, indent=2, ensure_ascii=False)
        # Unique per process AND per thread: two legs of the overnight batch, or a
        # future threaded caller, must never pick the same scratch name.
        tmp_path = self.path.with_name(
            f"{self.path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        try:
            tmp_path.write_text(serialized, encoding="utf-8")
            # os.replace is an atomic rename on POSIX and a MoveFileEx/REPLACE_EXISTING
            # on Windows: readers see either the whole old file or the whole new one.
            # It can still lose to a transient share violation from an indexer or a
            # virus scanner holding the destination open, which is why this retries a
            # couple of times before giving up rather than falling back to a
            # truncating write (a truncating write is the corruption we are avoiding).
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    os.replace(tmp_path, self.path)
                    return
                except OSError as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(0.2 * (attempt + 1))
            raise RuntimeError(
                f"Could not atomically replace enrichment cache at {self.path}: {last_exc}"
            )
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:  # noqa: PERF203 - best effort; a stray temp file is harmless
                pass


def _extract_go_bins(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    go_mf: List[str] = []
    go_bp: List[str] = []
    go_cc: List[str] = []
    for ref in _safe_list(payload.get("uniProtKBCrossReferences")):
        if not isinstance(ref, dict):
            continue
        db = _canonical(ref.get("database")).upper()
        gid = _canonical(ref.get("id"))
        if db != "GO" or not gid:
            continue
        term = ""
        for prop in _safe_list(ref.get("properties")):
            if not isinstance(prop, dict):
                continue
            key = _canonical(prop.get("key")).upper()
            if key in {"GO TERM", "TERM", "GOTERM"}:
                term = _canonical(prop.get("value"))
                break
        if term.startswith("F:"):
            go_mf.append(gid)
        elif term.startswith("P:"):
            go_bp.append(gid)
        elif term.startswith("C:"):
            go_cc.append(gid)
        else:
            go_bp.append(gid)
    return {
        "go_mf_ids": _dedupe_preserve(go_mf),
        "go_bp_ids": _dedupe_preserve(go_bp),
        "go_cellular_component_ids": _dedupe_preserve(go_cc),
    }


def _extract_uniprot_ec_numbers(payload: Dict[str, Any]) -> List[str]:
    ec_ids: List[str] = []

    def _scan(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                k = _canonical(key).casefold()
                if k in {"ecnumber", "ec"}:
                    if isinstance(value, dict):
                        text = _canonical(value.get("value"))
                        if text:
                            ec_ids.append(text)
                    else:
                        text = _canonical(value)
                        if text:
                            ec_ids.append(text)
                else:
                    _scan(value)
        elif isinstance(node, list):
            for item in node:
                _scan(item)

    _scan(payload)
    cleaned: List[str] = []
    for item in ec_ids:
        parts = re.findall(r"\d+\.\d+\.\d+\.[\d-]+", item)
        if parts:
            cleaned.extend(parts)
            continue
        text = _canonical(item)
        if text:
            cleaned.append(text)
    return _dedupe_preserve(cleaned)


def _parse_uniprot_record(accession: str, payload: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    protein_desc = _safe_dict(payload.get("proteinDescription"))
    recommended = _safe_dict(protein_desc.get("recommendedName"))
    recommended_name = _canonical(_safe_dict(recommended.get("fullName")).get("value"))

    alt_names: List[str] = []
    for row in _safe_list(protein_desc.get("alternativeNames")):
        if not isinstance(row, dict):
            continue
        alt_names.append(_canonical(_safe_dict(row.get("fullName")).get("value")))
    alt_names = _dedupe_preserve(alt_names)

    gene_names: List[str] = []
    for gene in _safe_list(payload.get("genes")):
        if not isinstance(gene, dict):
            continue
        gene_names.append(_canonical(_safe_dict(gene.get("geneName")).get("value")))
        for syn in _safe_list(gene.get("synonyms")):
            if isinstance(syn, dict):
                gene_names.append(_canonical(syn.get("value")))
    gene_names = _dedupe_preserve(gene_names)

    organism = _safe_dict(payload.get("organism"))
    organism_name = _canonical(organism.get("scientificName"))
    taxonomy_id = _canonical(organism.get("taxonId"))

    function_texts: List[str] = []
    pathways: List[str] = []
    catalytic_activity: List[str] = []
    subcellular_locations: List[str] = []

    for comment in _safe_list(payload.get("comments")):
        if not isinstance(comment, dict):
            continue
        ctype = _canonical(comment.get("commentType")).upper()
        text_values = [
            _canonical(_safe_dict(text_row).get("value"))
            for text_row in _safe_list(comment.get("texts"))
            if isinstance(text_row, dict)
        ]
        text_values = [t for t in text_values if t]

        if ctype == "FUNCTION":
            function_texts.extend(text_values)
        elif ctype == "PATHWAY":
            pathways.extend(text_values)
        elif ctype == "CATALYTIC ACTIVITY":
            reaction = _safe_dict(comment.get("reaction"))
            catalytic_activity.append(_canonical(reaction.get("name")))
            catalytic_activity.extend(text_values)
        elif ctype == "SUBCELLULAR LOCATION":
            for row in _safe_list(comment.get("subcellularLocations")):
                if not isinstance(row, dict):
                    continue
                location = _canonical(_safe_dict(row.get("location")).get("value"))
                if location:
                    subcellular_locations.append(location)

    function_text = ""
    if function_texts:
        joined = " ".join(function_texts)
        function_text = joined[:2000]

    feature_rows = _safe_list(payload.get("features"))
    tm_count = 0
    signal_peptide = {"present": False, "region": ""}
    for feature in feature_rows:
        if not isinstance(feature, dict):
            continue
        ftype = _canonical(feature.get("type")).casefold()
        if "transmembrane" in ftype:
            tm_count += 1
        if ftype == "signal peptide":
            signal_peptide["present"] = True
            location = _safe_dict(feature.get("location"))
            start = _canonical(_safe_dict(location.get("start")).get("value"))
            end = _canonical(_safe_dict(location.get("end")).get("value"))
            if start and end:
                signal_peptide["region"] = f"{start}-{end}"

    go_bins = _extract_go_bins(payload)
    pdb_ids: List[str] = []
    reactome_ids: List[str] = []
    string_id = ""
    keywords: List[str] = []

    for row in _safe_list(payload.get("keywords")):
        if isinstance(row, dict):
            keywords.append(_canonical(row.get("name")))

    for ref in _safe_list(payload.get("uniProtKBCrossReferences")):
        if not isinstance(ref, dict):
            continue
        db = _canonical(ref.get("database")).upper()
        rid = _canonical(ref.get("id"))
        if not rid:
            continue
        if db == "PDB":
            pdb_ids.append(rid)
        elif db == "REACTOME":
            reactome_ids.append(rid)
        elif db == "STRING" and not string_id:
            string_id = rid

    protein_existence = _first_non_empty(
        _safe_dict(payload.get("proteinExistence")).get("value"),
        payload.get("proteinExistence"),
    )
    sequence_length = _to_int(_safe_dict(payload.get("sequence")).get("length"))

    return {
        "status": "ok",
        "uniprot_id": accession,
        "recommended_name": recommended_name,
        "alternative_names": alt_names,
        "gene_names": gene_names,
        "organism": organism_name,
        "taxonomy_id": taxonomy_id,
        "function_text": function_text,
        "catalytic_activity": _dedupe_preserve([v for v in catalytic_activity if v]),
        "enzyme_commission": _extract_uniprot_ec_numbers(payload),
        "pathways": _dedupe_preserve([v for v in pathways if v]),
        "keywords": _dedupe_preserve([v for v in keywords if v]),
        "subcellular_locations": _dedupe_preserve([v for v in subcellular_locations if v]),
        "go_mf_ids": go_bins["go_mf_ids"],
        "go_bp_ids": go_bins["go_bp_ids"],
        "go_cellular_component_ids": go_bins["go_cellular_component_ids"],
        "protein_existence": protein_existence,
        "sequence_length": sequence_length if sequence_length is not None else 0,
        "transmembrane_regions_count": tm_count,
        "signal_peptide": signal_peptide,
        "pdb_ids": _dedupe_preserve(pdb_ids),
        "reactome_ids": _dedupe_preserve(reactome_ids),
        "string_id": string_id,
        "provenance": {
            "source": "UniProt",
            "query": accession,
            "source_url": source_url,
            "retrieved_at": _utc_now_iso(),
        },
    }


def _fetch_uniprot_enrichment(client: HttpClient, uniprot_id: str) -> Dict[str, Any]:
    """One UniProtKB entry fetch, returning both the parsed view and the raw payload.

    NOT BATCHED, DELIBERATELY (2026-07-28). UniProt does offer batch retrieval --
    ``/uniprotkb/search?query=accession:A OR accession:B`` and the ``/stream``
    endpoint -- and 21 accessions in one request would obviously beat 21 requests.
    It was rejected because this function caches ``raw`` (the complete entry JSON)
    verbatim, and that raw blob is what makes ``enrichment_cache.json`` 25.9 MB and
    what ``_parse_uniprot_record`` reads. The search endpoint returns entries wrapped
    in a ``results`` envelope and applies its own default field projection, so the
    per-entry JSON is not guaranteed byte-identical to ``/uniprotkb/{acc}.json``; any
    difference would change ``parsed``, the cached ``raw``, and
    ``provenance.source_url``, which the 2026-07-28 throughput work was explicitly
    not allowed to do. The bounded pool in ``_SERVICE_LANES["uniprot"]`` recovers
    most of the wall clock with none of that exposure.
    """

    accession = _canonical(uniprot_id).upper()
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"request_failed:{exc}",
            "provenance": {
                "source": "UniProt",
                "query": accession,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    if resp.status_code != 200:
        return {
            "status": "error",
            "error_message": f"HTTP {resp.status_code}",
            "provenance": {
                "source": "UniProt",
                "query": accession,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    try:
        payload = _safe_dict(resp.json())
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"json_parse_failed:{exc}",
            "provenance": {
                "source": "UniProt",
                "query": accession,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    parsed = _parse_uniprot_record(accession, payload, url)
    return {
        "status": "ok",
        "parsed": parsed,
        "raw": payload,
    }


def _xml_tag(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _normalize_chebi_id(raw: str) -> str:
    text = _canonical(raw).upper()
    if not text:
        return ""
    if text.startswith("CHEBI:"):
        return text
    if text.startswith("CHEBI"):
        digits = re.sub(r"[^0-9]", "", text)
        return f"CHEBI:{digits}" if digits else text
    if text.isdigit():
        return f"CHEBI:{text}"
    return text


def _fetch_chebi_enrichment(client: HttpClient, chebi_id: str) -> Dict[str, Any]:
    """One ChEBI ``getCompleteEntity`` call, parsed into the merged-compound shape.

    KNOWN BROKEN UPSTREAM, LEFT ALONE DELIBERATELY (observed 2026-07-28). A probe of
    the exact URL below with ``chebiId=CHEBI:15422`` returned HTTP 500 in 0.99s, and
    all 230 ChEBI entries in ``data/enrichment_cache.json`` carry
    ``status="error"`` / ``request_failed`` -- not one is ``ok``. Because
    ``HttpClient`` retries 5xx three times with 0.6s/1.2s backoff, each dead ChEBI id
    costs ~4.8s and returns nothing, so ChEBI is the most expensive per-id service in
    the module while contributing zero fields. Repointing this at the current ChEBI
    API would change enrichment *output*, and the 2026-07-28 concurrency work was
    scoped to throughput only, so the URL is untouched here and the breakage is
    reported separately.

    NOT BATCHED, DELIBERATELY. ChEBI does expose a list-oriented entry point
    (``getCompleteEntityByList``), but the parser below is a flat ``root.iter()``
    tag scrape that takes the first occurrence of each tag name in the whole
    document. Handed a multi-entity response it would silently blend one compound's
    InChI with another's formula -- exactly the class of correctness change this
    change was forbidden to make -- and the recorded ``provenance.source_url`` would
    stop identifying the single entity it describes. Concurrency at
    ``_SERVICE_LANES["chebi"]`` gets the wall-clock win without that risk.
    """

    cid = _normalize_chebi_id(chebi_id)
    if not cid:
        return {"status": "error", "error_message": "missing_chebi_id"}

    url = "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity"
    try:
        resp = client.get(url, params={"chebiId": cid})
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"request_failed:{exc}",
            "provenance": {"source": "ChEBI", "id": cid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }
    if resp.status_code != 200:
        status = "not_found" if resp.status_code in {400, 404} else "error"
        return {
            "status": status,
            "error_message": f"HTTP {resp.status_code}",
            "provenance": {"source": "ChEBI", "id": cid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }
    try:
        root = ElementTree.fromstring(resp.text)
    except ElementTree.ParseError as exc:
        return {
            "status": "error",
            "error_message": f"xml_parse_error:{exc}",
            "provenance": {"source": "ChEBI", "id": cid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }

    tag_values: Dict[str, List[str]] = {}
    for node in root.iter():
        text = _canonical(node.text)
        if not text:
            continue
        tag_values.setdefault(_xml_tag(node.tag).casefold(), []).append(text)

    primary_name = _first_non_empty(
        (_safe_list(tag_values.get("chebiasciiname")) or [""])[0],
        (_safe_list(tag_values.get("chebiname")) or [""])[0],
    )
    synonyms = _dedupe_preserve(_safe_list(tag_values.get("synonyms")) + _safe_list(tag_values.get("synonym")))
    inchi = _first_non_empty((_safe_list(tag_values.get("inchi")) or [""])[0])
    inchikey = _first_non_empty((_safe_list(tag_values.get("inchikey")) or [""])[0])
    smiles = _first_non_empty((_safe_list(tag_values.get("smiles")) or [""])[0])
    formula = _first_non_empty(
        (_safe_list(tag_values.get("formula")) or [""])[0],
        (_safe_list(tag_values.get("chemicalformula")) or [""])[0],
    )
    charge = _to_int((_safe_list(tag_values.get("charge")) or [""])[0])
    mass_mono = _to_float((_safe_list(tag_values.get("monoisotopicmass")) or [""])[0])
    mass_avg = _to_float((_safe_list(tag_values.get("mass")) or [""])[0])

    roles: List[str] = []
    ontology_parents: List[str] = []
    for node in root.iter():
        if _xml_tag(node.tag).casefold() != "ontologyparent":
            continue
        row = {_xml_tag(ch.tag).casefold(): _canonical(ch.text) for ch in list(node)}
        pid = _canonical(row.get("chebiid"))
        pname = _canonical(row.get("chebiname"))
        ptype = _canonical(row.get("type")).casefold()
        if pid or pname:
            ontology_parents.append(_first_non_empty(f"{pid}:{pname}" if pid and pname else "", pid, pname))
        if "role" in ptype and pname:
            roles.append(pname)

    cross_refs: Dict[str, str] = {"chebi_id": cid}
    for node in root.iter():
        if _xml_tag(node.tag).casefold() != "databaselink":
            continue
        db_name = ""
        accession = ""
        for child in list(node):
            ctag = _xml_tag(child.tag).casefold()
            ctext = _canonical(child.text)
            if ctag in {"type", "dbname"}:
                db_name = ctext.casefold()
            elif ctag in {"data", "accessionnumber"}:
                accession = ctext
        if not db_name or not accession:
            continue
        if "kegg" in db_name:
            cross_refs.setdefault("kegg_id", accession)
        elif "hmdb" in db_name:
            cross_refs.setdefault("hmdb_id", accession)
        elif "drugbank" in db_name:
            cross_refs.setdefault("drugbank_id", accession)
        elif "pubchem" in db_name:
            cross_refs.setdefault("pubchem_id", accession)

    return {
        "status": "ok",
        "source": "ChEBI",
        "id": cid,
        "primary_name": primary_name,
        "synonyms": synonyms,
        "inchi": inchi,
        "inchikey": inchikey,
        "smiles": smiles,
        "formula": formula,
        "charge": charge if charge is not None else 0,
        "mass_monoisotopic": mass_mono if mass_mono is not None else 0.0,
        "mass_average": mass_avg if mass_avg is not None else 0.0,
        "chebi_roles": _dedupe_preserve(roles),
        "ontology_parents": _dedupe_preserve(ontology_parents),
        "cross_references": cross_refs,
        "provenance": {
            "source": "ChEBI",
            "id": cid,
            "source_url": url,
            "retrieved_at": _utc_now_iso(),
            "status": "ok",
        },
    }


def _normalize_kegg_id(raw: str) -> str:
    text = _canonical(raw).upper()
    if not text:
        return ""
    return text.replace("CPD:", "")


def _parse_kegg_entry_text(text: str) -> Dict[str, Any]:
    rows: Dict[str, List[str]] = {}
    current_key = ""
    for line in text.splitlines():
        key = line[:12].strip()
        value = _canonical(line[12:])
        if key:
            current_key = key
            rows.setdefault(key, []).append(value)
        elif current_key:
            rows.setdefault(current_key, []).append(value)

    names: List[str] = []
    for row in rows.get("NAME", []):
        names.extend([_canonical(part) for part in row.split(";")])
    names = [n for n in names if n]

    pathways: List[str] = []
    for row in rows.get("PATHWAY", []):
        parts = row.split(" ", 1)
        pathways.append(parts[1].strip() if len(parts) > 1 else row)

    kegg_class = _dedupe_preserve(rows.get("CLASS", []) + rows.get("BRITE", []))
    exact_mass = _to_float((rows.get("EXACT_MASS", [""])[0] if rows.get("EXACT_MASS") else ""))
    avg_mass = _to_float((rows.get("MOL_WEIGHT", [""])[0] if rows.get("MOL_WEIGHT") else ""))
    formula = _canonical(rows.get("FORMULA", [""])[0] if rows.get("FORMULA") else "")

    cross_refs: Dict[str, str] = {}
    for row in rows.get("DBLINKS", []):
        if ":" not in row:
            continue
        key, value = row.split(":", 1)
        db = _canonical(key).casefold()
        vid = _canonical(value).split(" ")[0]
        if "drugbank" in db:
            cross_refs.setdefault("drugbank_id", vid)
        elif "pubchem" in db:
            cross_refs.setdefault("pubchem_id", vid)
        elif "chebi" in db:
            cross_refs.setdefault("chebi_id", vid)
        elif "hmdb" in db:
            cross_refs.setdefault("hmdb_id", vid)

    return {
        "primary_name": names[0] if names else "",
        "synonyms": _dedupe_preserve(names),
        "formula": formula,
        "mass_monoisotopic": exact_mass if exact_mass is not None else 0.0,
        "mass_average": avg_mass if avg_mass is not None else 0.0,
        "pathways": _dedupe_preserve([p for p in pathways if p]),
        "kegg_class": kegg_class,
        "cross_references": cross_refs,
    }


def _fetch_kegg_enrichment(client: HttpClient, kegg_id: str) -> Dict[str, Any]:
    """One KEGG COMPOUND flat-file fetch, parsed into the merged-compound shape.

    NOT BATCHED, DELIBERATELY (2026-07-28). ``rest.kegg.jp/get/`` accepts up to ten
    ``+``-joined entries per request, and KEGG's own etiquette would rather have one
    ten-entry request than ten requests. It was still rejected: KEGG answers a batch
    with a single ``///``-delimited blob and no per-entry status, and a batch
    containing one unknown accession fails as a unit. Today each bad id independently
    records ``status="not_found"`` for exactly that compound while its neighbours
    still record ``ok``; batching would convert that into all-or-nothing, and would
    also force ``provenance.source_url`` to name a URL covering ten unrelated
    compounds. Of the 311 cached KEGG entries every single one is ``ok`` and the
    measured latency is 1.79s, so KEGG is not the bottleneck anyway -- it gets the
    smallest lane (2 in flight, ~3 request starts/second) in ``_SERVICE_LANES``.
    """

    kid = _normalize_kegg_id(kegg_id)
    if not kid:
        return {"status": "error", "error_message": "missing_kegg_id"}
    url = f"https://rest.kegg.jp/get/cpd:{kid}"
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"request_failed:{exc}",
            "provenance": {"source": "KEGG", "id": kid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }
    if resp.status_code != 200 or not _canonical(resp.text):
        status = "not_found" if resp.status_code in {400, 404} else "error"
        return {
            "status": status,
            "error_message": f"HTTP {resp.status_code}",
            "provenance": {"source": "KEGG", "id": kid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }
    parsed = _parse_kegg_entry_text(resp.text)
    return {
        "status": "ok",
        "source": "KEGG",
        "id": kid,
        **parsed,
        "cross_references": {"kegg_id": kid, **_safe_dict(parsed.get("cross_references"))},
        "provenance": {"source": "KEGG", "id": kid, "source_url": url, "retrieved_at": _utc_now_iso(), "status": "ok"},
    }


def _normalize_hmdb_id(raw: str) -> str:
    text = _canonical(raw).upper()
    if not text:
        return ""
    if text.startswith("HMDB"):
        return text
    digits = re.sub(r"[^0-9]", "", text)
    return f"HMDB{digits}" if digits else text


def _fetch_hmdb_json_if_configured(client: HttpClient, hmdb_id: str) -> Optional[Dict[str, Any]]:
    api_url = _canonical(os.getenv("HMDB_API_URL", ""))
    if not api_url:
        return None
    api_key = _canonical(os.getenv("HMDB_API_KEY", ""))
    headers: Dict[str, str] = {"Accept": "application/json"}
    if api_key:
        auth_header = _canonical(os.getenv("HMDB_API_AUTH_HEADER", "X-API-Key")) or "X-API-Key"
        headers[auth_header] = api_key
    try:
        resp = client.get(
            api_url,
            params={
                "id": hmdb_id,
                "hmdb_id": hmdb_id,
                "accession": hmdb_id,
                "query": hmdb_id,
                "q": hmdb_id,
            },
            headers=headers,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return None

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

    if not rows:
        return None

    exact = None
    for row in rows:
        rid = _normalize_hmdb_id(_first_non_empty(row.get("hmdb_id"), row.get("accession"), row.get("id")))
        if rid == hmdb_id:
            exact = row
            break
    return exact if isinstance(exact, dict) else rows[0]


def _fetch_hmdb_html(client: HttpClient, hmdb_id: str) -> Optional[str]:
    url = f"https://hmdb.ca/metabolites/{hmdb_id}"
    try:
        resp = client.get(url)
    except Exception:  # noqa: BLE001
        return None
    if resp.status_code != 200:
        return None
    return resp.text


def _extract_html_field(html: str, header_text: str) -> str:
    pattern = (
        r"<th[^>]*>\s*" + re.escape(header_text) + r"\s*</th>\s*"
        r"<td[^>]*>\s*(.*?)\s*</td>"
    )
    match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    value = re.sub(r"<[^>]+>", " ", match.group(1))
    return _canonical(value)


def _fetch_hmdb_enrichment(client: HttpClient, hmdb_id: str) -> Dict[str, Any]:
    hid = _normalize_hmdb_id(hmdb_id)
    if not hid:
        return {"status": "error", "error_message": "missing_hmdb_id"}

    row = _fetch_hmdb_json_if_configured(client, hid)
    if isinstance(row, dict):
        name = _first_non_empty(row.get("name"), row.get("metabolite_name"))
        synonyms = _split_text_list(row.get("synonyms"))
        pathways: List[str] = []
        for pathway in _safe_list(row.get("pathways")):
            if isinstance(pathway, dict):
                pathways.append(_first_non_empty(pathway.get("name"), pathway.get("pathway_name")))
            else:
                pathways.append(_canonical(pathway))
        pathways = _dedupe_preserve([p for p in pathways if p])
        tissue_locations = _split_text_list(row.get("tissue_locations"))
        biofluid_locations = _split_text_list(row.get("biofluid_locations"))
        cellular_locations = _split_text_list(
            _first_non_empty(row.get("cellular_locations"), row.get("cellular_location"), row.get("subcellular_locations"))
        )
        description = _first_non_empty(row.get("description"), row.get("summary"), row.get("desc"))
        taxonomy_raw = row.get("taxonomy")
        taxonomy: Dict[str, Any] = {}
        if isinstance(taxonomy_raw, dict):
            taxonomy = {k: v for k, v in taxonomy_raw.items() if _canonical(v)}
        else:
            taxonomy = {
                "kingdom": _first_non_empty(row.get("kingdom")),
                "super_class": _first_non_empty(row.get("super_class")),
                "class": _first_non_empty(row.get("class")),
                "sub_class": _first_non_empty(row.get("sub_class")),
                "direct_parent": _first_non_empty(row.get("direct_parent")),
            }
            taxonomy = {k: v for k, v in taxonomy.items() if _canonical(v)}
        cross_refs = {
            "hmdb_id": hid,
            "kegg_id": _first_non_empty(row.get("kegg_id")),
            "chebi_id": _first_non_empty(row.get("chebi_id")),
            "pubchem_id": _first_non_empty(row.get("pubchem_id")),
            "drugbank_id": _first_non_empty(row.get("drugbank_id")),
        }
        for key in ["xrefs", "cross_references", "external_links"]:
            links = row.get(key)
            if isinstance(links, dict):
                for lk, lv in links.items():
                    normalized_key = _canonical(lk).casefold()
                    text = _canonical(lv)
                    if not text:
                        continue
                    if "kegg" in normalized_key:
                        cross_refs.setdefault("kegg_id", text)
                    elif "chebi" in normalized_key:
                        cross_refs.setdefault("chebi_id", text)
                    elif "pubchem" in normalized_key:
                        cross_refs.setdefault("pubchem_id", text)
                    elif "drugbank" in normalized_key:
                        cross_refs.setdefault("drugbank_id", text)
            elif isinstance(links, list):
                for item in links:
                    if not isinstance(item, dict):
                        continue
                    db_name = _canonical(item.get("database")).casefold()
                    db_value = _first_non_empty(item.get("id"), item.get("accession"), item.get("value"))
                    if not db_name or not db_value:
                        continue
                    if "kegg" in db_name:
                        cross_refs.setdefault("kegg_id", db_value)
                    elif "chebi" in db_name:
                        cross_refs.setdefault("chebi_id", db_value)
                    elif "pubchem" in db_name:
                        cross_refs.setdefault("pubchem_id", db_value)
                    elif "drugbank" in db_name:
                        cross_refs.setdefault("drugbank_id", db_value)
        cross_refs = {k: v for k, v in cross_refs.items() if v}
        return {
            "status": "ok",
            "source": "HMDB",
            "id": hid,
            "accession": hid,
            "primary_name": name,
            "synonyms": synonyms,
            "description": description,
            "taxonomy": taxonomy,
            "ontology_classification": taxonomy,
            "inchi": _first_non_empty(row.get("inchi")),
            "inchikey": _first_non_empty(row.get("inchikey")),
            "smiles": _first_non_empty(row.get("smiles")),
            "formula": _first_non_empty(row.get("formula")),
            "charge": _to_int(row.get("charge")) or 0,
            "mass_monoisotopic": _to_float(
                _first_non_empty(row.get("monoisotopic_molecular_weight"), row.get("monoisotopic_mass"))
            )
            or 0.0,
            "mass_average": _to_float(_first_non_empty(row.get("average_molecular_weight"), row.get("molecular_weight")))
            or 0.0,
            "pathways": pathways,
            "cellular_locations": cellular_locations,
            "tissue_locations": tissue_locations,
            "biofluid_locations": biofluid_locations,
            "cross_references": cross_refs,
            "external_links": _safe_dict(row.get("external_links")),
            "provenance": {
                "source": "HMDB",
                "id": hid,
                "source_url": _canonical(os.getenv("HMDB_API_URL", "")) or "configured_hmdb_api",
                "retrieved_at": _utc_now_iso(),
                "status": "ok",
            },
        }

    url = f"https://hmdb.ca/metabolites/{hid}"
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"hmdb_request_failed:{exc}",
            "provenance": {
                "source": "HMDB",
                "id": hid,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    if resp.status_code == 404:
        return {
            "status": "not_found",
            "error_message": "hmdb_not_found",
            "provenance": {
                "source": "HMDB",
                "id": hid,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    if resp.status_code != 200:
        status = "not_found" if 400 <= int(resp.status_code) < 500 else "error"
        return {
            "status": status,
            "error_message": f"HTTP {resp.status_code}",
            "provenance": {
                "source": "HMDB",
                "id": hid,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }

    html = resp.text
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = _canonical(re.sub(r"\s*-\s*HMDB.*$", "", title_match.group(1))) if title_match else ""
    return {
        "status": "ok",
        "source": "HMDB",
        "id": hid,
        "accession": hid,
        "primary_name": title,
        "synonyms": _split_text_list(_extract_html_field(html, "Synonyms")),
        "description": _extract_html_field(html, "Description"),
        "taxonomy": {},
        "ontology_classification": {},
        "inchi": _extract_html_field(html, "InChI Identifier"),
        "inchikey": _extract_html_field(html, "InChI Key"),
        "smiles": _extract_html_field(html, "SMILES"),
        "formula": _extract_html_field(html, "Chemical Formula"),
        "charge": _to_int(_extract_html_field(html, "Formal Charge")) or 0,
        "mass_monoisotopic": _to_float(_extract_html_field(html, "Monoisotopic Molecular Weight")) or 0.0,
        "mass_average": _to_float(_extract_html_field(html, "Average Molecular Weight")) or 0.0,
        "pathways": _split_text_list(_extract_html_field(html, "Pathways")),
        "cellular_locations": _split_text_list(_extract_html_field(html, "Cellular Locations")),
        "tissue_locations": _split_text_list(_extract_html_field(html, "Tissue Locations")),
        "biofluid_locations": _split_text_list(_extract_html_field(html, "Biofluid Locations")),
        "cross_references": {"hmdb_id": hid},
        "external_links": {},
        "provenance": {
            "source": "HMDB",
            "id": hid,
            "source_url": url,
            "retrieved_at": _utc_now_iso(),
            "status": "ok",
        },
    }


def _init_compound_enrichment(mapped_ids: Dict[str, Any]) -> Dict[str, Any]:
    cross_refs = {}
    for key, value in _safe_dict(mapped_ids).items():
        text = _canonical(value)
        if not text:
            continue
        normalized = key.strip().lower()
        if normalized in {"chebi", "hmdb", "kegg", "pubchem", "drugbank"}:
            cross_refs[f"{normalized}_id"] = text
        else:
            cross_refs[normalized] = text
    return {
        "enrichment_status": "ok",
        "primary_name": "",
        "synonyms": [],
        "description": "",
        "taxonomy": {},
        "ontology_classification": {},
        "inchi": "",
        "inchikey": "",
        "smiles": "",
        "formula": "",
        "charge": 0,
        "mass_monoisotopic": 0.0,
        "mass_average": 0.0,
        "chebi_roles": [],
        "ontology_parents": [],
        "kegg_class": [],
        "pathways": [],
        "cellular_locations": [],
        "tissue_locations": [],
        "biofluid_locations": [],
        "cross_references": cross_refs,
        "kegg": {},
        "chebi": {},
        "hmdb": {},
        "is_generic_class": False,
        "sources": [],
    }


def _is_generic_class_compound(name: str) -> bool:
    norm = _canonical(name).casefold()
    generic_markers = [
        "triacylglycerol",
        "diacylglycerol",
        "monoglyceride",
    ]
    return any(marker in norm for marker in generic_markers)


def _merge_compound_source(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    status = _canonical(source.get("status")).lower() or "error"
    if status not in {"ok", "not_found", "error"}:
        status = "error"
    provenance = _safe_dict(source.get("provenance"))
    source_name = _canonical(source.get("source")) or _canonical(provenance.get("source")) or "unknown"
    source_id = _canonical(source.get("id")) or _canonical(provenance.get("id"))
    source_key = source_name.strip().lower()

    source_payload = deepcopy(source)
    source_payload["status"] = status
    target[source_key] = source_payload

    source_entry = {
        "source": source_name,
        "id": source_id,
        "retrieved_at": _canonical(provenance.get("retrieved_at")) or _utc_now_iso(),
        "status": status,
        "error_if_any": _canonical(source.get("error_message")),
    }
    target["sources"].append(source_entry)

    cross_refs = _safe_dict(target.get("cross_references"))
    for key, value in _safe_dict(source.get("cross_references")).items():
        text = _canonical(value)
        if text and key not in cross_refs:
            cross_refs[key] = text
    target["cross_references"] = cross_refs


def _source_block(target: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    block = _safe_dict(target.get(source_name))
    if _canonical(block.get("status")).lower() != "ok":
        return {}
    return block


def _pick_scalar_from_sources(target: Dict[str, Any], key: str, *, numeric: bool = False) -> Any:
    order = ["chebi", "pubchem", "hmdb", "kegg"]
    for src in order:
        block = _source_block(target, src)
        if not block:
            continue
        value = block.get(key)
        if numeric:
            if isinstance(value, (int, float)) and float(value) != 0.0:
                return value
        else:
            text = _canonical(value)
            if text:
                return text
    return 0.0 if numeric else ""


def _recompute_compound_merged(target: Dict[str, Any]) -> None:
    target["primary_name"] = _pick_scalar_from_sources(target, "primary_name")
    target["description"] = _pick_scalar_from_sources(target, "description")
    target["inchi"] = _pick_scalar_from_sources(target, "inchi")
    target["inchikey"] = _pick_scalar_from_sources(target, "inchikey")
    target["smiles"] = _pick_scalar_from_sources(target, "smiles")
    target["formula"] = _pick_scalar_from_sources(target, "formula")
    target["charge"] = int(_pick_scalar_from_sources(target, "charge", numeric=True) or 0)
    target["mass_monoisotopic"] = float(_pick_scalar_from_sources(target, "mass_monoisotopic", numeric=True) or 0.0)
    target["mass_average"] = float(_pick_scalar_from_sources(target, "mass_average", numeric=True) or 0.0)

    all_synonyms: List[str] = []
    for src in ["chebi", "hmdb", "kegg"]:
        all_synonyms.extend(_clean_list_of_strings(_safe_dict(target.get(src)).get("synonyms")))
    target["synonyms"] = _dedupe_preserve(all_synonyms)

    chebi = _source_block(target, "chebi")
    hmdb = _source_block(target, "hmdb")
    kegg = _source_block(target, "kegg")

    target["chebi_roles"] = _clean_list_of_strings(chebi.get("chebi_roles"))
    target["ontology_parents"] = _clean_list_of_strings(chebi.get("ontology_parents"))
    target["kegg_class"] = _clean_list_of_strings(kegg.get("kegg_class"))
    target["pathways"] = _dedupe_preserve(
        _clean_list_of_strings(kegg.get("pathways")) + _clean_list_of_strings(hmdb.get("pathways"))
    )
    target["cellular_locations"] = _clean_list_of_strings(hmdb.get("cellular_locations"))
    target["tissue_locations"] = _clean_list_of_strings(hmdb.get("tissue_locations"))
    target["biofluid_locations"] = _clean_list_of_strings(hmdb.get("biofluid_locations"))
    target["taxonomy"] = _safe_dict(hmdb.get("taxonomy"))
    target["ontology_classification"] = _safe_dict(hmdb.get("ontology_classification"))

    statuses = [_canonical(row.get("status")).lower() for row in _safe_list(target.get("sources")) if isinstance(row, dict)]
    ok_count = sum(1 for status in statuses if status == "ok")
    non_ok_count = sum(1 for status in statuses if status in {"not_found", "error"})

    if ok_count == 0:
        target["enrichment_status"] = "error"
    elif non_ok_count > 0:
        target["enrichment_status"] = "ok_with_warnings"
    else:
        target["enrichment_status"] = "ok"


def _best_id_for_compound_dump(enrichment: Dict[str, Any]) -> str:
    refs = _safe_dict(enrichment.get("cross_references"))
    for key in ["hmdb_id", "chebi_id", "kegg_id", "pubchem_id", "drugbank_id"]:
        value = _canonical(refs.get(key))
        if value:
            return value
    return "unmapped"


def _first_text_from_containers(containers: List[Dict[str, Any]], aliases: List[str]) -> str:
    for container in containers:
        if not isinstance(container, dict):
            continue
        for alias in aliases:
            text = _canonical(container.get(alias))
            if text:
                return text
    return ""


def _compound_source_containers(compound: Dict[str, Any], enrichment: Dict[str, Any]) -> List[Dict[str, Any]]:
    containers: List[Dict[str, Any]] = [
        compound,
        _safe_dict(compound.get("mapped_ids")),
        enrichment,
        _safe_dict(enrichment.get("cross_references")),
    ]
    for source_name in ["chebi", "hmdb", "kegg", "pubchem"]:
        block = _safe_dict(enrichment.get(source_name))
        if block:
            containers.append(block)
            containers.append(_safe_dict(block.get("cross_references")))
    return containers


def _normalize_promoted_compound_id(key: str, value: str) -> str:
    if key == "hmdb_id":
        return _normalize_hmdb_id(value)
    if key == "kegg_id":
        return _normalize_kegg_id(value)
    if key == "chebi_id":
        return _normalize_chebi_id(value)
    return _canonical(value)


def _collect_entity_element_states(
    payload: Dict[str, Any],
    *,
    location_key: str,
    name_keys: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    locations = _safe_dict(payload.get("element_locations"))
    for row in _safe_list(locations.get(location_key)):
        if not isinstance(row, dict):
            continue
        entity_name = _first_non_empty(*(row.get(key) for key in name_keys))
        if not entity_name:
            continue
        state: Dict[str, Any] = {}
        for src, dst in [
            ("biological_state", "biological_state"),
            ("state", "biological_state"),
            ("biological_state_id", "biological_state_id"),
            ("pathbank_biological_state_id", "pathbank_biological_state_id"),
            ("pw_biological_state_id", "pw_biological_state_id"),
            ("subcellular_location", "subcellular_location"),
            ("species", "species"),
            ("species_id", "species_id"),
            ("pathbank_species_id", "pathbank_species_id"),
        ]:
            value = row.get(src)
            if isinstance(value, (int, float)) and value:
                state[dst] = int(value) if float(value).is_integer() else value
            else:
                text = _canonical(value)
                if text:
                    state[dst] = text
        if state:
            out.setdefault(_canonical(entity_name).casefold(), []).append(state)
    return out


def _merge_state_lists(existing: Any, discovered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    for item in _safe_list(existing) + discovered:
        if not isinstance(item, dict):
            continue
        normalized = {k: v for k, v in item.items() if _canonical(v)}
        if not normalized:
            continue
        key = json.dumps(normalized, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def _promote_compound_fields(compound: Dict[str, Any], element_states_by_name: Dict[str, List[Dict[str, Any]]]) -> None:
    enrichment = _safe_dict(_safe_dict(compound.get("enrichment")).get("compound"))
    containers = _compound_source_containers(compound, enrichment)

    id_aliases = {
        "hmdb_id": ["hmdb_id", "hmdb"],
        "kegg_id": ["kegg_id", "kegg"],
        "chebi_id": ["chebi_id", "chebi"],
        "pubchem_cid": ["pubchem_cid", "pubchem_id", "pubchem"],
        "cas": ["cas", "cas_number"],
        "biocyc_id": ["biocyc_id", "biocyc"],
        "chemspider_id": ["chemspider_id", "chemspider"],
    }
    for out_key, aliases in id_aliases.items():
        value = _first_text_from_containers(containers, aliases)
        value = _normalize_promoted_compound_id(out_key, value)
        if value:
            compound[out_key] = value

    inchi = _first_text_from_containers(containers, ["moldb_inchi", "inchi"])
    inchikey = _first_text_from_containers(containers, ["moldb_inchikey", "inchikey", "inchi_key"])
    if inchi:
        compound["moldb_inchi"] = inchi
    if inchikey:
        compound["moldb_inchikey"] = inchikey

    short_name = _first_non_empty(
        compound.get("short_name"),
        _first_text_from_containers(_safe_list(_safe_dict(compound.get("mapping_meta")).get("candidates")), ["short_name"]),
        enrichment.get("primary_name"),
    )
    if short_name:
        compound["short_name"] = short_name

    synonyms: List[str] = []
    synonyms.extend(_split_text_list(compound.get("synonyms")))
    synonyms.extend(_clean_list_of_strings(enrichment.get("synonyms")))
    for source_name in ["chebi", "hmdb", "kegg"]:
        synonyms.extend(_clean_list_of_strings(_safe_dict(enrichment.get(source_name)).get("synonyms")))
    compound["synonyms"] = _dedupe_preserve(synonyms)

    taxonomy = _safe_dict(enrichment.get("taxonomy")) or _safe_dict(enrichment.get("ontology_classification"))
    compound_class = _first_non_empty(
        compound.get("class"),
        compound.get("compound_class"),
        taxonomy.get("class"),
        taxonomy.get("direct_parent"),
        taxonomy.get("super_class"),
        _first_list_item(enrichment.get("kegg_class")),
        _first_list_item(enrichment.get("ontology_parents")),
    )
    if compound_class:
        compound["class"] = compound_class
        compound["compound_class"] = compound_class

    compound_type = _first_non_empty(
        compound.get("type"),
        compound.get("compound_type"),
        "Compound",
    )
    if compound_type:
        compound["type"] = compound_type
        compound["compound_type"] = compound_type

    name_key = _canonical(compound.get("name")).casefold()
    states = _merge_state_lists(compound.get("element_states"), element_states_by_name.get(name_key, []))
    if states:
        compound["element_states"] = states


def _promote_protein_fields(protein: Dict[str, Any]) -> None:
    enrichment = _safe_dict(_safe_dict(_safe_dict(protein.get("enrichment")).get("protein")).get("uniprot"))
    mapped_ids = _safe_dict(protein.get("mapped_ids"))

    uniprot = _first_non_empty(protein.get("uniprot"), protein.get("uniprot_id"), mapped_ids.get("uniprot"), enrichment.get("uniprot_id"))
    if uniprot:
        protein["uniprot"] = uniprot.upper()

    gene_name = _first_non_empty(
        protein.get("gene_name"),
        protein.get("gene"),
        mapped_ids.get("gene_name"),
        _first_list_item(enrichment.get("gene_names")),
    )
    if gene_name:
        protein["gene_name"] = gene_name

    ec_numbers = _dedupe_preserve(
        _split_text_list(protein.get("ec_number"))
        + _clean_list_of_strings(protein.get("ec_numbers"))
        + _clean_list_of_strings(enrichment.get("enzyme_commission"))
        + _clean_list_of_strings(enrichment.get("ec_numbers"))
    )
    if ec_numbers:
        protein["ec_number"] = ec_numbers[0]
        protein["ec_numbers"] = ec_numbers

    species_id = (
        _to_int(protein.get("species_id"))
        or _to_int(protein.get("pathbank_species_id"))
        or _to_int(_safe_dict(protein.get("species_ref")).get("species_id"))
        or _to_int(_safe_dict(protein.get("species_ref")).get("pathbank_species_id"))
        or _to_int(_safe_dict(protein.get("mapping_meta")).get("species_id"))
    )
    if species_id:
        protein["species_id"] = species_id

    synonyms: List[str] = []
    synonyms.extend(_split_text_list(protein.get("synonyms")))
    synonyms.extend(_clean_list_of_strings(enrichment.get("alternative_names")))
    synonyms.extend(_clean_list_of_strings(enrichment.get("gene_names")))
    recommended_name = _canonical(enrichment.get("recommended_name"))
    if recommended_name and recommended_name != _canonical(protein.get("name")):
        synonyms.append(recommended_name)
    protein["synonyms"] = _dedupe_preserve(synonyms)


def _promote_enriched_fields(payload: Dict[str, Any]) -> None:
    entities = _safe_dict(payload.get("entities"))
    compound_states = _collect_entity_element_states(
        payload,
        location_key="compound_locations",
        name_keys=["compound", "name", "entity", "element"],
    )
    for compound in _safe_list(entities.get("compounds")):
        if isinstance(compound, dict):
            _promote_compound_fields(compound, compound_states)
    for protein in _safe_list(entities.get("proteins")):
        if isinstance(protein, dict):
            _promote_protein_fields(protein)


def _build_enrichment_dump(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    entities = _safe_dict(payload.get("entities"))

    for row in _safe_list(entities.get("compounds")):
        if not isinstance(row, dict):
            continue
        name = _canonical(row.get("name"))
        enrichment = _safe_dict(_safe_dict(row.get("enrichment")).get("compound"))
        if not name or not enrichment:
            continue
        best_id = _best_id_for_compound_dump(enrichment)
        key = f"compound:{name}|{best_id}"
        out[key] = enrichment

    for row in _safe_list(entities.get("proteins")):
        if not isinstance(row, dict):
            continue
        name = _canonical(row.get("name"))
        uniprot = _safe_dict(_safe_dict(_safe_dict(row.get("enrichment")).get("protein")).get("uniprot"))
        if not name or not uniprot:
            continue
        uid = _canonical(uniprot.get("uniprot_id")) or "unmapped"
        key = f"protein:{name}|{uid}"
        out[key] = uniprot

    return out


# ---------------------------------------------------------------------------
# Concurrent prefetch (2026-07-28)
# ---------------------------------------------------------------------------


class _ServiceLane:
    """A per-service concurrency cap plus a floor on the gap between request starts.

    Two separate knobs because they answer two separate questions. ``max_workers``
    bounds how many sockets we hold open against one host at once; it is what stops
    us looking like a small scraping botnet. ``min_interval_seconds`` bounds the
    *rate* of new requests independently of how fast the host answers; without it a
    host that starts replying in 30ms (or replying 403 in 0.35s, which is what
    hmdb.ca does to us today) would let a two-worker lane fire dozens of requests a
    second.

    Every endpoint this module touches is a free public academic service that owes
    us nothing. Getting Project14 rate-limited or IP-banned would cost far more than
    a slow run costs, so where a service's published guidance is vague the number
    here is deliberately on the timid side. ``wait_turn`` reserves the next start
    slot under the lock and sleeps outside it, so N workers spread out rather than
    all waking at once.
    """

    def __init__(self, service: str, max_workers: int, min_interval_seconds: float) -> None:
        self.service = service
        self.max_workers = max(1, int(max_workers))
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait_turn(self) -> None:
        if self.min_interval_seconds <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_start)
            self._next_start = start_at + self.min_interval_seconds
        delay = start_at - now
        if delay > 0:
            time.sleep(delay)


def _build_service_lanes() -> Dict[str, _ServiceLane]:
    """Per-service budgets, justified one service at a time.

    UniProt (4 in flight, >=0.15s apart, so <= ~6.7 request starts/second).
      UniProt publishes a REST API meant for programmatic access and rate-limits it
      per IP, answering 429 with ``Retry-After`` rather than banning; their guidance
      to users is to avoid hammering it and to prefer the batch/stream endpoints for
      bulk work. We are not bulk -- 21 accessions for the 2026-07-28 PMC12444477 leg
      -- and at 1.10s measured latency four in flight finishes 21 accessions in about
      six rounds. Four is the widest lane here because UniProt is the endpoint most
      clearly engineered for programmatic clients and the one actually returning data
      (356 of 366 cached protein entries are ``ok``).

    ChEBI / EBI (3 in flight, >=0.25s apart, so <= 4 request starts/second).
      EMBL-EBI's web-services etiquette asks callers to keep concurrent connections
      to a handful and to back off on errors rather than retry hard; they do not
      publish a precise per-IP number for this endpoint, so we stay low. It matters
      more than usual here because the endpoint is currently answering HTTP 500 to
      every request (see ``_fetch_chebi_enrichment``): a failing service is exactly
      the wrong thing to point a wide pool at, and ``HttpClient`` already multiplies
      each id into three attempts.

    HMDB (2 in flight, >=0.50s apart, so <= 2 request starts/second).
      Deliberately the most restrictive lane, and the one lane that is SLOWER THAN
      SERIAL when measured on its own. Do the arithmetic honestly: hmdb.ca answers
      in 0.35s, so 13 ids issued back to back cost 13 x 0.35 = ~4.55s, while this
      lane's 0.50s pacing floor puts the 13th start at 6.0s and the answer at
      ~6.35s. The pacer, not the network, is the bound here, and that is a choice
      rather than an oversight -- there is no public HMDB API on this path
      (``_fetch_hmdb_enrichment`` falls back to scraping the human-facing
      ``hmdb.ca/metabolites/<id>`` HTML page, a small academic site with no
      published programmatic quota, which already answers our User-Agent with HTTP
      403 for 197 of the 201 ids in the cache), and scrapers get blocked for being
      fast far sooner than APIs do. The ~1.8s this costs is invisible against the
      ~29s ChEBI leg that sets the length of the whole prefetch, so the trade buys
      politeness for nothing. The win for HMDB is that its 13 ids no longer sit
      *behind* the other 55 lookups; HMDB itself got slightly slower, on purpose.

    KEGG (2 in flight, >=0.34s apart, so <= ~3 request starts/second).
      KEGG's REST API is free for academic use under terms that explicitly exclude
      bulk/automated harvesting, and the commonly cited ceiling for the service is
      around three requests per second. Two in flight paced at 0.34s sits right at
      that ceiling and no higher. KEGG is also the healthiest service we call (311
      of 311 cached entries ``ok``), which is a reason to protect the relationship,
      not to lean on it.
    """

    return {
        "uniprot": _ServiceLane("uniprot", max_workers=4, min_interval_seconds=0.15),
        "chebi": _ServiceLane("chebi", max_workers=3, min_interval_seconds=0.25),
        "hmdb": _ServiceLane("hmdb", max_workers=2, min_interval_seconds=0.50),
        "kegg": _ServiceLane("kegg", max_workers=2, min_interval_seconds=0.34),
    }


# Fallback for a service name that somehow reaches the pool without a budget: one
# worker, half a second apart. Strictly slower than the serial path it replaces, so
# an unmapped service can never become an accidental flood.
_FALLBACK_LANE_SETTINGS = (1, 0.5)


class _ClientPool:
    """One ``HttpClient`` -- and therefore one ``requests.Session`` -- per thread.

    ``HttpClient`` wraps a ``requests.Session`` for connection reuse. ``Session`` is
    not documented as thread-safe: its cookie jar and its urllib3 connection pool are
    shared mutable state, and sharing one across a pool is the textbook way to get a
    response delivered to the wrong caller. Since the whole point of the 2026-07-28
    change is that results must be *identical*, every worker gets its own session and
    the main thread gets its own for the serial fallback path. At the widest
    configuration that is 4 + 3 + 2 + 2 = 11 sessions for the lifetime of one
    ``enrich_payload`` call, all closed in its ``finally``.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._created: List[HttpClient] = []

    def get(self) -> HttpClient:
        client = getattr(self._local, "client", None)
        if client is None:
            client = HttpClient()
            self._local.client = client
            with self._lock:
                self._created.append(client)
        return client

    def close(self) -> None:
        with self._lock:
            created = list(self._created)
            self._created.clear()
        for client in created:
            try:
                client.session.close()
            except Exception:  # noqa: BLE001 - closing a socket must never fail a run
                pass


class _FetchTarget(NamedTuple):
    """One upstream lookup the merge pass is going to need and the cache does not have."""

    service: str  # "uniprot" | "chebi" | "hmdb" | "kegg"
    section: str  # EnrichmentCache section: "proteins" | "compounds"
    cache_key: str  # byte-for-byte the key the merge pass will look up
    entity_id: str  # normalized accession handed to the matching _fetch_* function


def _protein_uniprot_accession(protein: Any) -> str:
    """The UniProt accession the merge pass will enrich this protein row with, or "".

    Extracted so phase 1 (planning) and phase 3 (merge) cannot disagree about which
    rows get fetched. The three ways a row drops out are kept in the merge pass's
    order because it needs to distinguish them for its counters, but the *predicate*
    lives here: no name, the PathBank Unknown-protein sentinel (fetching UniProt for
    it is what ``test_unknown_sentinel_is_not_sent_to_uniprot_enrichment`` forbids),
    or no mapped accession.
    """

    if not isinstance(protein, dict):
        return ""
    if not _canonical(protein.get("name")):
        return ""
    if is_pathbank_unknown_protein(protein):
        return ""
    return _canonical(_safe_dict(protein.get("mapped_ids")).get("uniprot")).upper()


def _compound_candidate_ids(compound: Any) -> Dict[str, str]:
    """The ChEBI/HMDB/KEGG ids the merge pass will try for this compound row.

    Returns ``{}`` when the merge pass would skip the row entirely, so phase 1 and
    phase 3 stay in lockstep. Normalization happens here (``CHEBI:15422``,
    ``HMDB0000538``, ``C00002``) because the cache key is built from the normalized
    form and a planner that normalized differently would prefetch a key nobody looks
    up.
    """

    if not isinstance(compound, dict):
        return {}
    if not _canonical(compound.get("name")):
        return {}
    mapped_ids = _safe_dict(compound.get("mapped_ids"))
    candidate_ids = {
        "chebi": _normalize_chebi_id(_canonical(mapped_ids.get("chebi"))),
        "hmdb": _normalize_hmdb_id(_canonical(mapped_ids.get("hmdb"))),
        "kegg": _normalize_kegg_id(_canonical(mapped_ids.get("kegg"))),
    }
    if not any(candidate_ids.values()):
        return {}
    return candidate_ids


def _plan_enrichment_fetches(
    proteins: List[Any],
    compounds: List[Any],
    cache: EnrichmentCache,
) -> List[_FetchTarget]:
    """Phase 1: the de-duplicated list of cache misses, in first-encounter order.

    De-duplication is not an optimization, it is a correctness requirement. In the
    serial code the second protein carrying an accession the first protein already
    fetched counts as a ``cache_hits`` increment, not an ``api_calls`` increment,
    because the first one's ``cache.set`` landed before the second one's
    ``cache.get``. Prefetching a key twice would still produce identical *payload*
    output but would make two network calls where the report claims one. Emitting
    each key once keeps the request count equal to the serial request count exactly.
    """

    targets: List[_FetchTarget] = []
    seen: set = set()

    def _add(service: str, section: str, cache_key: str, entity_id: str) -> None:
        if cache_key in seen:
            return
        seen.add(cache_key)
        if cache.get(section, cache_key) is not None:
            return
        targets.append(
            _FetchTarget(service=service, section=section, cache_key=cache_key, entity_id=entity_id)
        )

    for protein in proteins:
        uniprot_id = _protein_uniprot_accession(protein)
        if not uniprot_id:
            continue
        _add("uniprot", "proteins", f"uniprot:{uniprot_id}", uniprot_id)

    for compound in compounds:
        candidate_ids = _compound_candidate_ids(compound)
        if not candidate_ids:
            continue
        for source_name in ["chebi", "hmdb", "kegg"]:
            source_id = candidate_ids.get(source_name, "")
            if not source_id:
                continue
            _add(source_name, "compounds", f"{source_name}:{source_id}", source_id)

    return targets


def _dispatch_fetch(client: HttpClient, target: _FetchTarget) -> Dict[str, Any]:
    """Call the one ``_fetch_*`` function this target names.

    Written as an if/elif chain over module globals rather than a lookup table built
    at import time so that ``unittest.mock.patch("...._fetch_uniprot_enrichment")``
    still intercepts the call -- ``tests/test_pathbank_unknown_fallback.py`` patches
    exactly that name and asserts it is never called, and a table captured at import
    would hold the unpatched function object forever.
    """

    if target.service == "uniprot":
        return _fetch_uniprot_enrichment(client, target.entity_id)
    if target.service == "chebi":
        return _fetch_chebi_enrichment(client, target.entity_id)
    if target.service == "hmdb":
        return _fetch_hmdb_enrichment(client, target.entity_id)
    return _fetch_kegg_enrichment(client, target.entity_id)


def _prefetch_worker(target: _FetchTarget, lane: _ServiceLane, clients: _ClientPool) -> Any:
    lane.wait_turn()
    return _dispatch_fetch(clients.get(), target)


def _prefetch_enrichment_sources(
    targets: List[_FetchTarget],
    clients: _ClientPool,
    *,
    max_workers: Optional[int] = None,
) -> Dict[Tuple[str, str], Any]:
    """Phase 2: fetch every planned miss concurrently, one bounded pool per service.

    One pool per service rather than one shared pool with a semaphore, because the
    services have wildly different costs and a shared pool lets the slowest one
    starve the rest. On 2026-07-28 a ChEBI id cost ~4.8s (HTTP 500 x3 attempts with
    backoff) while an HMDB id cost 0.35s (HTTP 403, no retry); in a shared pool the
    ChEBI failures would occupy every slot and the cheap lookups would queue behind
    them, which is the serial behaviour we are removing.

    Failures are *not* recorded. The ``_fetch_*`` functions already convert network
    and parse errors into ``{"status": "error", ...}`` dictionaries, so an exception
    escaping one of them is a genuine bug (``_parse_uniprot_record`` walking a shape
    it does not expect, say). Dropping such a key from the result dict makes the
    merge pass call the same fetcher inline, at the same entity, and raise the same
    exception at the same point in the same order the serial code would have -- one
    wasted request in exchange for the crash path staying observationally identical.
    """

    results: Dict[Tuple[str, str], Any] = {}
    if not targets:
        return results

    lanes = _build_service_lanes()
    by_service: Dict[str, List[_FetchTarget]] = {}
    for target in targets:
        by_service.setdefault(target.service, []).append(target)

    executors: List[ThreadPoolExecutor] = []
    futures: List[Tuple[Future, _FetchTarget]] = []
    try:
        for service, items in by_service.items():
            lane = lanes.get(service)
            if lane is None:
                lane = _ServiceLane(service, *_FALLBACK_LANE_SETTINGS)
            workers = min(lane.max_workers, len(items))
            if max_workers is not None:
                workers = min(workers, max(1, int(max_workers)))
            executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix=f"t2pw-enrich-{service}"
            )
            executors.append(executor)
            for target in items:
                futures.append((executor.submit(_prefetch_worker, target, lane, clients), target))

        for future, target in futures:
            try:
                results[(target.section, target.cache_key)] = future.result()
            except Exception:  # noqa: BLE001 - see docstring: merge pass re-raises this
                continue
    finally:
        for executor in executors:
            executor.shutdown(wait=True)

    return results


def enrich_payload(
    payload: Dict[str, Any],
    *,
    cache_path: Path,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Enrich every mapped protein and compound in ``payload``; never mutates the input.

    ``max_workers`` is a plumbing seam, not a product knob, and defaults to ``None``
    meaning "use the per-service budgets in ``_build_service_lanes``". Passing ``1``
    skips the plan and prefetch phases outright and fetches inline exactly where the
    pre-2026-07-28 code did, which is what
    ``tests/test_enrich_entities_concurrency.py`` compares the concurrent output
    against. Any other integer clamps every lane to at most that many workers, for an
    operator who needs to be gentler than the defaults on a shared network.

    Note for anyone looking for a switch to skip enrichment entirely: there is none.
    ``src/t2pw/config.py`` (checked 2026-07-28) defines only ``RESOLUTION_DB_ENV``
    and ``RAG_ENV``/``RAG_DEFAULTS``; it has no enrichment key of any kind, and the
    only caller, ``t2pw/app/streamlit_app.py``, calls ``run_enrichment``
    unconditionally. Whether enrichment should be skippable or deferred is a product
    decision and is deliberately not invented here.
    """

    working = deepcopy(payload)
    cache = EnrichmentCache(cache_path)
    clients = _ClientPool()
    try:
        return _enrich_payload_inner(working, cache, clients, max_workers=max_workers)
    finally:
        clients.close()


def _enrich_payload_inner(
    working: Dict[str, Any],
    cache: EnrichmentCache,
    clients: _ClientPool,
    *,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:

    report: Dict[str, Any] = {
        "summary": {
            "proteins_total": 0,
            "proteins_with_uniprot": 0,
            "proteins_enriched_ok": 0,
            "proteins_enriched_error": 0,
            "proteins_skipped_pathbank_unknown": 0,
            "compounds_total": 0,
            "compounds_with_ids": 0,
            "compounds_enriched_ok": 0,
            "compounds_enriched_ok_with_warnings": 0,
            "compounds_enriched_error": 0,
            "cache_hits": 0,
            "api_calls": 0,
        },
        "calls": {"uniprot": 0, "chebi": 0, "hmdb": 0, "kegg": 0},
        "entities": [],
    }

    entities = _safe_dict(working.get("entities"))
    proteins = _safe_list(entities.get("proteins"))
    compounds = _safe_list(entities.get("compounds"))

    report["summary"]["proteins_total"] = len(
        [row for row in proteins if isinstance(row, dict) and _canonical(row.get("name"))]
    )
    report["summary"]["compounds_total"] = len(
        [row for row in compounds if isinstance(row, dict) and _canonical(row.get("name"))]
    )

    # Phases 1 and 2 (see the module docstring). Everything below this block is the
    # original serial walk; the only thing it does differently is look in
    # ``prefetched`` before reaching for the network. ``max_workers=1`` opts out
    # entirely and leaves an empty dict, which makes every miss fall through to the
    # inline fetch -- i.e. the literal pre-2026-07-28 code path.
    prefetched: Dict[Tuple[str, str], Any] = {}
    if max_workers is None or int(max_workers) > 1:
        prefetched = _prefetch_enrichment_sources(
            _plan_enrichment_fetches(proteins, compounds, cache),
            clients,
            max_workers=max_workers,
        )

    for idx, protein in enumerate(proteins):
        if not isinstance(protein, dict):
            continue
        name = _canonical(protein.get("name"))
        if not name:
            continue
        if is_pathbank_unknown_protein(protein):
            report["summary"]["proteins_skipped_pathbank_unknown"] += 1
            report["entities"].append(
                {
                    "entity_type": "protein",
                    "name": name,
                    "json_pointer": f"/entities/proteins/{idx}",
                    "status": "skipped_pathbank_unknown_sentinel",
                }
            )
            continue
        # Call the SHARED helper rather than re-deriving the accession inline.
        # The two checks above stay where they are because the merge pass has to
        # distinguish their outcomes for its counters and its report rows, and by
        # this point both of the helper's own early returns are provably no-ops --
        # so what it returns here is exactly the accession. Review on 2026-07-28
        # found this line re-implemented as
        # ``_canonical(_safe_dict(protein.get("mapped_ids")).get("uniprot")).upper()``
        # while the module docstring claimed "phase 1 builds keys with the same
        # helpers phase 3 does, so the two cannot drift apart". They agreed on the
        # 2026-07-28 PMC12444477 leg (68 planned == 68 issued), but the stated
        # safety property was not actually held by the code, and the failure mode
        # it guards is silent in the direction that matters: an OVER-planning
        # phase 1 issues real requests to UniProt/ChEBI/HMDB/KEGG that
        # ``report["summary"]["api_calls"]`` -- incremented only here in phase 3 --
        # never counts, so the enrichment report would understate the traffic we
        # actually sent to four free academic services.
        uniprot_id = _protein_uniprot_accession(protein)
        if not uniprot_id:
            continue
        report["summary"]["proteins_with_uniprot"] += 1
        cache_key = f"uniprot:{uniprot_id}"
        cached = cache.get("proteins", cache_key)
        status = "error"
        if cached is not None:
            report["summary"]["cache_hits"] += 1
            enrichment_obj = _safe_dict(cached.get("parsed"))
            status = _canonical(cached.get("status")).lower() or _canonical(enrichment_obj.get("status")).lower()
        else:
            report["summary"]["api_calls"] += 1
            report["calls"]["uniprot"] += 1
            # Membership test, not truthiness: a prefetched value is whatever the
            # fetcher returned, and a legitimately falsy return (or a test double's
            # return) must still count as prefetched or this row would be fetched
            # twice and any call-count assertion would double.
            prefetch_key = ("proteins", cache_key)
            if prefetch_key in prefetched:
                fetched = prefetched[prefetch_key]
            else:
                fetched = _fetch_uniprot_enrichment(clients.get(), uniprot_id)
            cache.set("proteins", cache_key, fetched)
            enrichment_obj = _safe_dict(fetched.get("parsed")) if _canonical(fetched.get("status")).lower() == "ok" else fetched
            status = _canonical(fetched.get("status")).lower()

        protein.setdefault("enrichment", {})
        protein["enrichment"].setdefault("protein", {})
        if status == "ok" and enrichment_obj:
            protein["enrichment"]["protein"]["uniprot"] = enrichment_obj
            report["summary"]["proteins_enriched_ok"] += 1
        else:
            protein["enrichment"]["protein"]["uniprot"] = {
                "status": "error",
                "uniprot_id": uniprot_id,
                "error_message": _canonical(_safe_dict(cached).get("error_message"))
                if cached is not None
                else _canonical(_safe_dict(enrichment_obj).get("error_message")),
                "provenance": {
                    "source": "UniProt",
                    "query": uniprot_id,
                    "retrieved_at": _utc_now_iso(),
                },
            }
            report["summary"]["proteins_enriched_error"] += 1

        report["entities"].append(
            {
                "entity_type": "protein",
                "name": name,
                "json_pointer": f"/entities/proteins/{idx}",
                "status": status or "error",
                "cache_key": cache_key,
            }
        )

    for idx, compound in enumerate(compounds):
        if not isinstance(compound, dict):
            continue
        name = _canonical(compound.get("name"))
        if not name:
            continue
        # Same correction as the protein loop above: the candidate ids come from
        # the SHARED helper instead of being re-normalized inline, so that the
        # normalization phase 1 uses to build a cache key and the normalization
        # phase 3 uses to look one up are the same code, not two copies of it.
        # ``mapped_ids`` is still read separately because ``_init_compound_
        # enrichment`` wants the raw mapping (every id the row carries, including
        # ones no lane fetches), not the three normalized lookup ids.
        candidate_ids = _compound_candidate_ids(compound)
        if not candidate_ids:
            continue
        mapped_ids = _safe_dict(compound.get("mapped_ids"))
        report["summary"]["compounds_with_ids"] += 1
        enrichment = _init_compound_enrichment(mapped_ids)
        enrichment["is_generic_class"] = _is_generic_class_compound(name)

        for source_name in ["chebi", "hmdb", "kegg"]:
            source_id = candidate_ids.get(source_name, "")
            if not source_id:
                continue
            cache_key = f"{source_name}:{source_id}"
            cached = cache.get("compounds", cache_key)
            if cached is not None:
                report["summary"]["cache_hits"] += 1
                source_obj = cached
            else:
                report["summary"]["api_calls"] += 1
                report["calls"][source_name] += 1
                prefetch_key = ("compounds", cache_key)
                if prefetch_key in prefetched:
                    source_obj = prefetched[prefetch_key]
                elif source_name == "chebi":
                    source_obj = _fetch_chebi_enrichment(clients.get(), source_id)
                elif source_name == "hmdb":
                    source_obj = _fetch_hmdb_enrichment(clients.get(), source_id)
                else:
                    source_obj = _fetch_kegg_enrichment(clients.get(), source_id)
                cache.set("compounds", cache_key, source_obj)
            _merge_compound_source(enrichment, source_obj)

        if not enrichment.get("sources"):
            enrichment["sources"] = [
                {
                    "source": "none",
                    "id": "",
                    "retrieved_at": _utc_now_iso(),
                    "status": "error",
                    "error_if_any": "no_supported_compound_ids",
                }
            ]
            enrichment["enrichment_status"] = "error"
        else:
            _recompute_compound_merged(enrichment)

        compound.setdefault("enrichment", {})
        compound["enrichment"]["compound"] = enrichment

        if enrichment["enrichment_status"] == "error":
            report["summary"]["compounds_enriched_error"] += 1
        else:
            report["summary"]["compounds_enriched_ok"] += 1
            if enrichment["enrichment_status"] == "ok_with_warnings":
                report["summary"]["compounds_enriched_ok_with_warnings"] += 1

        report["entities"].append(
            {
                "entity_type": "compound",
                "name": name,
                "json_pointer": f"/entities/compounds/{idx}",
                "status": enrichment["enrichment_status"],
                "cross_references": _safe_dict(enrichment.get("cross_references")),
            }
        )

    _promote_enriched_fields(working)
    cache.save()
    return {"payload": working, "report": report}


def run_enrichment(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    cache_path: Path,
    dump_path: Optional[Path] = None,
    qa_report: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mapped input JSON must be an object.")

    # max_workers is forwarded, not interpreted: see enrich_payload's docstring. The
    # only caller in the tree (t2pw/app/streamlit_app.py) omits it and gets the
    # per-service defaults.
    result = enrich_payload(payload, cache_path=cache_path, max_workers=max_workers)
    enriched_payload = _safe_dict(result.get("payload"))
    report = _safe_dict(result.get("report"))
    output_path.write_text(json.dumps(enriched_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    resolved_dump_path = dump_path or Path("out") / "enrichment_dump.json"
    resolved_dump_path.parent.mkdir(parents=True, exist_ok=True)
    enrichment_dump = _build_enrichment_dump(enriched_payload)
    resolved_dump_path.write_text(json.dumps(enrichment_dump, indent=2, ensure_ascii=False), encoding="utf-8")
    report["artifacts"] = {
        **_safe_dict(report.get("artifacts")),
        "enrichment_dump_path": str(resolved_dump_path),
    }
    if qa_report is not None:
        report["qa_report_summary"] = _safe_dict(qa_report.get("summary"))
        report["qa_flags_count"] = {
            flag_name: len(flag_list)
            for flag_name, flag_list in _safe_dict(qa_report.get("flags")).items()
        }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-effort enrichment for mapped proteins and compounds.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input mapped JSON path")
    parser.add_argument("--out", dest="output_path", default="final.enriched.json", help="Output enriched JSON path")
    parser.add_argument(
        "--report",
        dest="report_path",
        default="enrichment_report.json",
        help="Enrichment report JSON path",
    )
    parser.add_argument(
        "--cache",
        dest="cache_path",
        default="enrichment_cache.json",
        help="Enrichment cache JSON path",
    )
    parser.add_argument(
        "--dump",
        dest="dump_path",
        default="out/enrichment_dump.json",
        help="Consolidated enrichment dump JSON path",
    )
    args = parser.parse_args()
    run_enrichment(
        Path(args.input_path),
        Path(args.output_path),
        Path(args.report_path),
        cache_path=Path(args.cache_path),
        dump_path=Path(args.dump_path),
    )


if __name__ == "__main__":
    main()
