"""Bounded, cached identity evidence for the protein identity ladder (D-003).

The defect this exists for, in ``runs_verify/2026-08-04_1306/papers/
PMC12452463/research/final_mapped.json``: the protein ``Fur`` reached the ladder
carrying ``uniprot:P0A9A9`` -- the *correct* accession for the ferric uptake
regulator of *Escherichia coli*. Rung 1 passed. Rung 2 could not run, because
the candidate pool held ten iron-sulfur proteins and P0A9A9 was never among
them, so the verdict read ``candidate_evidence:
no_candidate_describes_the_shipped_identifier`` -> ``identity_evidence_missing``
and the row shipped ``mapped_ids: {}``.

The ladder was right to fail closed; it genuinely had no evidence. What it
lacked was a way to go and get some. This module is that way and nothing more:
it supplies one candidate row describing the shipped accession, and **every
existing rung then runs unchanged** -- species, name/gene, score, margin. An
accession whose record says "Homo sapiens" for an *E. coli* entity is refuted
exactly as before.

It does not fetch and it does not cache. ``_fetch_uniprot_enrichment``
(``enrich_entities.py:507``) and ``EnrichmentCache`` (``:208``) already do both,
against the same ``uniprot:<ACC>`` key ``enrich_entities`` writes (``:1745``,
``:1967``) and ``batch/runner.py:87`` snapshots; ``map_protein_by_ids``
(``map_ids.py:1822``) already does the accession-keyed database lookup. All
three are called here, none reimplemented. The ``enrich_entities`` imports are
deferred to call time because ``enrich_entities`` imports ``HttpClient`` from
``map_ids`` (``:125``) and ``map_ids`` calls into this module -- a module-level
import would close that cycle.

Three boundaries D-003 draws, and how they hold:

* **Pre-freeze only.** The one consumer is the identity ladder, which runs
  during mapping. No exporter imports this module and a test asserts that.
* **Not a hidden live dependency.** Nothing is consulted unless a source is
  installed, and no source fetches unless built with ``allow_network=True``.
  With neither, the answer is ``not_evaluated`` and behaviour is what it was --
  which is what lets saved canonical JSON re-export with every resolver off.
* **A lookup failure is never a refutation.** This module returns ``ok``,
  ``unavailable`` or ``not_evaluated``; it cannot return ``rejected``. Only a
  rung judging real content may say that. An HTTP 404 is ``unavailable`` too: a
  missing, deleted or demerged entry is an absence of evidence, not proof.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from t2pw.paths import PROJECT_ROOT
from t2pw.pipeline.entity_identity import (
    VERIFICATION_NOT_EVALUATED,
    VERIFICATION_UNAVAILABLE,
)

#: Evidence was found. Deliberately not ``verified``: this module supplies a
#: candidate row; only the ladder's rungs decide whether it verifies.
EVIDENCE_OK = "ok"

#: Score for a candidate built from an exact-accession retrieval -- the
#: resolver's own value for the same class of evidence (``map_protein_by_ids``
#: scores an exact ``uniprot_id`` column hit 1.0, ``map_ids.py:1877``). It is
#: not a reward for being well formed: a well-formed accession nothing describes
#: yields no candidate at all, and one whose record names another organism or
#: another molecule is still refused by rungs 3 and 4.
EXACT_ACCESSION_MATCH_SCORE = 1.0

#: Absent/``0``/``off`` -> no source, no I/O, base behaviour. ``1``/``on``/
#: ``cache`` -> the on-disk enrichment cache only. ``network`` -> additionally
#: allows one relocated ``_fetch_uniprot_enrichment`` per uncached accession.
ENV_FLAG = "T2PW_IDENTITY_EVIDENCE"
ENV_CACHE_PATH = "T2PW_ENRICHMENT_CACHE"

_MODE_OFF = "off"
_MODE_CACHE = "cache"
_MODE_NETWORK = "network"
_OFF_VALUES = frozenset({"", "0", "off", "false", "no", "none"})
_NETWORK_VALUES = frozenset({"network", "live", "fetch", "2"})

_INSTALLED: ContextVar[Optional[Any]] = ContextVar("t2pw_identity_evidence_source", default=None)
_DEFAULT_LOCK = threading.RLock()
_DEFAULT_CACHE: Dict[str, Any] = {}


def evidence_record(
    status: str,
    *,
    accession: str = "",
    lookup: str = "",
    detail: str = "",
    candidate: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    cached: Optional[bool] = None,
) -> Dict[str, Any]:
    """One evidence answer, in the shape the identity verdict records.

    ``candidate`` is deep-copied in and out (:func:`evidence_candidate`), so the
    row a ladder judges is never the object a cache holds. ``MappingProxyType``
    would be the obvious way to say "immutable" and is the wrong one: every
    consumer in ``map_ids`` guards with ``isinstance(row, dict)``, so a proxy
    would be dropped silently and the hydration would fail as a no-op.
    """

    record: Dict[str, Any] = {
        "status": str(status or VERIFICATION_NOT_EVALUATED),
        "accession": str(accession or ""),
        "lookup": str(lookup or ""),
    }
    if detail:
        record["detail"] = str(detail)
    if cached is not None:
        record["cached"] = bool(cached)
    if provenance:
        record["provenance"] = deepcopy(provenance)
    if candidate:
        record["candidate"] = deepcopy(candidate)
    return record


def evidence_candidate(record: Any) -> Optional[Dict[str, Any]]:
    """The candidate row an ``ok`` record carries, as a fresh dict, else None."""

    if not isinstance(record, dict) or record.get("status") != EVIDENCE_OK:
        return None
    candidate = record.get("candidate")
    return deepcopy(candidate) if isinstance(candidate, dict) and candidate else None


def uniprot_cache_key(accession: Any) -> str:
    """The ``EnrichmentCache`` key, identical to ``enrich_entities.py:1967``.

    Sharing the key is the point: an accession this module fetches is one the
    enrichment pass then finds already cached, and vice versa.
    """

    return f"uniprot:{str(accession or '').strip().upper()}"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def candidate_from_uniprot_record(parsed: Any) -> Dict[str, Any]:
    """A ladder-shaped candidate row from a parsed UniProtKB record.

    Only fields the rungs already read are emitted -- ``uniprot``/``accession``
    (rung 2), ``organism``/``taxonomy_id`` (3), ``protein_name``/``gene_names``
    (4), ``score`` (5). Nothing is invented and nothing renamed.

    ``requested_taxonomy_id`` is deliberately not stamped:
    ``_candidate_species_verdict`` compares taxon ids first when both sides carry
    one, and UniProt answers at strain rank (P0A9A9 is 83333, *E. coli* K-12)
    where payloads ask at species rank (562). Stamping it would turn a correct
    strain-of-species answer into a mismatch; leaving it off lets the existing
    strain-qualification branch resolve it, which is what that branch is for.

    Returns ``{}`` when the record names nothing: a row with no name and no gene
    symbol gives rung 4 nothing to judge, and a candidate that cannot be judged
    only converts one honest failure into another.
    """

    row = parsed if isinstance(parsed, dict) else {}
    accession = _text(row.get("uniprot_id") or row.get("accession")).upper()
    name = _text(row.get("recommended_name"))
    genes = _string_list(row.get("gene_names"))
    if not accession or (not name and not genes):
        return {}
    candidate: Dict[str, Any] = {
        "uniprot": accession,
        "accession": accession,
        "score": EXACT_ACCESSION_MATCH_SCORE,
        "matched_on": "uniprot_accession",
        "identity_evidence": True,
        "evidence_source": "UniProt",
    }
    if name:
        candidate["protein_name"] = name
        candidate["name"] = name
    if genes:
        candidate["gene_names"] = list(genes)
        candidate["gene_name"] = genes[0]
    for key, value in (
        ("organism", _text(row.get("organism"))),
        ("taxonomy_id", _text(row.get("taxonomy_id"))),
    ):
        if value:
            candidate[key] = value
    provenance = row.get("provenance")
    if isinstance(provenance, dict) and provenance:
        candidate["provenance"] = deepcopy(provenance)
    return candidate


def pathbank_row_for_accession(result: Any, accession: str) -> Optional[Dict[str, Any]]:
    """The ``map_protein_by_ids`` candidate whose accession is exactly ours."""

    wanted = _text(accession).casefold()
    rows = result.get("candidates") if isinstance(result, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not wanted:
            continue
        if _text(row.get("uniprot") or row.get("accession")).casefold() == wanted:
            return row
    return None


def pathbank_evidence_is_adequate(row: Any) -> bool:
    """Whether a PathBank row can actually answer the rungs it will face.

    D-003 goes on to UniProt only "if PathBank evidence is inadequate", and this
    is what inadequate means, stated against the rungs rather than taste: rung 3
    needs an organism or a taxon id (a bare ``species_id`` foreign key is the gap
    ``tests/test_db_candidate_species_evidence.py`` documents) and rung 4 needs a
    name or a gene symbol. A row missing either is not evidence.
    """

    candidate = row if isinstance(row, dict) else {}
    named = _text(candidate.get("name")) or _text(candidate.get("protein_name"))
    genes = _text(candidate.get("gene_name")) or _string_list(candidate.get("gene_names"))
    species = _text(candidate.get("organism")) or _text(candidate.get("taxonomy_id"))
    return bool((named or genes) and species)


class UniProtIdentityEvidenceSource:
    """D-003's evidence flow, in order, over collaborators it does not own.

    ``entity accession -> PathBank by exact identifier -> UniProt by exact
    accession if PathBank evidence is inadequate -> cache and persist with
    response provenance -> one candidate``.

    Every collaborator is injected: ``db`` is anything with
    ``map_protein_by_ids``, ``cache`` an ``EnrichmentCache``, ``client`` an
    ``HttpClient``. An absent collaborator makes its own step ``not_evaluated``
    and the flow moves on -- an absent database is not a verdict.
    """

    def __init__(
        self,
        *,
        db: Any = None,
        cache: Any = None,
        cache_path: Optional[Path] = None,
        client: Any = None,
        allow_network: bool = False,
        persist: bool = True,
    ) -> None:
        self.db = db
        self.client = client
        self.allow_network = bool(allow_network)
        self.persist = bool(persist)
        self._cache = cache
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache_lock = threading.RLock()

    def cache(self) -> Any:
        """The ``EnrichmentCache``, opened on first genuine need and never before.

        Opening it reads a file that has been 25.9-39 MB across this project's
        life, so it must not happen because a module was imported or a test
        touched the ladder -- only because an accession actually needs a lookup.
        """

        with self._cache_lock:
            if self._cache is None and self._cache_path is not None:
                from t2pw.mapping.enrich_entities import EnrichmentCache

                self._cache = EnrichmentCache(self._cache_path)
            return self._cache

    def evidence_for(self, accession: Any, *, organism: str = "") -> Dict[str, Any]:
        """One evidence answer for one accession. Never raises, never refutes."""

        acc = _text(accession).upper()
        if not acc:
            return evidence_record(VERIFICATION_NOT_EVALUATED, detail="no_accession")

        pathbank = self._pathbank_evidence(acc, organism)
        if pathbank.get("status") == EVIDENCE_OK:
            return pathbank
        uniprot = self._uniprot_evidence(acc)
        if uniprot.get("status") == EVIDENCE_OK:
            return uniprot

        # Neither produced a record. Say which kind of silence it was: a step
        # ATTEMPTED and failed is ``unavailable``, one never attempted is
        # ``not_evaluated``. Both distinct from ``rejected``, which cannot arise.
        attempted = [r for r in (pathbank, uniprot) if r.get("status") == VERIFICATION_UNAVAILABLE]
        return attempted[-1] if attempted else uniprot

    def _pathbank_evidence(self, accession: str, organism: str) -> Dict[str, Any]:
        lookup = "pathbank_exact_identifier"
        if self.db is None:
            return evidence_record(
                VERIFICATION_NOT_EVALUATED, accession=accession, lookup=lookup,
                detail="no_pathbank_resolver",
            )
        try:
            result = self.db.map_protein_by_ids({"uniprot": accession}, species=organism or None)
        except Exception as exc:  # noqa: BLE001 - a dead DB is not a verdict
            return evidence_record(
                VERIFICATION_UNAVAILABLE, accession=accession, lookup=lookup,
                detail=f"pathbank_lookup_failed:{type(exc).__name__}",
            )
        row = pathbank_row_for_accession(result, accession)
        if row is None:
            return evidence_record(
                VERIFICATION_UNAVAILABLE, accession=accession, lookup=lookup,
                detail="no_pathbank_row_for_accession",
            )
        if not pathbank_evidence_is_adequate(row):
            return evidence_record(
                VERIFICATION_UNAVAILABLE, accession=accession, lookup=lookup,
                detail="pathbank_row_cannot_answer_the_rungs",
            )
        candidate = dict(row)
        candidate.setdefault("accession", accession)
        candidate["identity_evidence"] = True
        candidate["evidence_source"] = "PathBankDB"
        return evidence_record(
            EVIDENCE_OK, accession=accession, lookup=lookup, cached=False, candidate=candidate,
            provenance={"source": "PathBankDB", "query": accession},
        )

    def _uniprot_evidence(self, accession: str) -> Dict[str, Any]:
        lookup = "uniprot_exact_accession"
        cache = self.cache()
        key = uniprot_cache_key(accession)
        record = cache.get("proteins", key) if cache is not None else None
        cached = record is not None

        if record is None:
            if not self.allow_network or self.client is None:
                return evidence_record(
                    VERIFICATION_NOT_EVALUATED, accession=accession, lookup=lookup, cached=False,
                    detail="not_cached_and_network_not_permitted",
                )
            from t2pw.mapping.enrich_entities import _fetch_uniprot_enrichment

            try:
                record = _fetch_uniprot_enrichment(self.client, accession)
            except Exception as exc:  # noqa: BLE001 - an outage is not a verdict
                return evidence_record(
                    VERIFICATION_UNAVAILABLE, accession=accession, lookup=lookup,
                    detail=f"uniprot_fetch_failed:{type(exc).__name__}",
                )
            if cache is not None:
                # Persisted verbatim, provenance and all, under the key the
                # enrichment pass reads -- one fetch serves both.
                cache.set("proteins", key, record)
                if self.persist:
                    try:
                        cache.save()
                    except Exception:  # noqa: BLE001 - an unsaved memo is not a verdict
                        pass

        payload = record if isinstance(record, dict) else {}
        parsed = payload.get("parsed")
        if _text(payload.get("status")).casefold() != EVIDENCE_OK or not isinstance(parsed, dict):
            return evidence_record(
                VERIFICATION_UNAVAILABLE, accession=accession, lookup=lookup, cached=cached,
                detail=_text(payload.get("error_message")) or "uniprot_record_unusable",
                provenance=payload.get("provenance"),
            )
        candidate = candidate_from_uniprot_record(parsed)
        if not candidate:
            return evidence_record(
                VERIFICATION_UNAVAILABLE, accession=accession, lookup=lookup, cached=cached,
                detail="uniprot_record_names_nothing", provenance=parsed.get("provenance"),
            )
        return evidence_record(
            EVIDENCE_OK, accession=accession, lookup=lookup, cached=cached, candidate=candidate,
            provenance=parsed.get("provenance"),
        )


def install_identity_evidence_source(source: Any) -> Any:
    """Install the ambient source for this context. Returns a reset token."""

    return _INSTALLED.set(source)


def reset_identity_evidence_source(token: Any) -> None:
    _INSTALLED.reset(token)


@contextmanager
def identity_evidence_source(source: Any) -> Iterator[Any]:
    """Scope a source to a block, restoring the previous one afterwards."""

    token = install_identity_evidence_source(source)
    try:
        yield source
    finally:
        reset_identity_evidence_source(token)


def _env_mode() -> str:
    raw = str(os.getenv(ENV_FLAG, "") or "").strip().casefold()
    if raw in _OFF_VALUES:
        return _MODE_OFF
    return _MODE_NETWORK if raw in _NETWORK_VALUES else _MODE_CACHE


def default_identity_evidence_source() -> Optional[Any]:
    """The environment's source, or ``None`` when the switch is off (the default).

    Off by default on purpose: it is what keeps the change inert for every caller
    that has not asked for it -- an offline re-export, a test, an exporter -- and
    it is why enabling evidence is a visible decision, not a hidden dependency.

    No ``PathBankDbResolver`` is built here. The ladder's call sites hold no
    database handle, and building one from the ambient environment inside a gate
    is exactly the hidden dependency D-003 refuses; a caller that has a live
    resolver passes it to :class:`UniProtIdentityEvidenceSource` instead.
    """

    mode = _env_mode()
    if mode == _MODE_OFF:
        return None
    with _DEFAULT_LOCK:
        cache_path = Path(
            str(os.getenv(ENV_CACHE_PATH, "") or "").strip()
            or (PROJECT_ROOT / "data" / "enrichment_cache.json")
        )
        signature = f"{mode}|{cache_path}"
        if _DEFAULT_CACHE.get("signature") != signature:
            client = None
            if mode == _MODE_NETWORK:
                from t2pw.mapping.map_ids import HttpClient

                client = HttpClient()
            _DEFAULT_CACHE["signature"] = signature
            _DEFAULT_CACHE["source"] = UniProtIdentityEvidenceSource(
                cache_path=cache_path, client=client, allow_network=mode == _MODE_NETWORK,
            )
        return _DEFAULT_CACHE.get("source")


def identity_evidence_for(accession: Any, *, organism: str = "", source: Any = None) -> Dict[str, Any]:
    """Evidence about one accession, from whichever source is in force.

    Order: the explicit ``source``, then the installed ambient one, then the
    environment's. With none, the answer is ``not_evaluated`` and no I/O of any
    kind has happened.
    """

    acc = _text(accession).upper()
    provider = source or _INSTALLED.get() or default_identity_evidence_source()
    if provider is None:
        return evidence_record(
            VERIFICATION_NOT_EVALUATED, accession=acc, detail="no_identity_evidence_source"
        )
    try:
        record = provider.evidence_for(acc, organism=str(organism or ""))
    except Exception as exc:  # noqa: BLE001 - a broken source is not a verdict
        return evidence_record(
            VERIFICATION_UNAVAILABLE, accession=acc, detail=f"evidence_source_failed:{type(exc).__name__}"
        )
    if isinstance(record, dict):
        status = _text(record.get("status"))
        if status in (EVIDENCE_OK, VERIFICATION_UNAVAILABLE, VERIFICATION_NOT_EVALUATED):
            return record
        # A source reports what it found; it may not report a refutation. Only a
        # rung judging real content may say ``rejected``, so any other status is
        # clamped rather than passed through -- an injected source must not be
        # able to turn an absence of evidence into a biological verdict, which
        # would also suppress the preserved claim D-003 requires.
        clamped = dict(record)
        clamped["status"] = VERIFICATION_NOT_EVALUATED
        clamped["detail"] = f"non_conforming_source_status:{status or 'missing'}"
        return clamped
    return evidence_record(
        VERIFICATION_NOT_EVALUATED, accession=acc, detail="evidence_source_returned_no_record"
    )
