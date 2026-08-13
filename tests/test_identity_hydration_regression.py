"""G9 REGRESSION PROOF for C-033 — the pinned ``Fur`` -> P0A9A9 strip.

**This file is a genuine G9 regression proof, not a new-capability test.** The
behaviour it pins is pre-existing and observable on a committed artifact:
``runs_verify/2026-08-04_1306/papers/PMC12452463/research/final_mapped.json``
ships the protein ``Fur`` with ``mapped_ids: {}``, having stripped
``uniprot:P0A9A9`` -- the *correct* accession for the ferric uptake regulator of
*Escherichia coli*. Its own ``identity_verdict`` records why:

    identifier_resolution : ok
    candidate_evidence    : no_candidate_describes_the_shipped_identifier
    reason                : identity_evidence_missing

The candidate pool held ten iron-sulfur proteins (``sdhB``, ``iscS``, ``frdB``,
...) and P0A9A9 was never among them, so rung 2 had nothing to judge and the
ladder fell closed on a correct answer.

The test replays that row and that pool **out of the committed artifact**, and
asserts on artifact content: the accession must survive into ``mapped_ids``.
At the dispatch base it does not -- the row is stripped exactly as the artifact
records -- and the failure is on the stripped accession, not on an import.

**No network.** The evidence is a cached ``EnrichmentCache`` record under
``uniprot:P0A9A9`` in a tmp file, read by the real cache class. The source is
constructed with ``allow_network=False``, so a fetch is not merely unused, it is
not permitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping.enrich_entities import EnrichmentCache  # noqa: E402
from t2pw.mapping.map_ids import _enforce_shipped_identity_names  # noqa: E402
from t2pw.mapping.uniprot_evidence import (  # noqa: E402
    UniProtIdentityEvidenceSource,
    identity_evidence_source,
)

PINNED_ARTIFACT = (
    ROOT / "runs_verify" / "2026-08-04_1306" / "papers" / "PMC12452463" / "research" / "final_mapped.json"
)

#: The ``EnrichmentCache`` record for P0A9A9, in the exact shape
#: ``_fetch_uniprot_enrichment`` returns and ``enrich_entities`` caches. Field
#: values are UniProtKB's own for this entry (FUR_ECOLI).
CACHED_P0A9A9: Dict[str, Any] = {
    "status": "ok",
    "parsed": {
        "status": "ok",
        "uniprot_id": "P0A9A9",
        "recommended_name": "Ferric uptake regulation protein",
        "alternative_names": ["Ferric uptake regulator"],
        "gene_names": ["fur"],
        "organism": "Escherichia coli (strain K12)",
        "taxonomy_id": "83333",
        "provenance": {
            "source": "UniProt",
            "query": "P0A9A9",
            "source_url": "https://rest.uniprot.org/uniprotkb/P0A9A9.json",
            "retrieved_at": "2026-08-12T00:00:00Z",
        },
    },
    "raw": {},
}


def cached_source(tmp_path: Path, records: Dict[str, Any]) -> UniProtIdentityEvidenceSource:
    """A real ``EnrichmentCache`` over a tmp file. Network is not permitted."""

    cache_file = tmp_path / "enrichment_cache.json"
    cache_file.write_text(json.dumps({"proteins": records, "compounds": {}}), encoding="utf-8")
    return UniProtIdentityEvidenceSource(cache=EnrichmentCache(cache_file), allow_network=False)


def pinned_fur_row_and_result() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reconstruct the Fur row and pool as they were when the strip happened.

    Both halves come out of the committed artifact: ``rejected_mapped_ids`` is
    the accession the row carried, and ``rejected_candidates`` is the pool the
    ladder judged it against. Nothing here is invented.
    """

    payload = json.loads(PINNED_ARTIFACT.read_text(encoding="utf-8"))
    for row in payload["entities"]["proteins"]:
        if row.get("name") == "Fur":
            meta = row["mapping_meta"]
            assert row["mapped_ids"] == {}, "the artifact must still show the strip"
            assert meta["rejected_mapped_ids"] == {"uniprot": "P0A9A9"}
            row["mapped_ids"] = dict(meta["rejected_mapped_ids"])
            result = {
                "status": "unmapped",
                "reason": "novel_protein",
                "provider": "PathBankDB",
                "source": "db",
                "candidates": list(meta["rejected_candidates"]),
            }
            return row, result
    raise AssertionError("Fur is not in the pinned artifact")


def test_the_pinned_fur_accession_survives_the_ladder_with_cached_evidence(tmp_path: Path) -> None:
    """G9 REGRESSION. P0A9A9 must reach ``mapped_ids``; at base it is stripped."""

    row, result = pinned_fur_row_and_result()
    pool_accessions = {c.get("uniprot") for c in result["candidates"]}
    assert "P0A9A9" not in pool_accessions, "the defect requires a pool that cannot describe it"

    with identity_evidence_source(cached_source(tmp_path, {"uniprot:P0A9A9": CACHED_P0A9A9})):
        report = _enforce_shipped_identity_names(
            row, result, kind="protein", entity_name="Fur", organism="Escherichia coli"
        )

    # The artifact-content assertion. This is the whole regression.
    assert row["mapped_ids"] == {"uniprot": "P0A9A9"}
    assert report["rejected"] is False

    # ... and it survived because every rung passed on the hydrated record, not
    # because a rung was relaxed. Each of these is an affirmative pass.
    checks = report["identity_verdict"]["checks"]
    assert checks["candidate_evidence"] == "identity_evidence_candidate"
    assert checks["species"] == "ok"
    assert checks["name"] == "keep"
    assert checks["score"] == "ok"
    assert report["identity_verdict"]["verified"] is True

    # The audit trail now names what the row actually ships.
    assert row["mapping_meta"]["resolved_name"] == "Ferric uptake regulation protein"
    assert row["mapping_meta"]["identity_status"] == "verified"
    provenance = report["identity_verdict"]["identity_evidence"]["provenance"]
    assert provenance["source"] == "UniProt"
    assert provenance["query"] == "P0A9A9"
