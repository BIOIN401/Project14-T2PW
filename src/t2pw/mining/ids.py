"""Stable identifier minting for mining artifacts.

Every id is a deterministic function of content, so two runs over the same
evidence produce the same ids and a diff between runs is meaningful. Nothing
here uses a counter, a clock or a UUID -- a resumed run must mint the same id
for the same thing as the run it is resuming.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

_WS = re.compile(r"\s+")


def _blob(*parts: Any) -> str:
    return "\x1f".join(_WS.sub(" ", str(p if p is not None else "")).strip() for p in parts)


def digest(*parts: Any, length: int = 16) -> str:
    """Lowercase hex sha256 of ``parts``, truncated to ``length`` characters."""
    return hashlib.sha256(_blob(*parts).encode("utf-8", "replace")).hexdigest()[:length]


def sha256_bytes(data: bytes) -> str:
    """Full sha256 of raw bytes -- used for uploaded-file provenance."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", "replace")).hexdigest()


def paper_id(*, pmid: str = "", pmcid: str = "", doi: str = "", url: str = "",
             file_sha256: str = "", title: str = "") -> str:
    """Prefer the strongest stable handle a paper has.

    PMCID > PMID > DOI > uploaded-file hash > URL > title hash. A paper that
    later gains a PMCID keeps its original id for the run it was discovered in
    (the caller carries the id forward); this function only decides the id at
    first sight.
    """
    for prefix, value in (
        ("PMC", pmcid),
        ("PMID", pmid),
        ("DOI", doi),
        ("FILE", file_sha256[:16] if file_sha256 else ""),
        ("URL", digest(url) if url else ""),
    ):
        text = str(value or "").strip()
        if text:
            return f"{prefix}:{text}"
    return f"TITLE:{digest(title)}"


def evidence_id(paper: str, chunk: str, text: str) -> str:
    return f"ev_{digest(paper, chunk, text)}"


def claim_id(identity: str, paper: str, round_index: int) -> str:
    return f"cl_{digest(identity, paper, round_index)}"


def decision_id(claim: str, round_index: int, status: str) -> str:
    return f"dc_{digest(claim, round_index, status)}"


def entity_id(canonical_name: str, entity_type: str) -> str:
    return f"en_{digest(canonical_name, entity_type)}"


def reaction_id(identity: str) -> str:
    return f"rx_{digest(identity, length=20)}"


def frontier_id(kind: str, target: str, round_index: int) -> str:
    return f"fr_{digest(kind, target, round_index)}"


def query_id(text: str, round_index: int) -> str:
    return f"qr_{digest(text, round_index)}"


def component_id(seed: str) -> str:
    return f"cmp_{digest(seed, length=12)}"


def conflict_id(attribute: str, identity: str) -> str:
    return f"cf_{digest(attribute, identity)}"


def call_id(round_index: int, purpose: str, ordinal: int) -> str:
    return f"mc_{int(round_index):03d}_{purpose}_{int(ordinal):04d}"


def run_id(timestamp: str, species: str, topic: str) -> str:
    return f"{timestamp}_{digest(species, topic, length=8)}"


def stable_join(items: Iterable[Any]) -> str:
    """Deterministic joined form of a multiset of names (order-insensitive)."""
    return "+".join(sorted(_WS.sub(" ", str(i or "")).strip().lower() for i in items if str(i or "").strip()))


__all__ = [
    "digest",
    "sha256_bytes",
    "sha256_text",
    "paper_id",
    "evidence_id",
    "claim_id",
    "decision_id",
    "entity_id",
    "reaction_id",
    "frontier_id",
    "query_id",
    "component_id",
    "conflict_id",
    "call_id",
    "run_id",
    "stable_join",
]
