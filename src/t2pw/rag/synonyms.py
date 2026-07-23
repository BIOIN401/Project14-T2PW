"""Offline synonym-canonical resolver for RAG synthesis *grouping* (follow-up to
the direction-aware ``conflict_key`` fix).

Why this exists
---------------
``synthesize`` under-merges cross-paper duplicate reactions: two rows that are
identical except that one names a compound ``UDP-GlcNAc`` and the other
``UDP-N-acetylglucosamine`` (the SAME molecule, ChEBI:16264) stay as separate
reactions because ``canonical_name`` only consults the tiny hand-curated
``BIOCHEMICAL_ALIAS_MAP``. This module builds a **grouping key** resolver from the
project's EXISTING offline synonym data (``data/id_mapping_cache.json``) so those
duplicates collapse — *without* renaming anything the pipeline emits.

Grouping-only contract
----------------------
The returned callable is used ONLY to compute merge/grouping KEYS
(``_Reaction.conflict_key`` / ``signature`` and, optionally, the ``_build_entities``
key). It never rewrites a reaction/entity display name: survivors keep their own
real names (the merge winner's), so locked-reaction matching (name + inputs +
outputs on the RAW names) still works and the "locked reaction accounting failed"
gate stays green. See ``docs/rag/agents/wp5_synthesis.md`` step 2.

Offline / determinism
---------------------
``build_offline_synonym_resolver`` reads only local JSON files and returns a pure
function of their bytes — no network, no live LLM, no heavy deps. It is injected
into ``synthesize`` as an OPTIONAL dependency (default ``None`` = today's exact
behavior), mirroring the existing ``prose_extractor`` seam.

Extension point (NOT enabled by default)
-----------------------------------------
``build_offline_synonym_resolver(..., llm_fallback=...)`` accepts a
``str -> Optional[str]`` callable that COULD map names the offline index misses to
a canonical token (an LLM-backed lookup, say). It is ``None`` by default and MUST
stay so for the default/offline path, because a live model would break both
determinism and the offline guarantee. The hook is wired so a future caller can
chain it explicitly; this module never calls a model itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Name normalization.
#
# SOURCE OF TRUTH: ``t2pw.mapping.map_ids._normalize_name`` —
#   ``re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", value.strip().casefold()))``
# Replicated here (2 lines) rather than imported so building this resolver never
# drags map_ids' heavy / network-capable stack (openai/requests) into the offline
# ``synthesize`` import path (separation invariant + offline guarantee). Keep the
# two definitions in lock-step: the cache keys were built with the map_ids form,
# so a lookup only hits when the two normalizations agree.
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_NONALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_name(value: Any) -> str:
    lowered = _WS_RE.sub(" ", str(value or "").strip().casefold())
    return _NONALNUM_RE.sub("", lowered)


# Priority order for choosing the ONE stable external id that keys a compound /
# protein. First present wins (brief-prescribed). Two cache entries that resolve to
# the same stable id are synonyms and share a grouping token.
_ID_PRIORITY = ("chebi", "kegg", "hmdb", "pubchem", "cas", "pathbank_compound_id")

# Only these cache sections carry mapped compound / protein synonyms.
_CACHE_SECTIONS = ("compounds", "proteins")

# Grouping tokens are namespaced so a resolved (synonym) token can NEVER collide
# with the normalized-name fallback returned for an unmapped name (which is bare
# ``[a-z0-9 ]``). Prevents an unmapped compound literally named e.g. "C00043" from
# accidentally grouping with the KEGG-keyed token for that id.
_TOKEN_PREFIX = "syn::"

# Parenthetical abbreviation, e.g. the "UDP-GlcNAc" in
# "uridine diphosphate-N-acetyl-glucosamine (UDP-GlcNAc)".
_PAREN_RE = re.compile(r"\(([^()]*)\)")


def _key_to_norm(cache_key: str) -> str:
    """Recover the normalized input name from a cache key.

    Keys look like ``db::udpglcnac::::{}`` or ``api::<norm>::<params>``. Strip the
    leading ``db::`` / ``api::`` source tag, then take the segment up to the next
    ``::`` (the normalized name the entry was cached under).
    """
    text = str(cache_key or "")
    for prefix in ("db::", "api::"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    segment = text.split("::", 1)[0]
    return _normalize_name(segment)


def _stable_token(mapped_ids: Dict[str, Any]) -> Optional[str]:
    """First-present external id (per ``_ID_PRIORITY``) as a namespaced token."""
    for field in _ID_PRIORITY:
        value = mapped_ids.get(field)
        if value:
            return f"{_TOKEN_PREFIX}{field}:{value}"
    return None


def _name_variants(name: str) -> List[str]:
    """Deterministic structural sub-forms of ``name`` to try when the full form misses.

    Handles the common "long name (ABBREV)" shape: the parenthetical abbreviation
    (e.g. ``UDP-GlcNAc``) and the long-form text with parentheticals removed are
    each normalized and returned, so a raw name whose *whole* normalized form is
    unmapped can still resolve via one of its parts. Abbreviation-first, then the
    stripped long form — a fixed order, so resolution is deterministic.
    """
    raw = str(name or "")
    variants: List[str] = []
    for inner in _PAREN_RE.findall(raw):
        norm = _normalize_name(inner)
        if norm:
            variants.append(norm)
    stripped = _normalize_name(_PAREN_RE.sub(" ", raw))
    if stripped:
        variants.append(stripped)
    return variants


def _index_cache(data: Dict[str, Any], index: Dict[str, str]) -> None:
    """Populate ``norm_name -> stable_token`` from an id_mapping_cache dict.

    For every ``status=="mapped"`` compound / protein entry we register (a) the
    normalized name parsed out of the cache key and (b) the normalized
    ``candidates[*].name`` display names — both pointing at the entry's stable
    token. ``setdefault`` keeps first-seen on the (rare) event a normalized name
    collides across entries; iteration follows the file's stable insertion order,
    so the built index is a pure function of the file bytes.
    """
    for section in _CACHE_SECTIONS:
        entries = data.get(section)
        if not isinstance(entries, dict):
            continue
        for cache_key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("status") or "").casefold() != "mapped":
                continue
            mapped_ids = entry.get("mapped_ids")
            if not isinstance(mapped_ids, dict):
                continue
            token = _stable_token(mapped_ids)
            if not token:
                continue
            norm = _key_to_norm(str(cache_key))
            if norm:
                index.setdefault(norm, token)
            for cand in entry.get("candidates") or []:
                if isinstance(cand, dict):
                    cand_norm = _normalize_name(cand.get("name"))
                    if cand_norm:
                        index.setdefault(cand_norm, token)


def _index_pathwhiz(data: Dict[str, Any], index: Dict[str, str]) -> None:
    """OPTIONAL broader coverage from ``data/pathwhiz_id_db.json`` (offline).

    The pathwhiz db maps external ids to internal PathWhiz numbers
    (``{"compounds": {"chebi": {"CHEBI:16264": 196, ...}, ...}}``) and carries no
    names, so it cannot add name→token entries directly. It is consulted only to
    *alias* one external-id token onto another that shares the same PathWhiz
    compound/protein number — widening which stable ids count as the same species.
    Kept behind the opt-in ``extra_db_path`` param; the primary cache index is what
    the verified UDP-GlcNAc case needs, so the default path never loads this.
    """
    # Reserved, defensive extension. Left as a no-op alias pass: the cache already
    # links this pathway's verified synonyms, and a name-free id table adds risk
    # without adding NAME coverage. Present so a caller can pass ``extra_db_path``
    # and have it fail safe rather than error. See module docstring.
    return None


def _make_resolver(
    index: Dict[str, str],
    llm_fallback: Optional[Callable[[str], Optional[str]]] = None,
) -> Callable[[str], str]:
    """Close over the reverse index and return the grouping resolver."""

    def resolve(name: str) -> str:
        norm = _normalize_name(name)
        if not norm:
            return ""
        token = index.get(norm)
        if token:
            return token
        for variant in _name_variants(name):
            token = index.get(variant)
            if token:
                return token
        # Extension point (default off): an explicit, caller-supplied lookup for
        # names the offline index misses. Never a live model on the default path —
        # determinism + offline. Fails closed to the normalized-name fallback.
        if llm_fallback is not None:
            try:
                hit = llm_fallback(name)
            except Exception:  # noqa: BLE001 - a fallback must never break grouping
                hit = None
            if hit:
                return str(hit)
        # Unmapped: group by the name's own normalized form. This is the effective
        # grouping today's index-free behavior would give an uncovered name.
        return norm

    return resolve


def build_offline_synonym_resolver(
    cache_path: str = "data/id_mapping_cache.json",
    *,
    extra_db_path: Optional[str] = None,
    llm_fallback: Optional[Callable[[str], Optional[str]]] = None,
) -> Callable[[str], str]:
    """Build a deterministic, offline synonym-canonical grouping resolver.

    Loads ``cache_path`` (an id_mapping_cache) ONCE and builds a reverse index
    ``normalized name -> stable synonym token``, where the token is derived from
    the first present of ``(chebi, kegg, hmdb, pubchem, cas, pathbank_compound_id)``
    on each ``status=="mapped"`` entry. The returned ``resolve(name) -> str``:

    * returns the stable synonym **token** (e.g. ``"syn::chebi:CHEBI:16264"``) when
      the name — or one of its parenthetical sub-forms — resolves, so synonyms
      share a grouping key; else
    * returns ``_normalize_name(name)`` so an uncovered name still groups by its
      own normalized form (today's effective grouping for that name).

    The callable is a pure function of the loaded file(s). Malformed / missing
    files degrade gracefully to an identity-ish resolver that only normalizes
    (every name maps to its normalized form, so nothing merges across names) — it
    never raises. ``extra_db_path`` optionally consults ``pathwhiz_id_db.json`` for
    broader id coverage; ``llm_fallback`` is a reserved, default-off extension hook
    (see module docstring) — pass neither for the deterministic offline resolver.
    """
    index: Dict[str, str] = {}
    try:
        raw = Path(cache_path).read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:  # noqa: BLE001 - a missing/broken cache must not break synthesis
        data = None
    if isinstance(data, dict):
        _index_cache(data, index)

    if extra_db_path:
        try:
            extra_raw = Path(extra_db_path).read_text(encoding="utf-8")
            extra = json.loads(extra_raw)
        except Exception:  # noqa: BLE001 - optional db is best-effort only
            extra = None
        if isinstance(extra, dict):
            _index_pathwhiz(extra, index)

    return _make_resolver(index, llm_fallback=llm_fallback)


__all__ = ["build_offline_synonym_resolver"]
