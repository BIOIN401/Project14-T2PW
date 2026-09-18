"""``t2pw.mining`` -- experimental literature-driven pathway mining.

**Experimental, additive and OFF by default.** With
``T2PW_PATHWAY_MINING_ENABLED`` unset, nothing in this package runs: no existing
command, default, import, output, cache or PWML behaviour changes, and importing
this module does no work beyond defining names.

What it is
----------
A research-graph builder. Given a species and one or more starting points (a
topic, seed metabolites/enzymes/reactions, explicit papers, an existing pathway
graph), it iteratively mines the literature -- acquire, retrieve, extract,
admit, expand -- and produces an audited graph of paper-supported reactions
plus quarantined candidate subgraphs.

What it is NOT
--------------
* **Not a PWML mode.** The PWML exporter is never invoked, PathWhiz identifiers
  are never required, and no importer gate runs. See ``docs/pathway_mining/``.
* **Not database-validated.** External biochemical identifier services
  (PathBank, PWC, HMDB, KEGG, ChEBI, PubChem, UniProt) are deliberately not
  consulted in version 1; admission is on paper evidence alone and every entity
  carries ``external_refs = {"status": "not_evaluated"}``.

Import cost
-----------
This module imports only :mod:`t2pw.mining.flags`. The LLM client, the vector
store, ``chromadb`` and the paper fetchers are all constructed lazily inside the
call that needs them, so a disabled feature costs one module import and nothing
else.
"""

from __future__ import annotations

from t2pw.mining.flags import (
    FEATURE_ENV_VAR,
    PathwayMiningDisabled,
    mining_enabled,
    require_mining_enabled,
)

__all__ = [
    "FEATURE_ENV_VAR",
    "PathwayMiningDisabled",
    "mining_enabled",
    "require_mining_enabled",
]
