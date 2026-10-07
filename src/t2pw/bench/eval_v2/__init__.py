"""Evaluation v2 (ORCH-742c) -- a NEW, separately versioned evaluator. EVALUATION-ONLY.

Nothing in production imports this package, and it changes no runtime behaviour.
It never re-scores or replaces an earlier result: T-107/T-108/T-109 and every
committed benchmark report keep their meaning. Numbers produced here are labelled
with :data:`EVALUATOR_VERSION` and written to new paths (``docs/evaluation_v2/``).

Modules
-------
``fetch``     read-only EuropePMC full-text fetch (texts live OUTSIDE the repo)
``matcher``   strict, documented name normalization; no congener credit by overlap
``gold``      the v2 supported-reaction supplement (``gold_v2_supported_reactions.json``)
``reactions`` reaction P/R/F1, entity accuracy, evidence support, mapped-ID
              correctness and the un-gated RAG influence audit, per committed leg
``rag``       retrieval metrics, the A/B/C/D failure split, chunk provenance checks
``repro``     LLM reproducibility-metadata completeness over committed run artifacts
``report``    the CLI that writes the versioned sample report
"""

EVALUATOR_VERSION = "eval-v2.0 (ORCH-742c)"
