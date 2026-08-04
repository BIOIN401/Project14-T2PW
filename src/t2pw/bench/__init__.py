"""Evidence-backed scientific benchmark for the T2PW pipeline.

Why this package exists
=======================

Until now a batch run answered one question -- "did the export gate let this
paper through?" -- and reported it as a single pass/fail. That number conflates
five unrelated things:

1. whether the search found a paper that *could* describe the pathway at all,
2. whether extraction produced a structurally valid payload,
3. whether the payload is **biologically about the requested pathway**,
4. whether it satisfies PathWhiz's importer-shape rules,
5. whether research mode produced a reviewable candidate.

A run can score 0/16 on (4) while being perfect on (3), or score well on (4)
while having invented a metabolite -- and the old report cannot tell those apart.
Worse, the old report has no notion of *scientific* error at all: a pathway whose
"proteins" include coenzyme A, whose reaction output is named "LpxA product", and
whose enzymes are Unknown-backed placeholders shipping a real-looking accession
counts exactly the same as a correct one, as long as the gate is happy.

This package separates those questions and grounds them in a **pinned gold set**:
a small number of papers whose expected biology was read out of the paper text by
hand, with a verbatim quote behind every expectation. Nothing here re-derives
biology at runtime, and nothing here calls an LLM -- the gold set is data, the
validator is deterministic, and a benchmark result is reproducible from stored
artifacts alone.

Layering
--------

Like ``t2pw.batch``, this package sits above every layer and may import anything.
It is *read-only* with respect to the pipeline: it consumes payloads, contract
reports and admission reports, and never mutates them.

Modules
-------

``goldset``    the pinned gold set: schema, loader, and name-matching rules
``semantic``   the semantic coverage validator (the seven checks)
``metrics``    separated denominators, scientific error counts, failure taxonomy
``acceptance`` scores a run directory against a gold set and renders the report
"""

from __future__ import annotations

from t2pw.bench.goldset import (
    GOLD_SET_VERSION,
    GoldCase,
    GoldSet,
    load_gold_set,
    pinned_gold_set_path,
)

__all__ = [
    "GOLD_SET_VERSION",
    "GoldCase",
    "GoldSet",
    "load_gold_set",
    "pinned_gold_set_path",
]
