"""Read-only literature-search probe (no model call, no cache write).

Runs pilot-001's three round-1 frontier items through mining's new query
formulation and the real PaperSource.search (EuropePMC + NCBI), and counts results
per source. Output: <out.json>.
"""
from t2pw.config import ensure_dotenv_loaded

ensure_dotenv_loaded()

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from t2pw.mining import queries as Q  # noqa: E402
from t2pw.mining.acquisition import PaperSource  # noqa: E402
from t2pw.mining.config import MiningConfig, StartingInput  # noqa: E402
from t2pw.mining.frontier import seed_frontier  # noqa: E402
from t2pw.rag import acquire as A  # noqa: E402

_real = A.search_candidates


def _nocache(context, **kwargs):
    kwargs["use_cache"] = False  # never read or write data/rag_index/acquire_cache
    return _real(context, **kwargs)


A.search_candidates = _nocache

config = MiningConfig.build(
    target_species="Arabidopsis thaliana", topic="jasmonate biosynthesis",
    starting_inputs=[StartingInput(kind="seed_metabolite", raw="12-OPDA", value="12-OPDA"),
                     StartingInput(kind="seed_enzyme", raw="AOS", value="AOS")],
)
out = []
for item in seed_frontier(config):
    for attempt in (1, 2):
        query = Q.build_query(item, config, round_index=1, attempt=attempt)
        if query is None:
            continue
        context = Q.build_context(item, config, attempt=attempt)
        found = PaperSource().search(context, sources=config.acquisition_sources,
                                     max_papers=config.budgets.max_results_per_query)
        row = {"target": item.target, "attempt": attempt, "query_text": query.query_text,
               "ncbi_term": A._ncbi_term(context), "results": len(found),
               "by_source": {s: sum(1 for c in found if c.source == s) for s in ("europepmc", "ncbi")},
               "ids": [c.id for c in found]}
        out.append(row)
        print(json.dumps({k: row[k] for k in ("target", "attempt", "query_text", "results", "by_source")}))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
