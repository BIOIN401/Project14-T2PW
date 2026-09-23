"""Evaluation v2 report CLI (ORCH-742c). EVALUATION-ONLY.

``evaluate``        deterministic, offline: reads ONLY git-tracked run artifacts and
                    writes a NEW versioned report. Re-running it on the same commit
                    reproduces every number.
``live-retrieval``  runs the production retriever (embeddings only, never chat) over
                    the committed paper texts and writes its own JSON, which
                    ``evaluate --live-json`` then includes, labelled with its
                    embedding mode.

Never re-scores or edits T-107/T-108/T-109: their committed reports are untouched and
these numbers are a different evaluator version, not a correction.

Usage::

    python -m t2pw.bench.eval_v2.report evaluate --repo . --out docs/evaluation_v2 [--live-json F]
    python -m t2pw.bench.eval_v2.report live-retrieval --repo . --cache C:/t/bt/eval/emb.json --out F
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from t2pw.bench.eval_v2 import EVALUATOR_VERSION
from t2pw.bench.eval_v2 import reactions as R
from t2pw.bench.eval_v2.gold import load_supplement
from t2pw.bench.eval_v2.rag import archived_failure_split, live_retrieval
from t2pw.bench.eval_v2.repro import completeness
from t2pw.bench.goldset import load_gold_set

#: Committed benchmark runs named in the ledger. Labels only; nothing is re-scored.
RUN_LABELS = {"runs_verify/2026-08-28_1816": "T-107", "runs_verify/2026-09-01_1612": "T-108"}


def _json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _paper_text(repo: Path, leg_dir: str) -> Optional[str]:
    return _text((repo / leg_dir).parent / "01_source_text.txt")


def _micro(legs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Micro-averaged over legs (sum numerators, sum denominators). Never across populations."""

    s = collections.Counter()
    for l in legs:
        m = l["metrics"]
        t1, t2, ea, ev = (m["table1_paper_extraction"], m["table2_final_support"], m["entity_accuracy"],
                          m["evidence"])
        s["legs"] += 1
        s["t1_rows"] += t1["rows_claimed_as_paper_extraction"]
        s["tp_rows"] += t1["true_positive_rows"]
        s["gold"] += t1["gold_reactions"]
        s["gold_matched"] += len(t1["gold_reactions_matched"])
        s["retained"] += t2["retained_unique"]
        s["unsupported"] += t2["unsupported"]
        s["indeterminate"] += t2["indeterminate_external"]
        s["external"] += t2["external_rag_supported"]
        s["off"] += t2["paper_stated_off_pathway"]
        s["partial"] += t2["partial_match"]
        s["summary"] += t2["paper_stated_summary"]
        s["sub_hit"] += ea["substrate_hits"]
        s["sub_tot"] += ea["substrate_total"]
        s["prod_hit"] += ea["product_hits"]
        s["prod_tot"] += ea["product_total"]
        for k, v in ea["enzyme_verdicts"].items():
            s["enz_" + k] += v
        for k, v in ea["species_verdicts"].items():
            s["sp_" + k] += v
        for k, v in ev["verdicts"].items():
            s["ev_" + k] += v
        s["old_gold_matched"] += len(m["old_lenient_matcher_same_rows"]["gold_reactions_matched"])
        for x in m["old_lenient_matcher_same_rows"]["rows_where_old_and_v2_disagree"]:
            s["old_disagreements"] += 1
            v2_tp = x["v2_class"] == R.TP
            s["old_only" if x["old"] and not v2_tp else "v2_only" if v2_tp and not x["old"] else "old_other"] += 1
        s["dups"] += m["duplicate_rows_collapsed"]
        s["cofactor_gaps"] += len(m["minor_cofactors_missing_on_matched_reactions"])
    r = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    p, rc = r(s["tp_rows"], s["t1_rows"]), r(s["gold_matched"], s["gold"])
    enz_den = s["enz_correct"] + s["enz_wrong"] + s["enz_missing"]
    return {
        "legs": s["legs"],
        "reaction_precision": p, "precision_num_den": [s["tp_rows"], s["t1_rows"]],
        "reaction_recall": rc, "recall_num_den": [s["gold_matched"], s["gold"]],
        "reaction_f1": (round(2 * p * rc / (p + rc), 4) if p and rc else (0.0 if p is not None and rc is not None else None)),
        "unsupported_rate": r(s["unsupported"], s["retained"]), "unsupported_num_den": [s["unsupported"], s["retained"]],
        "unsupported_rate_upper_bound": r(s["unsupported"] + s["indeterminate"], s["retained"]),
        "external_rag_supported_rows": s["external"],
        "paper_stated_off_pathway_rows": s["off"],
        "partial_match_rows": s["partial"],
        "paper_stated_summary_rows": s["summary"],
        "substrate_accuracy": r(s["sub_hit"], s["sub_tot"]), "substrate_num_den": [s["sub_hit"], s["sub_tot"]],
        "product_accuracy": r(s["prod_hit"], s["prod_tot"]), "product_num_den": [s["prod_hit"], s["prod_tot"]],
        "enzyme_accuracy": r(s["enz_correct"], enz_den),
        "enzyme_verdicts": {k[4:]: v for k, v in s.items() if k.startswith("enz_")},
        "species_accuracy": r(s["sp_correct"], s["sp_correct"] + s["sp_wrong"]),
        "species_verdicts": {k[3:]: v for k, v in s.items() if k.startswith("sp_")},
        "evidence_support_rate": r(s["ev_verbatim_in_paper"], s["retained"]),
        "evidence_verdicts": {k[3:]: v for k, v in s.items() if k.startswith("ev_")},
        "matched_reactions_missing_minor_cofactors": s["cofactor_gaps"],
        "duplicate_rows_collapsed": s["dups"],
        "old_lenient_matcher_recall": r(s["old_gold_matched"], s["gold"]),
        "old_vs_v2_row_disagreements": s["old_disagreements"],
        "old_credits_v2_refuses": s["old_only"],
        "v2_credits_old_refuses": s["v2_only"],
        "both_credit_different_gold_reaction": s["old_other"],
    }


def evaluate(repo: Path, live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rd = R.load_lineage(repo)
    paths = rd.committed_paths(repo)
    tracked = set(paths)
    supplement = load_supplement()
    pinned = {c.paper_id: c for c in load_gold_set(repo / "src/t2pw/bench/gold/pinned_v1.json").cases}
    adm_index = rd.admission_index(paths, repo, [])

    legs_out: List[Dict[str, Any]] = []
    split_out: List[Dict[str, Any]] = []
    for leg in rd.discover_legs(paths, []):
        case = supplement.get(leg["target_paper"])
        if case is None or not case.complete:
            continue
        payload = rd.load_json(repo / leg["leg_dir"] / leg["payload_file"])
        if not isinstance(payload, dict):
            continue
        idx = rd.entity_provenance_index(payload)
        rows = [r for r in (payload.get("processes") or {}).get("reactions") or [] if isinstance(r, dict)]
        lin = [rd.build_record(r, leg, i, idx, adm_index) for i, r in enumerate(rows)]
        text = _paper_text(repo, leg["leg_dir"])
        pin = pinned.get(case.paper_id)
        m = R.score_leg(case, payload, lin, text, expected_species=case.species)
        if leg["run"] not in RUN_LABELS:
            # Row-level detail is kept for the two labelled runs only, to keep the
            # committed report small; every leg's metrics are kept in full.
            m = {k: v for k, v in m.items() if k != "rows"}
        legs_out.append({**{k: leg[k] for k in ("leg_dir", "run", "population", "payload_file", "target_paper")},
                         "run_label": RUN_LABELS.get(leg["run"], ""),
                         "requested_pathway": pin.requested_pathway if pin else "",
                         "metrics": m})
        adm_path = f"{leg['leg_dir']}/rag_admission_report.json"
        if adm_path in tracked:
            s1 = rd.load_json(repo / leg["leg_dir"] / "stage1_payload.json") \
                if f"{leg['leg_dir']}/stage1_payload.json" in tracked else None
            split_out.append({"leg_dir": leg["leg_dir"], "run": leg["run"], "target_paper": case.paper_id,
                              "stage1_available": s1 is not None,
                              **archived_failure_split(case, rd.load_json(repo / adm_path) or {}, s1)})

    # Aggregates: per population over the whole committed corpus, and per labelled run.
    agg: Dict[str, Any] = {}
    for pop in ("canonical", "fallback"):
        agg[pop] = _micro([l for l in legs_out if l["population"] == pop])
    for run, label in RUN_LABELS.items():
        for pop in ("canonical", "fallback"):
            chosen = [l for l in legs_out if l["run"] == run and l["population"] == pop]
            if chosen:
                agg[f"{label}:{pop}"] = _micro(chosen)

    split_counts: Dict[str, Any] = {}
    for pop in ("untruncated", "truncated"):
        chosen = [s for s in split_out if s["truncated"] == (pop == "truncated")]
        c = collections.Counter()
        for s in chosen:
            c.update(s["counts"])
        prov = collections.Counter()
        for s in chosen:
            prov.update(s["chunk_provenance"])
        split_counts[pop] = {
            "legs": len(chosen),
            "legs_without_stage1_payload": sum(1 for s in chosen if not s["stage1_available"]),
            "per_gold_reaction_outcomes": dict(c),
            "accepted_candidates": sum(s["accepted_candidates"] for s in chosen),
            "D_accepted_unsupported": sum(s["D_accepted_unsupported"] for s in chosen),
            "accepted_external_biology": sum(s["accepted_external_biology"] for s in chosen),
            "candidates_persisted": sum(s["candidates_persisted"] for s in chosen),
            "chunk_provenance": dict(prov),
        }

    # Mapped-ID correctness over EVERY committed canonical leg.
    ref = R.build_reference_index(repo)
    idc = collections.Counter()
    wrong: List[Dict[str, Any]] = []
    id_legs = 0
    for p in paths:
        if not p.endswith("/final_mapped.json"):
            continue
        payload = rd.load_json(repo / p)
        if not isinstance(payload, dict):
            continue
        id_legs += 1
        res = R.id_correctness(payload, ref)
        idc.update(res["counts"])
        for w in res["wrong"]:
            wrong.append({"leg_dir": p.rpartition("/")[0], **w})

    # Un-gated RAG influence audit over every leg that has all three artifacts.
    leak_legs, leak_rows = [], []
    for p in paths:
        if not p.endswith("/rag_admission_report.json"):
            continue
        leg_dir = p.rpartition("/")[0]
        if f"{leg_dir}/stage1_payload.json" not in tracked:
            continue
        s1 = rd.load_json(repo / leg_dir / "stage1_payload.json") or {}
        adm = rd.load_json(repo / p) or {}
        seed = _paper_text(repo, leg_dir)
        for fname, label in (("merged_payload.json", "stage2_merge_output"), ("final_mapped.json", "canonical_final")):
            if f"{leg_dir}/{fname}" not in tracked:
                continue
            res = R.leakage_audit(s1, rd.load_json(repo / leg_dir / fname) or {}, adm, seed,
                                  target_paper=rd.target_paper_id(leg_dir) or "")
            leak_legs.append({"leg_dir": leg_dir, "artifact": label, "stage2_rows": res["stage2_rows"],
                              "rows_absent_from_stage1": res["rows_absent_from_stage1"],
                              "candidates": len(res["ungated_rag_influenced_candidates"]),
                              "seed_text_available": res["seed_text_available"]})
            for c in res["ungated_rag_influenced_candidates"]:
                leak_rows.append({"leg_dir": leg_dir, "artifact": label, **c})
    leak_summary = {}
    for label in ("stage2_merge_output", "canonical_final"):
        chosen = [l for l in leak_legs if l["artifact"] == label]
        leak_summary[label] = {
            "legs": len(chosen),
            "rows_total": sum(l["stage2_rows"] for l in chosen),
            "rows_absent_from_stage1": sum(l["rows_absent_from_stage1"] for l in chosen),
            "ungated_rag_influenced_candidates": sum(l["candidates"] for l in chosen),
            "legs_with_candidates": sum(1 for l in chosen if l["candidates"]),
            "candidates_with_external_evidence": sum(
                1 for r in leak_rows if r["artifact"] == label and r["external_evidence"]),
            "by_criterion": dict(collections.Counter(
                c for r in leak_rows if r["artifact"] == label for c in r["criteria"])),
        }

    repro = completeness(repo, paths)
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "notice": ("NEW evaluator version. Not a re-score, replacement or correction of T-107, T-108, "
                   "T-109 or any committed benchmark result; those keep their meaning and were not "
                   "re-run. Every number below is computed from git-tracked artifacts at the commit "
                   "this file was generated on. 'PWML produced' is not biological correctness and is "
                   "not reported as such."),
        "gold_supplement": {
            "complete_cases": sorted(pid for pid, c in supplement.items() if c.complete),
            "in_scope_reactions": {pid: len(c.reactions) for pid, c in supplement.items()},
            "off_pathway_reactions": {pid: len(c.off_pathway) for pid, c in supplement.items()},
        },
        "reaction_metrics": {"aggregate": agg, "legs": legs_out},
        "rag_archived_failure_split": {"by_population": split_counts, "legs": split_out},
        "rag_live_retrieval": live if live is not None else {
            "status": "not_run", "reason": "no --live-json supplied; retrieval metrics were not measured"},
        "mapped_id_correctness": {"canonical_legs": id_legs, "counts": dict(idc), "wrong": wrong},
        "ungated_rag_influence_audit": {"summary": leak_summary, "legs": leak_legs, "candidates": leak_rows},
        "reproducibility_metadata": repro,
    }


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def render_markdown(res: Mapping[str, Any], commit: str = "") -> str:
    """The human-readable sample report. Every number is read from ``res``."""

    L: List[str] = [f"# Evaluation v2 -- sample report ({res['evaluator_version']})", ""]
    if commit:
        L += [f"Run artifacts read: git-tracked files at commit `{commit}` (the run trees are unchanged on this branch). Regenerate: `python -m t2pw.bench.eval_v2.report evaluate --repo . --out docs/evaluation_v2 --live-json docs/evaluation_v2/retrieval_live.json`.", ""]
    L += [f"> {res['notice']}", ""]
    g = res["gold_supplement"]
    L += ["## 1. Newly complete positive gold cases", "",
          "| case | in-scope reactions | off-pathway statements |", "|---|---|---|"]
    for pid in g["complete_cases"]:
        L.append(f"| {pid} | {g['in_scope_reactions'][pid]} | {g['off_pathway_reactions'][pid]} |")
    L += ["", "## 2. Reaction-level metrics (strict v2 matcher; micro-averaged; populations never summed)", "",
          "| population | legs | precision | recall | F1 | unsupported rate (upper bound) | off-pathway rows | "
          "summary rows | partial rows | substrate acc | product acc | enzyme acc | species acc | "
          "evidence support | OLD lenient recall |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pop, a in res["reaction_metrics"]["aggregate"].items():
        L.append(
            f"| {pop} | {a['legs']} | {_pct(a['reaction_precision'])} ({a['precision_num_den'][0]}/{a['precision_num_den'][1]}) "
            f"| {_pct(a['reaction_recall'])} ({a['recall_num_den'][0]}/{a['recall_num_den'][1]}) | {_pct(a['reaction_f1'])} "
            f"| {_pct(a['unsupported_rate'])} ({a['unsupported_num_den'][0]}/{a['unsupported_num_den'][1]}; "
            f"<= {_pct(a['unsupported_rate_upper_bound'])}) | {a['paper_stated_off_pathway_rows']} "
            f"| {a['paper_stated_summary_rows']} | {a['partial_match_rows']} "
            f"| {_pct(a['substrate_accuracy'])} | {_pct(a['product_accuracy'])} | {_pct(a['enzyme_accuracy'])} "
            f"| {_pct(a['species_accuracy'])} | {_pct(a['evidence_support_rate'])} | {_pct(a['old_lenient_matcher_recall'])} |")
    L += ["", "Old lenient matcher vs strict v2 matcher on the SAME rows (old credits / v2 refuses; "
          "v2 credits / old refuses): " + "; ".join(
              f"{pop}: {a['old_credits_v2_refuses']} / {a['v2_credits_old_refuses']}"
              for pop, a in res["reaction_metrics"]["aggregate"].items()), ""]
    idc = res["mapped_id_correctness"]
    L += ["## 3. Mapped-ID correctness (every committed canonical leg)", "",
          f"canonical legs: {idc['canonical_legs']}", "", "| verdict | entities |", "|---|---|"]
    for k, v in sorted(idc["counts"].items()):
        L.append(f"| {k} | {v} |")
    live = res["rag_live_retrieval"]
    L += ["", "## 4. RAG retrieval (live production retriever)", ""]
    if live.get("status") == "not_run":
        L.append(f"NOT RUN: {live['reason']}")
    else:
        L += [f"embedding_mode: **{live['embedding_mode']}** ({live['embedding_model']}); corpus "
              f"{len(live['corpus_papers'])} committed paper texts, {live['corpus_chunks']} chunks; top_k {live['top_k']}", "",
              "| relevance label | positive queries | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR |",
              "|---|---|---|---|---|---|---|"]
        for label, key in (("target_quote", "metrics"), ("participants_named (any paper)", "metrics_any_paper")):
            m = live[key]
            L.append(f"| {label} | {m['positive_queries']} | {_pct(m['recall_at_1'])} | {_pct(m['recall_at_3'])} "
                     f"| {_pct(m['recall_at_5'])} | {_pct(m['precision_at_5'])} | {m['mrr']} |")
        L += ["", f"Negative-query rejection: {_pct(live['negative_query_rejection_rate'])} "
                  f"({live['negative_query_rejected']}/{live['negative_queries']}). {live['score_note']}", "",
              "Per retrieved chunk: " + ", ".join(f"{k}: {v}" for k, v in sorted(live["chunk_provenance"].items()))]
    L += ["", "## 5. Retrieval vs LLM-use vs admission failure (archived admission reports)", "",
          "| population | legs | already in Stage 1 | A not retrieved | B retrieved, not used | "
          "C proposed, rejected | admitted | accepted candidates | D accepted unsupported | accepted external biology |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for pop, s in res["rag_archived_failure_split"]["by_population"].items():
        o = s["per_gold_reaction_outcomes"]
        L.append(f"| {pop} | {s['legs']} | {o.get('already_in_stage1', 0)} | {o.get('A_not_retrieved', 0)} "
                 f"| {o.get('B_retrieved_not_used', 0)} | {o.get('C_proposed_but_rejected', 0)} | {o.get('ok_admitted', 0)} "
                 f"| {s['accepted_candidates']} | {s['D_accepted_unsupported']} | {s['accepted_external_biology']} |")
    L += ["", "Per persisted candidate chunk: " + "; ".join(
        f"{pop}: " + ", ".join(f"{k} {v}" for k, v in sorted(s["chunk_provenance"].items()))
        for pop, s in res["rag_archived_failure_split"]["by_population"].items()), ""]
    L += ["## 6. Un-gated RAG influence audit (report only)", "",
          "| artifact | legs | rows | rows absent from Stage 1 | un-gated RAG-influenced candidates "
          "| of which external retrieved evidence | legs with any |",
          "|---|---|---|---|---|---|---|"]
    for art, s in res["ungated_rag_influence_audit"]["summary"].items():
        L.append(f"| {art} | {s['legs']} | {s['rows_total']} | {s['rows_absent_from_stage1']} "
                 f"| {s['ungated_rag_influenced_candidates']} | {s['candidates_with_external_evidence']} "
                 f"| {s['legs_with_candidates']} |")
    rp = res["reproducibility_metadata"]
    L += ["", "## 7. LLM reproducibility-metadata completeness", "",
          f"model_attempt records: {rp['model_attempt_records']} from {rp['legs_with_leg_trace']} leg traces "
          f"(runs: {', '.join(rp['runs_with_leg_trace']) or 'none'}); legs with a payload: {rp['legs_with_payload']}. "
          f"Audit rounds carrying temperature: {rp['audit_rounds_with_temperature']}/{rp['audit_rounds']}.", "",
          "| field | status | attempts with field | recorded as | future hook |", "|---|---|---|---|---|"]
    for f in rp["fields"]:
        L.append(f"| {f['field']} | {f['status']} | {f['model_attempts_with_field']}/{f['model_attempts']} "
                 f"| {f['recorded_as']} | {f['hook'] or '-'} |")
    L.append("")
    return "\n".join(L)


def run_live(repo: Path, cache: Path) -> Dict[str, Any]:
    rd = R.load_lineage(repo)
    paths = rd.committed_paths(repo)
    supplement = {k: v for k, v in load_supplement().items() if v.complete}
    pinned = {c.paper_id: c for c in load_gold_set(repo / "src/t2pw/bench/gold/pinned_v1.json").cases}
    texts: Dict[str, str] = {}
    for pid in pinned:
        # The most recent committed copy of each gold paper's stored text.
        cands = sorted(p for p in paths if p.endswith("/01_source_text.txt") and f"/{pid}" in p)
        if cands:
            texts[pid] = _text(repo / cands[-1]) or ""
    meta = {pid: {"title": c.title, "pathway": c.requested_pathway,
                  "organism": c.actual_organism or c.requested_organism} for pid, c in pinned.items()}
    out = live_retrieval(supplement, texts, meta, cache)
    out["evaluator_version"] = EVALUATOR_VERSION
    out["corpus_source"] = "latest committed 01_source_text.txt per gold paper (git ls-files)"
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--repo", default=".")
    e.add_argument("--out", required=True)
    e.add_argument("--live-json", default=None)
    lv = sub.add_parser("live-retrieval")
    lv.add_argument("--repo", default=".")
    lv.add_argument("--cache", required=True, help="embedding cache path OUTSIDE the repo")
    lv.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    repo = Path(a.repo).resolve()
    if a.cmd == "live-retrieval":
        cache = Path(a.cache).resolve()
        if repo in cache.parents:
            raise SystemExit("refusing: the embedding cache must live outside the repository")
        res = run_live(repo, cache)
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: res[k] for k in ("embedding_mode", "embedding_model", "embedder_stats",
                                              "corpus_chunks", "metrics", "metrics_any_paper", "negative_queries",
                                              "negative_query_rejection_rate")}, indent=1))
        return 0
    live = _json(Path(a.live_json)) if a.live_json else None
    res = evaluate(repo, live)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample_report.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                            text=True).stdout.strip()
    (out / "SAMPLE_REPORT.md").write_text(render_markdown(res, commit), encoding="utf-8")
    print(json.dumps(res["reaction_metrics"]["aggregate"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
