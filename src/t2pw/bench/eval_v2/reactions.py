"""Evaluation v2 per-leg analyses over COMMITTED run artifacts (ORCH-742c).

EVALUATION-ONLY: reads committed payloads, never writes into a run, calls no model.

Three analyses live here because all three walk the same payload rows:

* :func:`score_leg` -- reaction precision/recall/F1 against the v2 complete gold,
  entity accuracy (substrate, product, enzyme, species), evidence support, minor
  cofactors kept apart from defining chemistry, and the pinned-v1 lenient matcher
  on the same rows so the difference is visible;
* :func:`id_correctness` -- does each mapped identifier correspond to the entity
  that claims it (correct / unresolved-or-novel / wrong / not evaluable);
* :func:`leakage_audit` -- un-gated RAG-influenced reaction candidates.

ROW CLASSES (complete cases), in precedence order -- documented in
``docs/evaluation_v2/README.md``:

``target_paper_supported``   every defining participant of an in-scope gold reaction on
                             the right side (reversible gold may match backwards)
``paper_stated_off_pathway`` the same, against a reaction the paper states for another
                             pathway, an assay or transport
``paper_stated_summary``     the same, against an overall multi-step conversion the paper
                             states (a lumped row: not invented, never a true positive)
``name_conflict``            it would be one of the above (or a partial match) only by
                             crediting a parenthetical's INSIDE whose outer name does not
                             match (``heptanoyl pyrophosphate (HepPP)`` for heptaprenyl
                             diphosphate); never a true positive, listed for hand review
``partial_match``            at least one defining substrate AND one defining product of
                             an in-scope gold reaction, not all of them
``external_rag_supported``   no gold match; the R-D092-1 lineage classifier finds an
                             accepted reaction-specific external record
``indeterminate_external``   no gold match; external lineage the classifier cannot settle
``unsupported``              everything else: the exhaustively read paper does not state
                             it and nothing external defends it

D-093's two tables are never summed. TABLE 1 (paper extraction) excludes rows whose
lineage names an external source unless they matched gold; TABLE 2 (final support)
counts every retained row.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from t2pw.bench.eval_v2 import matcher as M
from t2pw.bench.eval_v2.gold import CaseV2, Reaction, Term
from t2pw.bench.goldset import GoldTerm, SupportedReaction, fold_for_quote

TP = "target_paper_supported"
OFF = "paper_stated_off_pathway"
SUMMARY = "paper_stated_summary"
CONFLICT = "name_conflict"
PARTIAL = "partial_match"
EXT = "external_rag_supported"
INDET = "indeterminate_external"
UNSUP = "unsupported"
ROW_CLASSES = (TP, OFF, SUMMARY, CONFLICT, PARTIAL, EXT, INDET, UNSUP)

#: An evidence string longer than this is a document dump, not a quote (R-D092-1 fact 4:
#: 35k- and 108k-character "evidence" strings exist and carry external text).
MAX_QUOTE_CHARS = 2000
MIN_QUOTE_FOLDED = 20

LINEAGE_REL = Path("docs/pwml_recovery_sprint/evidence/rd092_1_reaction_lineage.py")


def load_lineage(repo_root: Path) -> Any:
    """The committed R-D092-1 classifier, loaded by path and reused, not rebuilt."""

    name = "rd092_1_reaction_lineage"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(repo_root) / LINEAGE_REL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- matching
def _side_hits(terms: Sequence[Term], names: Sequence[str]) -> List[str]:
    """Per gold term: ``match`` | ``conflict`` | ``''`` (see :func:`matcher.name_verdict`)."""

    out = []
    for t in terms:
        verdicts = {M.name_verdict(n, M.term_keys(t.name, t.aliases)) for n in names}
        out.append("match" if "match" in verdicts else "conflict" if "conflict" in verdicts else "")
    return out


def _kind(hits_in: Sequence[str], hits_out: Sequence[str], ok: Tuple[str, ...]) -> str:
    i, o = [h in ok for h in hits_in], [h in ok for h in hits_out]
    return "full" if all(i) and all(o) else "partial" if any(i) and any(o) else ""


def match_reaction(g: Reaction, row: Mapping[str, Any]) -> Tuple[str, bool]:
    """``("full"|"name_conflict"|"partial"|"", backwards)`` for one row against one gold
    reaction. ``name_conflict``: the outcome changes only if a parenthetical's inside is
    allowed to credit a term its outer name does not match."""

    ins = M.participant_names(row.get("inputs"))
    outs = M.participant_names(row.get("outputs"))
    best = ("", False)
    for backwards in ((False, True) if g.reversible else (False,)):
        a, b = (outs, ins) if backwards else (ins, outs)
        hi = _side_hits(g.defining("inputs"), a)
        ho = _side_hits(g.defining("outputs"), b)
        strict = _kind(hi, ho, ("match",))
        if strict == "full":
            return "full", backwards
        kind = CONFLICT if _kind(hi, ho, ("match", "conflict")) != strict else strict
        if kind and (not best[0] or (best[0] == "partial" and kind == CONFLICT)):
            best = (kind, backwards)
    return best


def _row_key(row: Mapping[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    side = lambda v: tuple(sorted(M.strict_key(n) for n in M.participant_names(v)))  # noqa: E731
    return side(row.get("inputs")), side(row.get("outputs"))


def _old_signature(g: Reaction) -> SupportedReaction:
    t = lambda x: GoldTerm(name=x.name, aliases=x.aliases)  # noqa: E731
    return SupportedReaction(
        inputs=tuple(t(x) for x in g.defining("inputs")),
        outputs=tuple(t(x) for x in g.defining("outputs")),
        enzyme=t(g.enzyme) if g.enzyme else None,
        reversible=g.reversible, quote=g.quote, label=g.label)


def _old_matches(g: Reaction, row: Mapping[str, Any]) -> bool:
    from t2pw.bench.semantic import _signature_matches  # the pinned-v1 scorer's own matcher

    return bool(_signature_matches(_old_signature(g), row))


# --------------------------------------------------------------------------- helpers
def _species_of(row: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    for key in ("species", "organism"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key]
    state = row.get("biological_state")
    for st in payload.get("biological_states") or ():
        if isinstance(st, dict) and state and st.get("name") == state and st.get("species"):
            return str(st["species"])
    return ""


def _evidence_spans(row: Mapping[str, Any]) -> List[str]:
    ev = row.get("evidence")
    items = ev if isinstance(ev, list) else [ev]
    out = []
    for it in items:
        if isinstance(it, str) and it.strip():
            out.append(it)
        elif isinstance(it, dict):
            for k in ("text", "span", "quote"):
                if isinstance(it.get(k), str) and it[k].strip():
                    out.append(it[k])
                    break
    return out


def evidence_verdict(row: Mapping[str, Any], folded_paper: str) -> str:
    """``verbatim_in_paper`` | ``not_in_paper`` | ``dump_only`` | ``missing`` | ``paper_text_missing``."""

    spans = _evidence_spans(row)
    if not spans:
        return "missing"
    quotes = [s for s in spans if len(s) <= MAX_QUOTE_CHARS and len(fold_for_quote(s)) >= MIN_QUOTE_FOLDED]
    if not quotes:
        return "dump_only"
    if not folded_paper:
        return "paper_text_missing"
    return "verbatim_in_paper" if any(fold_for_quote(q) in folded_paper for q in quotes) else "not_in_paper"


def species_key(name: Any) -> str:
    """Binomial key: genus + epithet. Strain/subspecies qualifiers are dropped
    ('Lactococcus lactis subsp. lactis KF147' == 'Lactococcus lactis'). Species only."""

    return " ".join(M.strict_key(name).split()[:2])


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


# --------------------------------------------------------------------------- scoring
def score_leg(case: CaseV2, payload: Mapping[str, Any], lineage_records: Sequence[Mapping[str, Any]],
              paper_text: Optional[str], expected_species: str = "") -> Dict[str, Any]:
    """Every v2 reaction metric for one leg of one complete case. Pure function."""

    rows = [r for r in ((payload.get("processes") or {}).get("reactions") or []) if isinstance(r, dict)]
    lin_by_index = {int(r.get("row_index", -1)): r for r in lineage_records}
    folded_paper = fold_for_quote(paper_text) if paper_text else ""
    cplx = M.complex_index(payload)
    want_species = species_key(expected_species or case.species)

    seen: Dict[Tuple, int] = {}
    per_row: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        key = _row_key(row)
        dup_of = seen.get(key)
        seen.setdefault(key, i)
        rec: Dict[str, Any] = {
            "row_index": i, "name": row.get("name") or "",
            "inputs": M.participant_names(row.get("inputs")),
            "outputs": M.participant_names(row.get("outputs")),
            "enzymes": M.enzyme_names(row), "duplicate_of": dup_of,
        }
        full = [(g, bw) for g in case.reactions for kind, bw in [match_reaction(g, row)] if kind == "full"]
        off = [g for g in case.off_pathway if match_reaction(g, row)[0] == "full"]
        summ = [g for g in case.summaries if match_reaction(g, row)[0] == "full"]
        conflict = [(g, bw) for g in (*case.reactions, *case.off_pathway, *case.summaries)
                    for kind, bw in [match_reaction(g, row)] if kind == CONFLICT]
        part = [(g, bw) for g in case.reactions for kind, bw in [match_reaction(g, row)] if kind == "partial"]
        lin = lin_by_index.get(i, {})
        resolutions = lin.get("source_resolutions")
        external = isinstance(resolutions, list) and "external" in resolutions
        if full:
            cls, gid = TP, [g.id for g, _ in full]
        elif off:
            cls, gid = OFF, [g.id for g in off]
        elif summ:
            cls, gid = SUMMARY, [g.id for g in summ]
        elif conflict:
            cls, gid = CONFLICT, [g.id for g, _ in conflict]
            gold_keys = set().union(*(M.term_keys(t.name, t.aliases) for g, _ in conflict
                                      for t in (*g.defining("inputs"), *g.defining("outputs"))))
            rec["conflicting_names"] = [n for n in rec["inputs"] + rec["outputs"]
                                        if M.name_verdict(n, gold_keys) == "conflict"]
        elif part:
            cls, gid = PARTIAL, [g.id for g, _ in part]
        elif external and lin.get("support_class") == "external_rag_supported":
            cls, gid = EXT, []
        elif external and lin.get("support_class") == "indeterminate":
            cls, gid = INDET, []
        else:
            cls, gid = UNSUP, []
        rec.update({
            "class": cls, "gold_ids": gid, "lineage_class": lin.get("support_class", "unavailable"),
            "lineage_external": external, "table1": (not external) or cls in (TP, OFF, SUMMARY, CONFLICT, PARTIAL),
            "old_matcher_gold_ids": [g.id for g in case.reactions if _old_matches(g, row)],
            "evidence": evidence_verdict(row, folded_paper),
        })
        sp = _species_of(row, payload)
        rec["species"] = sp
        rec["species_verdict"] = ("unknown" if not sp else
                                  "correct" if species_key(sp) == want_species else "wrong")
        # Defining chemistry vs minor cofactors, for full matches only.
        if full:
            g, bw = full[0]
            a, b = ("outputs", "inputs") if bw else ("inputs", "outputs")
            ins, outs = M.participant_names(row.get(a)), M.participant_names(row.get(b))
            miss = [t.name for t in g.cofactors("inputs") if M.any_match(ins, t.name, t.aliases) is None]
            miss += [t.name for t in g.cofactors("outputs") if M.any_match(outs, t.name, t.aliases) is None]
            everything = ins + outs + rec["enzymes"]
            rec["missing_minor_cofactors"] = miss
            rec["catalytic_cofactors_present"] = [t.name for t in g.catalytic_cofactors
                                                  if M.any_match(everything, t.name, t.aliases)]
            if g.enzyme is None:
                rec["enzyme_verdict"] = "not_applicable"
            elif not rec["enzymes"]:
                rec["enzyme_verdict"] = "missing"
            else:
                want = M.term_keys(g.enzyme.name, g.enzyme.aliases)
                got = set().union(*(M.enzyme_keys(e, cplx) for e in rec["enzymes"]))
                # An inside-only hit ('oxidoreductase (entA)' for EntA) is not credited:
                # the evaluator cannot tell a generic outer name from a contradicting one.
                rec["enzyme_verdict"] = ("correct" if want & got else CONFLICT
                                         if any(M.inner_key(e) in want for e in rec["enzymes"]) else "wrong")
        if (conflict or part) and not full:
            g, bw = (conflict or part)[0]
            a, b = ("outputs", "inputs") if bw else ("inputs", "outputs")
            ins, outs = M.participant_names(row.get(a)), M.participant_names(row.get(b))
            rec["missing_defining"] = (
                [t.name for t in g.defining("inputs") if M.any_match(ins, t.name, t.aliases) is None]
                + [t.name for t in g.defining("outputs") if M.any_match(outs, t.name, t.aliases) is None])
        per_row.append(rec)

    uniq = [r for r in per_row if r["duplicate_of"] is None]
    counts = collections.Counter(r["class"] for r in uniq)

    # TABLE 1 -- paper extraction.
    t1 = [r for r in uniq if r["table1"]]
    tp_rows = sum(1 for r in t1 if r["class"] == TP)
    matched_ids = {gid for r in uniq if r["class"] == TP for gid in r["gold_ids"]}
    n_gold = len(case.reactions)
    precision = _rate(tp_rows, len(t1))
    recall = _rate(len(matched_ids), n_gold)
    f1 = (round(2 * precision * recall / (precision + recall), 4)
          if precision and recall else (0.0 if precision is not None and recall is not None else None))

    # Entity accuracy over gold reactions, via each reaction's aligned row.
    sub_hit = sub_tot = prod_hit = prod_tot = 0
    enz = collections.Counter()
    cof_missing: List[Dict[str, Any]] = []
    for g in case.reactions:
        full_rows = [r for r in uniq if g.id in r["gold_ids"] and r["class"] == TP]
        part_rows = [r for r in uniq if g.id in r["gold_ids"] and r["class"] in (CONFLICT, PARTIAL)]
        di, do = g.defining("inputs"), g.defining("outputs")
        sub_tot += len(di)
        prod_tot += len(do)
        if full_rows:
            sub_hit += len(di)
            prod_hit += len(do)
            r = full_rows[0]
            enz[r.get("enzyme_verdict", "not_applicable")] += 1
            if r.get("missing_minor_cofactors"):
                cof_missing.append({"gold_id": g.id, "row_index": r["row_index"],
                                    "missing": r["missing_minor_cofactors"]})
        elif part_rows:
            r = part_rows[0]
            missing = set(r.get("missing_defining") or ())
            sub_hit += sum(1 for t in di if t.name not in missing)
            prod_hit += sum(1 for t in do if t.name not in missing)
            enz["reaction_not_matched"] += 1
        else:
            enz["reaction_not_matched"] += 1
    enz_den = enz["correct"] + enz["wrong"] + enz["missing"] + enz[CONFLICT]
    sp = collections.Counter(r["species_verdict"] for r in uniq)
    ev = collections.Counter(r["evidence"] for r in uniq)

    old_matched = {gid for r in uniq for gid in r["old_matcher_gold_ids"]}
    old_rows = sum(1 for r in uniq if r["old_matcher_gold_ids"])
    disagreements = [
        {"row_index": r["row_index"], "inputs": r["inputs"], "outputs": r["outputs"],
         "old": r["old_matcher_gold_ids"], "v2_class": r["class"], "v2_gold_ids": r["gold_ids"]}
        for r in uniq
        if set(r["old_matcher_gold_ids"]) != (set(r["gold_ids"]) if r["class"] == TP else set())]

    return {
        "retained_reactions": len(rows),
        "unique_reactions": len(uniq),
        "duplicate_rows_collapsed": len(rows) - len(uniq),
        "row_classes": {c: counts.get(c, 0) for c in ROW_CLASSES},
        "table1_paper_extraction": {
            "rows_claimed_as_paper_extraction": len(t1),
            "true_positive_rows": tp_rows,
            "gold_reactions": n_gold,
            "gold_reactions_matched": sorted(matched_ids),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "table2_final_support": {
            "retained_unique": len(uniq),
            "unsupported": counts.get(UNSUP, 0),
            "unsupported_rate": _rate(counts.get(UNSUP, 0), len(uniq)),
            "indeterminate_external": counts.get(INDET, 0),
            "unsupported_rate_upper_bound": _rate(counts.get(UNSUP, 0) + counts.get(INDET, 0)
                                                  + counts.get(CONFLICT, 0), len(uniq)),
            "external_rag_supported": counts.get(EXT, 0),
            "paper_stated_off_pathway": counts.get(OFF, 0),
            "off_pathway_rate": _rate(counts.get(OFF, 0), len(uniq)),
            "paper_stated_summary": counts.get(SUMMARY, 0),
            "partial_match": counts.get(PARTIAL, 0),
            "name_conflict": counts.get(CONFLICT, 0),
        },
        "name_conflicts_for_hand_review": [
            {k: r[k] for k in ("row_index", "inputs", "outputs", "enzymes", "gold_ids", "conflicting_names")}
            for r in uniq if r["class"] == CONFLICT],
        "entity_accuracy": {
            "substrate_accuracy": _rate(sub_hit, sub_tot), "substrate_hits": sub_hit, "substrate_total": sub_tot,
            "product_accuracy": _rate(prod_hit, prod_tot), "product_hits": prod_hit, "product_total": prod_tot,
            "enzyme_accuracy": _rate(enz["correct"], enz_den), "enzyme_verdicts": dict(enz),
            "species_accuracy": _rate(sp["correct"], sp["correct"] + sp["wrong"]),
            "species_verdicts": dict(sp),
        },
        "minor_cofactors_missing_on_matched_reactions": cof_missing,
        "evidence": {
            "evidence_support_rate": _rate(ev["verbatim_in_paper"], len(uniq)),
            "evidence_support_rate_on_true_positives": _rate(
                sum(1 for r in uniq if r["class"] == TP and r["evidence"] == "verbatim_in_paper"),
                sum(1 for r in uniq if r["class"] == TP)),
            "verdicts": dict(ev),
            "paper_text_available": bool(folded_paper),
        },
        "old_lenient_matcher_same_rows": {
            "matched_rows": old_rows,
            "gold_reactions_matched": sorted(old_matched),
            "recall": _rate(len(old_matched), n_gold),
            "rows_where_old_and_v2_disagree": disagreements,
        },
        "rows": per_row,
    }


# --------------------------------------------------------------------------- mapped-ID correctness
_ID_FIELDS = ("uniprot", "pathbank_protein_id", "pathbank_compound_id", "kegg", "chebi", "hmdb",
              "pubchem", "drugbank", "pathbank_complex_id", "pathbank_protein_complex_id")
_XREF_KINDS = ("kegg", "chebi", "hmdb", "pubchem", "drugbank")


def _norm_id(kind: str, value: Any) -> Tuple[str, str]:
    v = str(value).strip()
    if kind == "chebi":
        v = v.upper().replace("CHEBI:", "")
    return kind, v


def _usable(value: Any) -> bool:
    return value not in (None, "", 0) and str(value).strip().casefold() not in ("unknown", "none")


REFERENCE_TABLES = ("data/id_mapping_cache.json", "data/enrichment_cache.json", "data/pathwhiz_id_db.json")


def _committed_table(root: Path, rel: str) -> Tuple[str, Any]:
    """``(blob id, parsed JSON)`` of ``rel`` as committed at HEAD. The working-tree copy
    is never read: the two caches are rewritten by every pipeline run and never
    committed, so reading them from disk would make the ID numbers depend on local
    state (review finding B-R2-1). A table missing at HEAD raises; it is not skipped."""

    blob = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()
    data = subprocess.run(["git", "cat-file", "blob", blob], cwd=root, capture_output=True, check=True).stdout
    return blob, json.loads(data.decode("utf-8"))


def build_reference_index(repo_root: Path) -> Dict[str, Any]:
    """Reference names per identifier, from the tables AS COMMITTED AT HEAD (git blobs,
    never the working-tree copies; ``tables`` records each blob id), split by authority.

    ``independent[(kind, id)]`` -- a record fetched or curated BY that identifier:
    ``data/pathwhiz_id_db.json`` (keyed the way payloads key it: ``pathbank_compound_id``
    -> ``compounds.by_id``, ``pathbank_protein_id`` -> ``proteins.by_id``,
    ``pathbank_complex_id`` / ``pathbank_protein_complex_id`` -> ``protein_complexes.by_id``)
    and ``data/enrichment_cache.json`` (KEGG / ChEBI / HMDB / UniProt records).

    ``mapper_cache[(kind, id)]`` -- ``data/id_mapping_cache.json``, the mapper's own
    candidate cache. Not an independent authority: a verdict resting only on it is
    marked so.

    ``same_record[(kind, id)]`` -- identifiers one PathWhiz record declares for itself,
    so each is judged with the names of all of them.

    The PathWhiz table mixes real PathBank records with LOCAL ids from sample .pwml
    files whose numbers collide with PathBank's (compound 14 is 'ATP' there, PathBank
    compound 14 is Iodotyrosine; protein 13 is 'diglyceride lipase' there, PathBank
    protein 13 is ACAT2). Only a record carrying an external identifier (a compound
    with KEGG/ChEBI/HMDB/PubChem/DrugBank, a protein with a UniProt accession, a complex
    with member proteins) is used; the rest would convict correct mappings.
    Read-only; nothing is fixed or written.
    """

    indep: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    mapper: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    same: Dict[Tuple[str, str], Set[Tuple[str, str]]] = collections.defaultdict(set)

    def add(idx: Dict[Tuple[str, str], Set[str]], kind: str, value: Any, *names: Any) -> None:
        if not _usable(value):
            return
        for n in names:
            if isinstance(n, str) and n.strip():
                idx[_norm_id(kind, value)].add(n.strip())

    root = Path(repo_root)
    tables: Dict[str, str] = {}
    loaded: Dict[str, Any] = {}
    for rel in REFERENCE_TABLES:
        tables[rel], loaded[rel] = _committed_table(root, rel)
    cache = loaded["data/id_mapping_cache.json"]
    for section in cache.values() if isinstance(cache, dict) else ():
        for entry in section.values() if isinstance(section, dict) else ():
            for c in (entry or {}).get("candidates") or ():
                if not isinstance(c, dict):
                    continue
                names = (c.get("name"), c.get("short_name"), c.get("gene_name"))
                for kind in ("pathbank_protein_id", "pathbank_compound_id", "uniprot",
                             "pathbank_complex_id", "pathbank_protein_complex_id"):
                    add(mapper, kind, c.get(kind), *names)
                if c.get("database") and c.get("id"):
                    add(mapper, str(c["database"]).lower(), c["id"], c.get("name"))
                for kind, value in (c.get("mapped_ids") or {}).items():
                    add(mapper, kind, value, *names)
    enr = loaded["data/enrichment_cache.json"]
    for key, rec in ((enr.get("proteins") or {}).items() if isinstance(enr, dict) else ()):
        parsed = (rec or {}).get("parsed") or {}
        if key.startswith("uniprot:") and parsed.get("status") == "ok":
            add(indep, "uniprot", key.split(":", 1)[1], parsed.get("recommended_name"),
                *(parsed.get("alternative_names") or ()), *(parsed.get("gene_names") or ()))
    for key, rec in ((enr.get("compounds") or {}).items() if isinstance(enr, dict) else ()):
        if isinstance(rec, dict) and rec.get("status") == "ok" and ":" in key:
            kind, value = key.split(":", 1)
            add(indep, kind, value, rec.get("primary_name"), *(rec.get("synonyms") or ()))
    pw = loaded["data/pathwhiz_id_db.json"]
    pw = pw if isinstance(pw, dict) else {}

    def declare(members: List[Tuple[str, str]]) -> None:
        for m in members:
            same[m] |= set(members)

    for pw_id, rec in ((pw.get("compounds") or {}).get("by_id") or {}).items():
        xrefs = [(k, rec.get(k)) for k in _XREF_KINDS if _usable(rec.get(k))]
        if not xrefs:
            continue  # a local sample id, not a PathBank record (see docstring)
        members = [_norm_id("pathbank_compound_id", pw_id)] + [_norm_id(k, v) for k, v in xrefs]
        for kind, value in members:
            add(indep, kind, value, rec.get("name"))
        declare(members)
    proteins = (pw.get("proteins") or {}).get("by_id") or {}
    for pw_id, rec in proteins.items():
        if not _usable(rec.get("uniprot")):
            continue
        members = [_norm_id("pathbank_protein_id", pw_id), _norm_id("uniprot", rec["uniprot"])]
        for kind, value in members:
            add(indep, kind, value, rec.get("name"), rec.get("gene"))
        declare(members)
    for pw_id, rec in ((pw.get("protein_complexes") or {}).get("by_id") or {}).items():
        if any(_usable((proteins.get(str(p)) or {}).get("uniprot")) for p in rec.get("protein_ids") or ()):
            for kind in ("pathbank_complex_id", "pathbank_protein_complex_id"):
                add(indep, kind, pw_id, rec.get("name"))
    return {"independent": dict(indep), "mapper_cache": dict(mapper), "same_record": dict(same),
            "tables": tables}


def _declared_keys(name: Any) -> Set[str]:
    """Every name an entity or a record declares: whole, outer AND inside of a trailing
    parenthetical. Identity is about which names a thing carries; the B2 rule (an inside
    never credits a GOLD term its outer name contradicts) governs reaction scoring."""

    return {k for k in (*M.candidate_keys(name), M.inner_key(name)) if k}


def _name_hit(claim: Set[str], names: Iterable[str]) -> bool:
    ref_keys = set().union(*(_declared_keys(n) for n in names))
    # Equality only, plus one identity-check-only relaxation: spacing inside a name
    # ('lipid IV_A' vs 'lipid IVA') is compared with spaces removed.
    compact = lambda ks: {k.replace(" ", "") for k in ks}  # noqa: E731
    return bool(claim & ref_keys or compact(claim) & compact(ref_keys))


def _row_records(row: Mapping[str, Any]) -> Dict[Tuple[str, str], Set[str]]:
    """Names from the row's own enrichment sub-records, each fetched BY one identifier
    (KEGG / ChEBI / HMDB id, UniProt accession): record-side, independent."""

    out: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    enr = row.get("enrichment") if isinstance(row.get("enrichment"), dict) else {}
    comp = enr.get("compound") if isinstance(enr.get("compound"), dict) else {}
    for kind in ("kegg", "chebi", "hmdb"):
        sub = comp.get(kind) if isinstance(comp.get(kind), dict) else {}
        if sub.get("status") == "ok" and _usable(sub.get("id")):
            out[_norm_id(kind, sub["id"])] |= {n for n in [sub.get("primary_name"), *(sub.get("synonyms") or ())] if n}
    uni = ((enr.get("protein") or {}).get("uniprot") or {}) if isinstance(enr.get("protein"), dict) else {}
    if uni.get("status") == "ok" and _usable(uni.get("uniprot_id")):
        out[_norm_id("uniprot", uni["uniprot_id"])] |= {
            n for n in [uni.get("recommended_name"), *(uni.get("alternative_names") or ()),
                        *(uni.get("gene_names") or ())] if n}
    return out


def id_correctness(payload: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, Any]:
    """Per entity: ``correct_identity`` | ``wrong_identity`` | ``unresolved_or_novel`` |
    ``not_evaluable_no_reference``, judged PER IDENTIFIER against that identifier's own
    record -- never a union across identifiers, so one correct KEGG id cannot hide a
    wrong PathWhiz id.

    Each identifier: ``correct`` if the claim matches its independent record (plus the
    independent records of identifiers the same PathWhiz record declares), else
    ``correct`` if it matches the mapper's candidate cache (``independent`` False),
    else ``wrong`` when any reference exists (``independent`` True iff an independent
    record exists), else ``no_reference``. The entity is ``wrong_identity`` if ANY
    identifier is wrong. The claim side is the entity NAME only (never its
    ``synonyms``, which may have been written from the very record under test)."""

    indep = ref.get("independent") or {}
    mapper = ref.get("mapper_cache") or {}
    same = ref.get("same_record") or {}
    out: List[Dict[str, Any]] = []
    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    for bucket in ("compounds", "proteins", "protein_complexes"):
        for row in ents.get(bucket) or ():
            if not isinstance(row, dict) or not row.get("name"):
                continue
            ids: Dict[str, str] = {}
            mapped = row.get("mapped_ids") if isinstance(row.get("mapped_ids"), dict) else {}
            meta = row.get("mapping_meta") if isinstance(row.get("mapping_meta"), dict) else {}
            for k in _ID_FIELDS:
                v = mapped.get(k) or row.get(k) or meta.get(k)
                if _usable(v):
                    ids[k] = str(v)
            claim = _declared_keys(row["name"]) | (_declared_keys(row["raw_name"]) if row.get("raw_name") else set())
            rec: Dict[str, Any] = {"bucket": bucket, "name": row["name"], "ids": ids}
            if not ids or str(meta.get("chosen_rule", "")).startswith("novel") or row.get("generated"):
                rec["verdict"] = "unresolved_or_novel"
                out.append(rec)
                continue
            own = _row_records(row)
            per_id: List[Dict[str, Any]] = []
            for k, v in ids.items():
                key = _norm_id(k, v)
                ind: Set[str] = set()
                for m in same.get(key, set()) | {key}:
                    ind |= set(indep.get(m, ())) | own.get(m, set())
                # The DB records the mapper cached in the payload for this identifier.
                mc = set(mapper.get(key, ()))
                for c in meta.get("candidates") or ():
                    if isinstance(c, dict) and c.get(k) is not None and str(c.get(k)) == v:
                        mc |= {n for n in (c.get("name"), c.get("short_name"), c.get("gene_name")) if n}
                if ind and _name_hit(claim, ind):
                    v_rec = {"verdict": "correct", "independent": True}
                elif mc and _name_hit(claim, mc):
                    v_rec = {"verdict": "correct", "independent": False}
                elif ind or mc:
                    v_rec = {"verdict": "wrong", "independent": bool(ind),
                             "reference_names": sorted(ind or mc)[:12]}
                else:
                    v_rec = {"verdict": "no_reference", "independent": False}
                per_id.append({"id": f"{k}:{v}", **v_rec})
            rec["per_id"] = per_id
            wrong = [p for p in per_id if p["verdict"] == "wrong"]
            right = [p for p in per_id if p["verdict"] == "correct"]
            if wrong:
                rec["verdict"] = "wrong_identity"
                rec["independent"] = any(p["independent"] for p in wrong)
                rec["contradicting_ids"] = [p["id"] for p in wrong]
            elif right:
                rec["verdict"] = "correct_identity"
                rec["independent"] = any(p["independent"] for p in right)
            else:
                rec["verdict"] = "not_evaluable_no_reference"
            out.append(rec)
    tally = collections.Counter(r["verdict"] for r in out)
    mapper_only = collections.Counter(r["verdict"] for r in out if r.get("independent") is False)
    return {"counts": dict(tally), "counts_resting_on_mapper_cache_only": dict(mapper_only),
            "wrong": [r for r in out if r["verdict"] == "wrong_identity"], "entities": len(out)}


# --------------------------------------------------------------------------- leakage audit
def _admission_records(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    recs = []
    for group in ("accepted", "rejected"):
        for r in doc.get(group) or ():
            if isinstance(r, dict):
                recs.append({"group": group, "inputs": r.get("inputs") or [], "outputs": r.get("outputs") or [],
                             "span": str((r.get("evidence") or {}).get("span") or ""),
                             "source_id": (r.get("evidence") or {}).get("source_id") or ""})
    return recs


def _keyset(names: Iterable[str]) -> frozenset:
    return frozenset(M.strict_key(n) for n in names if M.strict_key(n))




def leakage_audit(stage1: Mapping[str, Any], stage2: Mapping[str, Any], admission: Mapping[str, Any],
                  seed_text: Optional[str], target_paper: str = "") -> Dict[str, Any]:
    """Un-gated RAG-influenced reaction candidates in one leg. Reports, changes nothing.

    A Stage-2 row is a candidate when ALL of: (1) its input/output key sets match no
    row of the strict Stage-1 payload; (2) it corresponds to RAG-provided evidence --
    ``admission_match`` (a persisted candidate, either verdict, with the same key sets),
    ``span_in_row_evidence`` (a retrieved span of >= 40 folded chars inside the row's
    own evidence) or ``participant_only_in_rag`` (a participant absent from the seed
    paper text but present in a retrieved span); (3) NO accepted admission record has
    its key sets.

    Each candidate also says WHOSE retrieved text it corresponds to: a retrieved chunk
    of the seed paper itself (``seed_paper`` or the target id) is retrieval context but
    not external biology, so ``external_evidence`` is reported separately.
    """

    s1 = {(_keyset(M.participant_names(r.get("inputs"))), _keyset(M.participant_names(r.get("outputs"))))
          for r in ((stage1.get("processes") or {}).get("reactions") or []) if isinstance(r, dict)}
    recs = _admission_records(admission)
    by_key = collections.defaultdict(list)
    for r in recs:
        by_key[(_keyset(r["inputs"]), _keyset(r["outputs"]))].append(r)
    spans = [(fold_for_quote(r["span"]), r["source_id"]) for r in recs if len(fold_for_quote(r["span"])) >= 40]
    keyed = [(M.text_key(r["span"]), r["source_id"]) for r in recs]
    own = {"seed_paper", str(target_paper or "")} - {""}
    keyed_seed = M.text_key(seed_text) if seed_text else ""
    rows = [r for r in ((stage2.get("processes") or {}).get("reactions") or []) if isinstance(r, dict)]
    candidates, new_rows = [], 0
    for i, row in enumerate(rows):
        ins, outs = M.participant_names(row.get("inputs")), M.participant_names(row.get("outputs"))
        key = (_keyset(ins), _keyset(outs))
        if key in s1:
            continue
        new_rows += 1
        matched = by_key.get(key, [])
        ev = fold_for_quote(" ".join(_evidence_spans(row)))
        criteria, sources = [], set()
        if matched:
            criteria.append("admission_match")
            sources |= {m["source_id"] for m in matched}
        hit = {sid for s, sid in spans if ev and s in ev}
        if hit:
            criteria.append("span_in_row_evidence")
            sources |= hit
        only = {sid for n in ins + outs if keyed_seed and not M.mentions(n, keyed_seed)
                for k, sid in keyed if M.mentions(n, k)}
        if only:
            criteria.append("participant_only_in_rag")
            sources |= only
        if not criteria or any(m["group"] == "accepted" for m in matched):
            continue
        candidates.append({"row_index": i, "name": row.get("name") or "", "inputs": ins, "outputs": outs,
                           "criteria": criteria,
                           "matched_admission_verdicts": sorted({m["group"] for m in matched}),
                           "evidence_sources": sorted(str(x) for x in sources),
                           "external_evidence": any(str(x) and str(x) not in own for x in sources),
                           "row_carries_rag_provenance": bool(row.get("rag_provenance"))})
    return {"stage2_rows": len(rows), "rows_absent_from_stage1": new_rows,
            "ungated_rag_influenced_candidates": candidates,
            "seed_text_available": bool(keyed_seed)}


__all__ = [
    "CONFLICT", "EXT", "INDET", "OFF", "PARTIAL", "ROW_CLASSES", "SUMMARY", "TP", "UNSUP",
    "species_key",
    "build_reference_index", "evidence_verdict", "id_correctness", "leakage_audit",
    "load_lineage", "match_reaction", "score_leg",
]
