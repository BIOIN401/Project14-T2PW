"""ORCH-725: why did identity resolution block strict PWML on the unseen pilot?

Read-only diagnostic tooling. Nothing under ``src/`` imports it. It writes no
cache, runs no pipeline leg and changes no production behaviour.

Two independent measurements, deliberately kept in one file so they are read
together:

PART A -- is the F-185 export block STALE, or does the shipped payload really
fail?
    The same question ``orch716_stale_verdict_probe.py`` settled for the
    development cohort, asked again of the four unseen legs the pilot filed
    ``fail`` with ``failure_kind=contract``. It imports the SAME production
    predicates ``process_normalizer`` calls -- ``protein_external_identity``
    and ``protein_species_context`` -- and applies them row by row to the
    payload that actually reached export, ``final_mapped.json``.

    Re-reading ``final_stage3_gate_report.json`` would only restate what that
    report already says (F-144: asserting that *a* report is clean is not
    evidence that *the payload* is clean), so the predicates are applied
    directly.

PART B -- when the resolver refused a protein, WHAT did it refuse?
    Every rejected protein row carries ``mapping_meta.identity_verdict``, which
    since D-003 retains ``judged_candidate`` and ``judged_candidates`` -- the
    rows the ladder actually judged -- even though ``mapping_meta.candidates``
    is emptied by the first enforcement pass. This part reconstructs, per
    refusal, whether the rivals that defeated the margin check were
    *the same gene symbol in the same organism* as the shipped accession.

    That distinction is the whole diagnosis. A rival that is a different
    protein is a real ambiguity and the refusal is correct. A rival that is the
    same gene in the same organism -- a TrEMBL duplicate of a Swiss-Prot entry,
    or a second strain of the same species -- is not an ambiguity about
    identity at all, and the refusal is an artifact of the scoring.

Neither part is a licence to change anything. Whether these legs SHOULD export
is a product question answered by PRODUCT_CONTRACT and the F-147 precondition,
not by this probe.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The unseen-pilot strict legs the run filed FAIL with failure_kind=contract,
#: plus the one strict leg that produced a PWML as a control.
LEGS: Tuple[Tuple[str, str, str], ...] = (
    ("PMC7232280", "strict", "FAIL -- 2 identity issue codes (Neurospora crassa)"),
    ("PMC8510960", "strict", "FAIL -- 7 identity issue codes (Catharanthus roseus)"),
    ("PMC12376012", "strict", "FAIL -- 10 identity issue codes (Homo sapiens)"),
    ("PMC11172790", "strict", "FAIL -- 1 registry-validation issue (P. aeruginosa)"),
    ("PMC12071552", "strict", "control: PASSED, produced the pilot's only PWML"),
)

#: Legs whose refusals PART B reads. Research legs are included here and only
#: here: research never exports, so they cannot appear in PART A, but their
#: refusals are the same resolver on the same corpus and are evidence.
VERDICT_LEGS: Tuple[Tuple[str, str], ...] = (
    ("PMC11172790", "research"),
    ("PMC11172790", "strict"),
    ("PMC11405693", "research"),
    ("PMC11405693", "strict"),
    ("PMC12071552", "strict"),
    ("PMC12376012", "research"),
    ("PMC12376012", "strict"),
    ("PMC12542839", "research"),
    ("PMC7232280", "strict"),
    ("PMC8510960", "strict"),
)

#: Keys an identity can arrive under. Printed per row so a reader can tell an
#: absent identity from one filed under a key some other reader does not
#: consult -- PMC12180156's ALAS2 carried ``uniprot`` while ``uniprot_id`` was
#: ``None``, and F-144 is the rule that a zero from a key that does not exist
#: looks exactly like a zero from a measurement.
IDENTITY_KEYS = ("uniprot", "uniprot_id", "drugbank", "drugbank_id", "mapped_ids")


def load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def binomial(organism: Any) -> str:
    """First two whitespace tokens of an organism string, stripped of any
    parenthetical. ``Staphylococcus aureus (strain NCTC 8325 / PS 47)`` and
    ``Staphylococcus aureus (MRSA N315)`` both reduce to ``staphylococcusaureus``.
    """
    text = re.sub(r"\(.*?\)", " ", str(organism or ""))
    tokens = [tok for tok in re.split(r"\s+", text) if tok]
    return norm(" ".join(tokens[:2]))


def gene_symbols(row: Dict[str, Any]) -> List[str]:
    return [norm(g) for g in (row.get("gene_names") or []) if norm(g)]


def part_a(root: str, run: str) -> Dict[str, str]:
    sys.path.insert(0, os.path.join(root, "src"))
    from t2pw.pipeline.process_normalizer import (  # noqa: E402
        protein_external_identity,
        protein_species_context,
    )

    print("=" * 92)
    print("PART A -- production identity/species predicates applied to the SHIPPED payload")
    print("=" * 92)
    print(f"root : {root}")
    print(f"run  : {run}")
    print()

    verdict: Dict[str, str] = {}

    for paper, mode, note in LEGS:
        base = os.path.join(root, run, "papers", paper, mode)
        payload = load(os.path.join(base, "final_mapped.json"))
        gate = load(os.path.join(base, "final_stage3_gate_report.json"))
        reports = load(os.path.join(base, "contract_reports.json"))

        print("#" * 92)
        print(f"{paper}/{mode}   [{note}]")
        if payload is None:
            print("  no final_mapped.json -- leg produced no canonical payload")
            verdict[f"{paper}/{mode}"] = "no_payload"
            print()
            continue

        if gate is not None:
            print(
                f"  final_stage3_gate_report : ok={gate.get('ok')} "
                f"errors={len(gate.get('errors') or [])} stage={gate.get('stage')}"
            )
        if reports:
            for key, value in reports.items():
                if not isinstance(value, dict) or key.endswith("runtime_schema_report"):
                    continue
                if value.get("ok") is False or value.get("errors"):
                    print(
                        f"  FAILING CONTRACT REPORT  : {key} "
                        f"phase={value.get('phase')} stage={value.get('stage')} "
                        f"errors={len(value.get('errors') or [])}"
                    )
                    for err in value.get("errors") or []:
                        detail = (
                            err
                            if isinstance(err, str)
                            else (err.get("message") or err.get("reason") or json.dumps(err)[:150])
                        )
                        pointer = "" if isinstance(err, str) else (err.get("pointer") or err.get("path") or "")
                        print(f"      - {pointer} {str(detail)[:120]}")

        entities = payload.get("entities") or {}
        proteins = entities.get("proteins") or []
        print(f"  proteins in shipped payload: {len(proteins)}")
        still_failing = 0
        for idx, row in enumerate(proteins):
            ext = protein_external_identity(row)
            species = protein_species_context(row)
            present = [k for k in IDENTITY_KEYS if row.get(k) not in (None, "", {}, [])]
            flag = ""
            if not ext:
                flag += "  <-- STILL MISSING IDENTIFIER"
                still_failing += 1
            if not species:
                flag += "  <-- STILL MISSING SPECIES"
                still_failing += 1
            print(
                f"    [{idx}] {str(row.get('name')):<28} identity={str(ext)[:34]:<36} "
                f"species={str(species)[:20]:<22} keys={present}{flag}"
            )

        complexes = entities.get("protein_complexes") or []
        if complexes:
            print(f"  protein_complexes ({len(complexes)}):")
            for cx in complexes:
                meta = cx.get("mapping_meta") or {}
                print(
                    f"      {str(cx.get('name'))[:44]:<46} "
                    f"rule={str(meta.get('chosen_rule') or '-'):<34} "
                    f"target_organism={meta.get('target_organism') or '-'}"
                )

        # Which of the entities the run's issue codes named are still present as
        # BARE proteins. A pointer like /entities/proteins/13 into an 8-row list
        # is by itself proof the report describes a payload that no longer exists.
        named: List[str] = []
        for value in (reports or {}).values():
            if not isinstance(value, dict):
                continue
            for err in value.get("errors") or []:
                if isinstance(err, dict):
                    detail = err.get("detail") or {}
                    if isinstance(detail, dict) and detail.get("name"):
                        named.append(str(detail["name"]))
        if named:
            protein_names = {norm(r.get("name")) for r in proteins}
            complex_names = {norm(c.get("name")) for c in complexes}
            print("  entities the FAILING report named, located in the shipped payload:")
            for name in sorted(set(named)):
                where = (
                    "STILL a bare protein"
                    if norm(name) in protein_names
                    else "now a protein_complex"
                    if norm(name) in complex_names
                    else "absent (dropped or quarantined before export)"
                )
                print(f"      {name:<44} -> {where}")
            # A pointer index the shipped list cannot host is, on its own, proof
            # that the failing report describes a payload that no longer exists.
            indices: List[int] = []
            for value in (reports or {}).values():
                if not isinstance(value, dict):
                    continue
                for err in value.get("errors") or []:
                    if not isinstance(err, dict):
                        continue
                    where = str(err.get("path") or err.get("pointer") or "")
                    match = re.search(r"/entities/proteins/(\d+)", where)
                    if match:
                        indices.append(int(match.group(1)))
            if indices:
                highest = max(indices)
                hostable = highest < len(proteins)
                print(
                    f"  highest /entities/proteins/N index in the failing report: {highest}; "
                    f"shipped protein rows: {len(proteins)} -> "
                    f"{'hostable' if hostable else 'THE SHIPPED LIST CANNOT HOST IT'}"
                )

        verdict[f"{paper}/{mode}"] = (
            "shipped payload STILL fails" if still_failing else "shipped payload PASSES"
        )
        print(f"  => {verdict[f'{paper}/{mode}']} ({still_failing} row-level objection(s))")
        print()

    print("=" * 92)
    print("PART A VERDICT")
    print("=" * 92)
    for leg, value in verdict.items():
        print(f"  {leg:<30} {value}")
    print()
    print("  'shipped payload PASSES' on a leg the run FAILED is the STALE reading: the run")
    print("  was failed on a superseded audit_round report, not on the payload that would")
    print("  have been exported. That is F-147, already registered, and PRODUCT_CONTRACT")
    print("  section 1 names a stale gate report an unacceptable terminal blocker.")
    print("  It is NOT on its own a licence to make these legs pass -- F-147 carries a")
    print("  documented precondition and it is a product decision, not this probe's.")
    print()
    return verdict


def part_b(root: str, run: str) -> None:
    print()
    print("=" * 92)
    print("PART B -- what did the resolver actually refuse? (identity_verdict, D-003 retained rows)")
    print("=" * 92)
    print()

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for paper, mode in VERDICT_LEGS:
        base = os.path.join(root, run, "papers", paper, mode)
        payload = load(os.path.join(base, "final_mapped.json"))
        if payload is None:
            continue
        entities = payload.get("entities") or {}
        pool: List[Tuple[str, Dict[str, Any]]] = [
            ("protein", r) for r in (entities.get("proteins") or []) if isinstance(r, dict)
        ]
        pool += [
            ("protein(quarantined)", (r.get("original_row") or r))
            for r in (payload.get("quarantined_proteins") or [])
            if isinstance(r, dict)
        ]
        # A protein the Unknown-sentinel fallback rerouted into a wrapper is GONE
        # from entities.proteins by the time final_mapped.json is written, so its
        # identity_verdict survives only in the pre-repair gate snapshot. Those are
        # exactly the rows that decide whether an export ships a real accession or
        # a placeholder, so leaving them out understates the census -- G10H, whose
        # judged candidate is the reviewed Swiss-Prot Q8VWZ7, is one of them.
        snapshot = load(os.path.join(base, "gate_fail_report.json")) or {}
        for err in snapshot.get("errors") or []:
            if isinstance(err, dict) and isinstance(err.get("detail"), dict):
                pool.append(("protein(pre-repair snapshot)", err["detail"]))
        for bucket, row in pool:
            meta = row.get("mapping_meta") or {}
            verdict = meta.get("identity_verdict") or {}
            if not verdict or verdict.get("verified"):
                continue
            key = (paper, mode, norm(row.get("name")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "paper": paper,
                    "mode": mode,
                    "bucket": bucket,
                    "name": row.get("name"),
                    "organism": row.get("organism") or row.get("species") or "",
                    "reason": verdict.get("reason"),
                    "verdict": verdict,
                    "queries": meta.get("queries_tried") or [],
                    "provider": meta.get("provider"),
                }
            )

    def classify(entry: Dict[str, Any]) -> Tuple[str, str]:
        """(mechanism, one-line justification) for one refusal."""
        verdict = entry["verdict"]
        reason = str(verdict.get("reason") or "")
        judged = verdict.get("judged_candidate")
        judged = judged if isinstance(judged, dict) else {}
        raw_pool = verdict.get("judged_candidates")
        pool = [c for c in (raw_pool or []) if isinstance(c, dict)] if isinstance(raw_pool, list) else []
        # The pre-repair gate snapshot elides long nested structures ("2 item(s),
        # 19 chars elided"), so the rival ROWS may be unavailable while the
        # verdict's own margin arithmetic survives. Classify from that rather
        # than dropping the row -- silence here would delete the very cases the
        # Unknown fallback rerouted, which are the ones that decide an export.
        pool_elided = not pool and bool(verdict.get("competing_accessions"))
        want_org = binomial(entry["organism"])

        if reason == "ambiguous_insufficient_margin" and pool_elided:
            margin = verdict.get("margin")
            return (
                "REDUNDANT-RIVAL (same gene, same organism as the shipped accession)"
                if judged.get("reviewed") is True
                else "MARGIN REJECT, rival rows elided in the snapshot",
                f"judged {judged.get('accession')} ({judged.get('organism')}) "
                f"reviewed={judged.get('reviewed')} margin={margin}; "
                f"{len(verdict.get('competing_accessions') or [])} competing accession(s)",
            )

        if reason == "ambiguous_insufficient_margin":
            j_genes = set(gene_symbols(judged))
            j_org = binomial(judged.get("organism"))
            competing = {str(a).casefold() for a in (verdict.get("competing_accessions") or [])}
            redundant, distinct = [], []
            for cand in pool:
                acc = str(cand.get("accession") or "").casefold()
                if acc not in competing:
                    continue
                same_gene = bool(j_genes & set(gene_symbols(cand))) if j_genes else False
                same_org = binomial(cand.get("organism")) == j_org and bool(j_org)
                (redundant if (same_gene and same_org) else distinct).append(
                    (acc, cand.get("organism"), cand.get("gene_names"), cand.get("reviewed"))
                )
            if redundant and not distinct:
                return (
                    "REDUNDANT-RIVAL (same gene, same organism as the shipped accession)",
                    f"{len(redundant)} rival(s), all same gene+organism; reviewed(judged)="
                    f"{judged.get('reviewed')}, margin={verdict.get('margin')}",
                )
            if redundant and distinct:
                return (
                    "MIXED-RIVAL",
                    f"{len(redundant)} redundant + {len(distinct)} genuinely distinct rival(s)",
                )
            return ("GENUINE AMBIGUITY", f"{len(distinct)} distinct rival(s)")

        if reason == "species_mismatch":
            j_org = binomial(judged.get("organism"))
            organism_match_in_pool = [
                c for c in pool if binomial(c.get("organism")) == want_org and want_org
            ]
            if organism_match_in_pool:
                return (
                    "WRONG-CANDIDATE-SELECTED (an organism-matching row WAS in the pool)",
                    f"judged {judged.get('accession')} ({judged.get('organism')}) but "
                    f"{len(organism_match_in_pool)} pool row(s) match {entry['organism']}: "
                    + ", ".join(str(c.get("accession")) for c in organism_match_in_pool[:4]),
                )
            strain_only = j_org == want_org and bool(want_org)
            if strain_only:
                return (
                    "STRAIN-PARENTHETICAL (binomial agrees; only the strain suffix differs)",
                    f"requested '{entry['organism']}' vs candidate '{judged.get('organism')}'",
                )
            return (
                "NO ORGANISM-MATCHING RECORD IN POOL",
                f"judged {judged.get('accession')} is {judged.get('organism')}; "
                f"0 of {len(pool)} pool rows are {entry['organism']}",
            )

        if reason == "implausible_name_match":
            j_org = binomial(judged.get("organism"))
            exact_gene = norm(entry["name"]) in set(gene_symbols(judged))
            stripped = norm(re.sub(r"^(Cr|At|Sc|Hs|Nc)(?=[A-Z0-9])", "", str(entry["name"])))
            prefix_gene = stripped in set(gene_symbols(judged)) or stripped == norm(judged.get("protein_name"))
            if judged.get("accession") and j_org == want_org and (exact_gene or prefix_gene):
                return (
                    "NAME-GATE vs EXACT GENE+ORGANISM MATCH",
                    f"{judged.get('accession')} gene={judged.get('gene_names')} "
                    f"org={judged.get('organism')} display='{judged.get('protein_name')}'",
                )
            if not judged.get("accession"):
                return ("NAME-GATE, no accession on the judged row", "generic/family-level name")
            return ("NAME-GATE reject", f"judged {judged.get('accession')} {judged.get('protein_name')}")

        if reason == "identity_evidence_missing":
            return (
                "NO CANDIDATE DESCRIBES THE SHIPPED ACCESSION",
                f"shipped {verdict.get('identity')}, pool of {len(pool)} describes none of it",
            )
        return (reason or "(no reason recorded)", "")

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for entry in rows:
        mech, why = classify(entry)
        entry["mechanism"] = mech
        entry["why"] = why
        buckets.setdefault(mech, []).append(entry)

    for mech in sorted(buckets, key=lambda k: -len(buckets[k])):
        entries = buckets[mech]
        papers = sorted({e["paper"] for e in entries})
        print("-" * 92)
        print(f"{mech}")
        print(f"  entities: {len(entries)}   papers: {len(papers)}  {papers}")
        for entry in entries:
            print(
                f"    {entry['paper']}/{entry['mode']:<8} {str(entry['name'])[:34]:<36} "
                f"org={str(entry['organism'])[:26]:<28} reason={entry['reason']}"
            )
            print(f"        {entry['why']}")
        print()

    print("=" * 92)
    print("PART B SUMMARY -- refusals by mechanism")
    print("=" * 92)
    for mech in sorted(buckets, key=lambda k: -len(buckets[k])):
        print(f"  {len(buckets[mech]):3d}  {mech}")
    print(f"  {len(rows):3d}  TOTAL refused protein rows")
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = args[0] if args else "."
    run = args[1] if len(args) > 1 else "runs_verify/2026-09-06_1425"
    part_a(root, run)
    part_b(root, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
