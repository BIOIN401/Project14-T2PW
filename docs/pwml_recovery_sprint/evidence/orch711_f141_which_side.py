"""F-141, corrected: WHICH SIDE of the species comparison was silent?

An earlier pass of this classification labelled 22 rows
``species_evidence_lost_across_stage_boundary`` on the strength of the ENTITY
carrying ``explicit_entity_species`` / matched / *E. coli* / tax 562 at
confidence 1.0 while the ladder's ``species`` rung read ``unknown``. That
inference does not hold. ``_candidate_species_verdict`` returns ``unknown`` when
EITHER side is silent:

    if not requested: return "unknown"     # the ENTITY side
    ...
    if not declared:  return "unknown"     # the CANDIDATE side

So the entity carrying a species proves nothing on its own. This probe measures
the side that was actually silent, which is the only thing that separates a
recoverable propagation loss from correct fail-closed withholding.

ASCII-only output on purpose.
"""
import glob
import io
import json
import re
import sys

PINNED = "runs/2026-08-02_2130"
ACC = re.compile(r"uniprot:([A-Z0-9]{6,10})")
REAL = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}$")


def d(x):
    return x if isinstance(x, dict) else {}


def cand_has_species(c):
    c = d(c)
    return bool(
        str(c.get("organism") or c.get("species") or "").strip()
        or str(c.get("taxonomy_id") or c.get("taxon_id") or "").strip()
    )


def withheld(root):
    for fp in sorted(glob.glob(root + "/papers/*/*/final_mapped.json")):
        p = fp.replace("\\", "/")
        parts = p.split("/")
        leg = parts[-3] + "/" + parts[-2]
        payload = json.load(io.open(p, encoding="utf-8"))
        ents = payload.get("entities") or {}
        for bucket in ("proteins", "protein_complexes"):
            for idx, r in enumerate(ents.get(bucket) or []):
                if not isinstance(r, dict):
                    continue
                meta = d(r.get("mapping_meta"))
                iv = d(meta.get("identity_verdict"))
                if not iv:
                    continue
                held = [a for a in ACC.findall(json.dumps(iv)) if REAL.match(a)]
                if not held:
                    continue
                mids = d(r.get("mapped_ids"))
                shipped = str(mids.get("uniprot") or r.get("uniprot_id") or r.get("uniprot") or "")
                if any(a == shipped for a in held):
                    continue
                yield leg, bucket, idx, r, meta, iv, held, shipped


def classify(r, meta, iv):
    checks = d(iv.get("checks"))
    rung = str(checks.get("species") or "")
    requested = str(iv.get("organism") or "").strip()
    pool = meta.get("candidates") or []
    jc = d(iv.get("judged_candidate"))
    jcs = [c for c in (iv.get("judged_candidates") or []) if isinstance(c, dict)]
    any_cand = bool(jc) or bool(jcs) or bool(pool)
    cand_species = cand_has_species(jc) or any(cand_has_species(c) for c in jcs) \
        or any(cand_has_species(c) for c in pool)

    if not rung:
        return "candidate_does_not_describe_shipped", requested, any_cand, cand_species
    if rung in ("mismatch", "conflict"):
        return "conflicting_species_evidence", requested, any_cand, cand_species
    if rung == "unknown":
        if not requested:
            return "entity_species_silent__propagation_loss", requested, any_cand, cand_species
        if not any_cand:
            return "no_candidate_record_at_all__withholding_correct", requested, any_cand, cand_species
        if not cand_species:
            return "candidate_carries_no_species__withholding_correct", requested, any_cand, cand_species
        return "both_sides_present_rung_still_unknown__INVESTIGATE", requested, any_cand, cand_species
    return "other_measured_mechanism", requested, any_cand, cand_species


def run(root, title, show=False):
    print("=" * 78)
    print(title)
    print("=" * 78)
    counts = {}
    n = 0
    for leg, bucket, idx, r, meta, iv, held, shipped in withheld(root):
        n += 1
        cls, requested, any_cand, cand_species = classify(r, meta, iv)
        counts[cls] = counts.get(cls, 0) + 1
        if show:
            print("  %-26s %-24s held=%-8s requested=%-18s cand=%s cand_species=%s"
                  % (leg, str(r.get("name") or "")[:24], held[0],
                     (requested or "(SILENT)")[:18], any_cand, cand_species))
            print("       -> %s" % cls)
    print("")
    print("population: %d" % n)
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("   %-52s %d" % (k, v))
    print("")
    return counts


def main():
    run(PINNED, "G1. The 24 pinned withheld rows -- which side was silent?", show=True)
    roots = sorted({p.replace("\\", "/").split("/papers/")[0]
                    for p in glob.glob("runs*/**/papers/*/*/final_mapped.json", recursive=True)})
    total = {}
    for root in roots:
        for leg, bucket, idx, r, meta, iv, held, shipped in withheld(root):
            cls, _, _, _ = classify(r, meta, iv)
            total[cls] = total.get(cls, 0) + 1
    print("=" * 78)
    print("G2. Corpus-wide (%d roots)" % len(roots))
    print("=" * 78)
    print("population: %d" % sum(total.values()))
    for k, v in sorted(total.items(), key=lambda kv: -kv[1]):
        print("   %-52s %d" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
