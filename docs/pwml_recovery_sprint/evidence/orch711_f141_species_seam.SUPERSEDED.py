"""F-141 sub-classification: was species EVIDENCE available on the 24 withheld rows?

Probe D established the population (24 pinned / 82 corpus-wide) under the criterion
"the identity verdict names a real accession and the row does not ship it". 22 of
the 24 failed the ladder's ``species`` rung with ``unknown``.

This asks the question that decides whether ANY production correction is
authorised (D-070 section O-1c, LEDGER F-141): on those same rows, what did
``hydrate_species_references`` / ``_stamp_entity_species`` already resolve? A row
whose ``species_ref`` is source-supported and matched, while the identity ladder's
species rung reads ``unknown``, is evidence LOST ACROSS A STAGE BOUNDARY -- not a
genuinely species-unknown row -- and restoring it infers nothing.

ASCII-only output on purpose (ORCH-710 job 02 died printing cp1252).
"""
import glob
import io
import json
import os
import re
import sys

PINNED = "runs/2026-08-02_2130"
ACC = re.compile(r"uniprot:([A-Z0-9]{6,10})")
REAL = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}$")

SOURCE_SUPPORTED = {
    "explicit_entity_species",
    "single_pathway_species",
    "biological_state_species",
}


def d(x):
    return x if isinstance(x, dict) else {}


def rows(root):
    for fp in sorted(glob.glob(root + "/papers/*/*/final_mapped.json")):
        p = fp.replace("\\", "/")
        parts = p.split("/")
        leg = parts[-3] + "/" + parts[-2]
        payload = json.load(io.open(p, encoding="utf-8"))
        ents = payload.get("entities") or {}
        for bucket in ("proteins", "protein_complexes"):
            for idx, r in enumerate(ents.get(bucket) or []):
                if isinstance(r, dict):
                    yield leg, bucket, idx, r


def withheld(root):
    for leg, bucket, idx, r in rows(root):
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
        yield leg, bucket, idx, r, iv, held, shipped


def species_view(r):
    meta = d(r.get("mapping_meta"))
    ref = d(r.get("species_ref")) or d(meta.get("species_resolution"))
    return {
        "ref_source": str(ref.get("source") or ""),
        "ref_status": str(ref.get("status") or ""),
        "ref_name": str(ref.get("name") or ""),
        "ref_taxonomy": str(ref.get("taxonomy_id") or ""),
        "ref_confidence": ref.get("confidence"),
        "row_species": str(r.get("species") or ""),
        "row_species_name": str(r.get("species_name") or ""),
        "row_organism": str(r.get("organism") or ""),
        "row_taxonomy": str(r.get("taxonomy_id") or ""),
    }


def main():
    print("=" * 78)
    print("F1. The 24 pinned withheld rows -- what species evidence did they carry?")
    print("=" * 78)
    buckets = {}
    detail = []
    for leg, bucket, idx, r, iv, held, shipped in withheld(PINNED):
        checks = d(iv.get("checks"))
        sp_rung = str(checks.get("species") or "")
        sv = species_view(r)
        supported = sv["ref_source"] in SOURCE_SUPPORTED and sv["ref_status"] == "matched"
        real_org = bool(sv["row_species"]) and sv["row_species"].casefold() not in (
            "unknown species", "unknown", "",
        )
        if sp_rung == "unknown":
            if supported and real_org:
                cls = "species_evidence_lost_across_stage_boundary"
            elif real_org:
                cls = "species_present_but_not_source_supported"
            else:
                cls = "species_unresolved_withholding_correct"
        elif not sp_rung:
            cls = "candidate_does_not_describe_shipped"
        elif sp_rung in ("mismatch", "conflict"):
            cls = "conflicting_species_evidence"
        else:
            cls = "other_measured_mechanism"
        buckets[cls] = buckets.get(cls, 0) + 1
        detail.append((leg, "/entities/%s/%d" % (bucket, idx), str(r.get("name") or "")[:24],
                       held[0], sp_rung or "-", cls, sv))

    total = sum(buckets.values())
    print("population: %d" % total)
    print("")
    for k, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print("   %-46s %d" % (k, n))
    print("")
    print("PER ROW:")
    for leg, ptr, name, acc, rung, cls, sv in detail:
        print("  %-26s %-24s held=%-8s rung=%-8s %s" % (leg, name, acc, rung, cls))
        print("       species_ref: source=%-24s status=%-10s name=%-22s tax=%s conf=%s"
              % (sv["ref_source"] or "-", sv["ref_status"] or "-",
                 sv["ref_name"][:22] or "-", sv["ref_taxonomy"] or "-", sv["ref_confidence"]))
        print("       row        : species=%-22s species_name=%-22s organism=%-22s tax=%s"
              % (sv["row_species"][:22] or "-", sv["row_species_name"][:22] or "-",
                 sv["row_organism"][:22] or "-", sv["row_taxonomy"] or "-"))

    print("")
    print("=" * 78)
    print("F2. Same question across the whole committed corpus")
    print("=" * 78)
    roots = sorted({p.replace("\\", "/").split("/papers/")[0]
                    for p in glob.glob("runs*/**/papers/*/*/final_mapped.json", recursive=True)})
    cb = {}
    for root in roots:
        for leg, bucket, idx, r, iv, held, shipped in withheld(root):
            checks = d(iv.get("checks"))
            sp_rung = str(checks.get("species") or "")
            sv = species_view(r)
            supported = sv["ref_source"] in SOURCE_SUPPORTED and sv["ref_status"] == "matched"
            real_org = bool(sv["row_species"]) and sv["row_species"].casefold() not in (
                "unknown species", "unknown", "",
            )
            if sp_rung == "unknown":
                cls = ("species_evidence_lost_across_stage_boundary" if (supported and real_org)
                       else "species_present_but_not_source_supported" if real_org
                       else "species_unresolved_withholding_correct")
            elif not sp_rung:
                cls = "candidate_does_not_describe_shipped"
            elif sp_rung in ("mismatch", "conflict"):
                cls = "conflicting_species_evidence"
            else:
                cls = "other_measured_mechanism"
            cb[cls] = cb.get(cls, 0) + 1
    print("corpus population: %d  (roots scanned: %d)" % (sum(cb.values()), len(roots)))
    for k, n in sorted(cb.items(), key=lambda kv: -kv[1]):
        print("   %-46s %d" % (k, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
