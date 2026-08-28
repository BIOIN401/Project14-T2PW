"""Independent confirmation of the wrapper species-source census (C-099 Finding 1)."""
import glob, io, json, sys

srcs = {}
total = 0
withref = 0
roots = sorted({p.replace("\\","/").split("/papers/")[0]
                for p in glob.glob("runs*/**/papers/*/*/final_mapped.json", recursive=True)})
for fp in sorted(glob.glob("runs*/**/papers/*/*/final_mapped.json", recursive=True)):
    d = json.load(io.open(fp, encoding="utf-8"))
    for r in (d.get("entities") or {}).get("protein_complexes") or []:
        if not isinstance(r, dict):
            continue
        meta = r.get("mapping_meta") if isinstance(r.get("mapping_meta"), dict) else {}
        reason = str(r.get("generation_reason") or meta.get("generation_reason") or "")
        if not (r.get("generated") is True and reason == "single_protein_pathwhiz_wrapper"):
            continue
        total += 1
        ref = r.get("species_ref") if isinstance(r.get("species_ref"), dict) else {}
        if not ref:
            ref = meta.get("species_resolution") if isinstance(meta.get("species_resolution"), dict) else {}
        if not ref:
            continue
        name = str(ref.get("name") or "")
        if not name or name.strip().casefold() == "unknown species":
            continue
        withref += 1
        srcs[str(ref.get("source") or "(none)")] = srcs.get(str(ref.get("source") or "(none)"), 0) + 1
print("roots scanned                  : %d" % len(roots))
print("generated single-protein wrappers: %d" % total)
print("with a RESOLVED species ref     : %d" % withref)
for k, v in sorted(srcs.items(), key=lambda kv: -kv[1]):
    print("   %-28s %d" % (k, v))
