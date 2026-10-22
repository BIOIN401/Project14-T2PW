"""ORCH-762: cut the replay payloads in tests/fixtures/orch762/ from the leg artifacts.

Usage (from the worktree root):
    python docs/pwml_recovery_sprint/evidence/orch762/make_fixtures.py <primary checkout root>

Offline, stdlib only, deterministic. Each output is the leg file with ONLY the changes
listed beside it; the sha256 of every source and output is printed.

* menaquinone_kf147_merged.json -- ORCH-756 1-draw menaquinone leg ``merged_payload.json``
  (the Stage-2 input). Species rows gain the ``taxonomy_id`` / ``classification`` the leg's
  own NCBI backfill wrote at Stage 2 (``final_mapped.json`` species
  ``mapping_meta.taxonomy_backfill``: 684738 and 1590, Prokaryote), because the replay
  runs with NCBI off.
* menaquinone_lactis_species.json -- the same payload in the ORCH-755-night shape: the
  organism read as "Lactococcus lactis" (ORCH-761 § 4.2, extract § B). Only the nine
  ``proteins[].organism`` strings and the species row change: name "Lactococcus lactis",
  taxonomy id 1358 (NCBI's id for that name, ORCH-761 orch761_uniprot_probe2.txt).
* enterobactin_orch755_merged.json -- ORCH-755 enterobactin leg ``merged_payload.json``
  (2026-10-20 night, the holo-EntB loss), species row gains taxonomy id 562, Prokaryote.
* enterobactin_orch755_resolutions.json -- the identity each protein row of that leg's
  ``final_mapped.json`` recorded (``tests/orch762_harness.recorded_resolutions``): the
  replay answers PathBank's part from it, as ORCH-759a did.
"""

import hashlib
import json
import sys
from pathlib import Path

PRIMARY = Path(sys.argv[1])
HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[4]
OUT = WORKTREE / "tests" / "fixtures" / "orch762"
sys.path.insert(0, str(WORKTREE / "tests"))
from orch762_harness import recorded_resolutions  # noqa: E402

MENA = PRIMARY / ".claude/worktrees/orch756/runs_papers/ORCH-756/legs/menaquinone-strict/2026-10-21_2048/papers/PMC12421875/strict"
ENTB = PRIMARY / "runs_papers/ORCH-755/legs/enterobactin-strict/2026-10-20_2201/papers/PMC12096016/strict"
KF147 = "Lactococcus lactis subsp. lactis KF147"


def load(path):
    raw = path.read_bytes()
    print("source", path.relative_to(PRIMARY).as_posix(), "sha256", hashlib.sha256(raw).hexdigest())
    return json.loads(raw.decode("utf-8"))


def write(name, payload):
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("wrote", name, "sha256", hashlib.sha256(path.read_bytes()).hexdigest(), "bytes", path.stat().st_size)


def backfill(payload, ids):
    for row in payload["entities"]["species"]:
        if row.get("name") in ids:
            row["taxonomy_id"] = ids[row["name"]]
            row["classification"] = "Prokaryote"


OUT.mkdir(parents=True, exist_ok=True)
mena = load(MENA / "merged_payload.json")
mena_final = load(MENA / "final_mapped.json")
recorded = {row.get("raw_name") or row.get("name"): (row.get("mapping_meta") or {}).get("taxonomy_backfill", {}).get("taxonomy_id")
            for row in mena_final["entities"]["species"]}
assert recorded.get(KF147) == "684738" and recorded.get("Lactiplantibacillus plantarum") == "1590", recorded
kf = json.loads(json.dumps(mena))
backfill(kf, {KF147: "684738", "Lactiplantibacillus plantarum": "1590"})
write("menaquinone_kf147_merged.json", kf)

lactis = json.loads(json.dumps(mena))
changed = 0
for row in lactis["entities"]["proteins"]:
    if row.get("organism") == KF147:
        row["organism"] = "Lactococcus lactis"
        changed += 1
for row in lactis["entities"]["species"]:
    if row.get("name") == KF147:
        row["name"] = "Lactococcus lactis"
        changed += 1
assert changed == 10, changed
backfill(lactis, {"Lactococcus lactis": "1358", "Lactiplantibacillus plantarum": "1590"})
write("menaquinone_lactis_species.json", lactis)

entb = load(ENTB / "merged_payload.json")
backfill(entb, {"Escherichia coli": "562"})
write("enterobactin_orch755_merged.json", entb)
write("enterobactin_orch755_resolutions.json", recorded_resolutions(load(ENTB / "final_mapped.json")))

# * kf147_leg_cache_entries.json -- the ORCH-756 leg's own mapping-cache entries for the nine
#   Men proteins (``api-v8::men*::lactococcus lactis subsp lactis kf147...`` E. coli/human
#   best efforts, ``db::...`` species_not_found; ORCH-761 extract § A), verbatim, so a test can
#   prove a stale free-text answer is never replayed under a taxonomy scope.
cache = load(PRIMARY / ".claude/worktrees/orch756/data/id_mapping_cache.json")
MEN = ("mena", "menb", "menc", "mend", "mene", "menf", "meng", "menh", "meni")
entries = {
    key: value
    for key, value in cache["proteins"].items()
    if "::lactococcus lactis subsp lactis kf147::" in key and key.split("::")[1 if key.startswith(("api-", "db::")) else 0] in MEN
}
assert len(entries) == 18, sorted(entries)
write("kf147_leg_cache_entries.json", {"proteins": entries})
