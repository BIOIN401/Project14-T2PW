"""Exhaustive differential probe on bench.semantic._check_id_conflicts.

Run in the base tree and the tip tree; the two dumps must be byte-identical,
which is what "extraction only, behaviour unchanged" means.
"""
import itertools, json, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
from t2pw.bench import semantic as S  # noqa: E402

BUCKETS = ("compounds", "proteins", "protein_complexes")
NAMES = ("EntB", "holo-EntB", "Pyridoxal 5'-phosphate", "entb", "ent B", "")
IDS = (
    {"uniprot": "P0ADI4"},
    {"drugbank": "DB00114"},
    {"uniprot": "P0ADI4", "drugbank": "DB00114"},
    {},
)


class Forbidden:
    def __init__(self, kind, reason=""):
        self.kind = kind
        self.reason = reason


class Case:
    """Duck-typed GoldCase. ``forbid`` is a set of normalized names to condemn."""

    def __init__(self, forbid=(), kind="modification_state"):
        self.forbid = {S.normalize_name(n) for n in forbid}
        self.kind = kind

    def forbidden_match(self, candidate):
        norm = S.normalize_name(candidate)
        if norm and norm in self.forbid:
            return Forbidden(self.kind)
        return None


def dump(check, false_real):
    return {
        "ok": check.ok, "summary": check.summary, "false_real": false_real,
        "findings": check.findings,
    }


rows_out = {}
count = 0
# ---- part 1: every 2..4-row combination of (bucket, name) sharing one accession
combos = list(itertools.product(BUCKETS, NAMES))
for size in (2, 3, 4):
    for pick in itertools.combinations_with_replacement(combos, size):
        entities = {b: [] for b in BUCKETS}
        for bucket, name in pick:
            entities[bucket].append({"name": name, "uniprot": "P0ADI4"})
        key = "A|%d|%s" % (size, ";".join(f"{b}:{n}" for b, n in pick))
        rows_out[key] = dump(*S._check_id_conflicts(Case(), entities))
        count += 1
# ---- part 2: mixed accession sets, 2 and 3 rows
for size in (2, 3):
    for pick in itertools.product(itertools.product(BUCKETS, NAMES[:4]), repeat=size):
        for ids in itertools.product(IDS, repeat=size):
            entities = {b: [] for b in BUCKETS}
            for (bucket, name), idmap in zip(pick, ids):
                row = {"name": name}
                row.update(idmap)
                entities[bucket].append(row)
            key = "B|%d|%s|%s" % (
                size, ";".join(f"{b}:{n}" for b, n in pick),
                ";".join(json.dumps(i, sort_keys=True) for i in ids))
            rows_out[key] = dump(*S._check_id_conflicts(Case(), entities))
            count += 1
    break  # size 2 only for the accession cross-product (already 9216 cases)
# ---- part 3: forbidden identifiers interacting with the collision branch
for size in (2, 3):
    for pick in itertools.combinations_with_replacement(combos[:12], size):
        for forbid in ((), ("EntB",), ("holo-EntB",), ("EntB", "holo-EntB")):
            for kind in ("modification_state", "class_term"):
                entities = {b: [] for b in BUCKETS}
                for bucket, name in pick:
                    entities[bucket].append({"name": name, "uniprot": "P0ADI4"})
                key = "C|%d|%s|%s|%s" % (
                    size, ";".join(f"{b}:{n}" for b, n in pick), ",".join(forbid), kind)
                rows_out[key] = dump(*S._check_id_conflicts(Case(forbid, kind), entities))
                count += 1

OUT.write_text(json.dumps(rows_out, indent=0, sort_keys=True, default=str), encoding="utf-8")
print("cases:", count, "keys:", len(rows_out))
print("wrote", OUT)
