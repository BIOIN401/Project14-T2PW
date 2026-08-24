"""Do the two seams now agree on the COLLISION verdict for every input shape?

One definition means the production gate and the acceptance scorer must reach the
same set of (namespace, identifier) collisions on identical entity buckets.
"""
import itertools, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
from t2pw.bench import semantic as S            # noqa: E402
from t2pw.bench import semantic_production as SP  # noqa: E402

COLLISION = "accession_claimed_by_multiple_entities"
BUCKETS = ("compounds", "proteins", "protein_complexes")
NAMES = ("EntB", "holo-EntB", "Pyridoxal 5'-phosphate", "entb", "ent B", "")
ACC = ({"uniprot": "P0ADI4"}, {"drugbank": "DB00114"},
       {"uniprot": "P0ADI4", "drugbank": "DB00114"}, {})


class Case:
    def forbidden_match(self, candidate):
        return None


def collisions_from(check):
    return sorted((f.get("namespace"), f.get("identifier"), tuple(sorted(f.get("entities") or ())))
                  for f in check.findings if f.get("kind") == COLLISION)


print("same predicate object:", getattr(SP._s, "accession_claimed_across_kinds")
      is getattr(S, "accession_claimed_across_kinds"))

n = disagree = withc = 0
for size in (2, 3, 4):
    for pick in itertools.combinations_with_replacement(
            list(itertools.product(BUCKETS, NAMES)), size):
        for accs in itertools.combinations_with_replacement(ACC, size):
            ents = {b: [] for b in BUCKETS}
            for (bucket, name), a in zip(pick, accs):
                row = {"name": name}
                row.update(a)
                ents[bucket].append(row)
            scorer, _ = S._check_id_conflicts(Case(), ents)
            gate, _ph, _c, _f, _b = SP._audit_entities(ents)
            a1, a2 = collisions_from(scorer), collisions_from(gate)
            n += 1
            if a1:
                withc += 1
            if a1 != a2:
                disagree += 1
                if disagree <= 5:
                    print("DISAGREE", pick, accs)
                    print("   scorer:", a1)
                    print("   gate  :", a2)
print("cases:", n)
print("cases where the scorer emits a collision:", withc)
print("DISAGREEMENTS between the two seams:", disagree)
