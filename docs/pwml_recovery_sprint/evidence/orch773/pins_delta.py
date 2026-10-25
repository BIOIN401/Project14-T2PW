"""ORCH-773: the ORCH-766 byte-identity pins under a tree, raw and with the version fields stripped.

Usage: python pins_delta.py --tree <tree>      (the tree's src and tests are imported)

For "base" and each ``ALL_KINDS`` payload of tests/test_orch766_unreferenced_species.py: the test's own
``_digest`` of the quarantine output, and the same digest after replacing the five fields this lane's two
version bumps move -- quarantine_report.policy_version, decision_inputs.policy_version, decision_input_hash,
decision_id, removed_entity_report.schema_version -- with constants. Run on the base and the tip trees: equal
normalized digests prove the pins move ONLY by the bumps. Prints only.
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    args = ap.parse_args()
    tree = Path(args.tree).resolve()
    sys.path.insert(0, str(tree / "tests"))
    sys.path.insert(0, str(tree / "src"))
    import t2pw
    import test_orch766_unreferenced_species as T
    from t2pw.pipeline import strict_quarantine as SQ

    print("t2pw:", Path(t2pw.__file__).resolve().parent, "policy", SQ.QUARANTINE_POLICY_VERSION)

    def normalized(result) -> str:
        q = deepcopy(result.quarantine_report)
        q["policy_version"] = "X"
        q["decision_inputs"]["policy_version"] = "X"
        q["decision_input_hash"] = "X"
        q["decision_id"] = "X"
        r = deepcopy(result.removed_entity_report)
        r["schema_version"] = "X"
        blob = json.dumps({"payload": result.payload, "removed": r, "closure": result.closure_report, "quarantine": q},
                          sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    for kind in ("base", *T.ALL_KINDS):
        payload = T._base() if kind == "base" else T._kind_payload(kind)
        context = None if kind == "base" else T._context(kind)
        result = SQ.quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
        print(f'PIN "{kind}": "{T._digest(result)}"  normalized={normalized(result)}')
    print("default decision_input_hash", SQ.decision_input_hash())
    return 0


if __name__ == "__main__":
    sys.exit(main())
