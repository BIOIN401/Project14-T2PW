"""ORCH-773 diagnosis: print named entity rows and named process rows from a captured freeze input.

Usage: python dump_rows.py <capture dir>/<slug>/freeze_input.json <bucket>:<name> [...]
<bucket> is an entity bucket (proteins, protein_complexes, ...) or a process bucket prefixed with
"p." (p.reactions, p.transports, p.interactions). Read-only; stdlib only.
"""

import json
import sys
from pathlib import Path


def main():
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    p = doc["payload"]
    for spec in sys.argv[2:]:
        bucket, name = spec.split(":", 1)
        group = "processes" if bucket.startswith("p.") else "entities"
        rows = (p.get(group) or {}).get(bucket[2:] if group == "processes" else bucket) or []
        for i, row in enumerate(rows):
            if isinstance(row, dict) and row.get("name") == name:
                slim = {k: v for k, v in row.items() if k not in ("provenance_lineage", "mapping_meta", "evidence_spans")}
                print(f"--- {group}/{bucket}/{i}")
                print(json.dumps(slim, indent=1, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
