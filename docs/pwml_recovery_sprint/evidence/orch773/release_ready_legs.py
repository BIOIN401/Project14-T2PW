"""ORCH-773: which corpus legs are release_ready, and what each quarantined or removed.

Usage: python release_ready_legs.py <corpus.json> <out-dir of that run>. Read-only.
"""

import json
import re
import sys
from pathlib import Path


def main():
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    for row in doc["legs"]:
        r = row.get("replay") or {}
        if (r.get("quarantine") or {}).get("release") != "release_ready":
            continue
        work = out / re.sub(r"[^A-Za-z0-9._-]+", "_", row["id"])
        q = json.loads((work / "outputs" / "quarantine_report.json").read_text(encoding="utf-8"))
        rem = json.loads((work / "outputs" / "removed_entity_report.json").read_text(encoding="utf-8"))
        print(row["id"], "pwml", r.get("pwml_written"), (r.get("export") or {}).get("pwml_name"))
        for x in q.get("quarantined") or []:
            print("   quarantined:", x.get("process_kind"), x.get("name"), x.get("state"), x.get("reason"))
        for x in rem.get("removed_entities") or []:
            print("   removed:", x.get("bucket"), x.get("name"), x.get("reason"))


if __name__ == "__main__":
    main()
