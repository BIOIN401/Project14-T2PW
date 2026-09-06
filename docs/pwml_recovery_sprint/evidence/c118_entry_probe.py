"""C-118: the full answer this tree gives for a handful of golden keys.

Orchestration tooling. Prints the relation, verdict and reason strings the tree
named by ``PYTHONPATH`` produces for each key given on the command line, so the
one changed-and-still-refused entry can be read side by side at base and tip.

Usage::

    PYTHONPATH=<tree>/src python c118_entry_probe.py <key> [<key> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main(argv) -> int:
    import c061_relation_golden as golden

    wanted = set(argv[1:])
    body = golden.build()
    for entry in body["entries"]:
        if entry["key"] in wanted:
            print(json.dumps(entry, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
