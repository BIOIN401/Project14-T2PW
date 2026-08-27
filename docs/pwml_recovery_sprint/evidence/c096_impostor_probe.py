"""C-096 / REV correction 1 -- what does a NON-SINGLETON ``_NoDbResolver`` do?

Read-only, offline, no database. Written because the class docstring is being
corrected to state this outcome, and `[S5]` forbids asserting a runtime
behaviour from a static code path alone.

The first draft of that docstring claimed a lost-identity sentinel "would fail
open, back onto the ambient database", and that anything receiving it "must fail
visibly". Both are false. A fresh ``_NoDbResolver()`` is neither the singleton
nor ``None``, so it reaches neither arm of the resolver selection; the
availability ladder then defaults its missing ``available`` to ``True``. This
prints the outcome so the corrected docstring quotes a measurement.

``from_env`` is stubbed to refuse, so the ambient PathBank cannot participate and
whatever is printed is attributable to the impostor alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.mapping.map_ids as map_ids  # noqa: E402
import t2pw.pwml.compound_resolution as cr  # noqa: E402

print("t2pw.__file__ :", cr.__file__)

# No ambient database may take part in this measurement.
map_ids.PathBankDbResolver.from_env = classmethod(lambda _cls, overrides=None: None)


def _run(label: str, db_resolver: object) -> None:
    report: dict = {}
    rows = cr._resolve_compound_rows(
        [{"name": "glycolate", "hmdb_id": "HMDB0000115"}],
        db_resolver=db_resolver,
        strict_db=False,
        report=report,
        pointer_prefix="/entities/compounds",
        name_index=None,
    )
    db_resolution = report["db_resolution"]
    print(f"{label:<16}: available={db_resolution.get('available')!r} "
          f"reason={db_resolution.get('reason', '<ABSENT>')!r} "
          f"name={rows[0].get('name')!r} "
          f"row_status={db_resolution['compounds'][0].get('status')!r} "
          f"row_reason={db_resolution['compounds'][0].get('reason')!r}")


_run("singleton", cr.NO_DB_RESOLVER)
_run("fresh impostor", cr._NoDbResolver())
_run("None", None)

# Correction 2: the two selection conditions are identity tests against distinct
# objects, so they are mutually exclusive and arm ORDER cannot matter. What is
# load-bearing is the ``elif``.
print("mutually exclusive:",
      not (cr.NO_DB_RESOLVER is None) and cr.NO_DB_RESOLVER is not None)
