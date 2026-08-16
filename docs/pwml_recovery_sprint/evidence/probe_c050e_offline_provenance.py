"""C-050e §2b -- the provenance contrast, driven through both trees.

Why this probe exists
---------------------
``REV-050c`` B-4: C-050's discriminating provenance fixture stopped
discriminating once **C-040a** (D-028) landed. That fixture drove the
``fuzzy_name`` path, and D-028 rule 1 now *refuses* a fuzzy match outright -- no
``pathwhiz_id`` is stamped, so pass 2 has nothing to read back and the row
observes the same value at C-050's tip and at the pre-correction module.

The **offline name-index** path is not gated by D-028's DB-match admission, so
the drift the snapshot/restore mechanism exists to prevent is still observable
there. That is what this measures.

The two legs, one payload
-------------------------
One compound row, one ChEBI id that hits the offline index, a PathBank resolver
that reports itself unavailable. **No ``element_locations`` anywhere:** the other
thing correction round 1 changed was the ``element_locations`` walk, and leaving
that section out of the payload means the only differing code this run can
exercise is the provenance mechanism itself.

* ``tip`` -- ``t2pw.pwml.prefreeze_resolution`` as imported from this tree.
* ``pre`` -- the same module as it stood at :data:`PRE_CORRECTION_SHA`, C-050's
  original commit, i.e. **before** ``e7f28e7`` added ``_snapshot_provenance`` /
  ``_authoritative_provenance`` / ``_restore_provenance``. Loaded from the git
  blob under a private module name, so it imports the **tip's**
  ``compound_resolution`` -- C-040a included. Everything except the module under
  test is held at the tip.

Behavioural, not symbolic (G9): the same row is driven through both and the two
observed ``db_status`` values are printed. Expected ``pre = unmatched`` and
``tip = matched_offline_name_index``; exit 0 only when both hold.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repo_root import REPO_ROOT, add_src_to_path  # noqa: E402

add_src_to_path()

#: C-050's original commit -- the last tree whose fixed-point loop had no
#: provenance snapshot/restore. ``e7f28e7`` (correction round 1) is its child.
PRE_CORRECTION_SHA = "768be7507ce5ad08f2a4efe72537347b1a363dec"

#: The contrast the card requires. ``pre`` is the behavioural base failure.
EXPECTED = {"pre": "unmatched", "tip": "matched_offline_name_index"}

RESULT_PATH = REPO_ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "c050e_offline_provenance.json"


class _UnavailableDbResolver:
    """Reachable object, unavailable database -- never opens a connection."""

    last_error = "db_not_configured_in_probe"

    def available(self) -> bool:
        return False


class _StubNameIndex:
    """Offline ChEBI -> canonical-name index; the one lookup this probe needs."""

    def compound_canonical(self, **ids: Any) -> Optional[Dict[str, Any]]:
        if str(ids.get("chebi") or "") == "15428":
            return {"id": 78, "name": "Glycine", "matched_on": "chebi"}
        return None


def _payload() -> Dict[str, Any]:
    return {
        "entities": {
            "compounds": [
                {"name": "gly", "chebi_id": "CHEBI:15428",
                 "mapping_meta": {"query": {"name": "gly"}}},
            ]
        },
        "processes": {"reactions": [{"name": "R1", "inputs": ["gly"], "outputs": []}]},
    }


def _load_pre_correction(tmp_dir: Path) -> Any:
    blob = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show",
         f"{PRE_CORRECTION_SHA}:src/t2pw/pwml/prefreeze_resolution.py"],
        check=True, stdout=subprocess.PIPE,
    ).stdout  # bytes on purpose: text=True decodes cp1252 and mangles the em dashes
    path = tmp_dir / "prefreeze_resolution_precorrection.py"
    path.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("_c050e_precorrection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observe(module: Any) -> Dict[str, Any]:
    payload = _payload()
    summary = module.resolve_compounds_prefreeze(
        payload, db_resolver=_UnavailableDbResolver(), strict_db=False,
        name_index=_StubNameIndex())
    row = payload["entities"]["compounds"][0]
    return {
        "module_file": str(module.__file__),
        "db_status": row.get("db_status"),
        "name": row.get("name"),
        "resolution_passes": summary.get("resolution_passes"),
        "rename_map": summary.get("rename_map"),
    }


def main() -> int:
    import t2pw
    from t2pw.pwml import prefreeze_resolution as tip_module

    with tempfile.TemporaryDirectory(prefix="c050e-") as tmp:
        detail = {"pre": _observe(_load_pre_correction(Path(tmp))),
                  "tip": _observe(tip_module)}

    ok = all(detail[leg]["db_status"] == expected for leg, expected in EXPECTED.items())
    result = {
        "probe": "C-050e-offline-provenance",
        "pre_correction_sha": PRE_CORRECTION_SHA,
        "t2pw_file": t2pw.__file__,
        "expected": EXPECTED,
        "observed": {leg: obs["db_status"] for leg, obs in detail.items()},
        "detail": detail,
        "result": "MEASURED" if ok else "UNEXPECTED",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"T2PW: {t2pw.__file__}")
    print(f"RESULT: {result['result']}  pre={detail['pre']['db_status']!r}  "
          f"tip={detail['tip']['db_status']!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
