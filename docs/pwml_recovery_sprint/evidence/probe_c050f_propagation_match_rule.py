"""C-050f -- the propagation match rule, measured at the base and at the tip.

G9, behavioural, on a **correction of pre-existing observable behaviour**.
``_propagate`` rewrote references by ``_canonical`` while
``_assert_fully_propagated`` audited them by ``_norm``, so the detection set was
strictly wider than the rewrite set: a reference spelled ``GLY`` for a compound
renamed from ``gly`` was never rewritten and then always raised
``PREFREEZE_RENAME_NOT_PROPAGATED`` -- a terminal abort on a reference that is
not dangling, since it resolves to the entity being renamed. The quieter half is
the auditor's ``_norm(old) != _norm(new)`` guard, which skipped pure case changes
entirely, so ``glycine -> Glycine`` left a ``GLYCINE`` reference neither
rewritten nor flagged.

Both legs run through the **real entry point** ``run_prefreeze_resolution`` with
an offline stub name index and a resolver reporting itself unavailable: no DB, no
network, so nothing here is vacuous under P4-01. ``base`` is the module blob at
:data:`BASE_SHA` loaded under a private name, so it imports the tip's
``compound_resolution`` -- everything except the module under test is held at the
tip. The rename map each case produced is captured alongside (A5). The **new**
``PREFREEZE_RENAME_MAP_COLLISION`` guard is labelled as new and has no base leg;
its acceptance test is in ``tests/test_prefreeze_compound_resolution.py``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repo_root import REPO_ROOT, add_src_to_path  # noqa: E402

add_src_to_path()

#: The approved C-050d tip -- this card's base for every base-vs-tip claim.
BASE_SHA = "a81b1d65dfd35415f8ae94f91093980f316acf97"

MODULE = "src/t2pw/pwml/prefreeze_resolution.py"
RESULT_PATH = REPO_ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "c050f_match_rule.json"

#: ``(case, entity, chebi, reference spelling, base outcome, tip outcome)``. An
#: outcome is ``ok:<what the reference ended up spelled as>`` or the raised code.
#: The first four are REV-050e's measured cases. In ``case_change_*`` the exact
#: spelling was already rewritten at base; the VARIANT is the silent miss -- no
#: rewrite, no flag, and the run still reports success. ``unrelated_dangling``
#: names nothing and is no variant of a renamed name: unchanged here, and neither
#: leg raises on it (reported as a deferred finding).
PROPAGATION: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    ("control_gly", "gly", "15428", "gly", "ok:Glycine", "ok:Glycine"),
    ("upper_GLY", "gly", "15428", "GLY", "PREFREEZE_RENAME_NOT_PROPAGATED", "ok:Glycine"),
    ("title_Gly", "gly", "15428", "Gly", "PREFREEZE_RENAME_NOT_PROPAGATED", "ok:Glycine"),
    ("punct_succinyl_CoA", "succinyl-CoA", "15380", "succinyl CoA",
     "PREFREEZE_RENAME_NOT_PROPAGATED", "ok:Succinyl coenzyme A"),
    ("case_change_exact", "glycine", "15428", "glycine", "ok:Glycine", "ok:Glycine"),
    ("case_change_variant", "glycine", "15428", "GLYCINE", "ok:GLYCINE", "ok:Glycine"),
    ("unrelated_dangling", "gly", "15428", "Serine", "ok:Serine", "ok:Serine"),
)

#: A3 -- ``(case, entity, stale spelling, rename map, base outcome, tip outcome)``.
#: Each stale spelling names something no entity carries once the row is
#: ``Glycine``, so each must stay fatal. ``case_change`` is the one the base could
#: not see at all, and ``clean`` is the propagated payload, which must be accepted.
AUDIT: Tuple[Tuple[str, str, str, Dict[str, str], str, str], ...] = (
    ("audit_stale_exact", "gly", "gly", {"gly": "Glycine"},
     "PREFREEZE_RENAME_NOT_PROPAGATED", "PREFREEZE_RENAME_NOT_PROPAGATED"),
    ("audit_stale_variant", "gly", "GLY", {"gly": "Glycine"},
     "PREFREEZE_RENAME_NOT_PROPAGATED", "PREFREEZE_RENAME_NOT_PROPAGATED"),
    ("audit_stale_case_change", "glycine", "GLYCINE", {"glycine": "Glycine"},
     "ok:no-raise", "PREFREEZE_RENAME_NOT_PROPAGATED"),
    ("audit_clean", "gly", "Glycine", {"gly": "Glycine"}, "ok:no-raise", "ok:no-raise"),
)


class _UnavailableDbResolver:
    """Reachable object, unavailable database -- never opens a connection."""

    last_error = "db_not_configured_in_probe"

    def available(self) -> bool:
        return False


class _StubNameIndex:
    """Offline ChEBI -> canonical-name index. Two ids, both fixed here."""

    _BY_CHEBI = {
        "15428": {"id": 78, "name": "Glycine", "matched_on": "chebi"},
        "15380": {"id": 79, "name": "Succinyl coenzyme A", "matched_on": "chebi"},
    }

    def compound_canonical(self, **ids: Any) -> Optional[Dict[str, Any]]:
        return self._BY_CHEBI.get(str(ids.get("chebi") or ""))


def _payload(entity: str, chebi: str, reference: str) -> Dict[str, Any]:
    """One compound, one reaction, one reference spelled however the case wants."""

    return {
        "entities": {"compounds": [
            {"name": entity, "chebi_id": f"CHEBI:{chebi}",
             "mapping_meta": {"query": {"name": entity}}},
        ]},
        "processes": {"reactions": [{"name": "R1", "inputs": [reference], "outputs": []}]},
    }


def _run(module: Any, payload: Dict[str, Any]) -> None:
    module.run_prefreeze_resolution(
        payload, db_resolver=_UnavailableDbResolver(), strict_db=False,
        name_index=_StubNameIndex())


def _observe(module: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Every arm against one module. Returns ``(outcomes, rename maps)``.

    The map is captured by wrapping the module's own ``_propagate`` attribute and
    delegating to it, so the observation does not change what runs -- and it is
    captured even for cases that raise, because propagation precedes the audit.
    """

    outcomes: Dict[str, str] = {}
    maps: Dict[str, str] = {}

    for case, entity, chebi, reference, _, _ in PROPAGATION:
        payload = _payload(entity, chebi, reference)
        captured: List[Dict[str, str]] = []
        original = module._propagate

        def spy(*args: Any, **kwargs: Any) -> Any:
            captured.append(dict(args[1]))
            return original(*args, **kwargs)

        module._propagate = spy
        try:
            _run(module, payload)
            outcomes[case] = "ok:" + payload["processes"]["reactions"][0]["inputs"][0]
        except module.PrefreezeResolutionError as exc:
            outcomes[case] = str(exc.code)
        finally:
            module._propagate = original
        maps[case] = json.dumps(captured[0] if captured else None, sort_keys=True)

    for case, entity, stale, rename, _, _ in AUDIT:
        payload = _payload(entity, "15428", entity)
        _run(module, payload)
        payload["processes"]["reactions"][0]["inputs"][0] = stale
        try:
            module._assert_fully_propagated(payload, rename)
            outcomes[case] = "ok:no-raise"
        except module.PrefreezeResolutionError as exc:
            outcomes[case] = str(exc.code)

    return outcomes, maps


def _load_base(tmp_dir: Path) -> Any:
    blob = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{BASE_SHA}:{MODULE}"],
        check=True, stdout=subprocess.PIPE,
    ).stdout  # bytes on purpose: text=True decodes cp1252 and mangles the em dashes
    path = tmp_dir / "prefreeze_resolution_base.py"
    path.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("_c050f_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import t2pw
    from t2pw.pwml import prefreeze_resolution as tip

    with tempfile.TemporaryDirectory(prefix="c050f-") as tmp:
        base_outcomes, base_maps = _observe(_load_base(Path(tmp)))
    tip_outcomes, tip_maps = _observe(tip)

    expected = {row[0]: {"base": row[-2], "tip": row[-1]}
                for arm in (PROPAGATION, AUDIT) for row in arm}
    observed = {case: {"base": base_outcomes.get(case), "tip": tip_outcomes.get(case)}
                for case in expected}
    mismatches = {case: {"expected": want, "observed": observed[case]}
                  for case, want in expected.items() if observed[case] != want}
    rename_map_drift = {case: {"base": value, "tip": tip_maps.get(case)}
                        for case, value in base_maps.items() if tip_maps.get(case) != value}

    ok = not mismatches and not rename_map_drift
    result = {
        "probe": "C-050f-propagation-match-rule",
        "base_sha": BASE_SHA,
        "t2pw_file": t2pw.__file__,
        "expected": expected,
        "observed": observed,
        "mismatches": mismatches,
        "rename_map_base": base_maps,
        "rename_map_identical": not rename_map_drift,
        "rename_map_drift": rename_map_drift,
        "result": "MEASURED" if ok else "UNEXPECTED",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"T2PW: {t2pw.__file__}")
    for case, seen in observed.items():
        print(f"  {case:24s} base={seen['base']!r:42s} tip={seen['tip']!r}")
    print(f"RESULT: {result['result']}  mismatches={len(mismatches)}  "
          f"rename_map_identical={result['rename_map_identical']}  -> {RESULT_PATH.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
