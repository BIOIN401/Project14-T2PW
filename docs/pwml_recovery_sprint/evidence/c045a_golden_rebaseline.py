"""C-045a: characterise, then perform, the C-040 golden's baseline move (D-016).

A **deliberate baseline move under permanent merge rule 4**, not a behavioural
correction. C-045 moved ``_canonicalize_species_offline`` upstream of the freeze,
so ``build_pwml_ir`` called *standalone* -- the configuration
``test_build_pwml_ir_matches_the_pre_extraction_golden`` pins -- can no longer
know a species was only deterministically normalized and stops publishing the
species half of ``report["preflight"]``. The golden therefore pins a
configuration that is no longer the production path. Re-baselining is how a real
regression gets buried, so the evidence is the *characterisation*, not a green
tick::

    --mode flatten [--production] --out F   # P1/P2 (P3 with --production); per SHA
    --mode delta --base F --tip F --out F   # the verdict; imports no t2pw
    --mode digest --out F                   # P5: regenerates all 32 GOLDEN digests

Run any mode with ``PYTHONPATH`` set to a tree's ``src``. ``digest`` alone
re-derives the whole golden from this repo -- it calls the test module's own
``_leg_digest`` over its own ``GOLDEN`` keys, so there is no second, drifting
implementation to trust -- and pointed at a pre-C-045 export it reproduces the
*superseded* digests exactly, which is what proves the recorded delta is the
whole delta. Every mode imports ``t2pw`` **before** the test module, whose
module-level ``sys.path.insert`` would otherwise put this worktree's ``src``
ahead of the ``PYTHONPATH`` pin and silently measure tip code in a base run
(shared block S2); the importing tree is printed from inside the process.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]


def _load() -> Any:
    """Import t2pw first (that pins the tree), then C-040's test module."""
    import t2pw
    import t2pw.mapping.map_ids as map_ids
    import t2pw.pwml.ir  # noqa: F401
    import t2pw.pwml.prefreeze_resolution  # noqa: F401
    print(f"T2PW: {t2pw.__file__}", flush=True)

    class _ForcedUnavailable:  # the autouse _no_live_db fixture, verbatim
        @classmethod
        def from_env(cls) -> Any:
            raise RuntimeError("harvest_forced_unavailable")

    map_ids.PathBankDbResolver = _ForcedUnavailable
    sys.path.insert(0, str(ROOT / "tests"))
    import test_compound_resolution_extraction as tm
    return tm


def _flatten(obj: Any, prefix: str, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        if not obj:
            out[prefix + "#empty"] = True
        for key, value in obj.items():
            _flatten(value, f"{prefix}/{key}", out)
    elif isinstance(obj, list):
        out[prefix + "#len"] = len(obj)
        for index, value in enumerate(obj):
            _flatten(value, f"{prefix}/[{index}]", out)
    else:
        out[prefix] = obj


def _leaves(tm: Any, built: Tuple[Any, ...]) -> Dict[str, Any]:
    """``/0/...`` is the IR, ``/1/...`` the report -- kept apart on purpose: P2 is
    the claim that nothing under ``/0`` moves, so they must never share a path."""
    out: Dict[str, Any] = {}
    for index, half in enumerate(json.loads(json.dumps(list(built), default=tm._nonjson))):
        _flatten(half, f"/{index}", out)
    return out


def _mode_flatten(tm: Any, production: bool) -> Dict[str, Any]:
    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    result: Dict[str, Any] = {}
    for leg in sorted(tm.GOLDEN):
        assert (ROOT / leg).is_file(), f"committed leg fixture is missing: {leg}"
        source, per_config = json.loads((ROOT / leg).read_text(encoding="utf-8")), {}
        # ``_leg_digest`` reuses ONE payload object across the five configs, so
        # the standalone sweep must too, or it is not measuring the golden.
        payload = copy.deepcopy(source)
        for name, kwargs in tm._configs():
            if production:
                payload = copy.deepcopy(source)
                try:
                    run_prefreeze_resolution(
                        payload, strict_db=kwargs["strict_db"],
                        db_resolver=kwargs["db_resolver"], name_index=kwargs["name_index"],
                    )
                except Exception as exc:  # a D-015 cl.6 stop is a result, not a crash
                    per_config[name] = {"#raised": f"{type(exc).__name__}: {exc}"}
                    continue
            try:
                per_config[name] = _leaves(tm, build_pwml_ir(payload, **kwargs))
            except Exception as exc:
                # C-051b. Without ``--production`` this is the RAW payload, and
                # since C-051 ``build_pwml_ir`` refuses a compound row carrying
                # no resolution verdict -- so at any post-C-051 tree this mode
                # died here on leg 1 and the tool had two dead modes, not one.
                # The refusal is the measured result of the standalone
                # configuration, recorded in the same shape ``--production``
                # already records a pre-freeze stop in, so a base-vs-tip
                # ``--mode delta`` still lines the two sweeps up instead of
                # having nothing to compare.
                per_config[name] = {"#raised": f"{type(exc).__name__}: {exc}"}
        result[leg] = per_config
        print(f"  {leg}", flush=True)
    return result


def _mode_delta(base: Dict[str, Any], tip: Dict[str, Any]) -> Dict[str, Any]:
    assert set(base) == set(tip), "base and tip cover different legs"
    paths: Dict[str, Dict[str, Any]] = {}
    per_leg: Dict[str, Dict[str, List[str]]] = {}
    for leg in sorted(base):
        for config in sorted(base[leg]):
            before, after = base[leg][config], tip[leg][config]
            for key in sorted(set(before) | set(after)):
                if before.get(key, "\0absent") == after.get(key, "\0absent"):
                    continue
                general = "/".join("[]" if p[:1] + p[-1:] == "[]" else p
                                   for p in key.split("/"))  # list index -> []
                entry = paths.setdefault(general, {"legs": set(), "configs": set(), "samples": {}})
                entry["legs"].add(leg)
                entry["configs"].add(config)
                entry["samples"].setdefault(
                    leg, {"config": config, "path": key,
                          "base": before.get(key, "<absent>"), "tip": after.get(key, "<absent>")})
                per_leg.setdefault(leg, {}).setdefault(general, []).append(config)

    summary = {general: {"legs": len(e["legs"]), "configs": sorted(e["configs"]),
                         "samples": e["samples"]} for general, e in sorted(paths.items())}
    return {
        "legs_total": len(base),
        "legs_changed": len(per_leg),
        "changed_paths": summary,
        # P2, the explicit negative, as a measured result rather than a claim:
        "P2_ir_paths_changed": sorted(p for p in summary if p.startswith("/0")),
        "P2_report_keys_changed": sorted({
            p.split("/")[2].split("#")[0] for p in summary if p.startswith("/1/")}),
        "per_leg": {leg: {k: sorted(set(v)) for k, v in sorted(d.items())}
                    for leg, d in sorted(per_leg.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("flatten", "delta", "digest"))
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--base")
    parser.add_argument("--tip")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode == "delta":
        payload: Any = _mode_delta(
            json.loads(Path(args.base).read_text(encoding="utf-8")),
            json.loads(Path(args.tip).read_text(encoding="utf-8")))
    else:
        tm = _load()
        if args.mode == "digest":
            payload = {leg: tm._leg_digest(json.loads((ROOT / leg).read_text(encoding="utf-8")))
                       for leg in sorted(tm.GOLDEN)}
        else:
            payload = _mode_flatten(tm, args.production)

    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
