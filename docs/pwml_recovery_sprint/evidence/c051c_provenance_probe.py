"""C-051c: does ``raw_name`` survive the pre-freeze fixed-point loop?

**SHA-invariant.** Every section measures *behaviour*, and nothing here imports a
name that exists at only one of the two SHAs, so the same file runs at the base
``b8f7902c`` and at this card's tip. Run the base leg from a hash-verified base
tree: ``_repo_root.add_src_to_path`` inserts *the containing checkout's* ``src``
at ``sys.path[0]``, so ``PYTHONPATH`` alone is silently overridden and a base leg
launched from the tip worktree measures tip code and passes.

Sections
--------
``mechanism``  A1 / A8 -- the G9 behavioural proof. One compound row through
               ``resolve_compounds_prefreeze``. BASE: ``raw_name`` comes out
               ``'Glycolic acid'``, the canonical name recorded as though the
               paper had said it. TIP: ``'glycolate'``. The ``db_resolver=None``
               control reads ``'glycolate'`` at both SHAs -- so the drift is the
               *second pass*, not the fixture: only a **reachable** resolver
               re-derives ``match['raw_name']`` from the row's current name
               (``db_resolver.py:279``, ``:450``).
``idem``       A5 -- the loop converges and a second ``run_prefreeze_resolution``
               leaves the frozen payload byte-identical.
``resolve``    A6 / A9 -- every committed leg resolved directly, under BOTH the
               production resolver (``None``: no ``.env``, so unreachable) and a
               **reachable** canned one -- the configuration C-051b's GOLDEN
               ``_leg_digest`` exercises as ``C_canned_defaultindex_lenient``.
               Records the rename map and every row's ``name``/``raw_name``/
               ``db_status``, so A6 identity and the A9 GOLDEN delta are read
               off one measurement.
``corpus``     A2 / A3 -- every committed leg through the CLI: the sha256 of
               EVERY artifact file written, the IR compound rows, and
               ``canonical_graph_sha256`` / ``canonical_payload_sha256`` /
               ``admitted_payload_hash`` of the frozen payload.

Usage: ``<py> c051c_provenance_probe.py --tmp <dir> --out <json>
[--corpus-root <dir>] [--section <name>]``. ``--corpus-root`` exists for the base
leg: ``runs/`` and ``runs_verify/`` are committed artifacts this card does not
touch and the hash-verified base tree does not carry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from _repo_root import REPO_ROOT, add_src_to_path

add_src_to_path()
import t2pw  # noqa: E402
from t2pw.pipeline.canonical_hash import (  # noqa: E402
    canonical_graph_sha256,
    canonical_payload_sha256,
)
from t2pw.pipeline.strict_quarantine import admitted_payload_hash  # noqa: E402
from t2pw.pwml.name_index import PathwhizNameIndex  # noqa: E402
from t2pw.pwml.prefreeze_resolution import (  # noqa: E402
    PrefreezeResolutionError,
    resolve_compounds_prefreeze,
    run_prefreeze_resolution,
)
from t2pw.pwml.writer import run_pwml_pipeline_export  # noqa: E402

#: Row keys a provenance change could move: an A2 delta is reportable per KEY.
ROW_KEYS = ("name", "raw_name", "db_status", "chosen_rule", "confidence",
            "aliases", "synonyms", "pathwhiz_id", "pathbank_compound_id")


class _EmptyDb:
    """A **reachable** PathBank DB that answers nothing. Reachability is the
    point: ``_resolve_compound_rows`` builds a resolver only when ``available()``
    is True, and it is that resolver whose ``resolve`` re-derives ``raw_name``
    from the row's *current* name on every pass."""

    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:  # noqa: ARG002
        return []


class _CannedDb(_EmptyDb):
    """C-051b's GOLDEN config C, verbatim: reachable, and it answers Glycine."""

    def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:  # noqa: ARG002
        return [{"id": 78, "name": "Glycine", "short_name": "Gly",
                 "hmdb_id": "HMDB0000123", "kegg_id": "C00037", "chebi_id": "15428",
                 "pubchem_cid": "750", "cas": "56-40-6", "biocyc_id": "GLY",
                 "chemspider_id": "730", "drugbank_id": "DB00145",
                 "pwc_id": "PW_C000123", "description": "canned",
                 "synonyms": "Glycine; Gly"}]


def _index() -> PathwhizNameIndex:
    return PathwhizNameIndex({
        "compounds": {"hmdb": {"HMDB0000115": 900},
                      "by_id": {"900": {"name": "Glycolic acid", "hmdb": "HMDB0000115"}}},
        "species": {"taxonomy": {}, "by_id": {}},
    })


def _payload() -> Dict[str, Any]:
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [{"name": "glycolate", "hmdb_id": "HMDB0000115"},
                          {"name": "L-KDP"}],
        },
        "biological_states": [{"name": "cyto_state", "species": "Homo sapiens",
                               "subcellular_location": "cytosol"}],
        "element_locations": {"compound_locations": [
            {"compound": "glycolate", "biological_state": "cyto_state"},
            {"compound": "L-KDP", "biological_state": "cyto_state"}]},
        "processes": {"reactions": [
            {"name": "Glycolate turnover", "inputs": ["glycolate"],
             "outputs": ["L-KDP"], "biological_state": "cyto_state"}],
            "transports": [], "interactions": []},
    }


def _rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(((payload.get("entities") or {}).get("compounds")) or [])


def _projection(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: r.get(k) for k in ROW_KEYS if k in r} for r in rows]


def _blob(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)

# --- mechanism: A1 / A8 ----------------------------------------------------

def _mechanism() -> Tuple[int, Dict[str, Any]]:
    print("=== mechanism: raw_name through the pre-freeze fixed-point loop ===")
    observed: Dict[str, Any] = {}
    for leg, resolver in (("reachable_resolver", _EmptyDb()), ("no_resolver", None)):
        payload = _payload()
        summary = resolve_compounds_prefreeze(
            payload, db_resolver=resolver, strict_db=False, name_index=_index())
        row = next((r for r in _rows(payload) if r.get("name") == "Glycolic acid"), {})
        observed[leg] = {
            "passes": summary.get("resolution_passes"),
            "rename_map": summary.get("rename_map"),
            "name": row.get("name"), "raw_name": row.get("raw_name"),
            "db_match_raw_name": (row.get("db_match") or {}).get("raw_name"),
            "aliases": row.get("aliases"), "synonyms": row.get("synonyms"),
        }
        print(f"    {leg}: " + "  ".join(f"{k}={v!r}" for k, v in observed[leg].items()))

    reachable, control = observed["reachable_resolver"], observed["no_resolver"]
    observed["_drifted"] = reachable["raw_name"] != "glycolate"
    observed["_control_holds"] = control["raw_name"] == "glycolate"
    observed["_renamed"] = reachable["name"] == "Glycolic acid"
    failures: List[str] = []
    if not observed["_renamed"]:
        failures.append("VACUOUS: no rename happened, so there is no second pass "
                        "to preserve anything across")
    if not observed["_control_holds"]:
        failures.append("VACUOUS: the db_resolver=None control already lost "
                        "raw_name, so the reachable leg proves nothing")
    if observed["_drifted"]:
        print("    LEG READS AS: BASE -- the loop overwrote the extraction name "
              f"with {reachable['raw_name']!r}")
        failures.append("A1: raw_name is the CANONICAL name; the extraction name "
                        "the paper supplied is gone from the frozen row")
    else:
        print("    LEG READS AS: TIP -- the extraction name survived the loop")
    for line in failures:
        print(f"  FAIL {line}")
    return (1 if failures else 0), observed


# --- idem: A5 --------------------------------------------------------------

def _idem() -> Tuple[int, Dict[str, Any]]:
    print("\n=== idem: the loop converges and a second call is a no-op ===")
    results: Dict[str, Any] = {}
    failures: List[str] = []
    for leg, resolver in (("reachable_resolver", _EmptyDb()), ("no_resolver", None)):
        payload = _payload()
        first = run_prefreeze_resolution(payload, strict_db=False,
                                         db_resolver=resolver, name_index=_index())
        frozen = _blob(payload)
        second = run_prefreeze_resolution(payload, strict_db=False,
                                          db_resolver=resolver, name_index=_index())
        entry = {
            "passes_first": (first.get("compounds") or {}).get("resolution_passes"),
            "passes_second": (second.get("compounds") or {}).get("resolution_passes"),
            "payload_equals_frozen": _blob(payload) == frozen,
            "rows": _projection(_rows(payload)),
        }
        results[leg] = entry
        print(f"    {leg}: passes {entry['passes_first']} -> {entry['passes_second']}"
              f", payload == frozen: {entry['payload_equals_frozen']}")
        if not entry["payload_equals_frozen"]:
            failures.append(f"A5: {leg}: a second pre-freeze call moved the payload")
    for line in failures:
        print(f"  FAIL {line}")
    return (1 if failures else 0), results


# --- resolve / corpus ------------------------------------------------------

def _legs(corpus_root: Path) -> List[str]:
    legs: List[str] = []
    for top in ("runs", "runs_verify"):
        base = corpus_root / top
        if base.is_dir():
            legs.extend(str(p.relative_to(corpus_root)).replace("\\", "/")
                        for p in sorted(base.glob("*/papers/*/*/final_mapped.json")))
    return legs


def _args(input_path: Path, out_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_path=str(input_path), out_dir=str(out_dir),
        ref=str(REPO_ROOT / "reference" / "PW000001.pwml"),
        name="Generated Pathway", subject="Metabolic", description="",
        width=3200, height=1400, background_color="#FFFFFF", non_strict_db=True)


def _resolve(corpus_root: Path) -> Tuple[int, Dict[str, Any]]:
    """A6 / A9. Every committed leg, unreachable DB *and* reachable canned DB."""

    print(f"\n=== resolve: rename maps + rows on every leg ({corpus_root}) ===")
    results: Dict[str, Any] = {}
    # ``empty_reachable`` is GOLDEN config D and the DECISIVE one: a reachable
    # resolver matching NOTHING lets the offline index rename the row without
    # stamping an id, so pass 2 re-queries by the NEW name. A canned match stamps
    # ``pathwhiz_id`` and pass 2 then takes the legacy-id branch, which never
    # reaches ``apply_compound_db_resolution``.
    configs = (("production_none", None), ("canned_reachable", _CannedDb()),
               ("empty_reachable", _EmptyDb()))
    for config, resolver in configs:
        legs: Dict[str, Any] = {}
        for leg in _legs(corpus_root):
            payload = json.loads((corpus_root / leg).read_text(encoding="utf-8"))
            try:
                summary = resolve_compounds_prefreeze(
                    payload, db_resolver=resolver, strict_db=False)
            except PrefreezeResolutionError as exc:
                legs[leg] = {"raised": exc.code}
                continue
            except Exception as exc:  # noqa: BLE001
                legs[leg] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                continue
            legs[leg] = {
                "rows": summary.get("rows"), "renamed": summary.get("renamed"),
                "rename_map": summary.get("rename_map"),
                "passes": summary.get("resolution_passes"),
                "aliases_preserved": summary.get("aliases_preserved"),
                "identity_projected": summary.get("identity_projected"),
                "resolved": _projection(_rows(payload)),
            }
        results[config] = legs
        renamed = sum(1 for e in legs.values() if e.get("renamed"))
        raised = sorted(l for l, e in legs.items() if e.get("raised"))
        print(f"    {config:<18}: legs={len(legs)} with_renames={renamed} "
              f"raised={len(raised)} {sorted(set(legs[l]['raised'] for l in raised))}")
    return (1 if not results else 0), results


def _corpus(tmp: Path, corpus_root: Path) -> Tuple[int, Dict[str, Any]]:
    print(f"\n=== corpus: every committed leg through the CLI ({corpus_root}) ===")
    results: Dict[str, Any] = {}
    for index, leg in enumerate(_legs(corpus_root)):
        out = tmp / "corpus" / f"leg{index:03d}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            result = run_pwml_pipeline_export(_args(corpus_root / leg, out))
        except Exception as exc:  # noqa: BLE001
            results[leg] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        # EVERY artifact the export wrote, so a delta is attributable to a FILE.
        entry: Dict[str, Any] = {
            "ok": result.get("ok"),
            "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(out.rglob("*")) if p.is_file()},
        }
        ir_path = result.get("pwml_ir")
        if ir_path and Path(str(ir_path)).is_file():
            ir = json.loads(Path(str(ir_path)).read_text(encoding="utf-8"))
            entry["ir_compound_rows"] = _projection(
                list(((ir.get("entities") or {}).get("compounds")) or []))
        # A3: the three hashes over the payload as the pre-freeze sequence froze it.
        try:
            frozen = json.loads((corpus_root / leg).read_text(encoding="utf-8"))
            run_prefreeze_resolution(frozen, strict_db=False)
            entry["canonical_graph_sha256"] = canonical_graph_sha256(frozen)
            entry["canonical_payload_sha256"] = canonical_payload_sha256(frozen)
            entry["admitted_payload_hash"] = admitted_payload_hash(frozen)
            entry["frozen_compound_rows"] = _projection(_rows(frozen))
        except Exception as exc:  # noqa: BLE001
            entry["hash_error"] = f"{type(exc).__name__}: {exc}"[:200]
        results[leg] = entry
    exported = sum(1 for e in results.values()
                   if "pathway.pwml" in (e.get("artifacts") or {}))
    errored = sorted(leg for leg, e in results.items() if e.get("error"))
    print(f"    legs={len(results)} produced_pathway_pwml={exported} "
          f"raised={len(errored)}")
    for leg in errored[:5]:
        print(f"        {leg}: {results[leg]['error'][:120]}")
    return (1 if not results else 0), results


# ---------------------------------------------------------------------------

SECTIONS = ("mechanism", "idem", "resolve", "corpus")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-051c provenance probe")
    parser.add_argument("--tmp", required=True)
    parser.add_argument("--corpus-root", default=str(REPO_ROOT))
    parser.add_argument("--section", default="all", choices=("all",) + SECTIONS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"T2PW: {t2pw.__file__}")

    root = Path(args.corpus_root)
    runners = {
        "mechanism": _mechanism, "idem": _idem,
        "resolve": lambda: _resolve(root), "corpus": lambda: _corpus(tmp, root),
    }
    code = 0
    payload: Dict[str, Any] = {"t2pw": t2pw.__file__}
    for name in SECTIONS:
        if args.section in ("all", name):
            rc, payload[name] = runners[name]()
            code |= rc

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True),
                                  encoding="utf-8")
        print(f"\nmeasurements written to {args.out}")

    print("\nC-051c PROBE: " + ("PASSED" if code == 0 else "FAILED"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
