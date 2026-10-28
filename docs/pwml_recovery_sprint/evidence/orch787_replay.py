"""ORCH-787: replay the ORCH-784 chlorophyll-strict leg (PMC5130987) from the stage that failed.

The leg stopped in the post-pipeline at ``run_prefreeze_resolution`` with
``PREFREEZE_DUPLICATE_CANONICAL_ROWS`` (``protoheme`` / ``heme``). Its only saved payload is
``merged_payload.json`` (pre-Stage-6: Stage 6 needs live UniProt and is not replayable, KI-058), so the
replay feeds that payload to the pre-freeze seam, then the freeze and the PWML export, through the
production functions of ``--src`` (loaded by ``orch768_replay_corpus.load_tree``, which also installs its
socket block: no network).

The PathBank compound DB the seam consults is served from ``orch787/pathbank_compound_queries.json``,
recorded ONCE by ``record`` (read-only SELECTs, the queries this payload makes). A query not in the file
is answered ``[]`` and listed under ``unrecorded``.

    python orch787_replay.py record --src <tree>/src        # online, read-only, writes the fixture
    python orch787_replay.py replay --src <tree>/src --out <dir>   # offline
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PRIMARY = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
LEG = PRIMARY / (".claude/worktrees/orch784run/runs_papers/ORCH-784/legs/chlorophyll-strict/2026-10-27_2027/"
                 "papers/PMC5130987/strict")
# The ORCH-784 leg wrote no quarantine report; its Stage-0 context is taken from the same paper's
# ORCH-781 strict leg (same requested pathway and organism).
CTX_FROM = PRIMARY / (".claude/worktrees/orch781/runs_papers/ORCH-781/legs/chlorophyll-strict/2026-10-26_1603/"
                      "papers/PMC5130987/strict/quarantine_report.json")
FIXTURE = HERE / "orch787" / "pathbank_compound_queries.json"


def key(sql: str, params) -> str:
    return json.dumps([" ".join(str(sql).split()), [str(p) for p in params]])


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


#: Stage-6 identity fields a stamped row carries into the pre-freeze seam.
STAGE6_FIELDS = ("pathbank_compound_id", "chebi_id", "kegg_id", "hmdb_id", "pubchem_cid", "mapped_ids")


def leg_payload():
    """``merged_payload.json`` with the Stage-6 stamp Stage 6 would add, as RECORDED, not recomputed.

    Stage 6 is not replayable (KI-058). The same paper's ORCH-781 strict leg recorded which compound
    rows Stage 6 stamps with a PathBank id (``db_status == legacy_id_unverified`` in its
    ``final_mapped.json``: the pre-freeze seam keeps such a row's name). Those ids are carried onto the
    rows of the same extraction name here; every other row arrives as saved.
    """

    payload = load(LEG / "merged_payload.json")
    recorded = {str(r.get("raw_name") or r.get("name")).lower(): r
                for r in load(CTX_FROM.with_name("final_mapped.json"))["entities"]["compounds"]
                if r.get("db_status") == "legacy_id_unverified"}
    carried = []
    for row in payload["entities"]["compounds"]:
        stamp = recorded.get(str(row.get("name")).lower())
        if stamp:
            row.update({f: deepcopy(stamp[f]) for f in STAGE6_FIELDS if stamp.get(f) not in (None, "")})
            carried.append(f"{row['name']}={stamp.get('pathbank_compound_id')}")
    return payload, carried


def record(args) -> int:
    sys.path.insert(0, str(Path(args.src).resolve()))
    from t2pw.mapping.map_ids import PathBankDbResolver
    from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError, run_prefreeze_resolution

    db = PathBankDbResolver.from_env()
    if db is None:
        raise SystemExit("PathBank DB not configured")
    seen = {}
    real = db._query

    def recording(sql, params):
        rows = real(sql, params)
        seen[key(sql, params)] = rows
        return rows

    db._query = recording
    try:
        run_prefreeze_resolution(leg_payload()[0], strict_db=True, db_resolver=db)
        print("record run: no raise")
    except PrefreezeResolutionError as exc:
        print("record run raised:", exc.code)
    finally:
        db.close()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(seen, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(f"recorded {len(seen)} queries; last_error={db.last_error[:80]!r}")
    return 0


class ReplayDb:
    last_error = ""

    def __init__(self, answers):
        self.answers, self.calls, self.unrecorded = answers, 0, []

    def available(self) -> bool:
        return True

    def _query(self, sql, params):
        self.calls += 1
        k = key(sql, params)
        if k not in self.answers:
            self.unrecorded.append(k)
            return []
        return deepcopy(self.answers[k])


def replay(args) -> int:
    import orch768_replay_corpus as C

    T = C.load_tree(Path(args.src))
    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    T.app.PROJECT_ROOT = work
    db = ReplayDb(load(FIXTURE))
    payload, carried = leg_payload()
    lines = [f"tree: {T.t2pw}", f"leg: {LEG.relative_to(PRIMARY).as_posix()} merged_payload.json",
             f"compound rows in: {len(payload['entities']['compounds'])}",
             f"Stage-6 stamps carried as recorded by ORCH-781 ({len(carried)}): {', '.join(carried)}"]
    try:
        pf = T.prefreeze(payload, strict_db=True, db_resolver=db)
    except Exception as exc:  # noqa: BLE001 -- the failure IS the base result
        lines += [f"PREFREEZE RAISED: {getattr(exc, 'code', type(exc).__name__)}: {str(exc)[:300]}",
                  "pathway stops here: no freeze, no PWML"]
        return finish(lines, db, work)
    comp = pf.get("compounds") or {}
    lines += [f"prefreeze ok={pf.get('ok')} failures={pf.get('failures')}",
              f"rows_consolidated={json.dumps(comp.get('rows_consolidated'))}",
              f"compound rows out: {len(payload['entities']['compounds'])}"]
    for row in payload["entities"]["compounds"]:
        if row.get("consolidated_rows"):
            lines.append("survivor: " + json.dumps({k: row.get(k) for k in (
                "name", "class", "db_status", "pathbank_compound_id", "chebi_id", "hmdb_id", "kegg_id", "synonyms")}))
    ctx = (load(CTX_FROM).get("coverage") or {}).get("requested_context")
    freeze = T.app.freeze_canonical_payload(payload, strict_db=True, outputs_dir=work / "outputs", pathway_context=ctx,
                                            export_mode=C.MODE, research_mode=False, tmp=work)
    qres = freeze["quarantine_result"]
    qrep = qres.get("quarantine_report") or {}
    canonical = freeze["payload"]
    lines.append(f"freeze: quarantine_ok={freeze['quarantine_ok']} refusal={sorted(qres.get('refusal_reasons') or [])} "
                 f"release={(qrep.get('release') or {}).get('status')}")
    pwml_name, ir = None, None
    if canonical:
        res = T.app.run_pwml_export(deepcopy(canonical), project_root=work, ref_path=T.app._pwml_reference_path(T.root),
                                    quarantine_report=qrep, quarantine_artifacts=qres.get("quarantine_artifacts") or {},
                                    pathway_context=ctx, export_mode=C.MODE, **C.EXPORT_ARGS)
        xml = T.driver._xml_bytes(res)
        lines.append(f"export: ok={bool(res.get('ok'))} error={str(res.get('error') or '')[:200]!r} "
                     f"pwml_bytes={len(xml or b'')}")
        if res.get("ok") and xml:
            pwml_name, ir = "pathway.pwml", res.get("pwml_ir")
            (work / pwml_name).write_bytes(xml)
            lines.append(f"PWML WRITTEN: {pwml_name}; compounds named protoheme in PWML: "
                         f"{xml.decode('utf-8', 'replace').lower().count('>protoheme<')}")
    arts = {}
    for name, path in sorted((qres.get("quarantine_artifacts") or {}).items()):
        if name in (T.evidence.COVERAGE_REPORT_FILENAME, T.evidence.QUARANTINE_REPORT_FILENAME,
                    T.evidence.REMOVED_ENTITY_REPORT_FILENAME):
            arts[name] = load(Path(path))
    arts[T.evidence.FINAL_MAPPED_FILENAME] = canonical or payload
    if pwml_name:
        arts[T.evidence.PWML_IR_FILENAME], arts[pwml_name] = ir, True
    report = T.evidence.build_report(arts, no_pwml_cause="" if pwml_name else "replay wrote no PWML")
    cov = report.get("pwml_coverage") or {}
    lines.append(f"text record coverage: drawn={cov.get('drawn')} total={cov.get('total')} "
                 f"merged_duplicates={cov.get('merged_duplicates')}")
    text = T.evidence.render_text(report)
    (work / "reaction_evidence.txt").write_text(text, encoding="utf-8")
    lines.append("text record lines naming the consolidation:")
    lines += ["  " + ln for ln in text.splitlines() if "consolidated" in ln or "merged duplicate" in ln]
    return finish(lines, db, work)


def finish(lines, db, work: Path) -> int:
    import orch768_replay_corpus as C

    lines += [f"db queries served: {db.calls}; unrecorded: {len(db.unrecorded)}",
              f"network refused: {len(C.Offline.refused)}"]
    text = "\n".join(lines) + "\n"
    (work / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=("record", "replay"))
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    return record(args) if args.cmd == "record" else replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
