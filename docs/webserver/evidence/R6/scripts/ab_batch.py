"""Offline A/B of one batch leg on the recorded sour-taste case: engine apptest vs engine run_pathway.

Both arms go through runner.run_single (the child's own entry), replaying the recorded LLM cassette, each in
its own project root, as tests/equivalence does. Own process (harness.replay_env refuses a process that has
already imported the LLM client). Writes a JSON report; compares the manifest row and every leg file.

usage: ab_batch.py OUT_JSON BASE_DIR [sour_taste_v2|farnesol_research]  (farnesol needs PYTHONHASHSEED=0)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "C:/t/ws-R6/tests")
sys.path.insert(0, "C:/t/ws-R6/src")

import pytest  # noqa: E402

from equivalence import harness, llm_replay  # noqa: E402


def norm(blob: bytes, roots) -> bytes:
    for root in roots:
        for form in (str(root), Path(root).as_posix(), str(root).replace("\\", "\\\\")):
            blob = blob.replace(form.encode("utf-8"), b"<ROOT>")
    return blob


def arm(engine: str, base: Path, fixture: dict) -> dict:
    from streamlit.testing.v1 import AppTest

    from t2pw.batch import runner
    from t2pw.service import defaults

    case = fixture["case"]
    root = base / engine / "root"
    run_dir = base / engine / "run"
    harness.new_root(root)
    slug = case["paper_id"] + "__sour"
    folder = run_dir / runner.PAPERS_DIRNAME / slug
    folder.mkdir(parents=True)
    (folder / runner.SOURCE_TEXT_NAME).write_text(case["full_text"], encoding="utf-8")
    record = {k: case[k] for k in ("paper_id", "title", "requested_pathway", "requested_organism", "topic")}
    record["slug"] = slug
    runner.save_plan(run_dir, {"modes": [case["mode"]], "papers": [record], **({"engine": engine} if engine else {})})
    cassette = llm_replay.Cassette("replay", fixture["entries"])
    with pytest.MonkeyPatch.context() as mp:
        cassette.install(mp)
        harness.isolate(mp, root)
        harness.freeze_clock(mp)
        mp.setattr(defaults, "AUDIT_TIMEOUT_SECONDS", harness.AUDIT_TIMEOUT_SECONDS)
        real = AppTest.from_file

        def from_file(*a, **k):
            at = real(*a, **k)
            at.session_state[harness.AUDIT_TIMEOUT_KEY] = harness.AUDIT_TIMEOUT_SECONDS
            return at
        mp.setattr(AppTest, "from_file", from_file)
        started = time.monotonic()
        row = runner.run_single(run_dir, slug, case["mode"], timeout=3000, engine=engine)
        seconds = time.monotonic() - started
    leg = run_dir / runner.PAPERS_DIRNAME / slug / case["mode"]
    files = {p.name: hashlib.sha256(norm(p.read_bytes(), [root, run_dir])).hexdigest()
             for p in sorted(leg.iterdir()) if p.is_file() and p.name != "leg_trace.jsonl" and not p.name.startswith("leg_trace")}
    result_txt = norm((leg / runner.RESULT_NAME).read_bytes(), [root, run_dir]).decode("utf-8", "replace")
    return {"engine": engine, "seconds": round(seconds, 1), "row": row, "files": files, "result_txt": result_txt,
            "leg_listing": sorted(p.name for p in leg.iterdir()),
            "misses": list(cassette.misses), "refused": list(cassette.refused), "unused": cassette.unused(),
            "pwml_sha": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in leg.glob("*.pwml")}}


CASE = sys.argv[3] if len(sys.argv) > 3 else 'sour_taste_v2'


def main(out: Path, base: Path) -> int:
    fixture = harness.load_fixture(harness.CASES[CASE]['fixture'])
    mp = pytest.MonkeyPatch()
    try:
        harness.replay_env(mp, fixture)
        a = arm("apptest", base, fixture)
        b = arm("run_pathway", base, fixture)
    finally:
        mp.undo()
    skip = {"seconds", "files", "dir"}
    row_a = {k: v for k, v in a["row"].items() if k not in skip}
    row_b = {k: v for k, v in b["row"].items() if k not in skip}
    files_a, files_b = a["files"], b["files"]
    report = {
        "row_keys_only_apptest": sorted(set(row_a) - set(row_b)),
        "row_keys_only_run_pathway": sorted(set(row_b) - set(row_a)),
        "row_fields_that_differ": {k: [row_a[k], row_b[k]] for k in row_a if k in row_b and row_a[k] != row_b[k]},
        "files_only_apptest": sorted(set(files_a) - set(files_b)),
        "files_only_run_pathway": sorted(set(files_b) - set(files_a)),
        "files_identical": sorted(k for k in files_a if k in files_b and files_a[k] == files_b[k]),
        "files_that_differ": sorted(k for k in files_a if k in files_b and files_a[k] != files_b[k]),
        "pwml_identical": a["pwml_sha"] == b["pwml_sha"] and bool(a["pwml_sha"]),
        "pwml_sha": [a["pwml_sha"], b["pwml_sha"]],
        "result_txt_identical": a["result_txt"] == b["result_txt"],
        "manifest_file_names_identical": [Path(f["name"]).name for f in a["row"]["files"]] == [Path(f["name"]).name for f in b["row"]["files"]],
        "replay_clean": {e: {"misses": len(x["misses"]), "refused": len(x["refused"]), "unused_llm": len([u for u in x["unused"] if u[0] == "llm"])}
                         for e, x in (("apptest", a), ("run_pathway", b))},
        "seconds": [a["seconds"], b["seconds"]],
        "status": [a["row"]["status"], b["row"]["status"]],
        "row_apptest": row_a, "row_run_pathway": row_b,
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out.parent / "ab_result_txt_apptest.txt").write_text(a["result_txt"], encoding="utf-8")
    (out.parent / "ab_result_txt_run_pathway.txt").write_text(b["result_txt"], encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("row_apptest") and not k.startswith("row_run")},
                     indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
