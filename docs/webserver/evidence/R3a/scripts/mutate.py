"""R3a mutation check: each mutation of extraction.py must make the unit tests fail.
Each pytest run goes through bounded_run + pinned_pytest. The file is restored after
every mutation (and in a finally)."""
import os
import re
import subprocess
import sys
from pathlib import Path

TREE = "C:/t/ws-R3a"
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
TARGET = Path(TREE, "src/t2pw/service/extraction.py")
EV = Path("C:/t/wsR3atmp/mut")
EV.mkdir(parents=True, exist_ok=True)

MUTATIONS = {
    "join-single-newline": ('text = "\\n\\n".join(text_parts)', 'text = "\\n".join(text_parts)'),
    "texts-after-pdfs": ('text_parts.append(_pdf["text"].strip())', 'text_parts.insert(0, _pdf["text"].strip())'),
    "pdf-part-unstripped": ('text_parts.append(_pdf["text"].strip())', 'text_parts.append(_pdf["text"])'),
    "text-entries-unstripped": ("text_parts = [entry.strip() for entry in text_entries if entry.strip()]",
                                "text_parts = [entry for entry in text_entries if entry.strip()]"),
    "no-pdf-does-not-stop": ("                return AssembledInputs(failure=InputFailure(KIND_NO_PDF, message))\n", ""),
    "pdf-error-does-not-stop": ("                return AssembledInputs(failure=InputFailure(KIND_PDF_FAILED, message))\n", ""),
    "no-text-not-a-failure": ("return AssembledInputs(text=text, failure=InputFailure(KIND_NO_TEXT, message))",
                              "return AssembledInputs(text=text)"),
    "page-end-dropped": ("page_end=_pe,", "page_end=None,"),
    "single-page-not-end": ("else _ps", "else None"),
    "skip-refs-inverted": ("SKIP_SECTIONS if pdf_skip_refs else set()", "set() if pdf_skip_refs else SKIP_SECTIONS"),
    "ocr-ignored": ("enable_ocr_fallback=bool(pdf_ocr)", "enable_ocr_fallback=False"),
    "caption-dropped": ('                reporter.caption(" | ".join(_mp))\n', "                pass\n"),
    "pdf-warnings-dropped": ('            for _w in _pdf.get("warnings", []):', '            for _w in []:'),
    "preview-not-truncated": ('_pdf["text"][:1000]', '_pdf["text"][:1001]'),
    "expander-expanded": ("(first 1000 chars)\", expanded=False)", "(first 1000 chars)\", expanded=True)"),
    "temp-not-removed": ("                    os.unlink(_tmp_path)\n", "                    pass\n"),
    "page-range-warning-skipped": ("                reporter.warning(f\"Invalid page range '{_pr}'; extracting all pages.\")\n",
                                   "                pass\n"),
    "reporter-optional": ("    reporter: Reporter,\n", "    reporter: Reporter = None,\n"),
}

orig = TARGET.read_bytes()
results = {}
try:
    for name, (old, new) in MUTATIONS.items():
        src = orig.decode("utf-8")
        assert src.count(old) == 1, (name, src.count(old))
        TARGET.write_bytes(src.replace(old, new).encode("utf-8"))
        env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
        cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"mut-{name}",
               "--timeout", "120", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", f"{TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
               "-q", "-p", "no:cacheprovider", f"--basetemp=C:/t/wsR3atmp/mut-tmp/{name}",
               "tests/test_service_extraction.py", "-k", "not real_app"]
        out = subprocess.run(cmd, cwd=TREE, env=env, capture_output=True, text=True, timeout=200).stdout
        (EV / f"{name}.log").write_text(out, encoding="utf-8")
        summary = re.findall(r"^\d+ (?:failed|passed).*$", out, re.M)
        surv = re.findall(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", out)
        results[name] = (summary[-1] if summary else "?", surv[-1] if surv else "?")
        print(f"{name:28s} {results[name][0]}  survivors={results[name][1]}", flush=True)
finally:
    TARGET.write_bytes(orig)
killed = [n for n, (s, _) in results.items() if "failed" in s]
print(f"killed {len(killed)}/{len(MUTATIONS)}; survived: {sorted(set(MUTATIONS) - set(killed))}")
sys.exit(0 if len(killed) == len(MUTATIONS) else 1)
