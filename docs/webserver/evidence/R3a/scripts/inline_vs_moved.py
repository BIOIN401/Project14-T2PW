"""R3a display-equivalence proof: the BASE app's inline input-assembly code vs the TIP
app's call site + assemble_inputs + StreamlitReporter, against the same recording fake
``st`` and fake ``parse_pdf``, over a grid of scenarios.

Compared per scenario: the exact ordered list of st.* calls (spinner/expander enter and
exit included), where st.stop() fired, the final ``text``, ``user_task_context``, and
every parse_pdf call (bytes + kwargs). Both snippets are cut out of the real app source
by marker lines and exec'd at module level, as Streamlit runs the script.

usage: inline_vs_moved.py BASE_TREE TIP_TREE     (PYTHONPATH must be TIP_TREE/src)
"""
import json
import sys
import textwrap
from pathlib import Path

base_tree, tip_tree = sys.argv[1], sys.argv[2]

from t2pw.extraction.pdf_parser import SKIP_SECTIONS  # noqa: E402
import t2pw.service.extraction as ex  # noqa: E402
from t2pw.service.extraction import assemble_inputs  # noqa: E402
from t2pw.service.reporter import StreamlitReporter  # noqa: E402


def cut(tree):
    lines = Path(tree, "src/t2pw/app/streamlit_app.py").read_text(encoding="utf-8").splitlines()
    start = lines.index("if submit:") + 1
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "reset_refinement_state()")
    return textwrap.dedent("\n".join(lines[start:end]))


class Stop(Exception):
    pass


class Ctx:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def __enter__(self):
        self.log.append([self.name + ":enter"])

    def __exit__(self, *a):
        self.log.append([self.name + ":exit"])


class FakeSt:
    def __init__(self):
        self.log = []

    def __getattr__(self, name):
        def call(*a, **k):
            self.log.append([name, list(a), k])
            if name == "stop":
                raise Stop()
            if name in ("spinner", "expander"):
                return Ctx(self.log, name)
        return call


class Up:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def read(self):
        return self._data


def res(text="BODY", **o):
    r = {"text": text, "metadata": {}, "total_pages": 9, "pages_used": 3,
         "sections": {"intro": [0]}, "method": "pypdf", "warnings": [], "error": ""}
    r.update(o)
    return r


S = [  # (name, mode, texts, pdfs[(name, bytes, parse result)], page_range, skip, ocr, utc)
    ("text-only", "Paste text", [" a ", "", "b"], [], "", True, False, "  ctx "),
    ("empty-paste", "Paste text", ["", "  "], [], "", True, False, ""),
    ("upload-none", "Upload PDF", [], [], "bad", True, False, None),
    ("tp-none-badrange", "Text + PDF", ["t"], [], "bad", True, False, None),
    ("tp-none-empty", "Text + PDF", [""], [], "", True, False, None),
    ("two-pdfs", "Text + PDF", ["T1"], [
        ("a.pdf", b"A", res(" P1 ", warnings=["w"], metadata={"title": "X", "doi": "D"})),
        ("b.pdf", b"B", res("y" * 1500, sections={}))], "2-4", False, True, "c"),
    ("pdf-error-mid", "Upload PDF", [], [
        ("a.pdf", b"A", res("ok")), ("b.pdf", b"B", res("", error="broken")),
        ("c.pdf", b"C", res("never"))], "", True, False, None),
    ("empty-pdf", "Upload PDF", [], [("s.pdf", b"S", res("  "))], "5", True, False, None),
    ("range-half-bad", "Upload PDF", [], [("a.pdf", b"A", res())], "3-x", True, 1, None),
    ("range-bad-start", "Upload PDF", [], [("a.pdf", b"A", res())], "x-3", False, "", None),
    ("meta-author", "Upload PDF", [], [("a.pdf", b"A", res(metadata={"author": "Q"}))], " 1-1 ", True, False, None),
]


def run(snippet, scen):
    name, mode, texts, pdfs, pr, skip, ocr, utc = scen
    st, calls, queue = FakeSt(), [], [p[2] for p in pdfs]

    def parse_pdf(path, **kw):
        calls.append([Path(path).read_bytes().decode(), {k: sorted(v) if isinstance(v, set) else v
                                                         for k, v in kw.items()}])
        return queue.pop(0)

    ex.parse_pdf = parse_pdf
    uses_pdf = mode in {"Upload PDF", "Text + PDF"}
    ns = dict(st=st, parse_pdf=parse_pdf, SKIP_SECTIONS=SKIP_SECTIONS, assemble_inputs=assemble_inputs,
              StreamlitReporter=StreamlitReporter, text_entries=list(texts),
              uploaded_pdfs=[Up(n, b) for n, b, _ in pdfs] if uses_pdf else [],
              uses_pdf_input=uses_pdf, input_mode=mode, pdf_page_range=pr, pdf_skip_refs=skip,
              pdf_ocr=ocr, user_task_context=utc)
    stopped = False
    try:
        exec(compile(snippet, f"<{name}>", "exec"), ns)
    except Stop:
        stopped = True
    return {"st": st.log, "stopped": stopped, "text": None if stopped else ns.get("text"),
            "user_task_context": ns.get("user_task_context"), "parse": calls}


old, new = cut(base_tree), cut(tip_tree)
assert "assemble_inputs" not in old and "assemble_inputs" in new
bad = 0
for scen in S:
    a, b = run(old, scen), run(new, scen)
    same = json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    bad += not same
    print(f"{scen[0]:18s} {'IDENTICAL' if same else 'DIFFERENT'}  st-calls={len(a['st'])} "
          f"stopped={a['stopped']} parse-calls={len(a['parse'])}")
    if not same:
        print("  base:", json.dumps(a, default=str)[:1500])
        print("  tip :", json.dumps(b, default=str)[:1500])
print(f"{len(S) - bad}/{len(S)} scenarios identical")
sys.exit(1 if bad else 0)
