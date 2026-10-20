"""NEW ACCEPTANCE TEST for ORCH-754d: production's name keys equal the evaluation matcher's.

G9 labelling: ``t2pw.pipeline.name_keys`` is a new module -- production's own copy of the
evaluation matcher's ``strict_key`` / ``text_key`` / ``mentions`` and of the v1 bench
normalization under them -- so this file is an explicitly labelled new acceptance test and
claims no base failure. The CORRECTION ORCH-754d makes (production no longer imports the
evaluator) is proven by ``tests/test_eval_v2_metrics.py::
test_no_production_module_imports_evaluation_v2``, which fails at ``adc49c40`` and passes at
the tip.

The copy exists so the evaluator stays independent of what it measures. The price is two
copies that must agree, and this file is the alarm when they drift: every probe below goes
through both sides and any difference fails. Tests may import both packages; production may
not. No model, network or database is called.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench import goldset as G  # noqa: E402
from t2pw.bench.eval_v2 import matcher as M  # noqa: E402
from t2pw.pipeline import name_keys as N  # noqa: E402

EVIDENCE = ROOT / "docs" / "pwml_recovery_sprint" / "evidence"
FIXTURE_GROUPS = {
    "orch754a fixtures (chlorophyll draw 1, pooled)": sorted((ROOT / "tests" / "fixtures" / "orch754a").glob("*_payload.json")),
    "chlorophyll draws": sorted((EVIDENCE / "orch752_chlorophyll_payloads").glob("control__run*__payload.json")),
    "enterobactin draws": sorted((EVIDENCE / "orch754a_inputs" / "PMC12096016_enterobactin").glob("control__run*__payload.json")),
    "menaquinone draws": sorted((EVIDENCE / "orch754a_inputs" / "PMC12421875_menaquinone").glob("control__run*__payload.json")),
}
PAPERS = sorted((EVIDENCE / "orch754a_inputs" / "papers").glob("*.txt"))

NBSP, EM_SPACE, COMBINING_ACUTE = chr(0x00A0), chr(0x2003), chr(0x0301)

#: Hand-written edge cases. Invisible characters are built with chr() so the file shows them.
EDGE: List[Any] = [
    # empty, None, non-strings
    None, "", " ", NBSP, 0, 7, 1.5, True,
    # Greek letters, both cases, spelled out and distinct
    "α-ketoglutarate", "β-ketoglutarate", "Α-KETOGLUTARATE", "beta-hydroxyacyl-ACP",
    "β-hydroxyacyl-ACP", "γ-aminobutyrate", "Δ-aminolevulinate", "δ-aminolevulinic acid",
    "μ-opioid", "ω-3 fatty acid", "σ70", "NF-κB", "λ", "ε-caprolactam", "Ω", "ΣX", "αβγ",
    # NFKC: ligatures, full width, super/subscripts, compatibility forms
    "ﬁbrinogen", "ＡＴＰ", "Fe³⁺", "Fe2+", "NAD⁺", "NAD+", "NADH", "NADP⁺", "H+", "+", "Cu²⁺ ",
    "H₂O", "CO₂", "½O2", "Å", "Ⅳ", "㎎", "ﬀ", "e" + COMBINING_ACUTE, "é", "É",
    # unicode dashes
    "2‐oxoglutarate", "2‑oxoglutarate", "2‒oxoglutarate", "2–oxoglutarate", "2—oxoglutarate",
    "2―oxoglutarate", "2−oxoglutarate", "2-oxoglutarate",
    # whitespace
    f"glucose{NBSP}6-phosphate", "glucose\t6-phosphate", "  glucose   6-phosphate  ",
    f"glucose{EM_SPACE}6-phosphate", "line\nbreak",
    # parentheses and trailing notes
    "LpxC (protein)", "LpxC (enzyme)", "entA (gene)", "EntB (complex)", "GSA (abbrev.)",
    "GSA (abbrev)", "LpxC (deacetylase)", "2,3-dihydroxybenzoic acid (2,3-DHB)",
    "Glu 1-semialdehyde (GSA)", "heptanoyl pyrophosphate (HepPP)", "oxidoreductase (entA)",
    "isochorismate synthase (EntB)", "((nested))", "(", ")", "()", "x (protein) y",
    # the -ic acid rule
    "2,3-dihydroxybenzoic acid", "2,3-dihydroxybenzoate", "Benzoic Acid", "BENZOIC ACID",
    "picric acidic", "ic acid", "Ic Acid", "boric acid ester", "isochorismic acid",
    "5-aminolevulinic acid", "nucleic acids", "acid",
    # pyrophosphate == diphosphate
    "pyrophosphate", "Pyrophosphate", "PPi", "inorganic pyrophosphate", "diphosphate",
    "geranylgeranyl pyrophosphate", "pyrophosphatase", "thiamine pyrophosphate (TPP)",
    # punctuation, digits, locants, case
    "N,N'-diacetylchitobiose", "CoA-SH", "acetyl-CoA", "Acetyl-CoA", "ACETYL-COA",
    "acetyl coenzyme A", "PGE2", "PGE1", "MK-7", "MK-8", "MK7", "2,5-DHB", "D-glucose",
    "L-serine", "serine", "[4Fe-4S]", "Mg2+/ATP", "a/b", "x_y", "x.y", "x:y", "x;y", "!!!",
    "--", "...", "PPI", "ppi", "NO", "no", "Straße", "STRASSE", "İstanbul", "ı",
]


def _strings(node: Any, out: List[str]) -> List[str]:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            _strings(value, out)
    elif isinstance(node, list):
        for value in node:
            _strings(value, out)
    return out


def _payload_strings() -> List[List[str]]:
    groups = []
    for name, files in FIXTURE_GROUPS.items():
        assert files, f"probe fixtures missing: {name}"  # an empty glob would make this test vacuous
        for path in files:
            groups.append(_strings(json.loads(path.read_text(encoding="utf-8")), []))
    return groups


def test_strict_key_and_text_key_agree_on_every_probe():
    papers = [p.read_text(encoding="utf-8") for p in PAPERS]
    assert len(papers) == 3
    probes: List[Any] = list(EDGE)
    for group in _payload_strings():
        probes.extend(group)
    for paper in papers:
        probes.append(paper)
        probes.extend(paper.splitlines())
    assert len(probes) > 2000

    diffs = []
    for p in probes:
        for side_n, side_m in ((N.strict_key, M.strict_key), (N.text_key, M.text_key),
                               (N.normalize_name, G.normalize_name), (N.canonical_text, G.canonical_text)):
            if side_n(p) != side_m(p):
                diffs.append((side_n.__name__, p if len(str(p)) < 200 else str(p)[:200] + "..."))
    assert diffs == []


def test_mentions_agrees_on_every_name_against_every_quote_and_paper():
    # Each side keys the text with its own text_key, as production composes them; keyed once.
    papers = [(N.text_key(t), M.text_key(t), t) for t in (p.read_text(encoding="utf-8") for p in PAPERS)]
    edge_names = [e for e in EDGE if isinstance(e, str)]
    outcomes = set()
    diffs = []

    def check(name: str, keyed: Any, aliases=()) -> None:
        got, want = N.mentions(name, keyed[0], aliases), M.mentions(name, keyed[1], aliases)
        outcomes.add(want)
        if got != want:
            diffs.append((name, aliases, keyed[2][:120]))

    for group in _payload_strings():
        names = list(dict.fromkeys(s for s in group if len(s) <= 100)) + edge_names
        quotes = [(N.text_key(s), M.text_key(s), s) for s in dict.fromkeys(s for s in group if len(s) > 40)]
        for name in names:
            for keyed in quotes + papers:
                check(name, keyed)
        for i, name in enumerate(names[:50]):
            check(name, quotes[i % len(quotes)], aliases=tuple(names[i + 1:i + 4]))
    assert diffs == []
    assert outcomes == {True, False}  # both answers occurred, so the probe set is not one-sided


def test_production_copy_imports_nothing_from_t2pw():
    tree = ast.parse(Path(N.__file__).read_text(encoding="utf-8"))
    roots = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    roots |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert roots <= {"__future__", "re", "typing", "unicodedata"}, roots
