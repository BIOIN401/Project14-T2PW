"""VD-18 (T2PW_VD18_SPAN_TEMPLATES, default off): two narrow span shapes, read only with the flag on and only
with an explicit protein-symbol catalyst. With the flag unset the parser is main's."""

import pytest

from t2pw.rag import admission as AD

POSITIVE = [
    ("include the conversion of chorismate to isochorismate (catalyzed by EntC), formation of DHB by EntB, "
     "activation of DHB to DHB-AMP by EntE, and final assembly", ["DHB"], ["DHB-AMP"], ["EntE"]),
    ("activation of DHB to DHB-AMP by EntE", ["DHB"], ["DHB-AMP"], []),
    ("the EntC-catalyzed isomerization of chorismate to isochorismate", ["chorismate"], ["isochorismate"], ["EntC"]),
    ("reduction of protochlorophyllide to chlorophyllide by POR1", ["protochlorophyllide"], ["chlorophyllide"], ["POR1"]),
]

NEGATIVE = [
    # other-pathway chemistry in the copula shape (the removed template); the replay's false class
    ("acetyl-Pi is then converted to acetate by AckA", ["acetyl-Pi"], ["acetate"], []),
    ("acetyl-Pi is then converted to acetate by AckA", ["acetyl-Pi"], ["acetate"], ["AckA"]),
    ("acetyl-CoA is first converted into acetyl-Pi by phosphotransacetylase", ["acetyl-CoA"], ["acetyl-Pi"], []),
    ("acetyl-Pi is further converted into acetate, catalyzed by acetate kinase", ["acetyl-Pi"], ["acetate"], []),
    # reagents and methods are not catalysts
    ("reduction of protochlorophyllide to chlorophyllide by NADH", ["protochlorophyllide"], ["chlorophyllide"], []),
    ("reduction of protochlorophyllide to chlorophyllide by NADH", ["protochlorophyllide"], ["chlorophyllide"], ["NADH"]),
    ("conversion of chorismate to isochorismate was measured by HPLC", ["chorismate"], ["isochorismate"], []),
    ("conversion of chorismate to isochorismate by HPLC", ["chorismate"], ["isochorismate"], ["HPLC"]),
    ("oxidation of NADH to NAD by NMR", ["NADH"], ["NAD"], []),
    # two-capital formulas are not catalysts (review R2 N4)
    ("reduction of protochlorophyllide to chlorophyllide by CO2", ["protochlorophyllide"], ["chlorophyllide"], []),
    ("reduction of protochlorophyllide to chlorophyllide by CO2", ["protochlorophyllide"], ["chlorophyllide"], ["CO2"]),
    ("oxidation of nitrite to nitrate by NO2", ["nitrite"], ["nitrate"], []),
    # a span cut inside the catalyst name
    ("activation of DHB to DHB-AMP by Ent", ["DHB"], ["DHB-AMP"], ["Ent"]),
    # negated
    ("no activation of DHB to DHB-AMP by EntE", ["DHB"], ["DHB-AMP"], []),
    # the F-179 nominalized-subject shape
    ("condensation of glycine and succinyl-CoA to produce ALA by ALAS2", ["glycine", "succinyl-CoA"], ["ALA"], []),
    # wrong direction, wrong catalyst
    ("activation of DHB to DHB-AMP by EntE", ["DHB-AMP"], ["DHB"], []),
    ("activation of DHB to DHB-AMP by EntE", ["DHB"], ["DHB-AMP"], ["MenF"]),
]


def ok(span, i, o, e=()):
    return AD.validate_evidence_span(span, inputs=list(i), outputs=list(o), enzymes=list(e)).ok


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(AD.VD18_ENV_FLAG, "1")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv(AD.VD18_ENV_FLAG, raising=False)


@pytest.mark.parametrize("span,i,o,e", POSITIVE)
def test_flag_on_reads_the_new_shapes(on, span, i, o, e):
    assert ok(span, i, o, e)


@pytest.mark.parametrize("span,i,o,e", NEGATIVE)
def test_flag_on_still_refuses(on, span, i, o, e):
    assert not ok(span, i, o, e)


@pytest.mark.parametrize("span,i,o,e", POSITIVE)
def test_flag_off_is_main(off, span, i, o, e):
    """Flag unset: main's template list, and none of the new readings."""
    assert AD._prose_patterns() is AD._ALL_PROSE_PATTERNS
    assert [n for n, _ in AD._ALL_PROSE_PATTERNS][-1] == "nominalized_conversion_catalyzed_by"
    assert not any(n in {x for x, _ in AD._VD18_PROSE_PATTERNS} for n, _ in AD._ALL_PROSE_PATTERNS)
    vd18 = {x for x, _ in AD._VD18_PROSE_PATTERNS}
    assert all(r.pattern not in vd18 for r in AD.parse_span_relations(span))
    assert AD.parse_span_relation(span) is None or AD.parse_span_relation(span).pattern not in vd18
    assert not ok(span, i, o, e)
