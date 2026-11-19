"""VD-18: three narrow span shapes now read, each only with an explicit protein-symbol catalyst."""

import pytest

from t2pw.rag import admission as AD


def ok(span, i, o, e=()):
    return AD.validate_evidence_span(span, inputs=list(i), outputs=list(o), enzymes=list(e)).ok


@pytest.mark.parametrize("span,i,o,e", [
    ("The initial substrate, chorismate, is first converted to isochorismate by EntC (isochorismate synthase;",
     ["chorismate"], ["isochorismate"], ["EntC"]),
    ("chorismate, is first converted to isochorismate by EntC", ["chorismate"], ["isochorismate"], []),
    ("include the conversion of chorismate to isochorismate (catalyzed by EntC), formation of DHB by EntB, "
     "activation of DHB to DHB-AMP by EntE, and final assembly", ["DHB"], ["DHB-AMP"], ["EntE"]),
    ("the EntC-catalyzed isomerization of chorismate to isochorismate", ["chorismate"], ["isochorismate"], ["EntC"]),
])
def test_new_shapes_read(span, i, o, e):
    assert ok(span, i, o, e)


@pytest.mark.parametrize("span,i,o,e", [
    # prose catalysts stay unread (on the replay these were another pathway's chemistry)
    ("acetyl-CoA is first converted into acetyl-Pi by phosphotransacetylase", ["acetyl-CoA"], ["acetyl-Pi"], []),
    ("acetyl-Pi is further converted into acetate, catalyzed by acetate kinase", ["acetyl-Pi"], ["acetate"], []),
    # a span cut inside the catalyst name
    ("activation of DHB to DHB-AMP by Ent", ["DHB"], ["DHB-AMP"], ["Ent"]),
    # negated, and no catalyst
    ("chorismate is not first converted to isochorismate by EntC", ["chorismate"], ["isochorismate"], []),
    ("chorismate is first converted to isochorismate", ["chorismate"], ["isochorismate"], []),
    # the F-179 nominalized-subject shape is still not read by these templates
    ("condensation of glycine and succinyl-CoA to produce ALA by ALAS", ["glycine", "succinyl-CoA"], ["ALA"], []),
    # wrong direction, wrong catalyst
    ("chorismate, is first converted to isochorismate by EntC", ["isochorismate"], ["chorismate"], []),
    ("chorismate, is first converted to isochorismate by EntC", ["chorismate"], ["isochorismate"], ["MenF"]),
])
def test_still_refused(span, i, o, e):
    assert not ok(span, i, o, e)
