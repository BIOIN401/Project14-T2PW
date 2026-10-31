"""ORCH-797c (D-120 § 1): the Stage-6 name sets ``entity_identity`` reads are folded with the one reference key.

**G9: A CORRECTION OF OBSERVABLE BEHAVIOUR.** Only names present at base ``5a66b2f5`` are imported. At base,
``map_ids._collect_protein_like_names`` folded with ``map_ids._normalize_name``, which STRIPS Greek letters, and
``entity_identity`` read the set through the ``_named_in`` bridge. Both ``test_g9_*`` fail there on values:

* ``IkappaB`` is not recognised as the declared protein ``IκB`` (base: ``ib`` in the set, ``ikappab`` asked);
* a compound ``IB`` is routed as the protein ``IκBα`` (base: both fold to ``ib``), a wrong identity route.

``test_preservation_*`` passes at base and tip. No network is called.
"""

from __future__ import annotations

from t2pw.mapping.map_ids import _collect_protein_like_names
from t2pw.pipeline.entity_identity import route_entity_for_mapping


def _payload(protein: str):
    return {"entities": {"proteins": [{"name": protein}], "compounds": [], "protein_complexes": []},
            "processes": {}, "element_locations": {}}


def _route(name: str, protein: str):
    return route_entity_for_mapping(name, "compound", protein_like_names=_collect_protein_like_names(_payload(protein)))


def test_g9_a_greek_letter_spelled_out_names_the_declared_protein():
    assert _route("IkappaB", "IκB")["reason"] == "known_protein_like"  # base: default_compound_route


def test_g9_a_name_without_the_greek_letter_is_not_that_protein():
    assert _route("IB", "IκBα")["reason"] != "known_protein_like"  # base: known_protein_like (both fold to "ib")


def test_preservation_the_declared_spelling_still_routes_as_the_protein():
    assert _route("IκB", "IκB")["reason"] == "known_protein_like"
    assert _route("glutathione", "IκB")["route"] == "compound"
