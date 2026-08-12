"""Acceptance tests for ``t2pw.pipeline.canonical.biological_equivalence`` (C-020).

NEW ACCEPTANCE TESTS -- NO REPLACED BASELINE BEHAVIOUR. ``src/t2pw/pipeline/canonical.py``
does not exist at the base SHA, so nothing here preserves or replaces prior observable
behaviour, and none of it is G9 regression evidence: an ImportError at the base SHA is
symbol absence, not a behavioural baseline.

Each mutation test changes exactly ONE dimension in exactly ONE format and asserts that the
comparator names that dimension. A dimension with no failing test is a dimension the
comparator does not really check.
"""

from __future__ import annotations

import copy
import socket
from pathlib import Path

import pytest

from t2pw.pipeline import canonical
from t2pw.pipeline.canonical import biological_equivalence

REPO = Path(__file__).resolve().parents[1]
STRICT = REPO / "runs_verify" / "2026-08-04_1647" / "papers" / "PMC12856317" / "strict"
REAL_JSON = STRICT / "final_mapped.json"
REAL_PWML = STRICT / "pathway.pwml"
REAL_SBML = REPO / "outputs" / "pathway.sbml"


def _json():
    """The canonical payload: ALAS2 condenses glycine and succinyl-CoA into ALA."""
    return {
        "metadata": {},
        "entities": {
            "species": [{"name": "Homo sapiens", "taxonomy_id": "9606"}],
            "subcellular_locations": [{"name": "mitochondrial matrix"}, {"name": "cytosol"}],
            "compounds": [
                {"name": "glycine", "mapped_ids": {"chebi": "CHEBI:15428"}},
                {"name": "succinyl-CoA", "mapped_ids": {"chebi": "CHEBI:15380"}},
                {"name": "aminolevulinic acid", "mapped_ids": {"chebi": "CHEBI:17549"}}],
            "proteins": [{"name": "ALAS2", "mapped_ids": {"uniprot": "P22557"}}],
            "protein_complexes": [],
        },
        "biological_states": [
            {"name": "mitochondrial matrix", "subcellular_location": "mitochondrial matrix"},
            {"name": "cytosol", "subcellular_location": "cytosol"}],
        "element_locations": {
            "compound_locations": [
                {"compound": name, "biological_state": "mitochondrial matrix"}
                for name in ("glycine", "succinyl-CoA", "aminolevulinic acid")],
            "protein_locations": [
                {"protein": "ALAS2", "biological_state": "mitochondrial matrix"}]},
        "processes": {"reactions": [{
            "name": "ALAS2-catalyzed condensation",
            "inputs": ["glycine", "succinyl-CoA"], "outputs": ["aminolevulinic acid"],
            "direction": "right", "reversible": False,
            "biological_state": "mitochondrial matrix",
            "enzymes": [{"entity": "ALAS2", "entity_type": "protein"}],
            "modifiers": [{"entity": "ALAS2", "entity_type": "protein"}]}]},
    }


# The same biology as _json(), carrying every difference PRODUCT_CONTRACT section 5 calls
# acceptable: unrelated generated XML ids, a timestamp, ragged whitespace, sections in a
# different order from the JSON, and x/y/width layout metadata on every location record.
_PWML = """<?xml version="1.0" encoding="UTF-8"?>
<super-pathway-visualization>
 <generated-at>2026-08-04T16:47:03Z</generated-at>
 <pathway-visualization-contexts><pathway-visualization-context>
  <pathway-visualization>
    <id>PathwayVisualization1</id>
    <subcellular-locations>
      <subcellular-location><id type="integer">2</id><name>cytosol</name></subcellular-location>
      <subcellular-location><id type="integer">1</id><name>mitochondrial matrix</name>
      </subcellular-location>
    </subcellular-locations>
    <species><species><id type="integer">1</id><name>Homo sapiens</name>
      <taxonomy-id>9606</taxonomy-id><common-name>Human</common-name></species></species>
    <biological-states>
      <biological-state><id type="integer">5</id>
        <subcellular-location-id type="integer">1</subcellular-location-id>
        <species-id type="integer">1</species-id><name>mitochondrial matrix</name></biological-state>
      <biological-state><id type="integer">6</id>
        <subcellular-location-id type="integer">2</subcellular-location-id>
        <species-id type="integer">1</species-id><name>cytosol</name></biological-state>
    </biological-states>
    <compounds>
      <compound><id type="integer">808</id><name>succinyl-CoA</name><chebi-id>15380</chebi-id>
        <description nil="true"/></compound>
      <compound><id type="integer">78</id><name>glycine</name><chebi-id>15428</chebi-id></compound>
      <compound><id type="integer">894</id><name>aminolevulinic acid</name>
        <chebi-id>17549</chebi-id></compound>
    </compounds>
    <proteins><protein><id type="integer">30000</id><name>ALAS2</name>
      <uniprot-id>P22557</uniprot-id><gene-name/></protein></proteins>
    <protein-complexes/>
    <compound-locations>
      <compound-location><id type="integer">24</id><compound-id type="integer">78</compound-id>
        <biological-state-id type="integer">5</biological-state-id>
        <x type="integer">240</x><y type="integer">1050</y><width>160</width></compound-location>
      <compound-location><id type="integer">25</id><compound-id type="integer">808</compound-id>
        <biological-state-id type="integer">5</biological-state-id>
        <x type="integer">240</x><y type="integer">1210</y><width>160</width></compound-location>
      <compound-location><id type="integer">26</id><compound-id type="integer">894</compound-id>
        <biological-state-id type="integer">5</biological-state-id>
        <x type="integer">700</x><y type="integer">1050</y><width>160</width></compound-location>
    </compound-locations>
    <protein-locations><protein-location><id type="integer">22</id>
      <protein-id type="integer">30000</protein-id>
      <biological-state-id type="integer">5</biological-state-id>
      <x type="integer">775</x><y type="integer">489</y></protein-location></protein-locations>
    <reactions>
      <reaction><id type="integer">50000</id><pwr-id>PW_R050000</pwr-id>
        <direction>Right</direction>
        <reaction-left-elements>
          <reaction-left-element><id type="integer">8</id>
            <element-id type="integer">78</element-id>
            <stoichiometry type="integer">1</stoichiometry>
            <element-type>Compound</element-type></reaction-left-element>
          <reaction-left-element><id type="integer">9</id>
            <element-id type="integer">808</element-id>
            <stoichiometry type="integer">1</stoichiometry>
            <element-type>Compound</element-type></reaction-left-element>
        </reaction-left-elements>
        <reaction-right-elements>
          <reaction-right-element><id type="integer">10</id>
            <element-id type="integer">894</element-id>
            <stoichiometry type="integer">1</stoichiometry>
            <element-type>Compound</element-type></reaction-right-element>
        </reaction-right-elements>
        <reaction-enzymes><reaction-enzyme><id type="integer">11</id>
          <protein-id type="integer">30000</protein-id></reaction-enzyme></reaction-enzymes>
      </reaction>
    </reactions>
    <transports/>
    <interactions/>
    <reaction-visualizations><reaction-visualization><id type="integer">36</id>
      <reaction-id type="integer">50000</reaction-id>
      <biological-state-id type="integer">5</biological-state-id></reaction-visualization>
    </reaction-visualizations>
  </pathway-visualization>
 </pathway-visualization-context></pathway-visualization-contexts>
</super-pathway-visualization>
"""

_SPECIES = ('<species boundaryCondition="false" constant="false" hasOnlySubstanceUnits="true"'
            ' metaid="{m}" compartment="BiologicalState1" name="{n}" id="{i}">'
            '<annotation><pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
            ' pathwhiz:species_type="{t}"/>'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
            ' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">'
            '<rdf:Description rdf:about="#{m}"><bqbiol:is><rdf:Bag>'
            '<rdf:li rdf:resource="urn:miriam:{d}"/></rdf:Bag></bqbiol:is>'
            '</rdf:Description></rdf:RDF></annotation></species>')

_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="PathwayModel" metaid="model1">
  <annotation><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
    <rdf:Description rdf:about="#model1"><bqbiol:hasTaxon><rdf:Bag>
    <rdf:li rdf:resource="urn:miriam:taxonomy:9606"/>
    </rdf:Bag></bqbiol:hasTaxon></rdf:Description></rdf:RDF></annotation>
  <listOfCompartments>
   <compartment id="BiologicalState1" name="mitochondrial matrix" constant="false"/>
   <compartment id="BiologicalState2" name="cytosol" constant="false"/>
  </listOfCompartments>
  <listOfSpecies>{species}</listOfSpecies>
  <listOfReactions>
   <reaction id="Reaction50000" name="condensation" reversible="false" fast="false"
     compartment="BiologicalState1">
    <annotation><pathwhiz:reaction xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"
      pathwhiz:reaction_type="reaction"/></annotation>
    <listOfReactants>
     <speciesReference species="Compound104" stoichiometry="1" constant="false"/>
     <speciesReference species="Compound808" stoichiometry="1" constant="false"/>
    </listOfReactants>
    <listOfProducts>
     <speciesReference species="Compound894" stoichiometry="1" constant="false"/>
    </listOfProducts>
    <listOfModifiers><modifierSpeciesReference species="Protein266"/></listOfModifiers>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
""".replace("{species}", "".join(
    _SPECIES.format(m="meta%d" % index, n=name, i=sbml_id, t=kind, d=identifier)
    for index, (name, sbml_id, kind, identifier) in enumerate((
        ("glycine", "Compound104", "compound", "chebi:CHEBI:15428"),
        ("succinyl-CoA", "Compound808", "compound", "chebi:CHEBI:15380"),
        ("aminolevulinic acid", "Compound894", "compound", "chebi:CHEBI:17549"),
        ("ALAS2", "Protein266", "protein", "uniprot:P22557")))))


def _report(json_payload=None, pwml=None, sbml=None):
    return biological_equivalence(json_payload=_json() if json_payload is None else json_payload,
                                  pwml=_PWML if pwml is None else pwml, sbml=sbml)


def _differing(report, dimension, *also):
    """``dimension`` is reported different, and only it and the ``also`` it entails are.

    ``also`` is named explicitly rather than allowed loosely: moving a participant across
    sides genuinely changes stoichiometry and the process's references too, and a test that
    tolerated any extra dimension would stop noticing when one leaks.
    """
    assert report.verdict == canonical.VERDICT_NOT_EQUIVALENT, report.to_dict()
    assert report.dimension_status[dimension] == canonical.STATUS_DIFFERENT, report.to_dict()
    assert {d.dimension for d in report.differences} == {dimension, *also}, report.to_dict()
    assert {d.formats for d in report.differences_for(dimension)} == {("json", "pwml")}
    return report.differences_for(dimension)


# --------------------------------------------------------------------- the equivalent case


def test_same_biology_in_json_and_pwml_is_equivalent_despite_cosmetic_differences():
    report = _report()
    assert report.verdict == canonical.VERDICT_EQUIVALENT, report.to_dict()
    assert report.differences == () and report.not_representable == ()
    assert set(report.dimension_status) == set(canonical.DIMENSIONS)
    assert set(report.dimension_status.values()) == {canonical.STATUS_EQUIVALENT}


def test_the_cosmetic_differences_the_contract_allows_are_really_present():
    """Guards the test above: it would prove nothing if the PWML were a JSON transcription."""
    assert "2026-08-04T16:47:03Z" in _PWML          # a timestamp
    assert "<x type=\"integer\">240</x>" in _PWML   # non-biological layout metadata
    assert "PW_R050000" in _PWML                    # a generated internal XML id
    assert _PWML.index("<subcellular-locations>") < _PWML.index("<species>")  # other order
    assert _PWML.index("succinyl-CoA") < _PWML.index(">glycine<")  # other element order


def test_three_formats_of_the_same_biology_report_no_difference_at_all():
    report = _report(sbml=_SBML)
    assert report.formats_compared == ("json", "pwml", "sbml")
    assert report.differences == (), report.to_dict()
    # Not "equivalent": one dimension could not be compared, and an unrepresentable
    # dimension is never counted as agreement.
    assert report.verdict == canonical.VERDICT_INCOMPLETE
    assert [(g.dimension, g.format, g.key) for g in report.not_representable] == [
        ("organism_context", "sbml", "names")]


def test_a_dimension_a_format_cannot_express_is_not_representable_not_agreement():
    payload = _json()
    payload["processes"]["interactions"] = [
        {"entity_1": "aminolevulinic acid", "entity_2": "ALAS2", "relationship": "inhibits"}]
    report = _report(payload, sbml=_SBML)
    gap = next(g for g in report.not_representable
               if g.dimension == "process_entity_references")
    assert (gap.format, gap.key) == ("sbml", "interactions")
    assert "no construct for a non-stoichiometric regulatory interaction" in gap.reason
    assert report.dimension_status["organism_context"] == canonical.STATUS_NOT_REPRESENTABLE


def test_an_unrepresentable_dimension_with_nothing_to_express_raises_nothing():
    """SBML cannot carry interactions -- but this pathway has none to lose."""
    report = _report(sbml=_SBML)
    assert [g.dimension for g in report.not_representable] == ["organism_context"]


# ------------------------------------------------- one test per PRODUCT_CONTRACT section 5


def test_dimension_1_a_reaction_whose_participant_changed_is_reported():
    pwml = _PWML.replace('<element-id type="integer">78</element-id>',
                         '<element-id type="integer">894</element-id>', 1)
    report = _report(pwml=pwml)
    assert report.dimension_status["reactions"] == canonical.STATUS_DIFFERENT
    assert any("glycine" in d.values[0] for d in report.differences_for("reactions"))


def test_dimension_2_a_reactant_moved_to_the_product_side_is_reported():
    payload = _json()
    reaction = payload["processes"]["reactions"][0]
    reaction["inputs"], reaction["outputs"] = ["glycine"], ["aminolevulinic acid", "succinyl-CoA"]
    differences = _differing(_report(payload), "reactants_and_products",
                             "stoichiometry", "process_entity_references")
    assert {d.key.rsplit("/", 1)[-1] for d in differences} == {"left", "right"}
    # The reaction itself is still the SAME reaction: its participants did not change.
    assert _report(payload).dimension_status["reactions"] == canonical.STATUS_EQUIVALENT


def test_dimension_3_directionality_and_reversibility_are_reported():
    pwml = _PWML.replace("<direction>Right</direction>", "<direction>Both</direction>", 1)
    differences = _differing(_report(pwml=pwml), "directionality_and_reversibility")
    reported = {d.key.rsplit("/", 1)[-1]: d.values for d in differences}
    assert reported["direction"] == ('"right"', '"both"')
    assert reported["reversible"] == ("false", "true")


def test_dimension_4_a_changed_stoichiometric_coefficient_is_reported():
    pwml = _PWML.replace("""<element-id type="integer">808</element-id>
            <stoichiometry type="integer">1</stoichiometry>""",
                         """<element-id type="integer">808</element-id>
            <stoichiometry type="integer">2</stoichiometry>""", 1)
    differences = _differing(_report(pwml=pwml), "stoichiometry")
    assert "succinyl coa" in differences[0].values[1] and "2" in differences[0].values[1]


def test_dimension_5_a_dropped_enzyme_is_reported():
    # Split first: the same protein id also appears in <protein-locations>, and dropping
    # THAT would mutate cellular locations instead of the enzyme this test is about.
    head, tail = _PWML.split("<reactions>", 1)
    pwml = head + "<reactions>" + tail.replace(
        '<protein-id type="integer">30000</protein-id>', "", 1)
    differences = _differing(_report(pwml=pwml), "enzymes_and_modifiers",
                             "process_entity_references")
    assert {d.key.rsplit("/", 1)[-1] for d in differences} == {"enzymes", "modifiers"}
    assert all("protein|alas2" in d.values[0] and d.values[1] == "[]" for d in differences)


def test_dimension_6_a_transport_the_export_dropped_is_reported_with_its_cargo():
    payload = _json()
    payload["processes"]["transports"] = [{
        "cargo": "ALAS2", "from_biological_state": "cytosol",
        "to_biological_state": "mitochondrial matrix",
        "transporters": [{"entity": "ALAS2", "entity_type": "protein"}]}]
    differences = _differing(_report(payload), "transports_and_cargo",
                             "process_entity_references")
    assert [d.key for d in differences] == ["protein|alas2"]
    assert "cytosol" in differences[0].values[0] and differences[0].values[1] == "<absent>"


def test_dimension_7_a_changed_biological_identifier_is_reported():
    pwml = _PWML.replace("<chebi-id>15428</chebi-id>", "<chebi-id>99999</chebi-id>", 1)
    differences = _differing(_report(pwml=pwml), "entities_and_identifiers")
    assert differences[0].key == "compound|glycine"
    assert "15428" in differences[0].values[0] and "99999" in differences[0].values[1]


def test_dimension_7_an_identifier_inside_a_list_container_is_read_and_compared():
    """`pwml/writer.py:380,1219` emits EC as <ec-numbers><ec-number>...</ec-number></...>.

    Reading direct children only loses every value in such a container, which made the
    comparator report a difference between two documents that both carry the identifier.
    """
    payload = {"entities": {"proteins": [{"name": "ALAS2", "ec_number": "2.3.1.37"}]},
               "processes": {}}
    carried = ("<pathway-visualization><proteins><protein><id>1</id><name>ALAS2</name>"
               "<ec-numbers><ec-number>2.3.1.37</ec-number></ec-numbers>"
               "</protein></proteins></pathway-visualization>")
    assert canonical._parse_pwml(canonical._xml_root(carried)).entities[
        "protein|alas2"]["ids"] == {"ec": "2.3.1.37"}
    assert biological_equivalence(json_payload=payload, pwml=carried).verdict == \
        canonical.VERDICT_EQUIVALENT
    # Mutation-killing: drop the container and the dimension must be reported different.
    dropped = carried.replace(
        "<ec-numbers><ec-number>2.3.1.37</ec-number></ec-numbers>", "")
    differences = _differing(biological_equivalence(json_payload=payload, pwml=dropped),
                            "entities_and_identifiers")
    assert differences[0].values == ('[["ec", "2.3.1.37"]]', "[]")


def test_dimension_8_a_complex_that_lost_a_component_is_reported():
    payload = _json()
    payload["entities"]["protein_complexes"] = [{
        "name": "ALAS2 complex",
        "components": [{"name": "ALAS2", "stoichiometry": 2}]}]
    report = _report(payload)
    assert report.dimension_status["complexes_and_components"] == canonical.STATUS_DIFFERENT
    difference = report.differences_for("complexes_and_components")[0]
    assert difference.key == "protein_complex|alas2 complex"
    assert difference.values[1] == "<absent>"


def test_dimension_9_an_entity_moved_to_another_compartment_is_reported():
    pwml = _PWML.replace("""<compound-id type="integer">78</compound-id>
        <biological-state-id type="integer">5</biological-state-id>""",
                         """<compound-id type="integer">78</compound-id>
        <biological-state-id type="integer">6</biological-state-id>""", 1)
    differences = _differing(_report(pwml=pwml), "cellular_locations")
    assert differences[0].key == "entity_locations/compound|glycine"
    assert differences[0].values == ('["mitochondrial matrix"]', '["cytosol"]')


def test_dimension_10_a_process_pointing_at_another_state_is_reported():
    pwml = _PWML.replace("""<reaction-id type="integer">50000</reaction-id>
      <biological-state-id type="integer">5</biological-state-id>""",
                         """<reaction-id type="integer">50000</reaction-id>
      <biological-state-id type="integer">6</biological-state-id>""", 1)
    differences = _differing(_report(pwml=pwml), "process_entity_references")
    assert "biological_state=mitochondrial matrix" in differences[0].values[0]
    assert "biological_state=cytosol" in differences[0].values[1]


def test_dimension_11_organism_is_reported_and_never_canonicalized():
    """D-016 / C-045: 'Homo sapiens' against 'human' is REPORTED, never resolved."""
    pwml = _PWML.replace("<name>Homo sapiens</name>", "<name>human</name>", 1)
    differences = _differing(_report(pwml=pwml), "organism_context")
    assert differences[0].key == "names"
    assert differences[0].values == ('["homo sapiens"]', '["human"]')


def test_dimension_11_organism_differences_survive_lexical_normalization_only():
    pwml = _PWML.replace("<name>Homo sapiens</name>", "<name>  HOMO   SAPIENS  </name>", 1)
    assert _report(pwml=pwml).dimension_status["organism_context"] == \
        canonical.STATUS_EQUIVALENT


def test_dimension_12_a_reference_to_a_missing_entity_is_reported_as_broken():
    pwml = _PWML.replace('<element-id type="integer">78</element-id>',
                         '<element-id type="integer">777</element-id>', 1)
    report = _report(pwml=pwml)
    assert report.dimension_status["no_broken_references"] == canonical.STATUS_DIFFERENT
    broken = report.differences_for("no_broken_references")
    assert broken[0].formats == ("pwml", "pwml")
    assert "777" in broken[0].key and broken[0].values == ("<broken reference>", "<resolves>")


# ------------------------------------------------------------------------ real artifacts


def test_the_json_parser_reads_an_artifact_this_repository_produced():
    report = biological_equivalence(json_payload=REAL_JSON, pwml=REAL_PWML)
    assert report.dimension_status["reactions"] == canonical.STATUS_EQUIVALENT
    assert report.dimension_status["stoichiometry"] == canonical.STATUS_EQUIVALENT
    assert report.dimension_status["complexes_and_components"] == canonical.STATUS_EQUIVALENT
    assert report.dimension_status["organism_context"] == canonical.STATUS_EQUIVALENT


def test_the_pwml_parser_reads_an_artifact_this_repository_produced():
    graph = canonical._parse_pwml(canonical._xml_root(REAL_PWML))
    assert graph.organism_names == ["homo sapiens"] and graph.organism_taxa == ["9606"]
    # The real <protein> carries <uniprot-id> as a direct child and EC inside an
    # <ec-numbers> container; <gene-name/> is empty, so `gene` is genuinely absent here.
    assert graph.entities["protein|alas2"]["ids"] == {"uniprot": "p22557", "ec": "2.3.1.37"}
    assert graph.entities["protein_complex|alas2 complex"]["ids"] == {}
    assert graph.complexes["protein_complex|alas2 complex"] == [("protein|alas2", 1)]
    assert [k for k, _ in graph.reactions[0]["right"]] == ["compound|aminolevulinic acid"]
    assert graph.transports[0]["to"] == "mitochondrial matrix state"
    assert graph.broken == []


def test_the_sbml_parser_reads_an_artifact_this_repository_produced():
    graph = canonical._parse_sbml(canonical._xml_root(REAL_SBML))
    assert len(graph.entities) == 42 and len(graph.locations) == 4
    assert len(graph.reactions) == 21 and len(graph.transports) == 2
    assert graph.entities["protein|comt"]["ids"] == {"uniprot": "p21964"}
    assert graph.broken == []
    # One SBML species binds to one compartment, so a transport cannot state both ends.
    assert graph.transports[0]["from"] is canonical.ABSENT
    assert graph.transports[0]["to"] is canonical.ABSENT
    # The exporter emits no MIRIAM taxonomy annotation, so organism is genuinely lost here.
    assert graph.organism_taxa == []


def test_the_real_round_trip_reports_the_losses_it_actually_has():
    """The comparator is useful only if it names real losses on a real pair."""
    report = biological_equivalence(json_payload=REAL_JSON, pwml=REAL_PWML)
    assert report.verdict == canonical.VERDICT_NOT_EQUIVALENT
    assert {d.dimension for d in report.differences} == {
        "cellular_locations", "directionality_and_reversibility",
        "entities_and_identifiers", "process_entity_references"}
    direction = report.differences_for("directionality_and_reversibility")
    assert [d.values for d in direction] == [("null", '"right"'), ("null", "false")]
    # Only `gene` is lost here: the PWML does carry EC, inside an <ec-numbers> container.
    assert report.differences_for("entities_and_identifiers")[0].values == (
        '[["ec", "2.3.1.37"], ["gene", "alas2"], ["uniprot", "p22557"]]',
        '[["ec", "2.3.1.37"], ["uniprot", "p22557"]]')


# ------------------------------------------------------------- contract-level obligations


def test_a_hash_is_not_the_equivalence_decision():
    """D-007: two payloads with the same graph hash still differ if the biology differs."""
    payload = _json()
    payload["processes"]["reactions"][0]["direction"] = "both"
    assert "hashlib" not in dir(canonical) and "canonical_hash" not in dir(canonical)
    assert _report(payload).verdict == canonical.VERDICT_NOT_EQUIVALENT


def test_one_document_cannot_be_compared_against_itself():
    with pytest.raises(ValueError, match="at least two"):
        biological_equivalence(json_payload=_json())


def test_the_same_inputs_produce_the_same_report_twice():
    first = biological_equivalence(json_payload=REAL_JSON, pwml=REAL_PWML, sbml=REAL_SBML)
    second = biological_equivalence(json_payload=REAL_JSON, pwml=REAL_PWML, sbml=REAL_SBML)
    assert first.to_dict() == second.to_dict()
    ordering = [(d.dimension, d.formats, d.key, d.values) for d in first.differences]
    assert ordering == sorted(ordering), "differences must be deterministically ordered"


def test_the_comparator_performs_no_network_or_database_lookup(monkeypatch):
    """PRODUCT_CONTRACT section 8. Asserted at runtime, not inferred from the source."""
    def refuse(*args, **kwargs):
        raise AssertionError("the comparator opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    report = biological_equivalence(json_payload=REAL_JSON, pwml=REAL_PWML, sbml=REAL_SBML)
    assert report.verdict == canonical.VERDICT_NOT_EQUIVALENT


def test_the_comparator_does_not_mutate_the_payload_it_was_given():
    """It reports differences; it never repairs its inputs into agreement."""
    payload = _json()
    before = copy.deepcopy(payload)
    biological_equivalence(json_payload=payload, pwml=_PWML, sbml=_SBML)
    assert payload == before


def test_the_comparator_does_not_import_the_rag_package():
    """The RAG -> core separation invariant holds the arrow one way."""
    source = (REPO / "src" / "t2pw" / "pipeline" / "canonical.py").read_text(encoding="utf-8")
    statements = [line for line in source.splitlines()
                  if line.startswith(("import ", "from "))]
    assert statements and not any("rag" in line for line in statements), statements
