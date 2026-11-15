"""Final payload P5 (D-W21 a/b): id-less locations, tissues and cell types leave the PWML only when no state could then bind
to another state (the PathWhiz importer treats an omitted component as a wildcard). Tiny synthetic payloads, no network."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "tests"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from test_pwml_writer import build_pwml_ir  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402

SP = {"name": "Homo sapiens", "pathwhiz_id": 1}
TI = {"name": "root", "pathwhiz_id": 5}
CT = {"name": "neuron", "pathwhiz_id": 7}
LOC = {"name": "cytosol", "pathwhiz_id": 3}
LOC2 = {"name": "nucleus", "pathwhiz_id": 4}


def _build(states, *, locs=(), tissues=(), cts=()):
    payload = {"entities": {"species": [SP], "subcellular_locations": list(locs), "tissues": list(tissues),
                            "cell_types": list(cts), "compounds": [{"name": "A", "pathbank_compound_id": 103}]},
               "biological_states": states, "processes": {"reactions": [], "transports": [], "interactions": []}}
    ir, report = build_pwml_ir(payload, strict_db=False)
    assert report["errors"] == []
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000", height=1400,
                           width=3200, background_color="#FFFFFF", ref=str(ROOT / "reference" / "PW000001.pwml"))
    b = DeterministicPwmlBuilder(extraction=ir, signature=discover_structure_signature(ROOT / "reference" / "PW000001.pwml"), args=args)
    return b, b.build().root


def _names(root, section):
    return [n.findtext("name") for n in root.findall(f".//{section}/{section[:-1]}")]


def _states(root):
    """state name -> (tissue, cell type, location) names as written ('' = component omitted)."""
    by_id = {}
    for sec in ("tissues", "cell-types", "subcellular-locations"):
        for n in root.findall(f".//{sec}/{sec[:-1]}"):
            by_id[(sec, (n.findtext("id") or "").strip())] = n.findtext("name")
    out = {}
    for st in root.findall(".//biological-states/biological-state"):
        got = []
        for sec, tag in (("tissues", "tissue-id"), ("cell-types", "cell-type-id"), ("subcellular-locations", "subcellular-location-id")):
            v = (st.findtext(tag) or "").strip()
            got.append(by_id[(sec, v)] if v else "")
        out[st.findtext("name")] = tuple(got)
    return out


def _s(name, **k):
    return {"name": name, "species": "Homo sapiens", **k}


def test_wildcard_subset_bind_keeps_the_location(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    # A without its id-less location would be (species, tissue), a subset of B (species, tissue, location): the importer would bind A to B
    states = [_s("A", tissue="root", subcellular_location="zz idless place"), _s("B", tissue="root", subcellular_location="cytosol")]
    b, root = _build(states, locs=[{"name": "zz idless place"}, LOC], tissues=[TI])
    assert _states(root) == {"A": ("root", "", "zz idless place"), "B": ("root", "", "cytosol")}
    assert b.layout_debug_counts["components_kept_wildcard_guard"] == 1
    assert "locations_dropped_no_id" not in b.layout_debug_counts


def test_idless_cell_type_dropped_when_safe(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    states = [_s("A", cell_type="zz idless cell", subcellular_location="cytosol"), _s("B", subcellular_location="nucleus")]
    b, root = _build(states, locs=[LOC, LOC2], cts=[{"name": "zz idless cell"}])
    assert _names(root, "cell-types") == []
    assert _states(root) == {"A": ("", "", "cytosol"), "B": ("", "", "nucleus")}
    assert b.layout_debug_counts["cell_types_dropped_no_id"] == 1 and "components_kept_wildcard_guard" not in b.layout_debug_counts


def test_idless_tissue_dropped_when_safe(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    states = [_s("A", tissue="zz idless tissue", subcellular_location="cytosol"), _s("B", tissue="root", subcellular_location="nucleus")]
    b, root = _build(states, locs=[LOC, LOC2], tissues=[{"name": "zz idless tissue"}, TI])
    assert _names(root, "tissues") == ["root"]
    assert _states(root)["A"] == ("", "", "cytosol")
    assert b.layout_debug_counts["tissues_dropped_no_id"] == 1


def test_states_identical_after_the_drop_keep_their_components(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    states = [_s("A", tissue="zz tissue one", subcellular_location="cytosol"), _s("B", tissue="zz tissue two", subcellular_location="cytosol")]
    b, root = _build(states, locs=[LOC], tissues=[{"name": "zz tissue one"}, {"name": "zz tissue two"}])
    assert _names(root, "tissues") == ["zz tissue one", "zz tissue two"]
    assert _states(root) == {"A": ("zz tissue one", "", "cytosol"), "B": ("zz tissue two", "", "cytosol")}
    assert b.layout_debug_counts["components_kept_wildcard_guard"] == 2 and "tissues_dropped_no_id" not in b.layout_debug_counts


def test_species_only_state_keeps_its_idless_component(monkeypatch) -> None:
    # every IR state has a location; A has only an id-less location besides the species, so dropping it leaves a species-only state
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    states = [_s("A", subcellular_location="zz idless place"), _s("B", subcellular_location="cytosol")]
    b, root = _build(states, locs=[{"name": "zz idless place"}, LOC])
    assert _states(root)["A"] == ("", "", "zz idless place")
    assert b.layout_debug_counts["components_kept_wildcard_guard"] == 1 and "locations_dropped_no_id" not in b.layout_debug_counts


def test_both_idless_components_leave_the_state_one_of_them(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    states = [_s("A", cell_type="zz idless cell", subcellular_location="zz idless place"), _s("B", subcellular_location="cytosol")]
    b, root = _build(states, locs=[{"name": "zz idless place"}, LOC], cts=[{"name": "zz idless cell"}])
    assert _states(root)["A"] == ("", "", "zz idless place")  # location restored first, cell type dropped
    assert b.layout_debug_counts["cell_types_dropped_no_id"] == 1 and b.layout_debug_counts["components_kept_wildcard_guard"] == 1


def test_joint_drop_is_evaluated_together_and_restores_one_component(monkeypatch) -> None:
    # A drops either id-less component safely on its own ((sp, ct, loc) and (sp, tissue, ct) both differ from B), but dropping both
    # leaves (sp, ct), a subset of B (sp, ct, cytosol): exactly one is restored, the location first
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    states = [_s("A", tissue="zz idless tissue", cell_type="neuron", subcellular_location="zz idless place"),
              _s("B", cell_type="neuron", subcellular_location="cytosol")]
    b, root = _build(states, locs=[{"name": "zz idless place"}, LOC], tissues=[{"name": "zz idless tissue"}], cts=[CT])
    assert _states(root) == {"A": ("", "neuron", "zz idless place"), "B": ("", "neuron", "cytosol")}
    assert b.layout_debug_counts["tissues_dropped_no_id"] == 1 and b.layout_debug_counts["components_kept_wildcard_guard"] == 1


def test_dropping_one_component_alone_is_safe_in_the_joint_setup(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    states = [_s("A", tissue="zz idless tissue", cell_type="neuron", subcellular_location="cytosol"),
              _s("B", cell_type="neuron", subcellular_location="nucleus")]
    b, root = _build(states, locs=[LOC, LOC2], tissues=[{"name": "zz idless tissue"}], cts=[CT])
    assert _states(root)["A"] == ("", "neuron", "cytosol") and b.layout_debug_counts["tissues_dropped_no_id"] == 1


def test_flag_off_writes_every_row_and_counts_nothing(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "0")
    states = [_s("A", tissue="zz idless tissue", cell_type="zz idless cell", subcellular_location="zz idless place"), _s("B", subcellular_location="cytosol")]
    b, root = _build(states, locs=[{"name": "zz idless place"}, LOC], tissues=[{"name": "zz idless tissue"}], cts=[{"name": "zz idless cell"}])
    assert _states(root)["A"] == ("zz idless tissue", "zz idless cell", "zz idless place")
    assert not [k for k in b.layout_debug_counts if "dropped_no_id" in k or "wildcard" in k or "kept_state" in k]


def test_rows_with_a_database_id_are_never_dropped(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    states = [_s("A", tissue="root", cell_type="neuron", subcellular_location="cytosol"), _s("B", subcellular_location="nucleus")]
    b, root = _build(states, locs=[LOC, LOC2], tissues=[TI], cts=[CT])
    assert _states(root)["A"] == ("root", "neuron", "cytosol")
    assert not [k for k in b.layout_debug_counts if "dropped_no_id" in k]
