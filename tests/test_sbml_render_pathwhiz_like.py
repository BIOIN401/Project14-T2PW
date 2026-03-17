from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sbml_add_pathwhiz_layout import NS as PW_NS_MAP  # noqa: E402
from sbml_add_pathwhiz_layout import add_pathwhiz_layout  # noqa: E402
from sbml_render_pathwhiz_like import build_render_artifacts  # noqa: E402
from sbml_render_pathwhiz_like import render_to_png_bytes  # noqa: E402


CORE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m1" name="demo">
    <listOfCompartments>
      <compartment id="c_cell" name="cell" spatialDimensions="3" size="1" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="m_a_c_cell" name="A" compartment="c_cell" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="p_b_c_cell" name="B" compartment="c_cell" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="m_c_c_cell" name="C" compartment="c_cell" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="r1" name="A to C" reversible="false" compartment="c_cell">
        <listOfReactants>
          <speciesReference species="m_a_c_cell" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="m_c_c_cell" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <listOfModifiers>
          <modifierSpeciesReference species="p_b_c_cell"/>
        </listOfModifiers>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_add_pathwhiz_layout_creates_location_elements() -> None:
    case_dir = _make_case_dir()
    try:
        in_path = case_dir / "core.sbml"
        out_path = case_dir / "core_with_layout.sbml"
        in_path.write_text(CORE_SBML, encoding="utf-8")

        add_pathwhiz_layout(str(in_path), str(out_path))

        root = ET.fromstring(out_path.read_text(encoding="utf-8"))
        location_elements = root.findall(".//pathwhiz:location_element", PW_NS_MAP)
        assert location_elements
        assert any(elem.get("{http://www.spmdb.ca/pathwhiz}element_type") == "edge" for elem in location_elements)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_render_to_png_bytes_renders_core_sbml_without_preexisting_layout() -> None:
    pytest.importorskip("matplotlib")
    case_dir = _make_case_dir()
    try:
        in_path = case_dir / "core.sbml"
        in_path.write_text(CORE_SBML, encoding="utf-8")

        png_bytes = render_to_png_bytes(str(in_path), dpi=120)

        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png_bytes) > 1000
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_build_render_artifacts_reports_synthesized_geometry() -> None:
    pytest.importorskip("matplotlib")
    case_dir = _make_case_dir()
    try:
        in_path = case_dir / "core.sbml"
        in_path.write_text(CORE_SBML, encoding="utf-8")

        artifacts = build_render_artifacts(str(in_path), dpi=120)
        summary = artifacts["layout_summary"]

        assert artifacts["png_bytes"].startswith(b"\x89PNG\r\n\x1a\n")
        assert b"location_element" in artifacts["render_ready_sbml_bytes"]
        assert summary["geometry_source"] == "synthesized"
        assert summary["original_has_pathwhiz_layout"] is False
        assert summary["render_input_has_pathwhiz_layout"] is True
        assert summary["has_drawable_geometry"] is True
        assert summary["visible_location_element_count"] > 0
        assert summary["edge_count"] > 0
        assert summary["node_count"] > 0
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def _make_case_dir() -> Path:
    case_dir = ROOT / "tmp" / f"pytest_sbml_render_{uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=False)
    return case_dir
