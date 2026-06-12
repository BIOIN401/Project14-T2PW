#!/usr/bin/env python
"""Confirmation script for outputs/pathway.pwml.

Parses the PWML (XML) output and verifies internal referential integrity and
basic layout sanity:

  1. Every compound-location-id referenced from reaction/transport/
     reaction-coupled-transport visualizations exists in <compound-locations>.
  2. Every edge-id referenced exists in <edges>.
  3. Every protein-location-id referenced (including inside
     protein_complex_protein_visualizations) exists in <protein-locations>.
  4. Reports canvas width/height and flags if width is unexpectedly inflated
     relative to the requested width (default 3200, per writer.py --width
     default).
  5. Computes an approximate length (first-point to last-point distance) for
     every edge's path and reports the longest edge, flagging edges over
     ~1000px (heads up) and non-transport edges over ~600px (hard issue).

Usage:
    python scripts/confirm_pwml.py [path/to/pathway.pwml]
"""

from __future__ import annotations

import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_PWML_PATH = Path("outputs/pathway.pwml")
DEFAULT_REQUESTED_WIDTH = 3200
WIDTH_TOLERANCE = 400  # px slack allowed over requested width before flagging
EDGE_HEADS_UP_THRESHOLD = 1000  # px - any edge over this is worth a heads-up
NON_TRANSPORT_EDGE_HARD_THRESHOLD = 600  # px - hard issue for non-transport edges


def find_pathway_visualization(root: ET.Element) -> ET.Element:
    pv = root.find(".//pathway-visualization")
    if pv is None:
        raise SystemExit("Could not find <pathway-visualization> element in PWML file.")
    return pv


def collect_ids(parent: ET.Element, child_tag: str, id_tag: str = "id") -> set[str]:
    """Collect the text of <id_tag> for each direct child named child_tag."""
    ids: set[str] = set()
    container = parent.find(child_tag)
    if container is None:
        return ids
    for item in container:
        id_el = item.find(id_tag)
        if id_el is not None and id_el.text is not None:
            ids.add(id_el.text.strip())
    return ids


def collect_referenced_ids(pv: ET.Element, ref_tag: str) -> list[tuple[str, str]]:
    """Find all elements named ref_tag anywhere under pv, returning
    (containing-element-tag, ref-value) tuples for context in error messages."""
    results: list[tuple[str, str]] = []
    for el in pv.iter(ref_tag):
        if el.text is not None and el.text.strip():
            # Find a human-readable "context" - nearest ancestor with an id.
            results.append((el.tag, el.text.strip()))
    return results


PATH_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def path_endpoints(path_str: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Extract the first and last (x, y) coordinate pairs from an SVG-like
    path string composed of M/C/L commands and numeric coordinate pairs."""
    if not path_str:
        return None
    nums = [float(n) for n in PATH_NUM_RE.findall(path_str)]
    if len(nums) < 4:
        return None
    # Pair up consecutive numbers as (x, y) coordinates.
    points = list(zip(nums[0::2], nums[1::2]))
    if len(points) < 2:
        return None
    return points[0], points[-1]


def edge_length(path_str: str) -> float | None:
    pts = path_endpoints(path_str)
    if pts is None:
        return None
    (x0, y0), (x1, y1) = pts
    return math.hypot(x1 - x0, y1 - y0)


def main(argv: list[str]) -> int:
    pwml_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PWML_PATH
    if not pwml_path.exists():
        raise SystemExit(f"PWML file not found: {pwml_path}")

    tree = ET.parse(pwml_path)
    root = tree.getroot()
    pv = find_pathway_visualization(root)

    overall_pass = True

    print(f"Confirming PWML file: {pwml_path}")
    print("=" * 70)

    # --- Build id sets ---------------------------------------------------
    compound_location_ids = collect_ids(pv, "compound-locations")
    edge_ids = collect_ids(pv, "edges")
    protein_location_ids = collect_ids(pv, "protein-locations")

    # --- Check 1: compound-location-id references ------------------------
    cl_refs = collect_referenced_ids(pv, "compound-location-id")
    cl_dangling = [(tag, ref) for tag, ref in cl_refs if ref not in compound_location_ids]
    check1_pass = not cl_dangling
    overall_pass &= check1_pass
    print(f"[CHECK 1] compound-location-id references valid: "
          f"{'PASS' if check1_pass else 'FAIL'}")
    print(f"          total references: {len(cl_refs)}, "
          f"known compound-locations: {len(compound_location_ids)}, "
          f"dangling: {len(cl_dangling)}")
    if cl_dangling:
        for tag, ref in cl_dangling[:20]:
            print(f"          dangling compound-location-id={ref} (in <{tag}>)")

    # --- Check 2: edge-id references ---------------------------------------
    edge_refs = collect_referenced_ids(pv, "edge-id")
    edge_dangling = [(tag, ref) for tag, ref in edge_refs if ref not in edge_ids]
    check2_pass = not edge_dangling
    overall_pass &= check2_pass
    print(f"[CHECK 2] edge-id references valid: "
          f"{'PASS' if check2_pass else 'FAIL'}")
    print(f"          total references: {len(edge_refs)}, "
          f"known edges: {len(edge_ids)}, dangling: {len(edge_dangling)}")
    if edge_dangling:
        for tag, ref in edge_dangling[:20]:
            print(f"          dangling edge-id={ref} (in <{tag}>)")

    # --- Check 3: protein-location-id references ---------------------------
    pl_refs = collect_referenced_ids(pv, "protein-location-id")
    pl_dangling = [(tag, ref) for tag, ref in pl_refs if ref not in protein_location_ids]
    check3_pass = not pl_dangling
    overall_pass &= check3_pass
    print(f"[CHECK 3] protein-location-id references valid: "
          f"{'PASS' if check3_pass else 'FAIL'}")
    print(f"          total references: {len(pl_refs)}, "
          f"known protein-locations: {len(protein_location_ids)}, "
          f"dangling: {len(pl_dangling)}")
    if pl_dangling:
        for tag, ref in pl_dangling[:20]:
            print(f"          dangling protein-location-id={ref} (in <{tag}>)")

    # --- Check 4: canvas dimensions -----------------------------------------
    width_el = pv.find("width")
    height_el = pv.find("height")
    width = int(width_el.text.strip()) if width_el is not None and width_el.text else None
    height = int(height_el.text.strip()) if height_el is not None and height_el.text else None

    check4_pass = (
        width is not None
        and width <= DEFAULT_REQUESTED_WIDTH + WIDTH_TOLERANCE
    )
    overall_pass &= check4_pass
    print(f"[CHECK 4] canvas width near requested width: "
          f"{'PASS' if check4_pass else 'FAIL'}")
    print(f"          canvas width={width}, height={height}")
    print(f"          requested width={DEFAULT_REQUESTED_WIDTH} "
          f"(+/- {WIDTH_TOLERANCE} tolerance)")
    if not check4_pass and width is not None:
        print(f"          FLAG: canvas width {width} exceeds "
              f"{DEFAULT_REQUESTED_WIDTH + WIDTH_TOLERANCE}")

    # --- Check 5: edge lengths -----------------------------------------------
    edges_container = pv.find("edges")
    transport_edge_ids: set[str] = set()

    # Collect edge ids referenced from transport-related visualizations.
    for container_tag in (
        "transport-visualizations",
        "reaction-coupled-transport_visualizations",
        "reaction-coupled-transport-visualizations",
    ):
        container = pv.find(container_tag)
        if container is None:
            continue
        for edge_id_el in container.iter("edge-id"):
            if edge_id_el.text:
                transport_edge_ids.add(edge_id_el.text.strip())

    edge_lengths: list[tuple[str, float, bool]] = []  # (edge_id, length, is_transport)
    if edges_container is not None:
        for edge_el in edges_container:
            id_el = edge_el.find("id")
            path_el = edge_el.find("path")
            if id_el is None or path_el is None:
                continue
            edge_id = id_el.text.strip() if id_el.text else "?"
            length = edge_length(path_el.text or "")
            if length is None:
                continue
            edge_lengths.append((edge_id, length, edge_id in transport_edge_ids))

    heads_up_edges = [e for e in edge_lengths if e[1] > EDGE_HEADS_UP_THRESHOLD]
    hard_issue_edges = [
        e for e in edge_lengths
        if e[1] > NON_TRANSPORT_EDGE_HARD_THRESHOLD and not e[2]
    ]

    check5_pass = not hard_issue_edges
    overall_pass &= check5_pass

    longest = max(edge_lengths, key=lambda e: e[1]) if edge_lengths else None

    print(f"[CHECK 5] edge lengths within thresholds: "
          f"{'PASS' if check5_pass else 'FAIL'}")
    print(f"          edges measured: {len(edge_lengths)}")
    if longest is not None:
        eid, elen, is_t = longest
        print(f"          longest edge: id={eid}, length={elen:.1f}px, "
              f"transport={'yes' if is_t else 'no'}")
    print(f"          edges > {EDGE_HEADS_UP_THRESHOLD}px (heads-up, any type): "
          f"{len(heads_up_edges)}")
    for eid, elen, is_t in sorted(heads_up_edges, key=lambda e: -e[1])[:10]:
        print(f"            edge id={eid}, length={elen:.1f}px, "
              f"transport={'yes' if is_t else 'no'}")
    print(f"          non-transport edges > {NON_TRANSPORT_EDGE_HARD_THRESHOLD}px "
          f"(hard issue): {len(hard_issue_edges)}")
    for eid, elen, is_t in sorted(hard_issue_edges, key=lambda e: -e[1])[:10]:
        print(f"            edge id={eid}, length={elen:.1f}px")

    # --- Summary --------------------------------------------------------------
    print("=" * 70)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
