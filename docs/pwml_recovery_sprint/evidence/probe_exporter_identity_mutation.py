"""The exporter mutates biological identity after the canonical freeze.

WHAT IT SHOWS
-------------
Two independent measurements of the same violation.

**A. Committed-artifact diff.** For a leg that PASSED and shipped a PWML, compare
the compound rows in ``final_mapped.json`` (the canonical payload, hash-bound to
the final Stage-3 gate report) against the same compounds in ``pwml_ir.json``
(what the exporter actually built). On committed
``runs_verify/2026-08-04_1647/papers/PMC12856317/strict`` the exporter renames
``glycine -> Glycine`` with no provenance record, attaches a fabricated ``db_row``
to that same compound, materializes ``pathwhiz_id``/``db_id``/``db_status``/
``chosen_rule`` on all four compounds, strips the ``CHEBI:`` prefix on all four,
and adds one identifier that the canonical payload does not carry.

**B. Live re-derivation.** Build the IR from a quarantined payload and print the
``db_resolution`` block, showing resolution happening at export time and its
``ambiguous`` outcomes.

CORRECTION 2026-08-06 (task H-005) -- WHY THE OLD NUMBER WAS WRONG
------------------------------------------------------------------
Until this revision, measurement A keyed **both sides on the raw ``name``**
(``{row["name"]: row for row in ...["compounds"]}``, old ``:78-79``). The exporter
renames ``glycine`` to ``Glycine``, so ``canonical_rows.get("Glycine")`` returned
``{}`` and **every field on that compound was reported as added**. The probe
printed "TOTAL identifiers added post-freeze: 10"; nine of those ten are Glycine's
own canonical identifiers, all present in the canonical row (only the ``chebi``
VALUE differs, ``CHEBI:15428`` -> ``15428``). The true count of identifiers added
post-freeze on this leg is **1** -- heme's ``mapped_ids.pubchem = '3334'``, itself
a re-projection of that same row's top-level ``pubchem_cid: '3334'``.

That was not merely an off-by-nine. **A probe that pairs on raw name can never
reach its own T-102 acceptance target**, because the metric it computes is a
function of naming rather than of mutation: the more faithfully the exporter
records a rename, the larger the "added" count it produces. The old prose in this
docstring, and ``MASTER_PLAN.md`` section 1.2, both rested on that artefact.

**The PRODUCT_CONTRACT section 5 violation is unchanged and still blocking.** It
is violated by KIND, not by count: a post-freeze entity rename with no provenance
record, a fabricated ``db_row``, ``CHEBI:`` stripping on 4 of 4 compounds, and
identity materialization on 4 of 4. Only the magnitude was overstated.

HOW ROWS ARE PAIRED NOW
-----------------------
Renaming is the behaviour under examination, so the pairing must not depend on the
post-rename name -- and it must not depend on any field the exporter itself
authors.

**The anchor is ``mapping_meta.query.name``, and nothing else.** It is written
upstream of the freeze, it is the extraction-time name the mapper was queried
with, and the exporter carries it through a rename untouched. Measured on the
committed leg: the canonical row named ``glycine`` and the IR row named
``Glycine`` both carry ``mapping_meta.query.name == "glycine"``.

**``raw_name`` is deliberately NOT used -- not as the anchor and not as a
fallback.** It is created by the code under examination, so it cannot establish
independent pre-canonical lineage: an exporter that renamed a row and simply did
not write ``raw_name`` would defeat it. The earlier revision of this probe used
``raw_name`` with a positional cross-check; that is superseded. Position is not
used as a gate either, because ``PRODUCT_CONTRACT`` section 5 explicitly permits
ordering to differ, so a positional disagreement is not evidence of anything.

Matching is **strict, unique and one-to-one**. An anchor must be present exactly
once on the canonical side and exactly once on the IR side for its rows to pair.

**Every lineage defect is loud, named, and fails closed:**

* ``LINEAGE_ANCHOR_MISSING`` -- a row carries no usable
  ``mapping_meta.query.name``. Presence is **verified, never assumed**: it is
  absent in production today (``runs/2026-08-02_2130/papers/PMC12856317/strict``
  has a protein row with ``mapping_meta.query`` of ``null``), so a missing anchor
  is its own condition and never silently degrades to another key.
* ``AMBIGUOUS_LINEAGE_ANCHOR`` -- the same anchor on two rows of one side.
* ``UNMATCHED_IR_ROW`` / ``UNMATCHED_CANONICAL_ROW`` -- an anchor present on one
  side only. Never counted as an addition or a removal.

**A lineage failure can never present as a clean result.** This is the property
the whole probe exists to protect, so it is worth stating precisely. On failure
the probe still classifies the rows that *did* pair -- suppressing them outright
would be blind exactly when the exporter is dishonest -- but it labels the counts
a **LOWER BOUND**, prints ``RESULT: UNDETERMINED`` instead of ``RESULT:
MEASURED``, and exits **2**. **Zero in every category alongside
``UNDETERMINED`` is not a zero-mutation result and must never be read as one.**
Only ``RESULT: MEASURED`` with exit ``0`` asserts that the whole population was
measured.

``--selfcheck`` runs every one of those conditions against crafted rows, plus the
two cases that matter most: that a rename which creates **no** ``raw_name`` still
pairs through the anchor, and that a population of one perfectly clean paired row
plus one unmatched row **cannot** produce a passing conclusion. Run it under the
wrapper like any other test.

WHAT IT REPORTS -- FIVE CATEGORIES, NEVER CONFLATED
---------------------------------------------------
1. **name changes** -- IR ``name`` differs from the canonical ``name``.
2. **mapped-ID changes** -- identifier keys/values added, removed or changed,
   *excluding* pure prefix normalization (category 4). Each addition is further
   marked as a within-row re-projection or as having no canonical source.
3. **synthetic database rows** -- a ``db_row`` with no canonical counterpart.
4. **prefix normalization** -- a value differing only by a registry prefix.
5. **identity materialization** -- ``pathwhiz_id`` / ``db_id`` / ``db_status`` /
   ``chosen_rule`` / ``pathbank_compound_id`` given a value the canonical row
   did not carry.

No single total is printed. These are different kinds of contract violation and
must not be traded against one another.

WHY IT IS COMMITTED
-------------------
PRODUCT_CONTRACT section 5 requires that reloading ``final_mapped.json`` and
exporting again produce a biologically equivalent pathway, and that exporters not
"add, remove, resolve or reinterpret biological content after the canonical graph
is frozen". Measurement A is the falsification of that today, and it is an
acceptance target for C-050/C-051: after those branches **all five category counts
must be 0** with ``RESULT: MEASURED``, which is the same requirement as "the diff
is empty", stated per category so no one category can be traded against another.

**This probe is NECESSARY BUT NOT SUFFICIENT for milestone T-102.** ``DECISIONS.md``
**D-016 (LOCKED)** rules that *T-102 equivalence is NOT narrowed to compounds*: it
must verify **both** compound identity **and** organism/species equivalence, across
canonical JSON, PWML **and** SBML. This probe measures compound rows in the JSON
only. **Passing it does not satisfy T-102.** Compound identity equivalence remains
required; organism/species equivalence across JSON/PWML/SBML remains required and
is measured elsewhere.

INVOCATION
----------
    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/probe_exporter_identity_mutation.py

Measurement A is pure file comparison -- no network, no database, always
reproducible. Measurement B calls ``build_pwml_ir``, which may attempt a PathBank
connection; where the database is unreachable it falls back to the offline name
index. Either way the point stands, and the fact that the OUTCOME DEPENDS ON
DATABASE REACHABILITY is itself part of the finding.

ARTIFACT DEPENDENCY
-------------------
Both legs used here (1647 and 1207) are already committed. This script needs
nothing from INIT-001.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Tuple

from _repo_root import add_src_to_path, require

add_src_to_path()

import t2pw.pipeline.strict_quarantine as SQ  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir  # noqa: E402

PASSING_LEG = "runs_verify/2026-08-04_1647/papers/PMC12856317/strict"
REFUSED_LEG = "runs_verify/2026-08-04_1207/papers/PMC12452463/strict/final_mapped.json"

IDENTITY_FIELDS = ("pathwhiz_id", "db_id", "db_status", "chosen_rule", "pathbank_compound_id")
TOP_LEVEL_ID_FIELDS = (
    "chebi_id", "hmdb_id", "kegg_id", "biocyc_id", "chemspider_id", "pubchem_cid", "cas",
)
# Written by the IR builder as an internal handle. PRODUCT_CONTRACT section 5 lists
# "generated internal XML IDs" as acceptable to differ, so this is reported as an
# observation and deliberately excluded from the five biological counts.
STRUCTURAL_FIELDS = ("key",)

_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*:(?P<tail>.+)$")


def _strip_prefix(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = _PREFIX.match(value.strip())
    return match.group("tail") if match else value.strip()


def _is_prefix_normalization(before: Any, after: Any) -> bool:
    """True when `before` and `after` differ only by a registry prefix."""

    if before is None or after is None or before == after:
        return False
    if not isinstance(before, str) or not isinstance(after, str):
        return False
    return _strip_prefix(before) == _strip_prefix(after)


LINEAGE_POINTER = "mapping_meta.query.name"


def _lineage_anchor(row: Dict[str, Any]) -> str | None:
    """The pre-freeze extraction name this row was mapped from, or None.

    ``raw_name`` is NOT consulted here, and must not be added as a fallback: it is
    written by the exporter under examination and therefore cannot establish
    independent pre-canonical lineage.
    """

    meta = row.get("mapping_meta")
    if not isinstance(meta, dict):
        return None
    query = meta.get("query")
    if not isinstance(query, dict):
        return None
    name = query.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    return name


def _index_by_anchor(
    rows: List[Dict[str, Any]], side: str
) -> Tuple[Dict[str, int], List[str]]:
    """Anchor -> row index, unique only. Missing and duplicated anchors are problems."""

    problems: List[str] = []
    seen: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        anchor = _lineage_anchor(row)
        if anchor is None:
            problems.append(
                f"LINEAGE_ANCHOR_MISSING: {side} row {index} name={row.get('name')!r} carries "
                f"no usable {LINEAGE_POINTER}; it cannot be paired and is NOT assumed "
                f"unchanged, added or removed"
            )
            continue
        seen.setdefault(anchor, []).append(index)

    unique: Dict[str, int] = {}
    for anchor, indices in seen.items():
        if len(indices) > 1:
            problems.append(
                f"AMBIGUOUS_LINEAGE_ANCHOR: {side} rows {indices} all carry "
                f"{LINEAGE_POINTER}={anchor!r}; one-to-one matching is impossible"
            )
            continue
        unique[anchor] = indices[0]
    return unique, problems


def _pair_rows(
    canonical: List[Dict[str, Any]], ir: List[Dict[str, Any]]
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[str]]:
    """Strict one-to-one pairing on the pre-freeze lineage anchor. Fails closed."""

    canonical_by_anchor, problems = _index_by_anchor(canonical, "canonical")
    ir_by_anchor, ir_problems = _index_by_anchor(ir, "IR")
    problems = problems + ir_problems

    for anchor in sorted(set(ir_by_anchor) - set(canonical_by_anchor)):
        row = ir[ir_by_anchor[anchor]]
        problems.append(
            f"UNMATCHED_IR_ROW: IR row {ir_by_anchor[anchor]} name={row.get('name')!r} carries "
            f"{LINEAGE_POINTER}={anchor!r}, which no canonical row uniquely carries. "
            f"NOT counted as an addition."
        )
    for anchor in sorted(set(canonical_by_anchor) - set(ir_by_anchor)):
        row = canonical[canonical_by_anchor[anchor]]
        problems.append(
            f"UNMATCHED_CANONICAL_ROW: canonical row {canonical_by_anchor[anchor]} "
            f"name={row.get('name')!r} carries {LINEAGE_POINTER}={anchor!r}, which no IR row "
            f"uniquely carries. NOT counted as a removal."
        )

    pairs = [
        (canonical[canonical_by_anchor[anchor]], ir[ir_by_anchor[anchor]])
        for anchor in sorted(set(canonical_by_anchor) & set(ir_by_anchor))
    ]
    return pairs, problems


def _classify(canonical_row: Dict[str, Any], ir_row: Dict[str, Any]) -> Dict[str, List[str]]:
    """Split every difference between one paired row into the five categories."""

    found: Dict[str, List[str]] = {
        "name_changes": [], "mapped_id_changes": [], "synthetic_db_rows": [],
        "prefix_normalizations": [], "identity_materializations": [],
        "structural_observations": [], "fields_dropped": [],
    }

    if canonical_row.get("name") != ir_row.get("name"):
        found["name_changes"].append(
            f"{canonical_row.get('name')!r} -> {ir_row.get('name')!r} "
            f"(raw_name={ir_row.get('raw_name')!r}; no provenance record accompanies it)"
        )

    canonical_ids = canonical_row.get("mapped_ids") or {}
    ir_ids = ir_row.get("mapped_ids") or {}
    for key in sorted(set(canonical_ids) | set(ir_ids)):
        before, after = canonical_ids.get(key), ir_ids.get(key)
        if before == after:
            continue
        if _is_prefix_normalization(before, after):
            found["prefix_normalizations"].append(f"mapped_ids.{key}: {before!r} -> {after!r}")
        elif before is None:
            source = next(
                (f for f in TOP_LEVEL_ID_FIELDS if canonical_row.get(f) == after), None
            )
            origin = (
                f"within-row re-projection of canonical {source}"
                if source else "NO CANONICAL SOURCE -- a new external claim"
            )
            found["mapped_id_changes"].append(f"mapped_ids.{key} ADDED {after!r} [{origin}]")
        elif after is None:
            found["mapped_id_changes"].append(f"mapped_ids.{key} REMOVED (was {before!r})")
        else:
            found["mapped_id_changes"].append(
                f"mapped_ids.{key} CHANGED {before!r} -> {after!r}"
            )

    for field in TOP_LEVEL_ID_FIELDS:
        before, after = canonical_row.get(field), ir_row.get(field)
        if before == after:
            continue
        if _is_prefix_normalization(before, after):
            found["prefix_normalizations"].append(f"{field}: {before!r} -> {after!r}")
        else:
            found["mapped_id_changes"].append(f"{field} CHANGED {before!r} -> {after!r}")

    if "db_row" in ir_row and "db_row" not in canonical_row:
        found["synthetic_db_rows"].append(
            f"db_row {json.dumps(ir_row['db_row'], sort_keys=True)} fabricated at export time"
        )

    for field in IDENTITY_FIELDS:
        before, after = canonical_row.get(field), ir_row.get(field)
        if before == after:
            continue
        if before is None:
            found["identity_materializations"].append(f"{field}: None -> {after!r}")
        else:
            found["identity_materializations"].append(f"{field}: {before!r} -> {after!r}")

    for field in STRUCTURAL_FIELDS:
        if field in ir_row and field not in canonical_row:
            found["structural_observations"].append(f"{field} = {ir_row[field]!r}")

    dropped = sorted(set(canonical_row) - set(ir_row))
    if dropped:
        found["fields_dropped"].append(", ".join(dropped))
    return found


CATEGORIES = (
    ("name_changes", "1. name changes"),
    ("mapped_id_changes", "2. mapped-ID changes"),
    ("synthetic_db_rows", "3. synthetic database rows"),
    ("prefix_normalizations", "4. prefix normalization"),
    ("identity_materializations", "5. identity materialization"),
)


def _report(
    canonical_rows: List[Dict[str, Any]], ir_rows: List[Dict[str, Any]]
) -> Tuple[List[str], int]:
    """Build the whole measurement-A body. Returns (lines, exit_code).

    Factored out of ``measurement_a`` so ``--selfcheck`` can assert on the VERDICT
    -- in particular that an unmatched or ambiguous row can never yield
    ``RESULT: MEASURED`` or exit 0, no matter how clean the rows that did pair are.
    """

    lines: List[str] = []
    pairs, problems = _pair_rows(canonical_rows, ir_rows)
    lines.append(f"    pairing anchor: {LINEAGE_POINTER} (pre-freeze); strict one-to-one;")
    lines.append("    raw_name is NOT used, as anchor or fallback -- the exporter writes it")
    lines.append(f"    {len(canonical_rows)} canonical rows, {len(ir_rows)} IR rows, "
                 f"{len(pairs)} paired, {len(problems)} lineage problem(s)")

    if problems:
        lines.append("")
        for problem in problems:
            lines.append(f"  !! {problem}")

    totals = {key: 0 for key, _ in CATEGORIES}
    rows_hit = {key: 0 for key, _ in CATEGORIES}
    lines.append("")
    for canonical_row, ir_row in pairs:
        found = _classify(canonical_row, ir_row)
        lines.append(f"  canonical {canonical_row.get('name')!r}  ->  IR {ir_row.get('name')!r}")
        for key, title in CATEGORIES:
            entries = found[key]
            totals[key] += len(entries)
            rows_hit[key] += 1 if entries else 0
            lines.append(f"      {title:<32}: {len(entries)}")
            for entry in entries:
                lines.append(f"          - {entry}")
        for entry in found["structural_observations"]:
            lines.append(f"      (structural, non-biological, not counted): {entry}")
        for entry in found["fields_dropped"]:
            lines.append(f"      (canonical fields absent from the IR row): {entry}")

    lines.append("")
    lines.append("  ---- FIVE CATEGORIES, REPORTED SEPARATELY, NEVER SUMMED ----")
    for key, title in CATEGORIES:
        lines.append(f"    {title:<32}: {totals[key]:>3} instance(s) across "
                     f"{rows_hit[key]} of {len(pairs)} paired compound row(s)")
    lines.append("")

    unpaired = len(problems)
    if problems:
        lines.append(f"  RESULT: UNDETERMINED -- {unpaired} lineage problem(s) above.")
        lines.append(f"  The five counts are a LOWER BOUND over the {len(pairs)} row(s) that")
        lines.append("  paired. They say NOTHING about the rows that did not. ZERO IN EVERY")
        lines.append("  CATEGORY HERE IS NOT A ZERO-MUTATION RESULT and must never be read as")
        lines.append("  one; an exporter that breaks lineage cannot thereby appear clean.")
        code = 2
    else:
        lines.append(f"  RESULT: MEASURED -- all {len(pairs)} row(s) paired one-to-one on "
                     f"{LINEAGE_POINTER}.")
        code = 0

    lines.append("")
    lines.append("  ACCEPTANCE for C-050/C-051: EVERY one of the five counts must be 0 AND")
    lines.append("  the result must be MEASURED. A single total is deliberately not printed:")
    lines.append("  the categories are different kinds of PRODUCT_CONTRACT section 5")
    lines.append("  violation and must not trade off against one another.")
    lines.append("  NECESSARY BUT NOT SUFFICIENT FOR T-102. DECISIONS.md D-016 (LOCKED):")
    lines.append("  T-102 is NOT narrowed to compounds -- it must also verify organism/species")
    lines.append("  equivalence across canonical JSON, PWML and SBML. This probe covers")
    lines.append("  compound rows in the JSON only. Passing it does not satisfy T-102.")
    return lines, code


def measurement_a() -> int:
    print("=== A. canonical payload vs shipped IR (committed artifacts) ===")
    print(f"    leg: {PASSING_LEG}")
    leg = require(PASSING_LEG)
    if leg is None:
        return 0
    canonical_path = leg / "final_mapped.json"
    ir_path = leg / "pwml_ir.json"
    if not canonical_path.is_file() or not ir_path.is_file():
        print("  [skip] leg is missing final_mapped.json or pwml_ir.json")
        return 0

    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    lines, code = _report(canonical["entities"]["compounds"], ir["entities"]["compounds"])
    for line in lines:
        print(line)
    print()
    return code


def measurement_b() -> None:
    print("=== B. resolution observed live at export time ===")
    path = require(REFUSED_LEG)
    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = SQ.quarantine_and_close(copy.deepcopy(payload), strict_db=True)
    gate_payload = copy.deepcopy(result.payload)
    gate_payload.setdefault("metadata", {}).update({
        "pathway_name": "Enterobactin biosynthesis", "name": "Enterobactin biosynthesis",
        "pathway_subject": "Metabolic", "subject": "Metabolic",
    })
    try:
        _ir, report = build_pwml_ir(gate_payload, db_resolver=None)
    except Exception as exc:  # noqa: BLE001
        print(f"  build_pwml_ir raised {type(exc).__name__}: {str(exc)[:200]}")
        return
    print("  ir report ok :", report.get("ok"), "| errors:", len(report.get("errors") or []))
    resolution = report.get("db_resolution") or {}
    for row in (resolution.get("compounds") or [])[:8]:
        print("    %-42s status=%-24s rule=%s" % (
            str(row.get("raw_name"))[:42], row.get("status"), row.get("chosen_rule")))
    for err in (report.get("errors") or [])[:3]:
        code = err.get("code") if isinstance(err, dict) else ""
        print(f"    ERROR {code}: {str(err)[:140]}")
    print()
    print("  Resolution is happening HERE, after the payload was frozen and hashed.")


def _selfcheck() -> int:
    """Prove each named loud condition actually fires. Runtime, not assertion by prose."""

    def row(name: str, anchor: str | None = None, **extra: Any) -> Dict[str, Any]:
        built: Dict[str, Any] = {"name": name, **extra}
        if anchor is not None:
            built["mapping_meta"] = {"query": {"name": anchor}}
        return built

    checks: List[Tuple[str, bool, str]] = []

    def record(label: str, passed: bool, detail: str) -> None:
        checks.append((label, passed, detail))
        print(f"  {'OK  ' if passed else 'FAIL'} {label:<46} {detail}")

    # --- every lineage defect must fire, by name -----------------------------
    named: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]] = [
        ("LINEAGE_ANCHOR_MISSING (canonical side)", [row("a")], [row("a", "a")]),
        ("LINEAGE_ANCHOR_MISSING (IR side)", [row("a", "a")], [row("a")]),
        ("AMBIGUOUS_LINEAGE_ANCHOR (canonical side)",
         [row("a", "x"), row("b", "x")], [row("a", "x")]),
        ("AMBIGUOUS_LINEAGE_ANCHOR (IR side)",
         [row("a", "x")], [row("a", "x"), row("b", "x")]),
        ("UNMATCHED_IR_ROW", [row("a", "a")], [row("Zed", "zed")]),
        ("UNMATCHED_CANONICAL_ROW", [row("a", "a"), row("b", "b")], [row("a", "a")]),
    ]
    for label, canonical, ir in named:
        condition = label.split(" ")[0]
        _pairs, problems = _pair_rows(canonical, ir)
        record(label, any(p.startswith(condition) for p in problems), str(problems))

    # --- a rename that creates NO raw_name still pairs, via the anchor -------
    canonical = [row("glycine", "glycine", mapped_ids={"chebi": "CHEBI:1"})]
    renamed_no_raw_name = [row("Glycine", "glycine", mapped_ids={"chebi": "CHEBI:1"})]
    assert "raw_name" not in renamed_no_raw_name[0]
    lines, code = _report(canonical, renamed_no_raw_name)
    body = "\n".join(lines)
    record(
        "rename WITHOUT raw_name still pairs via the anchor",
        code == 0 and "RESULT: MEASURED" in body and "1. name changes                 : 1" in body,
        f"exit={code} name_changes_reported={'1. name changes                 : 1' in body}",
    )

    # --- raw_name is never consulted, even when it lies ---------------------
    lying = [row("Glycine", "glycine", raw_name="something-else",
                 mapped_ids={"chebi": "CHEBI:1"})]
    _pairs, problems = _pair_rows(canonical, lying)
    record("a misleading raw_name does not affect pairing", not problems, str(problems))

    # --- THE CRITICAL ONE ---------------------------------------------------
    # One perfectly clean paired row plus one unmatched row. Every category is 0.
    # This must NOT read as a pass.
    clean_plus_unmatched_canonical = [row("a", "a", mapped_ids={"chebi": "CHEBI:1"}),
                                      row("b", "b")]
    clean_plus_unmatched_ir = [row("a", "a", mapped_ids={"chebi": "CHEBI:1"})]
    lines, code = _report(clean_plus_unmatched_canonical, clean_plus_unmatched_ir)
    body = "\n".join(lines)
    all_zero = all(f"{title:<32}:   0 instance" in body for _key, title in CATEGORIES)
    record(
        "clean paired row + unmatched row cannot pass",
        all_zero and code == 2 and "RESULT: UNDETERMINED" in body
        and "RESULT: MEASURED" not in body and "LOWER BOUND" in body,
        f"all_five_counts_zero={all_zero} exit={code} "
        f"undetermined={'RESULT: UNDETERMINED' in body} "
        f"measured={'RESULT: MEASURED' in body}",
    )

    # --- and the negative: genuinely identical rows report nothing ----------
    identical = [row("a", "a", mapped_ids={"chebi": "CHEBI:1"})]
    lines, code = _report(identical, [row("a", "a", mapped_ids={"chebi": "CHEBI:1"})])
    body = "\n".join(lines)
    record(
        "identical rows report nothing and are MEASURED",
        code == 0 and "RESULT: MEASURED" in body
        and all(f"{title:<32}:   0 instance" in body for _key, title in CATEGORIES),
        f"exit={code}",
    )

    passed = sum(1 for _l, ok, _d in checks if ok)
    print(f"  selfcheck: {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


def _demo_lineage_failure() -> int:
    """Print a real failing report and return its code, so the PROCESS exit is observable.

    ``--selfcheck`` asserts on ``_report``'s return value in-process; this makes the
    same guarantee visible in a wrapper cleanup report as a nonzero ``exit_code``.
    The population is one clean paired row, one row with a missing anchor and one
    unmatched row -- every category count is 0 and it still must not pass.
    """

    canonical = [
        {"name": "a", "mapping_meta": {"query": {"name": "a"}}},
        {"name": "b", "mapping_meta": {"query": {"name": "b"}}},
        {"name": "c"},
    ]
    ir = [{"name": "a", "mapping_meta": {"query": {"name": "a"}}}]
    lines, code = _report(canonical, ir)
    for line in lines:
        print(line)
    print(f"\n  process exit code: {code}")
    return code


def main(argv: List[str] | None = None) -> int:
    import sys

    args = sys.argv[1:] if argv is None else argv
    if "--selfcheck" in args:
        print("=== pairing selfcheck: every loud condition must fire ===")
        return _selfcheck()
    if "--demo-lineage-failure" in args:
        print("=== demo: lineage failure must exit nonzero, not look clean ===")
        return _demo_lineage_failure()
    code = measurement_a()
    measurement_b()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
