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
post-rename name. Two independent pairings are computed and required to agree.

1. **Rename lineage.** The exporter preserves the pre-rename name in ``raw_name``.
   The lineage key of an IR row is ``raw_name`` when present, else ``name``; the
   canonical key is ``name``. ``raw_name`` is by construction the name the
   canonical payload used, i.e. a lineage established before canonicalization.
2. **Position.** Both lists are the same ``entities.compounds`` array; index i
   pairs with index i. This consults no exporter-written field at all.

Neither alone is sufficient, which is the point of requiring both. ``raw_name`` is
written by the very code under examination, so a rename that simply omitted
``raw_name`` would be invisible to pairing (1) -- it would surface as an unmatched
row. Position is not written by the exporter, but it is meaningful only when the
two lists have equal length and preserve order. Requiring agreement turns a
dishonestly-recorded rename into a loud ``PAIRING_DISAGREEMENT`` rather than a
spurious pile of "added" identifiers.

**Unmatched and ambiguous rows fail loudly and by name.** They are never folded
into "added" or "removed". When any of ``ROW_COUNT_MISMATCH``,
``AMBIGUOUS_CANONICAL_NAME``, ``AMBIGUOUS_LINEAGE_KEY``, ``UNMATCHED_IR_ROW``,
``UNMATCHED_CANONICAL_ROW`` or ``PAIRING_DISAGREEMENT`` fires, the probe prints
``PROBE INTEGRITY: FAILED``, suppresses the category counts (they would be
meaningless) and exits **2**. Exit 2 means "this probe could not measure", NOT
"the exporter is clean" and NOT "the exporter is dirty". ``--selfcheck`` runs each
of those six conditions against crafted rows and asserts it fires; run it under
the wrapper like any other test.

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
is frozen". Measurement A is the falsification of that today. It is the acceptance
target for C-050/C-051 and for milestone T-102: after those branches **all five
category counts must be 0**, which is the same requirement as "the diff is empty",
stated per category so no one category can be traded against another.

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


def _lineage_key(ir_row: Dict[str, Any]) -> str:
    raw = ir_row.get("raw_name")
    return str(raw) if isinstance(raw, str) and raw.strip() else str(ir_row.get("name"))


def _pair_rows(
    canonical: List[Dict[str, Any]], ir: List[Dict[str, Any]]
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[str]]:
    """Pair by rename lineage, corroborate by position, fail loudly on anything else."""

    problems: List[str] = []

    canonical_by_name: Dict[str, int] = {}
    for index, row in enumerate(canonical):
        name = str(row.get("name"))
        if name in canonical_by_name:
            problems.append(
                f"AMBIGUOUS_CANONICAL_NAME: canonical rows {canonical_by_name[name]} and "
                f"{index} both named {name!r}; no unambiguous partner exists"
            )
        canonical_by_name[name] = index

    lineage_seen: Dict[str, int] = {}
    lineage_pairing: Dict[int, int] = {}
    for index, row in enumerate(ir):
        key = _lineage_key(row)
        if key in lineage_seen:
            problems.append(
                f"AMBIGUOUS_LINEAGE_KEY: IR rows {lineage_seen[key]} and {index} both trace "
                f"to canonical name {key!r}"
            )
        lineage_seen[key] = index
        if key in canonical_by_name:
            lineage_pairing[index] = canonical_by_name[key]
        else:
            problems.append(
                f"UNMATCHED_IR_ROW: IR row {index} name={row.get('name')!r} "
                f"raw_name={row.get('raw_name')!r} traces to canonical name {key!r}, which no "
                f"canonical row carries. NOT counted as an addition."
            )

    claimed = set(lineage_pairing.values())
    for index, row in enumerate(canonical):
        if index not in claimed:
            problems.append(
                f"UNMATCHED_CANONICAL_ROW: canonical row {index} name={row.get('name')!r} is "
                f"claimed by no IR row. NOT counted as a removal."
            )

    if len(canonical) != len(ir):
        problems.append(
            f"ROW_COUNT_MISMATCH: {len(canonical)} canonical compound rows vs {len(ir)} IR "
            f"rows; positional corroboration is unavailable"
        )
    else:
        for index in range(len(ir)):
            if lineage_pairing.get(index, index) != index:
                problems.append(
                    f"PAIRING_DISAGREEMENT: IR row {index} pairs to canonical row "
                    f"{lineage_pairing[index]} by rename lineage but to canonical row {index} "
                    f"by position; the exporter's own lineage record is not trustworthy here"
                )

    pairs = [(canonical[c_index], ir[index]) for index, c_index in sorted(lineage_pairing.items())]
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
    canonical_rows = canonical["entities"]["compounds"]
    ir_rows = ir["entities"]["compounds"]

    pairs, problems = _pair_rows(canonical_rows, ir_rows)
    print(f"    pairing: rename lineage (raw_name -> name), corroborated by position")
    print(f"    {len(canonical_rows)} canonical rows, {len(ir_rows)} IR rows, "
          f"{len(pairs)} paired, {len(problems)} integrity problem(s)")
    if problems:
        print()
        for problem in problems:
            print(f"  !! {problem}")
        print()
        print("  PROBE INTEGRITY: FAILED -- category counts suppressed, they would be")
        print("  meaningless with rows that could not be paired. This says nothing about")
        print("  whether the exporter mutates biology; it says this probe cannot tell.")
        return 2

    totals = {key: 0 for key, _ in CATEGORIES}
    rows_hit = {key: 0 for key, _ in CATEGORIES}
    print()
    for canonical_row, ir_row in pairs:
        found = _classify(canonical_row, ir_row)
        print(f"  canonical {canonical_row.get('name')!r}  ->  IR {ir_row.get('name')!r}")
        for key, title in CATEGORIES:
            entries = found[key]
            totals[key] += len(entries)
            rows_hit[key] += 1 if entries else 0
            print(f"      {title:<32}: {len(entries)}")
            for entry in entries:
                print(f"          - {entry}")
        for entry in found["structural_observations"]:
            print(f"      (structural, non-biological, not counted): {entry}")
        for entry in found["fields_dropped"]:
            print(f"      (canonical fields absent from the IR row): {entry}")

    print()
    print("  ---- FIVE CATEGORIES, REPORTED SEPARATELY, NEVER SUMMED ----")
    for key, title in CATEGORIES:
        print(f"    {title:<32}: {totals[key]:>3} instance(s) across "
              f"{rows_hit[key]} of {len(pairs)} compound rows")
    print()
    print("  ACCEPTANCE after C-050/C-051 (milestone T-102): EVERY one of the five")
    print("  counts above must be 0, and no pairing integrity problem may be reported.")
    print("  A single total is deliberately not printed: the categories are different")
    print("  kinds of PRODUCT_CONTRACT section 5 violation and must not trade off.")
    print()
    return 0


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

    def row(name: str, **extra: Any) -> Dict[str, Any]:
        return {"name": name, **extra}

    cases: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]] = [
        ("AMBIGUOUS_CANONICAL_NAME", [row("a"), row("a")], [row("a"), row("a")]),
        ("AMBIGUOUS_LINEAGE_KEY", [row("a"), row("b")],
         [row("A", raw_name="a"), row("B", raw_name="a")]),
        # A rename the exporter did NOT record: no raw_name, so lineage finds nothing.
        ("UNMATCHED_IR_ROW", [row("a")], [row("Zed")]),
        ("UNMATCHED_CANONICAL_ROW", [row("a"), row("b")], [row("a")]),
        ("ROW_COUNT_MISMATCH", [row("a"), row("b")], [row("a")]),
        # Lineage says IR row 0 came from canonical 'b'; position says 'a'.
        ("PAIRING_DISAGREEMENT", [row("a"), row("b")],
         [row("B", raw_name="b"), row("A", raw_name="a")]),
    ]

    failures = 0
    for expected, canonical, ir in cases:
        _pairs, problems = _pair_rows(canonical, ir)
        fired = any(problem.startswith(expected) for problem in problems)
        print(f"  {'OK  ' if fired else 'FAIL'} {expected:<28} -> {problems}")
        failures += 0 if fired else 1

    clean = [row("a", mapped_ids={"chebi": "CHEBI:1"})]
    same = [row("a", mapped_ids={"chebi": "CHEBI:1"})]
    _pairs, problems = _pair_rows(clean, same)
    counts = _classify(clean[0], same[0])
    quiet = not problems and not any(counts[key] for key, _ in CATEGORIES)
    sizes = {key: len(counts[key]) for key, _ in CATEGORIES}
    print(f"  {'OK  ' if quiet else 'FAIL'} identical rows report nothing -> "
          f"problems={problems} category_sizes={sizes}")
    failures += 0 if quiet else 1

    print(f"  selfcheck: {len(cases) + 1 - failures}/{len(cases) + 1} passed")
    return 1 if failures else 0


def main(argv: List[str] | None = None) -> int:
    import sys

    args = sys.argv[1:] if argv is None else argv
    if "--selfcheck" in args:
        print("=== pairing selfcheck: every loud condition must fire ===")
        return _selfcheck()
    code = measurement_a()
    measurement_b()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
