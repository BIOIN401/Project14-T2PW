"""A0-C1: the graph hash covers the identity fallback the EXPORTER consumes.

``ir._first_nonempty`` settles an entity's exported identity over four tiers --
the record, ``mapping_meta``, ``mapped_ids``, then the FIRST
``mapping_meta.candidates`` entry. ``graph_projection`` reached tiers 1 and 3
only, so on committed artifacts an identifier that decides what gets exported
could be rewritten without moving ``canonical_graph_sha256`` -- the digest
exporters bind to. This file measures those rows and proves all three directions:
the consumed value moves the hash, ranking/transient noise does not, and a
reorder moves it exactly when it moves WHICH value is consumed.

``ir.py`` is used here strictly as a READ-ONLY ORACLE. Nothing imported from it
is modified, and the projection deliberately does not import it: exporters bind
to this hash, so what the hash covers may not be defined by the exporter it is
checked against.

Labels, each verified by running THIS file against base e4eeef4's sources
(24 failed, 6 passed there; 30 passed at the tip):
  G9 REGRESSION  -- red on base on a VALUE the pipeline computed, never on an
                    import: ``test_every_committed_gap_row_is_now_covered``
                    (19 ids, on the graph hash),
                    ``test_mutating_the_consumed_fallback_moves_the_graph_hash``
                    and ``test_a_reorder_that_changes_the_consumed_value_moves_the_hash``
                    (on the graph hash), and
                    ``test_the_published_graph_hash_is_what_the_verdict_checks``
                    (on the verdict REASON: base reports
                    ``canonical_payload_mismatch_graph_intact`` for an edit that
                    moved the exported identity).
  NEW ACCEPTANCE -- everything else. Two of them are also red on base --
                    ``test_the_allowlist_names_every_identity_key_ir_consumes``
                    on allowlist membership, and
                    ``test_the_seam_resolves_its_hash_schema_import_when_exec_d_alone``
                    on a missing statement. Neither is offered as a G9 proof:
                    a symbol that is absent is not behaviour.

C-093 re-based the census onto a frozen cohort plus an attributed admission
register and added four tests. **None of them is offered as a G9 regression
proof and none was written to fail on the base SHA**: production behaviour does
not move -- ``git diff <base> HEAD -- src/`` is empty -- and the census's five
values are byte-identical over C-068's own 35 legs. Three of the four are
explicitly labelled NON-VACUITY: they exist to show the re-based assertions can
still go red, because a pin that cannot go red is the defect reproduced.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in (ROOT / "src",) if str(p) not in sys.path]

from t2pw.pipeline import canonical_hash as ch  # noqa: E402
from t2pw.pipeline import gate_reports as gr  # noqa: E402
from t2pw.pwml import ir  # noqa: E402

APP_REL = "src/t2pw/app/streamlit_app.py"
#: The worked counterexample: ``isochorismate`` carries no identifier of its own
#: and no ``mapped_ids``, so the 40741 that is exported comes from candidate 0.
COUNTEREXAMPLE = ROOT / "runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json"
COMPOUND = "isochorismate"
COMPOUND_KEYS = ["pathbank_compound_id", "pw_compound_id", "pathwhiz_id"]

#: Every ordered identity key list ``ir.py`` resolves, by the bucket it is applied
#: to. ``components`` is the nested list at ``ir.py:1204`` / ``:1212``.
BUCKET_KEYS: dict[str, tuple[tuple[str, ...], ...]] = {
    "cell_types": (("pathbank_cell_type_id", "pw_cell_type_id", "pathwhiz_id"),),
    "compounds": (tuple(COMPOUND_KEYS), ("hmdb_id", "hmdb"), ("kegg_id", "kegg"),
                  ("pubchem_cid", "pubchem"), ("pwc_id",)),
    "element_collections": (("pathbank_element_collection_id",
                             "pw_element_collection_id", "pathwhiz_id"),),
    "nucleic_acids": (("pathbank_nucleic_acid_id", "pw_nucleic_acid_id", "pathwhiz_id"),),
    "protein_complexes": (("pathbank_complex_id", "pathbank_protein_complex_id",
                           "pw_complex_id", "pathwhiz_id"),),
    "proteins": (("pathbank_protein_id", "pw_protein_id", "pathwhiz_id"),
                 ("uniprot", "uniprot_id", "uniprot-id"),
                 ("drugbank", "drugbank_id", "drugbank-id")),
    "species": (("pathbank_species_id", "pw_species_id", "pathwhiz_id"),),
    "subcellular_locations": (("pathbank_subcellular_location_id",
                               "pw_subcellular_location_id", "pathwhiz_id"),),
    "tissues": (("pathbank_tissue_id", "pw_tissue_id", "pathwhiz_id"),),
    "components": (("pathbank_protein_id", "pw_protein_id", "pathwhiz_id", "protein_id"),
                   ("uniprot", "uniprot_id", "uniprot-id")),
}


def _corpus() -> list[str]:
    listed = subprocess.run(["git", "ls-files", "*final_mapped.json"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return sorted(listed.stdout.split())


def _rows(payload: Any) -> Any:
    """Every (bucket, keys, row) ``ir.py`` resolves an identity for."""
    entities = payload.get("entities") if isinstance(payload, dict) else None
    for bucket, key_lists in BUCKET_KEYS.items():
        if bucket == "components":
            continue
        for row in (entities or {}).get(bucket) or []:
            if isinstance(row, dict):
                yield bucket, key_lists, row
                for component in (row.get("components") or []) if bucket == \
                        "protein_complexes" else []:
                    if isinstance(component, dict):
                        yield "components", BUCKET_KEYS["components"], component


def _blind(row: dict) -> dict:
    """``row`` as the OLD projection saw it: tiers 1 and 3, never ``mapping_meta``.

    Dropping the one container is exact because every identity key is in
    :data:`GRAPH_FIELDS` -- asserted by
    :func:`test_the_allowlist_names_every_identity_key_ir_consumes`, without which
    a tier-1 hit on an unnamed key would be counted visible when it is not.
    """
    return {key: value for key, value in row.items() if key != "mapping_meta"}


def _slot(row: dict, keys: tuple[str, ...]) -> tuple[Any, str]:
    """The (container, key) the ladder consumes, for a row that lost tiers 1 and 3.
    ``mapping_meta`` before candidate 0, because a whole tier is scanned first."""
    meta = row.get("mapping_meta") or {}
    candidates = meta.get("candidates") or []
    first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    for container in (meta, first):
        for key in keys:
            if container.get(key) not in (None, ""):
                return container, key
    raise AssertionError("a gap row with no fallback slot")


@lru_cache(maxsize=None)
def _gap_rows() -> tuple[tuple[str, str, int, str], ...]:
    """THE CENSUS. One entry per committed row whose EXPORTED identity comes from
    a container the projection could not see: (file, bucket, row index, key)."""
    found = []
    for relative in _corpus():
        payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for index, (bucket, key_lists, row) in enumerate(_rows(payload)):
            for keys in key_lists:
                consumed = ir._first_nonempty(row, list(keys))
                if consumed in (None, "") or consumed == ir._first_nonempty(
                        _blind(row), list(keys)):
                    continue
                found.append((relative, bucket, index, _slot(row, keys)[1]))
    return tuple(found)


def _mutated(value: Any) -> Any:
    return value + 1 if isinstance(value, int) and not isinstance(value, bool) \
        else f"{value}-c030"


def _counterexample() -> tuple[dict, dict]:
    payload = json.loads(COUNTEREXAMPLE.read_text(encoding="utf-8"))
    row = next(c for c in payload["entities"]["compounds"] if c.get("name") == COMPOUND)
    return payload, row


def _edit(edit) -> tuple[dict, dict, dict]:
    """``(base payload, moved payload, moved row)`` -- ``edit`` gets the moved row."""
    base, _ = _counterexample()
    moved, row = deepcopy(base), None
    row = next(c for c in moved["entities"]["compounds"] if c.get("name") == COMPOUND)
    edit(row)
    return base, moved, row


# ── the census ──────────────────────────────────────────────────────────────

#: The commit whose corpus C-068's five census equalities were measured over.
#: ``git ls-tree`` at this SHA IS the frozen cohort -- see
#: :func:`_frozen_cohort`. Recorded as a SHA rather than as 35 pasted paths so
#: the manifest cannot drift from the history it claims to quote.
FROZEN_CENSUS_COHORT_SHA = "50fb4b6762d4993b4d17d63cd01e1292c61b2ca9"

#: ``sha256("\n".join(sorted(cohort)))``. The cohort is DERIVED from git, so this
#: pins the derivation: a rewritten history, a wrong SHA or a truncated
#: ``ls-tree`` changes it and the test says so instead of quietly measuring a
#: different cohort.
FROZEN_CENSUS_COHORT_DIGEST = \
    "3de665440f046f61fbe2be3ade3b6ecb6779974a2e3409209cb1f36b01359a7b"

#: C-068's five values, re-measured by C-093 at ``b36f3c5`` over the SAME 35 legs
#: and **byte-identical**. See :func:`test_the_census_reproduces_over_the_committed_corpus`.
FROZEN_CENSUS: dict[str, Any] = {
    "legs": 35,
    "gap_rows": 55,
    "files_carrying_a_gap": 21,
    "buckets": {"compounds": 43, "protein_complexes": 12},
    "keys": {"pathbank_compound_id": 43, "pathbank_complex_id": 12},
}


class CensusAdmissionUnattributed(ValueError):
    """A census bucket or identity key appeared with no card and no merge SHA."""


def _admitted(*entries: tuple[str, str, str, str]) -> dict[str, dict[str, str]]:
    """Build :data:`CENSUS_ADMISSIONS` so a bucket CANNOT enter it unattributed.

    **Structural, not conventional**, in the shape ``_excluded`` uses in
    ``tests/test_compound_resolution_extraction.py``. There is no dict literal
    below for a later editor to append a bare key to: the register is built from
    ``(bucket_or_key, merge_sha, witness_leg, attribution)`` quadruples, and a
    bare string, a wrong arity, a non-string field, an empty or whitespace
    attribution, an attribution shorter than :data:`MIN_ADMISSION_CHARS`, a merge
    SHA that is not a 7-to-40 character hex abbreviation, a witness leg that is
    not a ``final_mapped.json`` path, or a duplicated entry raises
    :exc:`CensusAdmissionUnattributed` **at import time** -- so the module fails
    to collect rather than one test failing somewhere downstream.

    **``witness_leg`` was added by C-093 correction round 1.** REV-093 was right
    that the first form was "a rubber stamp with a bouncer at the door": every
    check was on the SHAPE of the citation -- real-looking SHA, 120 characters, a
    ``PMC`` number somewhere -- and **nothing tied the citation to an artifact**. Any
    plausible prose plus any leg-committing SHA was accepted. The witness closes
    that: it names one committed leg that must still exist AND must still
    contribute at least one census row under this exact bucket-or-key, so an
    admission cannot be written for rows that do not exist, cannot survive the
    rows disappearing, and cannot be pre-registered before the commit that
    produces it. Checked by
    :func:`test_the_admission_register_cannot_absorb_a_bucket_silently`.

    Proved by :func:`test_the_admission_register_cannot_absorb_a_bucket_silently`.
    """
    register: dict[str, dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 4:
            raise CensusAdmissionUnattributed(
                f"an admission is a (name, merge_sha, witness_leg, attribution) "
                f"quadruple; got {entry!r}")
        name, sha, witness, attribution = entry
        if not isinstance(name, str) or not name.strip():
            raise CensusAdmissionUnattributed(f"unnamed admission: {entry!r}")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{7,40}", sha or ""):
            raise CensusAdmissionUnattributed(
                f"{name}: {sha!r} is not a merge SHA. An identity-fallback census "
                f"bucket that cannot be traced to a commit is a FINDING, not a "
                f"baseline move (C-093 section 2a).")
        if not isinstance(witness, str) or not witness.endswith("final_mapped.json"):
            raise CensusAdmissionUnattributed(
                f"{name}: {witness!r} is not a committed leg. Name ONE leg whose "
                f"rows this admission traces, so the citation can be checked "
                f"against the artifact instead of taken on trust.")
        if not isinstance(attribution, str) or \
                len(attribution.strip()) < MIN_ADMISSION_CHARS:
            raise CensusAdmissionUnattributed(
                f"{name}: admitted with no usable attribution ({attribution!r}). "
                f"State the leg it came off and what it means.")
        if name in register:
            raise CensusAdmissionUnattributed(f"{name}: admitted twice")
        register[name] = {"merge_sha": sha, "witness_leg": witness,
                          "attribution": " ".join(attribution.split())}
    return register


#: Minimum length, in stripped characters, of an admission's attribution. Not a
#: style rule: it exists to make ``""``, ``" "`` and ``"corpus growth"``
#: structurally impossible, which is the whole difference between a baseline move
#: and a silenced one.
MIN_ADMISSION_CHARS = 120

#: Census buckets and identity keys present over TODAY's corpus but absent from
#: :data:`FROZEN_CENSUS`, each traced to the commit that committed the leg it
#: comes off.
#:
#: **C-093, and the point of the whole card.** ``_corpus()`` is
#: ``git ls-files "*final_mapped.json"``, so this census grows every time a
#: milestone run commits its legs; between C-068's 35 legs and C-093's base the
#: corpus reached 92. C-068 moved five equalities by hand and they went stale
#: again within a fortnight, which is the decay :func:`_admitted` exists to stop.
#:
#: An entry here is **not** a claim the row is correct. It is a claim that the
#: row was TRACED. ``proteins`` in particular is admitted with a defect named.
CENSUS_ADMISSIONS: dict[str, dict[str, str]] = _admitted(
    ("protein_complexes", "266aba6",
     "runs_verify/2026-08-21_2057/papers/PMC12452463/research/final_mapped.json",
     "NOT a new bucket -- the frozen cohort already carries 12; the count rose to "
     "22 because ten further legs were committed carrying one gap row each. "
     "Measured, every one of the ten: 8 x 'enterobactin synthase complex' -> "
     "pathbank_complex_id 3623 on PMC12452463 legs and 2 x 'ferrochelatase "
     "complex' -> 912 on PMC12180156 legs, each read from mapping_meta."
     "candidates[0] with no record-level or mapped_ids value. First committed by "
     "266aba6 (T-103, 2026-08-21); the rest by 2673067 (T-104), bb125d0 "
     "(C-072/C-073 paper validation), 9cb491c (T-105) and efca465 (T-106). "
     "UNLIKE C-068's re-baseline, all ten legs have quarantine_report.json -> ok "
     "TRUE with refusal_reasons []: these are ACCEPTED legs, so 'the census grew "
     "because two REFUSED legs were committed' no longer explains it and is not "
     "restated here. Nine of the ten carry resolution.status 'ambiguous' "
     "(issue ambiguous_complex_name_species) and one 'fallback' "
     "(pathbank_unknown_sentinel_component, confidence 0.0) -- that a top-ranked "
     "AMBIGUOUS candidate supplies an exported identity is a standing property of "
     "this pipeline, not something these legs introduced: it is present in the "
     "frozen cohort too, on legs already pinned in GOLDEN."),
    ("pathbank_complex_id", "266aba6",
     "runs_verify/2026-08-21_2057/papers/PMC12452463/research/final_mapped.json",
     "The key behind the ten protein_complexes rows above -- eight on PMC12452463 "
     "legs and two on PMC12180156 legs; same legs, same commits, same "
     "measurement. It is already in FROZEN_CENSUS at 12 and is listed here only "
     "because the count moved to 22, so a reader of the whole-corpus census can "
     "find its attribution in one place rather than inferring it from the bucket "
     "entry."),
    ("proteins", "aee228c",
     "runs_verify/2026-08-24_1203/papers/PMC12856317/strict/final_mapped.json",
     "A GENUINELY NEW BUCKET, and it comes off exactly ONE leg: "
     "runs_verify/2026-08-24_1203/papers/PMC12856317/strict/final_mapped.json, "
     "committed by aee228c (2026-08-24, affected-paper cohort A). Two rows, CLPX "
     "and CLPP: 'ATP-dependent Clp protease ATP-binding subunit clpX-like, "
     "mitochondrial' (pathbank_protein_id 8580 from mapping_meta, uniprot O76031 "
     "from candidates[0]) and 'Putative ATP-dependent Clp protease proteolytic "
     "subunit, mitochondrial' (3923 from mapping_meta, Q16740 from "
     "candidates[0]). Both resolutions are status 'matched' at confidence 1.0 and "
     "both accessions are the correct human proteins, so the IDENTITIES are not "
     "the defect. THE DEFECT IS THAT THE GATE CANNOT SEE THEM: that leg's "
     "quarantine_report.json is ok FALSE with refusal_reasons "
     "['unexportable_entity:2'], and the two unexportable entities are these same "
     "two rows, each with reason 'protein_missing_external_identity'. "
     "entity_identity.protein_external_identity scans row, mapped_ids, ids and "
     "mapping_meta and STOPS THERE, while ir._first_nonempty also reaches "
     "mapping_meta.candidates[0] -- so the exporter would have exported an "
     "accession the quarantine gate reports as absent. Registered by C-093 as a "
     "finding and NOT fixed here (C-093 may not touch src/). THE ASYMMETRY RUNS "
     "BOTH WAYS AND ONLY ONE DIRECTION HAS FIRED. On the candidates[0] tier the "
     "gate is stricter than the exporter, which is the safe direction: it refuses "
     "where the exporter would have exported. But ir._first_nonempty never reads "
     "row['ids'] while protein_external_identity does, so a uniprot or drugbank "
     "value present ONLY under 'ids' passes the gate and leaves the exporter with "
     "no identity -- the UNSAFE direction, latent with zero exposure today. "
     "Measured by REV-093 over ALL 92 committed final_mapped.json -- the whole "
     "corpus, 14 under runs/ and 78 under runs_verify/ -- 1011 protein and "
     "protein_complex rows including components: unsafe 0, safe 2, exactly O76031 "
     "and Q16740. (An earlier draft of this entry said 78, which is the "
     "runs_verify/ subtree and not the corpus; REV-093 re-ran it corpus-wide "
     "rather than relabel it, and independently confirmed that git grep for "
     "'\"ids\"' across all 92 legs returns zero, so the unsafe tier has no "
     "exposure at all today.) The measurement is an INTEGRATION-BRANCH artifact, "
     "task ORCH-093 label ladder-all92, at e9aa5c8; it is not in this branch's "
     "tree, so look for it under g11/ORCH-093/ after merge. So 'no gate is "
     "weakened' is TRUE AS A MEASUREMENT OVER THIS CORPUS, NOT AS A PROPERTY OF "
     "THE CODE; the follow-up card must close both tiers, not only the one that "
     "happened to fire. The leg is EXCLUDED from the export golden on that record "
     "rather than pinned."),
    ("pathbank_protein_id", "aee228c",
     "runs_verify/2026-08-24_1203/papers/PMC12856317/strict/final_mapped.json",
     "The mapping_meta half of the two proteins rows above -- 8580 and 3923, on "
     "the one PMC12856317 strict leg of 2026-08-24_1203, same commit. Separately "
     "named because a key can appear in a bucket that is already admitted, and an "
     "unattributed KEY is as much a finding as an unattributed bucket."),
    ("uniprot", "aee228c",
     "runs_verify/2026-08-24_1203/papers/PMC12856317/strict/final_mapped.json",
     "The candidates[0] half of the two proteins rows above -- O76031 and Q16740, "
     "on the same one PMC12856317 strict leg, same commit. This is the exact tier "
     "entity_identity.protein_external_identity does not scan, which is why these "
     "two accessions are simultaneously exportable by ir.py and reported missing "
     "by the strict quarantine gate."),
)


@lru_cache(maxsize=None)
def _frozen_cohort() -> tuple[str, ...]:
    """The 35 legs committed at :data:`FROZEN_CENSUS_COHORT_SHA`, from git."""
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", FROZEN_CENSUS_COHORT_SHA],
        cwd=ROOT, capture_output=True, text=True)
    assert listed.returncode == 0, (
        f"cannot read the frozen census cohort at {FROZEN_CENSUS_COHORT_SHA}: "
        f"{listed.stderr.strip()}. The cohort is the evidence base of these "
        f"equalities; without it they cannot be checked at all.")
    return tuple(sorted(p for p in listed.stdout.split()
                        if p.endswith("final_mapped.json")))


def _census(rows: tuple[tuple[str, str, int, str], ...]) -> dict[str, Any]:
    """``rows`` summarised the way :data:`FROZEN_CENSUS` records it."""
    buckets: dict[str, int] = {}
    keys: dict[str, int] = {}
    for _relative, bucket, _index, key in rows:
        buckets[bucket] = buckets.get(bucket, 0) + 1
        keys[key] = keys.get(key, 0) + 1
    return {"gap_rows": len(rows),
            "files_carrying_a_gap": len({r[0] for r in rows}),
            "buckets": buckets, "keys": keys}


def test_the_census_reproduces_over_the_committed_corpus() -> None:
    """The measurement A0-C1's acceptance is scoped to, in TWO halves.

    **RE-BASED BY C-093 under permanent merge rule 4** -- the second deliberate
    move of this pin, and the last one that should ever be needed. C-068 moved
    five equalities by hand (32 -> 35 legs, 49 -> 55 rows, and so on) and
    recorded the reason honestly, but it left them keyed to
    ``git ls-files "*final_mapped.json"``. That set grows whenever a milestone
    run commits its legs, so the pin went stale again the moment T-103 landed and
    was still red at C-093's base with the corpus at **92**. A pin that a
    routine, correct, unrelated commit turns red is camouflage: it is exactly the
    neighbourhood D-065's regression walked through.

    So the equality is not re-typed with bigger numbers. It is **re-based onto a
    frozen cohort**, the way H-001 froze the strict-replay gate's cohort to an
    explicit manifest (TRAP-2):

    **Half one, the equality.** Over the 35 legs committed at
    :data:`FROZEN_CENSUS_COHORT_SHA` -- C-068's own corpus, derived from git and
    digest-pinned -- the census must still be C-068's five values EXACTLY. That
    set cannot grow, so this half can only move if identity RESOLUTION moves,
    which is the event the pin was written to hear about. Re-measured at C-093's
    base ``b36f3c5``: 55 rows, 21 files, ``{compounds: 43, protein_complexes: 12}``,
    ``{pathbank_compound_id: 43, pathbank_complex_id: 12}`` -- **byte-identical to
    C-068**. Nothing regressed; only the corpus grew.

    **Half two, the property.** Over today's whole corpus the census may hold only
    buckets and keys that are in :data:`FROZEN_CENSUS` or in
    :data:`CENSUS_ADMISSIONS`, where an entry cannot exist without a merge SHA and
    an attribution. A genuinely new bucket therefore still fails here -- which is
    the property that matters -- while another benchmark run committing more legs
    does not. The whole-corpus counts are deliberately NOT pinned: they are a
    function of how many times the sprint ran the pipeline, and pinning that
    measures nothing about identity resolution.

    Measured delta at ``b36f3c5``, whole corpus: 92 legs, 69 gap rows, 32 files,
    ``{compounds: 43, protein_complexes: 22, proteins: 4}``,
    ``{pathbank_compound_id: 43, pathbank_complex_id: 22, pathbank_protein_id: 2,
    uniprot: 2}``. Every one of the 14 rows the frozen cohort does not carry is
    attributed in :data:`CENSUS_ADMISSIONS`, and the ``proteins`` bucket carries a
    registered production finding rather than a reassurance.
    """
    cohort = _frozen_cohort()
    assert len(cohort) == FROZEN_CENSUS["legs"]
    assert hashlib.sha256("\n".join(cohort).encode()).hexdigest() == \
        FROZEN_CENSUS_COHORT_DIGEST

    committed = set(_corpus())
    missing = sorted(set(cohort) - committed)
    assert not missing, (
        f"the frozen census cohort is no longer committed: {missing}. "
        f"runs/ and runs_verify/ are read-only evidence (D-055, F-055...F-064); "
        f"un-committing a leg to hold a number is forbidden.")

    frozen = _census(tuple(g for g in _gap_rows() if g[0] in set(cohort)))
    assert frozen["gap_rows"] == FROZEN_CENSUS["gap_rows"]
    assert frozen["files_carrying_a_gap"] == FROZEN_CENSUS["files_carrying_a_gap"]
    assert frozen["buckets"] == FROZEN_CENSUS["buckets"]
    assert frozen["keys"] == FROZEN_CENSUS["keys"]

    whole = _census(_gap_rows())
    present = set(whole["buckets"]) | set(whole["keys"])
    accounted = set(FROZEN_CENSUS["buckets"]) | set(FROZEN_CENSUS["keys"]) | \
        set(CENSUS_ADMISSIONS)
    unattributed = sorted(present - accounted)
    assert not unattributed, (
        f"unattributed in the identity-fallback census: {unattributed}. Trace each "
        f"to the card and merge SHA that committed the leg it comes off and add it "
        f"to CENSUS_ADMISSIONS -- or, if it cannot be traced, STOP: an "
        f"unattributed bucket is a finding, not a baseline move.")

    # ...and the other direction, which C-093 first omitted and REV-093 caught.
    # Without this the complete DISAPPEARANCE of proteins / uniprot /
    # pathbank_protein_id passes silently -- and those rows are the only census
    # evidence outside the frozen cohort and the exact bucket carrying C-093's
    # registered finding. A one-sided check also lets a future author admit a name
    # BEFORE the commit that produces it, with nothing ever noticing it went live.
    stale = sorted(set(CENSUS_ADMISSIONS) - present)
    assert not stale, (
        f"admitted but no longer in the census: {stale}. Either a leg was "
        f"un-committed -- forbidden, runs/ and runs_verify/ are read-only evidence "
        f"(D-055) -- or the rows this admission traced have disappeared, which is a "
        f"resolution change and must be measured, not deleted from the register.")


def test_the_repaired_census_assertion_itself_goes_red(monkeypatch: Any) -> None:
    """NON-VACUITY (C-093), permanent, and the strongest form of it.

    The perturbation tests below exercise :func:`_census`, the helper. This one
    drives **the repaired test function itself** -- the thing a reviewer actually
    trusts -- and asserts it RAISES. Five perturbations, one per assertion the
    re-based pin makes:

    1. the frozen cohort's census SHRINKS (a gap row stops being a gap);
    2. the frozen cohort's census GROWS in a bucket it already has;
    3. a bucket appears over the whole corpus that nothing admits;
    4. an admitted bucket's attribution is withdrawn -- so the register cannot be
       emptied without the pin noticing, which is what stops a later card
       "simplifying" it into an unconditional pass;
    5. an admitted bucket VANISHES from the census -- the direction C-093's first
       form missed and REV-093 caught.
    """
    real = _gap_rows()
    cohort = set(_frozen_cohort())
    from_cohort = tuple(g for g in real if g[0] in cohort)
    assert from_cohort, "the frozen cohort carries no census rows to perturb"
    synthetic = ("runs_verify/9999-99-99_9999/papers/PMCNONVACUITY/strict/"
                 "final_mapped.json", "nucleic_acids", 0, "pw_nucleic_acid_id")

    def _pin(rows: tuple[tuple[str, str, int, str], ...]) -> None:
        """Replace the census the test under perturbation reads.

        The module global is what the test function resolves at call time, so
        rebinding it here is what the test sees; ``_gap_rows``'s own
        ``lru_cache`` is left untouched and pytest restores the binding.
        """
        monkeypatch.setitem(globals(), "_gap_rows", lambda: rows)

    for perturbed in (tuple(r for r in real if r != from_cohort[0]),
                      real + (from_cohort[0],),
                      real + (synthetic,)):
        _pin(perturbed)
        with pytest.raises(AssertionError):
            test_the_census_reproduces_over_the_committed_corpus()

    # 5. an ADMITTED bucket disappears from the census entirely. C-093's first
    #    form passed this silently -- REV-093's F2, and the reason the staleness
    #    half exists: a regression confined to ir.py's proteins ladder would have
    #    been invisible everywhere in this module.
    _pin(tuple(r for r in real if r[1] != "proteins"))
    with pytest.raises(AssertionError, match="no longer in the census"):
        test_the_census_reproduces_over_the_committed_corpus()

    _pin(real)
    test_the_census_reproduces_over_the_committed_corpus()  # control: still green
    monkeypatch.setitem(globals(), "CENSUS_ADMISSIONS", {})
    with pytest.raises(AssertionError):
        test_the_census_reproduces_over_the_committed_corpus()


def _ir_declared_ladders() -> dict[str, tuple[str, ...]]:
    """The PRIMARY identity ladder ``ir.py`` declares per bucket, from its SOURCE.

    ``build_pwml_ir`` drives every entity and component bucket from two literal
    tables, ``component_specs`` (``ir.py``, four component buckets) and
    ``entity_specs`` (five entity buckets). Each row is
    ``(source_key, ..., prefix, [db_keys...], ...)`` and that ``db_keys`` list is
    the ordered ladder handed to ``_db_id`` / ``_entity_record`` at ``:1324`` and
    ``:1451``. Parsing them is what makes
    :func:`test_bucket_keys_still_matches_the_ladders_ir_declares` a real gate
    rather than a restatement of :data:`BUCKET_KEYS`.

    Read by AST, not imported: the tables are local to ``build_pwml_ir`` and are
    not reachable as module attributes.
    """
    source = (ROOT / "src/t2pw/pwml/ir.py").read_text(encoding="utf-8")
    ladders: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if not names & {"entity_specs", "component_specs"}:
            continue
        for row in node.value.elts:
            if not isinstance(row, ast.Tuple) or len(row.elts) < 4:
                continue
            bucket, keys = row.elts[0], row.elts[3]
            if not isinstance(bucket, ast.Constant) or not isinstance(keys, ast.List):
                continue
            if not all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                       for k in keys.elts):
                continue
            ladders[str(bucket.value)] = tuple(k.value for k in keys.elts)
    return ladders


def test_bucket_keys_still_matches_the_ladders_ir_declares() -> None:
    """NEW ACCEPTANCE (C-093). :data:`BUCKET_KEYS` is checked against ir.py itself.

    **This test replaces one C-093 wrote and REV-093 correctly rejected as
    tautological.** That version asserted every ``(bucket, key)`` the census
    reports appears in ``BUCKET_KEYS`` -- true by construction for any data and
    any change to ``ir.py``, because :func:`_gap_rows` takes its bucket from
    ``BUCKET_KEYS.items()`` and its key from the very tuple :func:`_slot` was
    handed. It claimed to close a hole ``_slot`` structurally cannot open, which
    is worse than no test: a guard that does not detect the loss it advertises.

    The real risk it was reaching for is DRIFT: ``BUCKET_KEYS`` is a hand-copy of
    ladders that live in ``ir.py``, and the census, ``_blind`` and every
    ``CENSUS_ADMISSIONS`` attribution are only as true as that copy. So the copy
    is now compared against :func:`_ir_declared_ladders`, which reads ``ir.py``'s
    own ``entity_specs`` and ``component_specs`` tables. A bucket added, renamed,
    dropped or re-ordered in ``ir.py`` fails here.

    **Scope, stated so the next reader does not over-trust it.** This gates the
    PRIMARY ladder of the nine buckets ``ir.py`` declares in those two tables. It
    does NOT gate the secondary ladders in :data:`BUCKET_KEYS` -- the compound
    external ids, which are built in ``compound_resolution.py``, or the
    ``uniprot`` / ``drugbank`` lists spelled inline at ``ir.py:1506``, ``:1546``,
    ``:2464``. Those are covered only by
    :func:`test_the_allowlist_names_every_identity_key_ir_consumes`, which proves
    each is hashed, not that the list is complete. Closing that gap needs the
    inline lists lifted into a table and is a separate card.
    """
    declared = _ir_declared_ladders()
    assert len(declared) == 9, sorted(declared)
    unknown = sorted(set(declared) - set(BUCKET_KEYS))
    assert not unknown, (
        f"ir.py drives buckets BUCKET_KEYS does not name: {unknown}. The census, "
        f"_blind and every CENSUS_ADMISSIONS attribution are measured through "
        f"BUCKET_KEYS, so a bucket missing from it is invisible to all of them.")
    for bucket, ladder in sorted(declared.items()):
        assert BUCKET_KEYS[bucket][0] == ladder, (
            f"{bucket}: ir.py resolves {list(ladder)}, BUCKET_KEYS copies "
            f"{list(BUCKET_KEYS[bucket][0])}")


def test_the_ir_ladder_comparison_is_not_vacuous() -> None:
    """NON-VACUITY (C-093), permanent, for the test immediately above.

    An AST reader that silently found nothing would make that comparison pass
    over an empty map, which is the failure mode of every source-parsing gate. So:
    the reader must find all nine buckets and their real contents, and each of the
    FOUR drifts the real test catches -- a dropped bucket, a renamed key, a
    re-ordered ladder, an unknown bucket -- is shown here to break it.

    **The surrogate models all THREE assertions the real test makes**, including
    its ``len(declared) == 9`` cardinality check. REV-093 round 2 caught that the
    first version omitted that line, so a DROPPED bucket did not break
    ``_matches`` and the "dropped bucket" in this docstring was a demonstration
    that was not there -- F1's own defect, one level up. The real gate always
    caught a drop; the proof did not.
    """
    declared = _ir_declared_ladders()
    assert declared["compounds"] == ("pathbank_compound_id", "pw_compound_id",
                                     "pathwhiz_id")
    assert declared["proteins"] == ("pathbank_protein_id", "pw_protein_id",
                                    "pathwhiz_id")
    assert set(declared) == set(BUCKET_KEYS) - {"components"}

    def _matches(candidate: dict[str, tuple[str, ...]]) -> bool:
        """The real test's three assertions, in the order it makes them."""
        return (len(candidate) == 9
                and not (set(candidate) - set(BUCKET_KEYS))
                and all(BUCKET_KEYS[b][0] == ladder
                        for b, ladder in candidate.items()))

    assert _matches(declared)
    # a DROPPED bucket -- caught by cardinality, which is why cardinality is in
    # the surrogate. ir.py silently ceasing to drive a bucket is the drift this
    # whole gate exists for, and it is the one the first surrogate could not see.
    assert not _matches({k: v for k, v in declared.items() if k != "compounds"})
    assert not _matches({**declared, "proteins": ("uniprot", "pw_protein_id",
                                                  "pathwhiz_id")})
    assert not _matches({**declared, "compounds": tuple(
        reversed(declared["compounds"]))})
    assert not _matches({**declared, "a_bucket_bucket_keys_does_not_name": ("x",)})


def test_the_frozen_cohort_equality_still_bites() -> None:
    """NON-VACUITY (C-093), permanent. The re-based equality can go RED.

    A frozen cohort is the right fix only if freezing it did not also freeze the
    assertion into something that cannot fail. Three perturbations, each the
    shape of a real regression, each shown to move a value the test compares:

    1. a gap row that stops being a gap -- identity resolution starts publishing
       the value at the record level, so the census SHRINKS;
    2. a gap row that changes which key it resolves through;
    3. a leg dropped from the cohort -- the shape ``_corpus()``'s growth used to
       produce, now caught as a file-count and row-count move rather than
       silently absorbed.
    """
    cohort = set(_frozen_cohort())
    rows = tuple(g for g in _gap_rows() if g[0] in cohort)
    assert _census(rows)["buckets"] == FROZEN_CENSUS["buckets"]

    dropped = _census(rows[1:])
    assert dropped["gap_rows"] != FROZEN_CENSUS["gap_rows"]
    assert dropped["buckets"] != FROZEN_CENSUS["buckets"]

    relative, bucket, index, key = rows[0]
    rekeyed = ((relative, bucket, index, "pathwhiz_id"),) + rows[1:]
    assert _census(rekeyed)["keys"] != FROZEN_CENSUS["keys"]

    one_leg = {r[0] for r in rows[:1]}
    without = _census(tuple(r for r in rows if r[0] not in one_leg))
    assert without["files_carrying_a_gap"] != FROZEN_CENSUS["files_carrying_a_gap"]


def test_an_unattributed_bucket_or_key_is_reported_not_absorbed() -> None:
    """NON-VACUITY (C-093), permanent. Half two of the census test is not vacuous.

    ``set(...) - accounted == []`` would look green forever if the accounting set
    absorbed whatever it was given. It does not: a bucket or a key in neither
    :data:`FROZEN_CENSUS` nor :data:`CENSUS_ADMISSIONS` is reported. Exercised on
    a SYNTHESIZED census row -- nothing is written under ``runs/`` or
    ``runs_verify/``, which are read-only evidence.
    """
    accounted = set(FROZEN_CENSUS["buckets"]) | set(FROZEN_CENSUS["keys"]) | \
        set(CENSUS_ADMISSIONS)
    whole = _census(_gap_rows())
    assert not (set(whole["buckets"]) | set(whole["keys"])) - accounted

    # SOFT COUPLING, recorded rather than engineered away (REV-093). These three
    # names are chosen because nothing admits them TODAY. If a later card
    # legitimately admits ``nucleic_acids`` or ``drugbank``, the ``assert
    # synthetic not in accounted`` below fails with that name in the message --
    # loudly, and pointing straight at the fix, which is to pick a name that is
    # still unadmitted. It is a maintenance cost, not a false alarm, and it is
    # cheaper than a synthetic name so exotic that it stops resembling the thing
    # the test is about.
    for synthetic in ("nucleic_acids", "pw_nucleic_acid_id", "drugbank"):
        assert synthetic not in accounted, synthetic
        invented = _gap_rows() + (
            ("runs_verify/9999-99-99_9999/papers/PMCNONVACUITY/strict/"
             "final_mapped.json", synthetic, 0, synthetic),)
        summary = _census(invented)
        assert (set(summary["buckets"]) | set(summary["keys"])) - accounted == \
            {synthetic}


def test_the_admission_register_cannot_absorb_a_bucket_silently() -> None:
    """NON-VACUITY (C-093), permanent. A bucket cannot enter the register bare.

    :data:`CENSUS_ADMISSIONS` is a constructor, not a literal, so there is no bare
    key to append: every unattributed shape raises, and it raises at IMPORT time,
    so the module fails to collect rather than one assertion failing downstream.

    **REV-093 round 1: the shape checks alone were theatre.** A real-looking SHA,
    120 characters and a ``PMC`` number anywhere in the prose were the whole bar,
    and nothing tied any of it to an artifact. The witness leg is what makes this
    a gate, and it is checked three ways -- each closing a different way a
    citation can be false:

    a. the witness is a **committed** leg (``git ls-files``), so an admission
       cannot cite a file that does not exist;
    b. the witness **actually contributes** at least one census row under this
       exact bucket-or-key, so an admission cannot be written for rows that are
       not there, and cannot be pre-registered ahead of the commit that produces
       them;
    c. the cited SHA is the commit that **ADDED that witness** -- REV-093's
       optional (c), taken because it is one ``git log`` per admission and it is
       the last gap between the citation and the artifact. Without it the SHA had
       only to be *some* leg-adding commit, which any of the ten would satisfy.
    """
    leg = "runs_verify/2026-08-24_1203/papers/PMC12856317/strict/final_mapped.json"
    for entry in ("proteins", ("proteins",), ("proteins", "aee228c"),
                  ("proteins", "aee228c", leg),
                  ("proteins", "aee228c", leg, ""),
                  ("proteins", "aee228c", leg, "   "),
                  ("proteins", "aee228c", leg, "corpus growth"),
                  ("proteins", "aee228c", leg, None),
                  ("proteins", "aee228c", leg, 0),
                  ("", "aee228c", leg, "z" * MIN_ADMISSION_CHARS),
                  ("proteins", "", leg, "z" * MIN_ADMISSION_CHARS),
                  ("proteins", "not-hex", leg, "z" * MIN_ADMISSION_CHARS),
                  ("proteins", "aee228c", "", "z" * MIN_ADMISSION_CHARS),
                  ("proteins", "aee228c", "notaleg.json", "z" * MIN_ADMISSION_CHARS),
                  ("proteins", "aee228c", None, "z" * MIN_ADMISSION_CHARS),
                  ("proteins", "aee228c", leg,
                   "z" * (MIN_ADMISSION_CHARS - 1))):
        with pytest.raises(CensusAdmissionUnattributed):
            _admitted(entry)  # type: ignore[arg-type]

    good = ("proteins", "aee228c", leg, "z" * MIN_ADMISSION_CHARS)
    assert _admitted(good) == {"proteins": {"merge_sha": "aee228c",
                                            "witness_leg": leg,
                                            "attribution": good[3]}}
    with pytest.raises(CensusAdmissionUnattributed):
        _admitted(good, good)

    committed = set(_corpus())
    rows = _gap_rows()
    for name, record in CENSUS_ADMISSIONS.items():
        witness = record["witness_leg"]
        assert witness in committed, (
            f"{name}: witness leg {witness} is not committed, so the admission "
            f"cites an artifact that cannot be read")

        # (b) the witness really carries rows under THIS bucket or key
        contributed = [g for g in rows
                       if g[0] == witness and name in (g[1], g[3])]
        assert contributed, (
            f"{name}: {witness} contributes no census row under this name. An "
            f"admission for rows that do not exist is exactly the pre-registration "
            f"REV-093 flagged.")

        # (c) the cited SHA is the commit that ADDED that witness
        added = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H", "--", witness],
            cwd=ROOT, capture_output=True, text=True)
        assert added.returncode == 0, f"{name}: cannot read the history of {witness}"
        shas = added.stdout.split()
        assert any(s.startswith(record["merge_sha"]) for s in shas), (
            f"{name}: {record['merge_sha']} did not add {witness} (added by "
            f"{[s[:7] for s in shas]}), so the citation names the wrong commit")

        assert re.search(r"PMC\d+", record["attribution"]), (
            f"{name}: the attribution names no paper, so the rows it admits "
            f"cannot be found again")


def test_the_allowlist_names_every_identity_key_ir_consumes() -> None:
    """NEW ACCEPTANCE (also red on base, on allowlist membership -- not a G9
    proof). GRAPH_FIELDS is the ONE gate on what reaches the graph hash, so an
    identity key it does not name is an identifier the exporter can change
    silently -- and it also makes :func:`_blind` an exact oracle."""
    every = {key for lists in BUCKET_KEYS.values() for keys in lists for key in keys}
    assert every <= ch.GRAPH_FIELDS, sorted(every - ch.GRAPH_FIELDS)


@pytest.mark.parametrize("relative", sorted({g[0] for g in _gap_rows()}))
def test_every_committed_gap_row_is_now_covered(relative: str) -> None:
    """G9 REGRESSION. For EVERY census row, rewriting the identifier the exporter
    consumes moves the graph hash. On base e4eeef4 not one of the 49 moves it."""
    payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    before = ch.canonical_graph_sha256(payload)
    covered = 0
    for _file, bucket, index, key in [g for g in _gap_rows() if g[0] == relative]:
        moved = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        _bucket, key_lists, row = list(_rows(moved))[index]
        assert _bucket == bucket
        keys = next(k for k in key_lists if key in k)
        container, slot = _slot(row, keys)
        container[slot] = _mutated(container[slot])
        # the slot really is the consumed one, and the export really moved
        assert ir._first_nonempty(row, list(keys)) == container[slot]
        assert ch.canonical_graph_sha256(moved) != before
        covered += 1
    assert covered


# ── the three directions, on the committed counterexample ───────────────────


def test_mutating_the_consumed_fallback_moves_the_graph_hash() -> None:
    """G9 REGRESSION. 40741 -> 40742 at candidate 0 changes what ``ir.py`` exports
    for ``isochorismate``. On base the graph hash does not move; here it must."""
    base, moved, row = _edit(
        lambda r: r["mapping_meta"]["candidates"][0].update(pathbank_compound_id=40742))
    original = next(c for c in base["entities"]["compounds"] if c.get("name") == COMPOUND)
    assert all(original.get(k) is None for k in COMPOUND_KEYS)
    assert "mapped_ids" not in original
    # the EXPORTED identity moved with it, at ir.py's own record builder
    assert ir._db_id(original, COMPOUND_KEYS) == 40741
    assert ir._db_id(row, COMPOUND_KEYS) == 40742
    exported = ir._entity_record(row, "cmp1", COMPOUND_KEYS, "pathbank_compound_id")
    assert exported["pathwhiz_id"] == exported["pathbank_compound_id"] == 40742
    assert ch.canonical_graph_sha256(moved) != ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) != ch.canonical_payload_sha256(base)


def test_a_reorder_that_changes_the_consumed_value_moves_the_hash() -> None:
    """G9 REGRESSION. A0-C1's second half. Promoting candidate 1 changes which
    identifier is exported, so the hash MUST follow it."""
    base, moved, row = _edit(
        lambda r: r["mapping_meta"]["candidates"].insert(
            0, r["mapping_meta"]["candidates"].pop(1)))
    assert ir._db_id(row, COMPOUND_KEYS) != 40741
    assert ch.canonical_graph_sha256(moved) != ch.canonical_graph_sha256(base)


def test_a_score_edit_is_ranking_noise_and_never_moves_the_graph_hash() -> None:
    """NEW ACCEPTANCE. The other half of A0-C1: ranking is not biology."""
    base, moved, row = _edit(
        lambda r: r["mapping_meta"]["candidates"][0].update(score=0.123456))
    assert ir._db_id(row, COMPOUND_KEYS) == 40741
    assert ch.canonical_graph_sha256(moved) == ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) != ch.canonical_payload_sha256(base)


@pytest.mark.parametrize("case", ["reorder_the_tail", "drop_the_tail", "append_a_candidate"])
def test_ranking_that_leaves_the_consumed_value_in_front_never_moves_the_hash(
        case: str) -> None:
    """NEW ACCEPTANCE. The candidate LIST, its ORDER, its LENGTH and its other
    entries stay out: only the one value ``ir.py`` consumes is hashed."""
    edits = {
        "reorder_the_tail": lambda c: c.__setitem__(slice(1, None), list(reversed(c[1:]))),
        "drop_the_tail": lambda c: c.__setitem__(slice(1, None), []),
        "append_a_candidate": lambda c: c.append({"pathbank_compound_id": 999999,
                                                  "score": 0.01}),
    }
    base, moved, row = _edit(lambda r: edits[case](r["mapping_meta"]["candidates"]))
    assert ir._db_id(row, COMPOUND_KEYS) == 40741
    assert ch.canonical_graph_sha256(moved) == ch.canonical_graph_sha256(base)


def test_provenance_beside_the_consumed_value_stays_out_of_the_graph_hash() -> None:
    """NEW ACCEPTANCE. Transient metadata sitting in the same container as a value
    that IS hashed must not be dragged in with it."""
    base, moved, _row = _edit(lambda r: r["mapping_meta"].update(
        resolution={"status": "novel"}, chosen_rule="rewritten", confidence=0.1))
    assert ch.canonical_graph_sha256(moved) == ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) != ch.canonical_payload_sha256(base)


# ── the seam publishes the hashes, and they are load-bearing ────────────────


def _seam() -> Any:
    """``freeze_canonical_payload`` alone, the way the AST harnesses load it."""
    source = (ROOT / APP_REL).read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "freeze_canonical_payload")
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, APP_REL, "exec"), namespace)  # noqa: S102
    return namespace["freeze_canonical_payload"]


def test_the_seam_resolves_its_hash_schema_import_when_exec_d_alone() -> None:
    """NEW ACCEPTANCE (also red on base, on a missing statement -- not a G9
    proof). The import is function-local precisely so the harnesses, which exec
    this FunctionDef with a hand-built globals dict, can still run it."""
    assert _seam() is not None
    body = ast.parse((ROOT / APP_REL).read_text(encoding="utf-8"))
    node = next(n for n in body.body if isinstance(n, ast.FunctionDef)
                and n.name == "freeze_canonical_payload")
    imports = [n for n in ast.walk(node) if isinstance(n, ast.ImportFrom)]
    assert [(n.module, [a.name for a in n.names]) for n in imports] == [
        ("t2pw.pipeline.canonical_hash", ["HASH_SCHEMA_VERSION"])]


def test_the_published_graph_hash_is_what_the_verdict_checks() -> None:
    """G9 REGRESSION, on the verdict REASON. The wiring is only worth anything if
    the consumer reads it: an edit that moved the exported identity must be caught
    as a GRAPH mismatch, and an evidence-only edit as the payload mismatch that
    leaves the graph intact. On base the first is misreported as the SECOND --
    "the biology is intact, only the evidence moved" -- about a payload whose
    exported compound identity had changed."""
    payload, _row = _counterexample()
    report = gr.stamp_report({"stage": "final", "ok": True, "errors": []},
                             phase=gr.PHASE_FINAL_PRE_EXPORT,
                             payload=payload, payload_hash=gr.payload_sha256(payload),
                             hash_schema=ch.HASH_SCHEMA_VERSION)

    def verdict(candidate: dict) -> gr.GateVerdict:
        return gr.gate_verdict(gr.stamp_artifact_set({
            gr.FINAL_GATE_REPORT_KEY: report, gr.CANONICAL_PAYLOAD_KEY: candidate}))

    assert verdict(payload).failed is False
    _base, biology, _r = _edit(
        lambda r: r["mapping_meta"]["candidates"][0].update(pathbank_compound_id=40742))
    _base, evidence, _r = _edit(lambda r: r.update(lineage=[{"stage": "rag"}]))
    assert verdict(biology).reason == gr.REASON_CANONICAL_GRAPH_MISMATCH
    assert verdict(evidence).reason == gr.REASON_CANONICAL_PAYLOAD_MISMATCH_GRAPH_INTACT
