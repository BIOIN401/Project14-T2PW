"""C-121 correction round -- REV-121 findings 1 and 2, applied as TEXT only.

Two edits, both inside C-121's declared boundary, neither touching a single line of
behaviour:

1. ``autostate_restoration_required``'s docstring restates the "exactly five" figure
   with the CONDITIONS it was measured under, and records the two measurement
   artifacts that return seven.
2. ``_VISIBLE_LOCATION_BUCKETS`` documents the scanned-vs-assigned asymmetry.

Scripted so a reviewer reads the exact substitutions rather than an in-place rewrite,
and refuses unless each target appears exactly once.
"""
import io

p = r"C:/t/c121/src/t2pw/pipeline/process_normalizer.py"
s = io.open(p, encoding="utf-8").read()

# ── FINDING 1 ───────────────────────────────────────────────────────────────
old1 = '''    **Why the guard is the whole fix and not a nicety.** Re-running
    :func:`ensure_autostates` unconditionally at the strict seam changes the
    payload on 40 archived production legs where this predicate is quiet, six of
    them legs that produce a PWML today: each would gain a spurious
    ``__auto_state__`` and a ``cell`` subcellular location and every graph hash
    would move. Measured against every archived leg's committed required-field
    gate report, this predicate fires on exactly the five legs that failed on an
    F-192 code and on no others -- no false positives, no false negatives.
'''
new1 = '''    **Why the guard is the whole fix and not a nicety.** Re-running
    :func:`ensure_autostates` unconditionally at the strict seam changes the
    payload on 40 archived production legs where this predicate is quiet, six of
    them legs that produce a PWML today: each would gain a spurious
    ``__auto_state__`` and a ``cell`` subcellular location and every graph hash
    would move.

    **FIVE archived legs fire, and the figure is only meaningful with its
    measurement conditions attached** (``ORCH-737``, which verified it at the call
    site after ``REV-121`` challenged it):

    * measured **at the production call site** -- driving :func:`quarantine_and_close`
      over all 154 archived legs and comparing the resulting payloads -- not by
      evaluating this predicate on a payload read off disk;
    * with **each leg in its own export mode**: ``research`` for a ``/research/``
      leg, ``pathwhiz`` for a ``/strict/`` one;
    * five payloads moved, **zero** legs changed their reaction set and **zero**
      changed their quarantine ``ok``.

    The **"no false positives, no false negatives"** qualifier is narrower than the
    firing count and must not be quoted without it: it is a cross-tabulation against
    each leg's **committed** ``pwml_required_field_gate_report.json``, and **99 of
    the 154 legs have no such report**. Over the 55 that do, the split is exact. On
    the other 99 the claim is that the predicate is quiet, which is measured -- not
    that a gate report agreed with it, which cannot be.

    **THE TRAP, recorded because two independent measurements have now fallen into
    it.** A naive probe returns **SEVEN**, and both extra legs are measurement
    artifacts rather than firings:

    * evaluating this predicate on the raw committed ``final_mapped.json`` instead of
      at the call site -- which is how this card's own census and the pre-dispatch
      census were taken. It happened to give the right answer for the five; it is
      not the same measurement;
    * driving a ``/research/`` leg through **strict** quarantine, which production
      never does. Research mode runs every decision and **applies none of them**
      (``export_mode.py``), so the same leg pushed down the strict path loses states
      production would have kept, arrives here with none, and fires clause 1. In the
      mode production actually runs them in, neither extra leg moves at all.

    So: seven is what you get by asking the wrong question two different ways, and
    the answer to the question production asks is five.
'''
assert s.count(old1) == 1, ("finding 1 anchor", s.count(old1))
s = s.replace(old1, new1, 1)

# ── FINDING 2 ───────────────────────────────────────────────────────────────
old2 = '''#: ``element_locations`` buckets whose rows the PWML required-field gate treats as
#: VISIBLE entities and therefore demands a ``biological_state`` on -- the gate's
#: own ``location_fields`` keys (``pwml/ir.py``, the
#: ``visible_entity_missing_location_state`` check). Duplicated here rather than
#: imported so this module keeps importing nothing from the exporter; the two are
#: locked together by a test that reads the gate's own table, so they cannot drift
#: apart silently.
'''
new2 = '''#: ``element_locations`` buckets whose rows the PWML required-field gate treats as
#: VISIBLE entities and therefore demands a ``biological_state`` on -- the gate's
#: own ``location_fields`` keys (``pwml/ir.py``, the
#: ``visible_entity_missing_location_state`` check). Duplicated here rather than
#: imported so this module keeps importing nothing from the exporter; the two are
#: locked together by a test that reads the gate's own table, so they cannot drift
#: apart silently.
#:
#: KNOWN ASYMMETRY -- the guard SCANS FOUR BUCKETS, ``ensure_autostates`` ASSIGNS TO
#: TWO (``REV-121`` finding 2, and it is a real limitation rather than a
#: misreading).
#:
#: :func:`ensure_autostates` iterates ``compound_locations`` and
#: ``protein_locations`` only. ``nucleic_acid_locations`` and
#: ``element_collection_locations`` are scanned by clause 2 of
#: :func:`autostate_restoration_required` and are never assigned to. So a payload
#: whose ONLY unassigned visible row sits in one of those two buckets would fire the
#: guard, be mutated -- a placeholder state and a ``cell`` compartment added -- and
#: **still fail the gate on the row that provoked it.**
#:
#: LATENT, NOT LIVE, and measured rather than assumed. Over the 154 archived
#: production legs (``evidence/c121_bucket_gap_census.py``,
#: ``evidence/g11/C-121/35`` and ``36``): ``nucleic_acid_locations`` rows appear on
#: **4** legs, ``element_collection_locations`` rows on **10**, and **not one leg in
#: either bucket carries a row missing a state.** Zero legs exercise the gap, so
#: C-121 changes no outcome anywhere in the corpus because of it.
#:
#: DELIBERATELY NOT FIXED HERE. Narrowing this tuple to the two assigned buckets
#: would make the guard blind to rows the gate still refuses; widening
#: ``ensure_autostates`` to assign the other two would change what that function
#: does to every payload in the pipeline, not only at C-121's seam. Both are
#: behaviour changes, both would need the whole five-leg / 40-leg verification run
#: again, and neither is authorized under ``D-099``.
#:
#: THE SPECIFIC HAZARD a future reader must not step on:
#: ``tests/test_c121_autostate_lifecycle.py::test_the_visible_location_buckets_are_the_gates_own_buckets``
#: locks this tuple to the gate's ``location_fields`` table. That lock is deliberate
#: -- it stops the guard going blind to a bucket -- but it is one-directional: if a
#: later card adds a FIFTH bucket to the gate, this tuple must widen to keep that
#: test green, the guard will then scan five buckets, and the assignment side will
#: still cover two. **The gap widens silently, and green tests will say nothing.**
#:
#: WHAT A FUTURE CARD SHOULD DO: make the assignment side and the scan side one
#: list rather than two, by teaching ``ensure_autostates`` to iterate
#: ``_VISIBLE_LOCATION_BUCKETS`` itself, and pin the equality with a test that fails
#: when the two disagree. That is a change to ``ensure_autostates``'s behaviour for
#: every payload, so it needs its own authorization, its own census over the quiet
#: set, and its own G9 proof -- which is exactly why C-121 records it instead of
#: taking it.
'''
assert s.count(old2) == 1, ("finding 2 anchor", s.count(old2))
s = s.replace(old2, new2, 1)

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("OK -- both corrections applied")
