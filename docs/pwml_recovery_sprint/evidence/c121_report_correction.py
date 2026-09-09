"""C-121 correction round -- section 3.2 of the implementation report.

REV-121's number (7) was adjudicated wrong by ORCH-737, but its method complaint was
right: section 3.2 stated a firing count without stating the measurement it came from.
This rewrites that section to name the conditions and record the trap. Text only.
"""
import io

p = r"C:/t/c121/docs/pwml_recovery_sprint/C-121-IMPLEMENTATION-REPORT.md"
s = io.open(p, encoding="utf-8").read()

old = """**The guard fires on exactly 5 production legs, and no leg outside those 5 changed by a single
byte.** 154 legs scanned (`.pytest_tmp_baseline` and worktrees excluded), cross-tabulated
against each leg's own committed `pwml_required_field_gate_report.json`:"""

new = """**The guard fires on exactly 5 production legs, and no leg outside those 5 changed** — and
that sentence is only worth anything with the measurement attached, which is what `REV-121`
was right to press on even though its own number was wrong.

**Under which measurement it is true.** `ORCH-737` verified it at the **production call site**:
`quarantine_and_close` driven over all 154 archived legs in both trees, **each leg in its own
export mode** — `research` for a `/research/` leg, `pathwhiz` for a `/strict/` one — comparing
the resulting payloads. **5 payloads moved, 0 legs changed their reaction set, 0 changed their
quarantine `ok`.** The five are the `D-099` § 5 population.

**Where my own census was weaker, and I am not defending it.** `c121_impl_census.py` evaluates
the predicate on the raw committed `final_mapped.json` rather than at the call site — as did the
orchestrator's pre-dispatch census. That method happened to return the right five here; it is
not the same measurement, and it should not be quoted as if it were.

**The trap, recorded so the next reader does not repeat it.** A naive probe returns **7**.
`REV-121` reached that number by driving two `/research/` legs through **strict** quarantine,
which production never does: research mode runs every decision and applies none of them, so such
a leg pushed down the strict path loses states production would have kept and then fires clause
1. In the mode production actually runs them in, `ORCH-737` measured both as **NOT MOVED**. The
same trap is reachable the other way, by evaluating the predicate on a payload read off disk.
Both extra legs are measurement artifacts, not firings. This is now stated in
`autostate_restoration_required`'s own docstring, not only here.

**The cross-tab qualifier is narrower than the firing count** and must not be quoted without it.
`no false positives, no false negatives` is a cross-tabulation against each leg's **committed**
`pwml_required_field_gate_report.json`, and **99 of the 154 legs carry no such report**. Over
the 55 that do, the split is exact:"""

assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)

old2 = """the `D-099` § 5 population, unchanged. **40** quiet legs would have been perturbed by an
unguarded re-run; the guarded entry point perturbed **0** of them."""
new2 = """the `D-099` § 5 population, unchanged. **40** quiet legs would have been perturbed by an
unguarded re-run; the guarded entry point perturbed **0** of them (payload-level census; the
call-site confirmation is `ORCH-737` above)."""
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2, 1)

# Record the correction round itself, and the one number I could not reproduce.
old3 = """## 6. Process lifecycle — 28 jobs, every one clean"""
new3 = """## 5.4 Correction round (`REV-121` / `ORCH-737`), applied 2026-09-09

Two text corrections, no behaviour change, no test-expectation change:

1. `autostate_restoration_required`'s docstring now states the **conditions** the "exactly
   five" figure was measured under — call site, per-leg export mode, and the 55-of-154 scope of
   the gate-report cross-tab — and records both ways a naive probe returns 7. §3.2 above was
   corrected the same way.
2. `_VISIBLE_LOCATION_BUCKETS` now documents the **scanned-vs-assigned asymmetry**: the guard
   scans four buckets, `ensure_autostates` assigns to two, so a payload whose only unassigned
   visible row is a nucleic-acid or element-collection row would fire the guard, be mutated and
   still fail the gate. Left unfixed on instruction — narrowing the guard or widening
   `ensure_autostates` are both behaviour changes and neither is authorized — with the hazard
   named: `test_the_visible_location_buckets_are_the_gates_own_buckets` locks the tuple to the
   gate's table, so a fifth gate bucket would widen the scan without widening the assignment and
   green tests would say nothing.

**One number I could not reproduce, reported rather than copied.** The correction brief gives
the archived population of those two buckets as 3 and 7 legs. Measured over all 154 production
legs (`evidence/c121_bucket_gap_census.py`, reports `35` and `36`) I get
**`nucleic_acid_locations` on 4 legs and `element_collection_locations` on 10**, and no scope I
sliced reproduces 3 and 7 — strict-only is 2 and 5, research-only is 2 and 5, and legs with a
committed gate report are 0 and 3. **The conclusion is unaffected and is what the docstring
rests on: 0 legs in either bucket carry a row missing a state, so the gap is latent.** The
docstring records my measured 4 and 10 with the scope named. Worth a glance in case the brief's
numbers came from a slice I have not thought of.

## 6. Process lifecycle — 31 jobs, every one clean"""
assert s.count(old3) == 1, s.count(old3)
s = s.replace(old3, new3, 1)

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("OK -- report section 3.2 corrected, 5.4 added")
