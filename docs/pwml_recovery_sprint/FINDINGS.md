# Deferred findings

Defects, shortcuts and debt a card **found but did not own**. Recorded so they are neither
lost nor silently absorbed into an unrelated card. Nothing here authorizes a code change.

**Process.** Workers **report findings to the orchestrator**; workers do **not** edit this
shared file. Findings accumulate across a pack, and this file is updated **once at pack
closeout** — not after every card.

**Entry format.** ID · severity · discovering card · exact path/symbol · observed vs
expected · minimal reproducer or evidence · current-card impact · future owner or
`unowned` · shortcut deliberately accepted, where applicable.

---

## F-001 — evidence-haystack ordering is not mutation-pinned

- **Severity** low
- **Discovered by** H-008
- **Path/symbol** `src/t2pw/rag/graph_delta.py` — the sorted evidence-haystack ordering
- **Observed vs expected** The sorted ordering is implemented, but nothing pins it, so a
  mutation that reorders the haystack would not be caught.
- **Evidence** No test asserts the sort order; H-008's tests exercise retention, not order.
- **Current-card impact** none
- **Owner** `unowned`
- **Escalation** Blocking only if a later card (C-043) exposes an actual deterministic
  correctness failure.

## F-002 — comment overstates scalar writer coverage

- **Severity** low · documentation only
- **Discovered by** H-008
- **Path/symbol** `src/t2pw/rag/graph_delta.py:99-100`
- **Observed vs expected** The comment claims wider scalar writer coverage than the code
  implements.
- **Evidence** the two cited lines
- **Current-card impact** none
- **Owner** `unowned`
