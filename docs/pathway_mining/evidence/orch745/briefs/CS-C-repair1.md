# ORCH-745 CS-C — repair round 1 (of at most 2)

Reviews of your tip `d01e0825`: code review **REQUEST_CHANGES** (1 blocking: F3-ORDER); bio-audit
**SOUND_WITH_FINDINGS** (no blocking). Orchestrator decision on ORCH-1: **keep the structural
thin-answer predicate; do not add a token gate** (both reviewers agree: the thin shape follows the
provider route — all 7 flagged pilot-003 answers share one alphabetically sorted key set — not
reasoning; and estimated usage equals visible size by construction). Same rules as before
(`C:/t/bt/orch745/COMMON_RULES.md`), same worktree `C:/t/m745-c`, same branch; add commits on top of
`d01e0825`, never rewrite history. Append new test nodes to `C:/t/bt/m745-c/g9_manifest.json` under a
new key `repair1_nodes` (or a sibling file `g9_manifest_repair1.json` if the manifest is a list),
saying for each whether it fails at `d01e0825` and/or `f9556288`.

## Blocking

**F3-ORDER** (`frontier.cap_frontier`, the interleave loop ~:398-404, and docs `architecture.md:131`,
`limits_and_stopping.md:119`, `known_limitations.md:121`). Every reserved species item is forced into
the 4th, 8th, ... place and any left over are appended at the end, even when its own priority (6.0)
is higher than the items placed ahead of it — also when the cap does not bind at all. In small or
mixed frontiers species searches end up behind enzyme (4.5) and cofactor (1.0) items: the reverse of
the PRIORITY table and of what F-3 protects; rounds ask in frontier order and stop when their budget
runs out, so this makes target-species searches LESS likely than at base. Required: a reserved item
takes the EARLIER of its priority-order position and its every-fourth slot (reviewer sketch: `for item
in rest: while reserved and (len(kept) % stride == stride - 1 or _rank(reserved[0]) <= _rank(item)):
kept.append(reserved.pop(0)); kept.append(item)`, then any remaining reserved items). Add a test: an
under-cap frontier containing species items plus lower-priority kinds keeps the base priority order
(passes at f9556288, fails at d01e0825, passes after the fix). The existing G9 and acceptance tests
must still pass. Fix the three doc locations to describe "the earlier of its priority position or every
fourth place".

## Non-blocking (fix in this round)

- **B2-RECORD** (`call_json` ~:423-438 with `_attempt` ~:527-530, :587-590): after a failed re-ask the
  record says status `ok` + `thin_after_retry` but `parse_error`/`schema_error` still hold the re-ask's
  error, and `record.provider` is the last provider reported, not necessarily the kept attempt's.
  Clear parse_error/schema_error when the kept answer is the earlier thin one (or document precisely in
  `artifacts.md`), and take `record.provider` from the kept attempt's log.
- **ORCH-1 docs**: update `evidence_policy.md` (~:275), `known_limitations.md` (~:100), `testing.md`
  (~:49) and the fixture note: the predicate is structural; pilot-003 had 7 such answers (not 8), 4 of
  them with 5k–24k output tokens, sharing one sorted-key shape that points to the provider route; a
  re-ask costs up to one extra call per thin answer. Say that when usage is not reported the structural
  rule decides alone, and that any future token rule needs an explicit per-attempt `usage_reported` flag.
- **BIO-1 docs**: replace "which is what keeps an uncited participant out of the graph" with wording
  like "which keeps it out of the target graph: with no species citation it can never be
  target_species_supported, so at most it is held as a quarantined lead".
- **B2-SCOPE**: one docstring line on `thin_answer`/`call_json`: it assumes evidence-cited record
  payloads; a new purpose whose payload has another shape must pass a predicate (or None).
- **SCHEMA-BUMP**: leave `ARTIFACT_SCHEMA_VERSION` as you have it; the orchestrator reconciles the
  version across change sets at integration.

Run the full mining suite at your new tip and report the count. Final report: same format as before,
covering only this round.
