# W6b: the results page says why a run needs review

Branch `ws/W6b-review-reasons` (from `518339dd`). Closes E2E-2 audit finding F11
(`docs/webserver/results/E2E-2-bio-audit.md` section 4).

## What changed

`src/t2pw/web/views.py`: new pure `review_why(res)`, called from `item_view` for a finished
`review_required` item. `src/t2pw/web/templates/result.html`: a "Why this needs review"
card directly under the "Review required before use" alert. New `tests/test_web_review_reasons.py`.

The section reads only the result's own `counts` and `release_status` (`missing_anchors`,
`reasons`). It opens no file and no path, so there is no new path surface.

| Line | Source | Wording |
|---|---|---|
| Reactions / Transports / Interactions / Proteins | `counts` (top level, else `counts.pwml`) | "Reactions: 0"; when the PWML has 0 reactions, "The PWML contains no reactions." (it makes no claim about the text) |
| Unplaced terms | `release_status.missing_anchors` | "The run looked for these terms but did not place them in this pathway. Check whether they belong:" then the list. It does not say the protein is part of the pathway or that anything was lost (audit F2: PKD2L1 is a false alarm). |
| Left out | `release_status.reasons` token `omitted_from_pwml:N_processes+M_actors` | "1 process found in the sources left out of the PWML by the strict quality checks." |

Anything absent, non-list, wrong type, a bool, or negative is skipped. If nothing is left
the card is not drawn. Anchor text goes through `scrub` (length 80, at most 20) and then
Jinja autoescape; no `|safe`.

## Deliberately withheld

`expansion_blocked_reason`, `retrieval_attempts` and `completeness` are not shown. The
audit found them misreporting (F4: the blocked-reason string names the wrong cause;
`retrieval_attempts` is not a retrieval counter; F5: completeness 0.8 is overstated). They
stay withheld until the biology side fixes them. A test asserts none of them reaches the page.

## Limits

- **Interactions**: no interactions count exists in the result today (`counts` has
  reactions, transports, proteins, compounds, entities, and `counts.pwml` has
  compounds/proteins/reactions/edges). The label is wired and appears if a
  `counts.interactions` is ever supplied; otherwise it is left out. For E2E-2 the page
  therefore does not state interactions. Wiring them needs the driver to emit the count
  (a `batch/` change, outside this track).
- **Plain reason for omission**: the report gives only the counts in the `reasons` token.
  The reason string in the report is the one audit F4 found wrong, so no reason is given
  beyond "the strict quality checks".
- Compounds are not listed (the brief named reactions, transports, interactions, proteins).

## Open questions

None for the PO. For the biology side: emit an interactions count in `counts` so the page
can state it.

## Tests

Through `bounded_run.py --heavy-lock W6b` wrapping `pinned_pytest.py --expect-tree`
(`T2PW_OFFLINE_CURATOR=1`). Exit 95 (lock held by R26) was retried until acquired.

| Run | Result | Exit | Survivors |
|---|---|---|---|
| `tests/test_web_review_reasons.py` | 8 passed | 0 | 0 |
| 13-file web+bridge list (`web.lst`) | 301 passed (unchanged count, no existing test edited) | 0 | 0 |

Evidence: `docs/webserver/evidence/W6b/` (outputs and wrapper reports). No screenshot: the
page is server-rendered text in an existing card; the test asserts the rendered HTML.

## Rendered text for the E2E-2 case

> **Why this needs review**
> What the PWML contains:
> - Reactions: **0**
> - Transports: **6**
> - Proteins: **6**
>
> **The PWML contains no reactions.**
>
> The run looked for these terms but did not place them in this pathway. Check whether they belong:
> - PKD2L1
> - SCN2A
>
> 1 process found in the sources left out of the PWML by the strict quality checks.

## Review round 1 (APPROVE WITH NOTES) and fixes, commit 658083a7

- F1: "No reactions were found in this text." became "The PWML contains no reactions." The data
  only says the PWML has 0 reactions; an omitted process may itself be a reaction.
- F2: omitted-token regex is now `\d{1,9}`; a 5000-digit token is ignored instead of raising.
- F3: tests now assert `retrieval_attempts`, "Retrieval", "Completeness" and the generic
  `reasons` strings (`semantic_evaluation_failed:...`) never reach the page, and that a missing
  reactions count gives no "no reactions" line.
- F4: for a `review_required` item with the card, the detail line ends "See why this needs
  review below." (otherwise it keeps "Read the warnings below."). No existing test pinned the old text.
- F6 disclosure: before the wrapped runs I ran the 8-test file once directly with pytest, outside
  the wrapper (about 0.6 s, no survivors, no lock involved). In round 2 I also ran one empty-stdin
  `python -I -` by mistake (exited at once, produced nothing); nothing else ran outside the wrapper.

Round 2 results (all through the wrapper, `--heavy-lock W6b`, evidence in `evidence/W6b/round2/`):

| Run | Result | Exit | Survivors |
|---|---|---|---|
| `tests/test_web_review_reasons.py` | 10 passed | 0 | 0 |
| 13-file web+bridge list | 301 passed | 0 | 0 |
| Scratch mutants (`C:/t/w6b_mut`, `w6b_mut2`, not in the repo) | rendering all of `reasons` fails 3 tests, including `test_e2e2_case_states_why` on the `semantic_evaluation_failed` assertion; rendering `retrieval_attempts` fails 6, including `test_e2e2_case_states_why` | 1 each, as intended | 0 |
