# PathWhiz importer changes (smpdb `lib/pwml_parser.rb`)

Three fixes to the PWML importer found while importing T2PW output, logged as separate patches. Each can become its own pull request. The product owner's earlier change (the polymorphic `element-id` remap) is in the base file and in none of the patches.

## Order and how to apply
Apply in order, from the smpdb repo root, on a checkout whose `lib/pwml_parser.rb` equals the 2026-11-11 backup (sha256 below):

    git apply --check 01-vcv-node-typo.patch && git apply 01-vcv-node-typo.patch
    git apply --check 02-transport-biological-state-remap.patch && git apply 02-transport-biological-state-remap.patch
    git apply --check 03-transport-transporter-remap.patch && git apply 03-transport-transporter-remap.patch

The file uses CRLF line endings and the patches carry them. Keep `core.autocrlf=false` while applying. Apply in order.

## Verification (2026-11-13)
Scratch repo containing only the 2026-11-11 backup as `lib/pwml_parser.rb`; all three `git apply --check` and `git apply` passed.

| state | sha256 |
|---|---|
| base (backup 2026-11-11) | cab1accdac417c01ea415d294707445cd14265e67e2994e90c1b99deaca1966b |
| after 01 | 4481d48c4c92e2fb321df628970dc865e1692e283f5a679f0e183bd4f9cc8ea9 |
| after 02 (equals backup 2026-11-12) | fd318fdf04e12a6c2aa70f00010830baceddf21f3e857be8901a8be622038788 |
| after 03 | eb00ccaaf04fcdf6d7454070bb7300787442ce16a200d16d58e7b2208ecfffae |
| current smpdb `lib/pwml_parser.rb` | eb00ccaaf04fcdf6d7454070bb7300787442ce16a200d16d58e7b2208ecfffae (identical to after 03) |

`ruby -c` on the result, inside image `smpdb-app:latest`: `Syntax OK`.

Base note: the 2026-11-11 file already contains the owner's `element-id` hunk (it differs from smpdb git HEAD by that hunk only) and none of our three fixes. No hunk of the owner's change is mixed into any patch.

## Harness command
From the T2PW repo, using the harness copy of smpdb (never the owner checkout), through the bounded wrapper:

    python tools/pathwhiz_import/run_import.py A.pwml B.pwml --list more_paths.txt --screenshots A.pwml --out tools/pathwhiz_import/out/run1 [--resume] [--down]

Details are in `tools/pathwhiz_import/README.md`. Harness, 59 files, writer flag on: 43 clean and 147 messages with all three patches (P3), against 29 clean and 211 messages before the importer patches (P2). Not measured per patch.

## Changes
| NN | patch | what | evidence |
|---|---|---|---|
| 01 | `01-vcv-node-typo.patch` | vacuous edge loop used the wrong variable | read from code, not reproduced |
| 02 | `02-transport-biological-state-remap.patch` | remap left/right biological state ids | 37 of 37 transports lost, empty DB |
| 03 | `03-transport-transporter-remap.patch` | remap transport-transporter ids, add `@transport_transporter_list` | 19 messages in 12 files down to 3 in 3 |
| 04 | `04-biological-state-exact-match.patch` (optional copy) | an omitted biological-state component matches nil, not any value | harness, 39 files: states created equal states in the file for the 3 large PathBank controls (262 to 273, 176 to 183, 342 to 351); 3 writer files gain a transport; 15 files without bindable states identical |

## Stacked branches (PO amendment 2026-11-13; the branches are now the source of truth, the .patch files are optional copies)
Repo: the PathWhiz app checkout (owner's folder, never touched). Worktrees under `C:/t/pwpr/NN`. Not pushed.

| order | branch | commit | on top of |
|---|---|---|---|
| 01 | `t2pw/01-vcv-node-typo` | `17af3272` | `test-prod-with-parser-fix` `6e11df0e` |
| 02 | `t2pw/02-transport-state-remap` | `48ab7499` | 01 |
| 03 | `t2pw/03-transport-transporter-remap` | `c5c12d69` | 02 |
| 04 | `t2pw/04-biological-state-exact-match` | `80bdaca7` | 03 |

The owner's uncommitted change in `lib/pwml_parser.rb` (the `element-id` branch, 10 lines) is in no branch. Check: tip of 03 (`c5c12d69`) plus that change (diff between the committed HEAD and the 2026-11-11 backup) gives a `lib/pwml_parser.rb` with sha256 `eb00ccaaf04fcdf6d7454070bb7300787442ce16a200d16d58e7b2208ecfffae`, equal to the file the harness imports (the owner's working copy, which the harness clones and patches with `git diff HEAD`). The harness keeps importing from the owner's working copy, unchanged.
