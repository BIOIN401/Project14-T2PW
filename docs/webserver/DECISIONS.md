# T2PW Web Server — Decisions

Append-only. **Product owner only.** Agents record open questions in their RESULT doc and
never append here.

---

**D-W01 · 2026-11-05 — Stand-alone server; PWML is a download.** T2PW renders its own
pathway image and exports PWML as a separate artifact. PathWhiz is a separate, older
application, and setting it up is out of scope. Users download the PWML and take it to
PathWhiz themselves.

**D-W02 · 2026-11-05 — PathBank hand-off is a curator queue, not a direct database
write.** Pathways that users consent to share go into a queue. A curator imports them. The
public server never writes to the PathBank database.

**D-W03 · 2026-11-05 — Name mode uses the existing RAG as it is.** When open-access full
text cannot be found, we produce what RAG can support and say so. No new acquisition
work is needed for the LOI.

**D-W04 · 2026-11-05 — Organism drop-down lists all PathBank species.**

**D-W05 · 2026-11-05 — Jobs are asynchronous.** A run takes 20–60 minutes. Users get a
job page and an optional email.

**D-W06 · 2026-11-05 — The results page offers the PWML download** plus the internal
rendered image. No in-browser PathWhiz viewer.

**D-W07 · 2026-11-05 — Failed runs auto-retry.** The limit and wall-clock cap are set in
W4 (default: 2 retries, infrastructure failures and no-PWML outcomes only).

**D-W08 · 2026-11-05 — Refactor now.** The Streamlit-embedded orchestration is extracted
into `t2pw.service.run_pathway` on the `webserver/refactor` branch, before the web server
is built on it. The PWML recovery sprint is closed and `main` is current.

**D-W09 · 2026-11-05 — The public site is a new, polished UI.** The Streamlit app stays as
the developer tool.

**D-W10 · 2026-11-05 — The LLM cost is funded.**

**D-W11 · 2026-11-10 — SBML is in scope.** T2PW will produce SBML files. This supersedes the
"PWML-first / SBML boundary" (no SBML implementation, extension or refactor) in
`docs/pwml_recovery_sprint/DECISIONS.md` § D-023 (line 859). PO, verbatim: "Yes we are overriing
sbml no decison now we will be doig sbml so cghange that in the docs". Written into this file by
the orchestrator at the PO's direction.

**D-W12 · 2026-11-10 — One final payload feeds every format, PWML included.** The pipeline is:
detailed extraction → ids assigned to all entities → the payload checked for completeness, fixed
and gap-filled (public databases included) → the final payload → every file type (PWML, SBML,
KGML, others). PWML is built from that final payload, not from a separate path. Ids found by
lookup are meant for the PWML too. Goal: as close to 100% text-to-PWML conversion as possible.
PO, verbatim: "we want to try our absoluite best to get 100% text to pwml conversion" and "having
a detailed payload and extracitona then assinging ids to all entities then checking this payload
for completeion and making any fixes and filling gaps will provide final payload that we use to
create the file types". Answers proposals Q-04 and question 1 of `docs/pwml_output/CLEAN-MODEL.md`
§ 8. The standing no-regression rule for PWML output still applies: each change is measured
before it replaces current output.

**D-W13 · 2026-11-10 — Unsigned relations go into every format that can carry them.** PWML
still leaves them out (D-104 Amendment 1). PO, verbatim: "yes if it allows for the interactions
allow it to carry them".

**D-W14 · 2026-11-10 — KEGG: no live KEGG calls for now** (option A in
`docs/pwml_output/assessment/07b-kegg-licence-and-access.md`). PO, verbatim: "yeah leave it for now".

**D-W15 · 2026-11-10 — PathWhiz import is tested automatically, using the local SMPDB copy**
(`SummerBIOIN/smpdb`). Edits to that code are allowed when needed for the test, and each one is
reported to the PO. PO, verbatim: "you can make an edit to the spmbdb code if u need to in order
to do that but let me know" and "automating the testing proces is vital".
