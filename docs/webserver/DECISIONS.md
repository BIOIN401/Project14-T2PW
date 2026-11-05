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
