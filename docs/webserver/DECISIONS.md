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

**D-W16 · 2026-11-10 — Gap filling also uses RAG and related papers.** The completeness step
of D-W12 fills gaps from three sources: public databases, RAG over the submitted paper, and
relevant papers found and retrieved when the paper itself leaves large gaps. The same route serves
a prompt that names a pathway with no paper (for example "give me pathway of tca cycle"). Content
from another paper is labelled with that paper's id and is never presented as stated by the
submitted paper. This extends D-W03 (name mode uses the existing RAG as it is) beyond the LOI; it
does not add work for the LOI. PO, verbatim: "but the gpas should also fill in using rag, and
fidning releveant papers if there are large gaps in the paper itself or perhaps the user prompted
with something like give me pathway of tca cycle".

**D-W17 · 2026-11-10 — Explore filling from databases of known pathways; name mode shares the
gap-filling machinery.** (a) Explore filling missing steps from curated pathway databases, as an
independent source that can raise accuracy. Candidates to assess: PathBank, Reactome, Rhea,
WikiPathways, MetaCyc; KEGG stays excluded under D-W14. Each candidate is assessed for coverage,
licence and how a step is matched to the paper's organism and entities before any build. Steps
taken from such a database are labelled with the database and its pathway id, never as stated by
the paper. (b) Name-only mode ("give me pathway of tca cycle") uses the same completeness-and-fill
machinery as a submitted paper (D-W12, D-W16), not a separate route. PO, verbatim: "i also think we
should explore fillig in from db of known pathways that would improve accuracy by using another
source, and the name only mode should use the same machinery that we use for gap filling for the
regular pdfs, addd that inot the docs too".


**D-W18 · 2026-11-11 — Known-pathway database fill: sources and limits** (answers the open
questions in `docs/pwml_output/assessment/08-known-pathway-db-fill.md` on `pwml/clean-model`).
(a) Sources: Rhea, Reactome, WikiPathways and PathBank. PathBank is the lab's own database, so its
licence is not a blocker; use the PathBank database we already have (read-only), accepting that it
is less reliable to reach. MetaCyc/EcoCyc are ignored. KEGG stays excluded (D-W14). No message to
the PathBank team is needed for now. (b) Filled steps go into the one final payload and from there
into every format, PWML included (D-W12), labelled `db` with database and accession. (c) Rhea, which
has no organism field, may fill a step only where a gap has been identified, and only when the
other parts of the reaction (substrates, products, enzyme) already match the paper's organism.
(d) Steps known only by cross-species inference (for example a Reactome human-to-rat ortholog step)
are not used unless the step is confirmed for that species; revisit later. (e) Size: by default a
fill only adds steps between entities the paper already names. When the user asks for a named
pathway, or says they want the entire pathway (for example "cholesterol pathway"), fetch the whole
pathway. Fetch as much as is available, then an agent selects what matters from the user's request
or the inferred context. PO, verbatim: "Yes to rhea and reactome and wikipathways and pathbank bro
we own pathbank the lab im in, but yeah it acts up use the database we have althouhg i dont think it
will work as well / the filling should go into the main payload used for all formats / 2. lets
leave pathbanks writes fr now / 3. Yes if its just filling in a gap and the other parts of the
reaction are clearly from that orgnaism by that i mean if everythging else matches chances are that
step is there, but we should prolly only use it if we have identified a gap right / 4. im not sure
like we should be able to see a connection but is that smart is it true that the rat has that step
in it, if its not confirmed lets leave it atleast for now lets log this decisions / 5. yeah
anywhere between the paper, but if the person asks for cholestrol pathway or has speciied thjey
want the entire cholestrol pathay we gotta fetch the entire thing, i thuink it would be smart to
like fetch as muhc data as we can and then allow an agent to select the stuff that is important
based on the users input or inferred context right / 7. ignore it". Evaluation method (question 6)
still open.

**D-W19 · 2026-11-11 — Final payload is the main path; PathWhiz importer fixes; fill rules for
enzymes, transporters and protein ids.** (a) The final payload (D-W12) is the main thing every
format is built from, PWML included: `T2PW_FINAL_PAYLOAD` is to become the default path. (b)
Renaming compounds to their database names (e.g. "H+" to "Hydrogen Ion") is accepted. (c) Protein
ids: match the paper's species; if that fails, an id from another species may be used, and this is
disclosed in the text version. (d) Reactions with no enzyme: first look for the enzyme in other
papers (RAG or whatever method works best); only if none is found mark spontaneous as unknown,
because it may truly be unknown. (e) Transporters are written in the form PathWhiz expects. (f)
PathWhiz importer bugs found by the import test (the `vcv_node` typo in `lib/pwml_parser.rb` and the
transport `left-/right-biological-state-id` never being remapped) are to be fixed, and recorded in
our docs; no smpdb commits (standing rule). (g) The PathWhiz import harness, its seed data and its
results may be committed, provided the repository is not bloated. (h) Evaluation of database fill
(D-W18 open question): hold-out ablation plus a hand-checked sample now; testing on new papers with
non-PathBank answers later. PO, verbatim: "For 6 yes i like ur plan on the checeking the datbase
filling, we can do ther eal testign later / 1. Ues commit the pathwhiz import / 2. yeah sure / 3.
dont explode my repo, but if github will accept all of that stuff then yeah commit it / 4. yeah i
mean the final payload is what we are gonna use as the main thig for buildin pwmland all other text
types / 5. yes / 6. we should try but if we cant we can take another species name if needed, and
then just disclose this in the text version / [transport bug] 1. Fix this but also make a note of
it in our docs / 2. we should try to find these enzymes and then yeah mark them as unknown because
they may truly be unknown but we need to check other papers for it firdst using rag or whayveer
metho works the btest / 3. match the pathwhiz format".

**D-W20 · 2026-11-12 — Unmatched locations, same-state transports, nucleic-acid type, keys for
live runs** (answers `FINAL-PAYLOAD-P3.md` open questions on `pwml/final-payload`). (a) Location,
cell-type and tissue names PathWhiz lacks: look up their ontology ids online and keep them in the
final payload; a location still unresolved is dropped only when converting the final payload to a
format that needs the id (e.g. PWML), not from the payload itself. (b) Transports whose start and
end state are the same: fix at the cause (the state filler in the pipeline) and test it; this is an
authorised pipeline change. (c) DNA/RNA type of a nucleic acid that the name does not settle: try to
fill it from the paper first; the fallback is decided after seeing how far that gets. (d) The `.env`
keys may be copied into a work tree for live runs, and the copy must be deleted and cleaned up
afterwards. PO, verbatim: "1. yes look up and then drop location after converting from final
payload to the specific file type / 2. yeah lets fix and test / 3. im not sure is there anyway to
fill this in from the paper? lets start with this / and yeah bro u can move the .env keys but you
have to delete this and clean it up".

**D-W21 · 2026-11-13 — Phase 4 open questions, PathWhiz changes as reviewable patches, internal
renderings** (answers `FINAL-PAYLOAD-P4.md` § 9 on `pwml/final-payload`). (a) Add the guard so an
id-less location is not dropped when that would leave a state PathWhiz can bind to another state.
(b) Cell types and tissues without an id stay in the final payload; they may be removed from the
PWML only. Find out and record why each has no id. (c) The 4 transports whose start and end are
the same: test a fix at the extraction step. It must fix them and keep everything we currently
have; a fix that loses anything is not adopted. (d) DNA/RNA type the paper does not settle is
flagged unknown; also try an online lookup to settle it. (e) Inner and outer membrane are
different things and are never merged. (f) The PO has given permission to edit the `smpdb` folder
and fix PathWhiz where needed. Every change there is logged on its own so that each one can
become its own pull request. (g) Show the paper pathways in our internal rendering. PO, verbatim:
"1. yeah sure / 2. Well why dont they have an ID, and they should still be in the final payload but
yeah night have to remove form the pwml / 3. Well lets test this change see if its fixes those
errors but also maintians everuthing we currently have / 4. Yeah flag it as unkown, is there anway
to search online and try to figure out what it would be? / 5. arent they differnt things inner and
outer membarnaes / and i want you to show me what these ppaer pathways would look like in our
intenral rendering / aSLO WE have receieved permission to edit the spmbd folder and fix pathwhiz so
maybe if changes eed to be made there w cna do that as well, and then we just ned to log all
changes so each change can act as its own pull requiest".
