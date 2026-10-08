# ORCH-745 CS-B — repair round 1 (of at most 2)

Your tip `467a9faa` was reviewed three ways:
- code review **REQUEST_CHANGES** (1 blocking: F1 negated conversion);
- bio conformance **SOUND_WITH_FINDINGS** (no blocking; it ACCEPTS your 5 TABLE_DELTAS #2/#4/#6/#29/#30 —
  record them as adjudicated);
- an adversarial red-team bio-auditor **UNSOUND: 9 blocking gate breaks (B1–B9)**, each a realistic sentence
  family that reaches `admission.judge -> accepted` for A. thaliana on evidence that must not be admitted.

The orchestrator has merged the repaired change sets into your branch: it is now at `4f8d33a3`
(= your 467a9faa + CS-A repair tip 92ab5be9 + CS-C repair tip beb6d70e). Mining suite there: **391 passed**.
Same rules (`C:/t/bt/orch745/COMMON_RULES.md`), same worktree `C:/t/m745-b`; add commits on top of `4f8d33a3`,
never rewrite history. Every repair test that proves a correction must FAIL behaviourally at `467a9faa`
(your first tip; at 4f8d33a3 too, since the merges do not touch species reading) and pass at your new tip;
list them in `C:/t/bt/m745-b/g9_manifest_repair1.json` (`{"nodes": [{"node", "finding", "fails_at_467a9faa",
"expected_failure", "fixtures"}]}`). The binding spec is still `F1_BIO_ADJUDICATION.md`; the T-rows (T1, T1b,
T2, T3, T4–T11, #33-or-inferred_only) and the full pilot-003 replay (exactly #3, #14, #34 admitted; #14
in_planta, #3/#34 review/review_article_only) must still hold after every change. **The full red-team
report with exact probe sentences, payload shapes and verified patterns is in
`C:/t/bt/orch745/reviews/CS-B-redteam.txt`; the code and conformance findings are in
`C:/t/bt/orch745/reviews/csb_r0_and_csc_r1.txt` (first 69 lines).** Read both. Every probe sentence quoted
there that the reviewer marks "admitted" becomes a `test_repair1_745b_*` regression test (parametrize by
family) asserting NOT target_species_supported; every control they mark correct stays correct.

## Blocking — positive credit is still too lenient (fix all; fail closed)

- **B2 / code F1 — negated or failed conversion earns in_planta.** In rule 5(a) refuse credit when a negation
  governs the conversion in the clause: `not|no|never|nor|without|fail(s|ed) to|unable to|lack(s|ing)?`
  near the conversion predicate. Keep 5(b) (loss-of-function) on its `_NO_EFFECT` guard only — "no JA-Ile was
  detected in gh3.10 jar1 lines" must stay credited (T1). The helpers P7 sentence "allene oxide cyclase, not
  allene oxide synthase, converts 13-HPOT to 12-OPDA" must keep its credit (negation not governing the
  conversion).
- **B1 / code F2 — cell-free preparations earn in_planta.** Organ/organelle/cell preparation nouns are
  IN_VITRO: `(leaf|leaves|flower|root|rosette|seedling|silique|plant|cell|chloroplast|plastid|tissue)
  (extract|homogenate|lysate|fraction|assay|disc|preparation|supernatant|powder)`, `(soluble|membrane|
  stromal|microsomal|crude|desalted|clarified|total|cell-free) (protein )?(extract|fraction)`,
  `(isolated|intact) (chloroplasts|plastids|peroxisomes|mitochondria)`, affinity tags (His/Strep/GST/FLAG/MBP),
  `produced in (E. coli|bacteria|yeast|Pichia|insect cells|Sf9|baculovirus)`, `insect-cell`, `assays? with`;
  also cell suspension/culture/callus (BIO-745B-5). Put "plant material" in `_PROCEDURAL`, not `_IN_VITRO`.
  Note: an in-planta metabolite measurement "JA-Ile in leaf extracts of jar1" is a real recall cost — accept it
  (fail closed); say so in `known_limitations.md`.
- **B3 — open hedge list.** Add: thought to, presumed, seem(s) to, appear(s), consistent with, point(s)
  to/towards, speculat*, impl(y|ies|ied), unclear, remain(s) unknown/unclear/to be/an open, open question,
  assume(d|s), conceivabl*. (Apply to positive credit and the review route.)
- **B4 — homology transfer without the word "ortholog".** A homology-transfer veto on positive credit and the
  review route: counterpart, closest … relative, relative of, corresponding … (enzyme|protein|gene|reaction),
  equivalent, by (sequence )?similarity, homologous, resembl*, analogous, the same (conversion|reaction).
  NOT "compared with/to" (normal loss-of-function prose). Prefer the claim to come out `inferred_only` (T10
  style).
- **B5 — transgenic host without the HOST words.** Extend HOST: carry/carrying, harbo(u)r*, introduc(ed|tion|
  ing), construct, promoter, cDNA, transgene; complementation with a FOREIGN gene (e.g. "complementation of the
  aos mutant with the MpAOS gene") must not credit A. thaliana. (If "complement" alone costs native-gene
  complementation recall, require a foreign-gene cue; document the choice.)
- **B6 — exogenous application synonyms.** Extend APPLICATION: suppl(y|ied|ies|ement*), infiltrat*, inject*,
  dosed, administered, watered, soaked, dipped, perfused, deuterated, labell?ed. State in `species_policy.md`
  that fed-precursor conversion is held back pending the product-owner hinge (the JA-conjugate rows).
- **B7 — review route admits negations, hedges and open questions.** Veto: fail(s|ed) to, unclear, open
  question, remain(s) unknown/unclear/to be, whether, plus B3's hedges. **Code F4:** the review route must also
  require a conversion predicate (convert*, conversion, reduc*, produc*, form(ed|s|ation), synthesi[sz]*,
  catalyz*/catalys*, oxidi[sz]*/oxygenat*, conjugat*, metaboli[sz]*, yield*, generat*, cycli[sz]*, give rise)
  when the sentence names only a substrate and a product — "In Arabidopsis, OPDA and OPC-8:0 accumulate in
  wounded leaves" must NOT admit; the T2/T3 sentence ("are reduced by OPR3 ... to produce") must.
- **B8 — non-target organism as a gene-symbol prefix.** Teach the rule-4 organism check gene-symbol species
  prefixes when the sentence does not name the target: `\b(Mp|Pp|Os|Sl|Zm|Ta|Hv|Gm|Nt|Nb|Bn|Ca|Vv|Md|St|Le|Cs|
  Sb|Br|Mt|Pt)([A-Z]{2,}[0-9A-Za-z.]*|\s[a-z]{2,}[0-9a-z]*)\b` (note `[a-z]{2,}` to catch "Mp gh3a"); a
  documented ceiling like `_EXTRA_HOSTS`. The adjudication's own W4 sentence "Genetic knockout of MpFAD5 nearly
  abolishes dn-OPDA accumulation after mechanical wounding" must give no credit even with empty
  `paper_observed_organisms`.
- **B9 / code F3 — citations the `_CITATION` list misses.** Add narrative forms (previously
  reported/shown/described/demonstrated, as shown/reported/described by/in, prior/earlier work/studies,
  has/had been shown/reported, co-workers), numeric parenthetical `\(\s*\d{1,3}(\s*[,–-]\s*\d{1,3})*\s*\)`, and
  superscript digits flattened after a period ("...12-OPDA.12,13").
- **N1 / code F5(1) — the colon split destroys compound names (recall, but it silences exactly the OPR3 /
  OPC-8:0 family the DoD topic leads to).** `_CLAUSE = ;|:(?![0-9])|\b(and|but|whereas|while|or)\b` (no split
  on a colon followed by a digit). Test: "In the opr3 knockout line, OPC-8:0 was undetectable in leaves."
  (enzyme/gene OPR3, OPDA -> OPC-8:0, species cited) -> target_species_supported / in_planta; and "In
  Arabidopsis leaves, OPDA is converted to OPC-8:0 by OPR3." -> in_planta. Also F5(2): split sentences after a
  single capital when the next token starts a sentence ("clade A. In opr3 mutant leaves ..."), and F5(3): do not
  glue tokens across a SPACED em/en dash ("JA-Ile — the").
- **N2 — W13 species-span check too weak.** `species_span_names_target` must reject a sentence naming the target
  only by comparison (resembl*, similar to, analogous, as in, also found/present in, both) and one naming another
  organism alongside the target (the condition the review route already applies).
- **N3 — section check is a prefix test.** Use `re.search(r'method|material|introduction|background',
  section.lower())`; add "plant material", "growth conditions", `(was|were) (\w+ )?(obtained|purchased|provided|
  surface-sterili[sz]ed|stratified|transferred)` to `_PROCEDURAL`. Do NOT add "body" (abstract-only papers are
  indexed as body).
- **Orchestrator decision (defence in depth for N5):** no positive in_planta credit from `discussion` or
  `conclusion` spans (interpretive sections; they restate others' work and hedge). Abstract, results, figure
  legends and body keep credit. Document it in `species_policy.md` as a decision the product owner may revisit.

## Blocking — support must not be overstated (A3 + orchestrator ORCH-2; admission.py / claims.py / graph.py /
controller.py are yours for this)

Today `judge` uses `store.supporting_papers(identity)`, i.e. EVERY sibling claim's paper whatever its species
state, and `graph._merge_into` folds a sibling's paper_ids and evidence_ids into the stored reaction; then
`controller._admit` sets the stored node's `decision_status` from the latest decision. Consequences, now common
because many At claims are non-admissible siblings of admitted ones:
(a) a reaction supported only by reviews grades `multiple_independent_papers` (A3); (b) an admitted reaction
lists another paper's in-vitro or ambiguous claim as independent support (`multiple_independent_papers`, its
spans in `evidence_ids`) — overstated evidence; (c) an admitted node is relabelled `candidate` by a later
non-admissible sibling while it stays admitted. Required:
1. Evidence level and the admitted reaction's `paper_ids` / `evidence_ids` / `claim_ids` count ONLY sibling
   claims that are themselves `target_species_supported`. Reviews never count toward independence: all
   supporting TSS papers are reviews -> `review_article_only`; >= 2 TSS primary (non-review) papers ->
   `multiple_independent_papers`; otherwise `single_paper` (a review alongside one primary adds a corroboration
   note, not independence). Record this rule in `evidence_policy.md` as a decision the product owner may revisit.
2. A non-admissible sibling claim never adds its paper/evidence to an admitted reaction as support. Keep it
   auditable on the node (e.g. `non_supporting_claim_ids` with each claim's species state) and in the history.
   When a quarantined reaction is lifted by a later TSS claim, its support becomes the TSS claims only; the
   earlier non-admissible claims move to the non-supporting list.
3. A later non-admissible sibling never downgrades an admitted reaction: the node stays admitted with an
   admitted status; the claim's own non-admissible verdict is still recorded (decision detail / history), and
   `summary.json` / `report.md` counts stay consistent (an admitted reaction is never counted as a candidate).
4. G9-style tests (fail at 467a9faa): two review-only TSS siblings -> `review_article_only`; review TSS + primary
   IVO sibling with the same identity -> `review_article_only` and the IVO paper not in the node's paper_ids;
   admitted primary TSS + later primary ambiguous sibling -> node still admitted/accepted, paper_ids unchanged,
   the sibling listed as non-supporting.

## Non-blocking (do them; they are small)
- **BIO-745B-1:** add bare X of an "X (Y)" name to the LOOSE (negative-marker) name set only; positive credit
  stays strict (full name and Y). Update TABLE_DELTAS (#10, #19, #22, #23 -> IVO; #33 -> inferred_only accepted
  by the bio-auditor) — the admitted set and grades must not change.
- **F6:** persist the legacy-claim floor in the merged reading (`species_support["floor_state"]`) and apply it
  on every later recompute; add a 3-batch permutation test.
- **F8:** rename the moved tests to match what they assert, fix the docstring (grade none via W13), drop the
  duplicate assertion.
- **N4:** `species_policy.md`: document the ceiling that a primary paper's in-planta sentence naming no organism
  is attributed via the paper-level organism screen.
- **BIO-745B-2 (escalation, NO code change):** record in `known_limitations.md` that a participant kept by
  `model_citation` is not checked against its cited span's text, so the other side of an admitted reaction can
  rest on the model's citation alone (e.g. #14's "jasmonic acid" where the abstract says "JA").
- Keep `ARTIFACT_SCHEMA_VERSION` at `t2pw.mining/1.2.0` unless you add persisted fields again; if you do, bump to
  1.3.0 and keep `artifacts.md` in step.

Run the full mining suite at your new tip. Final report: per finding (B1–B9, F1–F10, N1–N5, BIO-745B-*, A3/ORCH-2)
what changed and the test proving it; the T-row table and full replay result again; suite count; manifest path.
