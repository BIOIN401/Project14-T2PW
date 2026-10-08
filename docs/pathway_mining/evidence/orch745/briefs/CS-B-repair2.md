# ORCH-745 CS-B — repair round 2 (the LAST allowed round): narrow positive credit structurally

Re-review of your repair-1 tip `fd53a301` (full findings: `C:/t/bt/orch745/reviews/csb_r1.txt`):
- red team round 2 **UNSOUND** (R2-B1..B7: 52 of 57 new paraphrase probes still accepted);
- bio conformance **UNSOUND** (BIO-R1-1..4, same families);
- code **REQUEST_CHANGES** (R1-B1 a gate WEAKENED by the delta; R1-B2..B5 paraphrases; R1-B6 a lift bug).

All three agree the remaining breaks come from rule 5(a) (a conversion stated in an organ / genotype context)
guarded by OPEN word lists: every round closes the quoted sentences and paraphrases get through (red-team N-2:
"list extension alone will not converge"). No further repair round is allowed after this one, so this round
changes the SHAPE of positive credit instead of adding words. **Orchestrator decision (fail-closed; positive
credit only shrinks; documented in `species_policy.md` as a decision the product owner may revisit):**

## 1. Positive in-planta credit comes ONLY from loss-of-function genetics (rule 5(b)); rule 5(a) is removed

Under the adjudicated policy, feeding/supplied precursors are held back (hinge) and occurrence is not
conversion, so in a primary paper a bare "X is converted to Y in Arabidopsis leaves" is either background, a
fed-precursor experiment, or an in-vitro description. Remove the 5(a) branch entirely. A supporting sentence of a
NON-review paper earns IN_PLANTA only if, within one clause (your clause splitter, colon-safe):
1. a plant loss-of-function genotype marker: mutant(s), knockout, knock-out, knockdown, loss-of-function,
   null allele, T-DNA, SALK_, RNAi, amiRNA, CRISPR … — but NOT a protein/site-directed mutant: no
   "mutant protein/enzyme/variant/form", "variant", "site-directed", amino-acid substitution notation
   (`\b[A-Z]\d{2,4}[A-Z]\b` on the original text), "wild-type enzyme/protein", "(in)active";
2. the claim's own enzyme or gene, strictly named, and not a foreign-prefixed symbol (see §3);
3. a strictly-named substrate or product of the claim (cofactors / free amino acids never count);
4. a POSITIVE EFFECT phrase from a whitelist — absence forms: "not detected/detectable", "no detectable",
   "undetectable", "below the (detection|quantification) limit", "absent", "abolished", "abrogated",
   "eliminated", "lacked/lacking", "deficient", "depleted", "no <participant> was detected/found/formed";
   change forms: reduced, decreased, lower, diminished, impaired, compromised, accumulated, increased,
   elevated, higher — a CHANGE form only counts when no negation token (not|no|n't|never|neither|nor|without|
   none) precedes it in the clause (so "were not reduced" fails, while "No JA-Ile was detected" is an absence
   form and passes);
5. and NONE of (anywhere in the sentence): a no-effect phrase (unchanged, unaffected, not affected/changed/
   altered/different, not significantly, did not differ/change, similar to / comparable to wild type,
   wild-type levels, normal levels, no change/difference/effect); a hedge or modality word — use a BROAD
   stem set, not the current list: may|might|could|would|should|probabl|possibl|likely|unlikely|believ|thought|
   suggest|propos|postulat|hypothe|presum|assum|apparent|seem|appear|putative|potential|speculat|conceivabl|
   consider(ed)? to|yet to|remain|unclear|unknown|whether|debat|controvers|elusive|unresolved|expected; any
   IN_VITRO / HOST / APPLICATION / INFERENCE / HOMOLOGY / COMPARISON marker (as, as in, like, unlike, similar to,
   in contrast to, compared with/to, homolog(ue)s?, counterpart …); a citation (incl. "(ref. 12)",
   "well known", "it is (well) established that"); a procedural verb; a no-credit section
   (methods/material/introduction/background/discussion/conclusion — keep your `re.search`).
T1 ("In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit.") must
still earn in_planta via this rule; T1b, the full pilot-003 replay (exactly #3, #14, #34; #14 in_planta) and
every T-row must hold. Remove/convert the now-obsolete 5(a)-positive acceptance tests deliberately (list each
moved test and why, merge rule 4), and rewrite the `tests/helpers_mining.py` synthetic abstracts that were
phrased to earn 5(a) credit so they earn credit by loss-of-function genetics instead (e.g. "In Arabidopsis
thaliana aos knockout lines, 12-OPDA was undetectable in wounded leaves.") — keep the fixtures' biology
coherent. The colon-safe recall controls that use 5(b) (r1 "In the opr3 knockout line, OPC-8:0 was undetectable
in leaves.", r3) must keep credit; r4 (a 5(a) sentence) now correctly gets none.

## 2. The review route: strict and broad-vetoed (rule 7)

- Restore the adjudication's participant + enzyme branch exactly (code R1-B1, a weakening you introduced): `(sub
  and prod and _REVIEW_CONVERSION) or ((sub or prod) and enzyme_named and _CONVERSION)` with `_CONVERSION` the
  strict convert/conversion/formed from/metaboli[sz]ed to/produced from/synthesi[sz]ed from set. The four R1-B1
  probes ("JAR1 is required for JA-Ile production", "JA-Ile levels are reduced in jar1 mutants", "JA-Ile
  formation requires JAR1", the "catalyzes" step with a wrong substrate) must not admit.
- For the substrate-and-product case require the two to be LINKED, not co-occurring: "X … to/into Y", "Y …
  from X", or "X and Y … respectively" with a conversion verb between them — "OPDA and OPC-8:0 are produced in
  wounded leaves" / "… levels are reduced after wounding" must not admit (R1-N2); the T2/T3 sentence ("In
  Arabidopsis, OPDA and dn-cis-OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0,
  respectively.") must.
- Veto the review route on the same BROAD negation + modality stem set as §1.5, plus: not|no|never|neither|
  nor|none|without|cannot|unable|incapable|fail|lack|absent|negligible|undetect|rather than|to be (determined|
  established|shown|demonstrated); and on any comparison/homology/foreign-gene/other-organism mention.

## 3. Foreign gene symbols and organisms (R2-B5, R2-B6, R2-B7, BIO-R1-4)

- `_others_named` (rule 4, the review route, the W13 species-span check, and the host marker) ALWAYS includes
  foreign-prefixed gene symbols, whether or not the sentence also names the target — a sentence naming the target
  and a foreign gene is a host/comparison construct, never native evidence. Drop the "if target not in named"
  guard.
- Widen `_FOREIGN_GENE` to the canonical allele notations: `\b(?:Mp|Pp|Os|Sl|Zm|Ta|Hv|Gm|Nt|Nb|Bn|Ca|Vv|Md|St|Le|
  Cs|Sb|Br|Mt|Pt|Lj|Ph|Ps|Sm|Gh)(?:[A-Z]{2,}[0-9A-Za-z.]*|[\s-]?[a-z]{2,}\d[0-9a-z]*|[\s-][A-Z]{2,}\d?
  [0-9A-Za-z.]*)\b` ("MpFAD5", "Mp FAD5", "Mpgh3a", "Mp-aos1", "Mpfad5"); keep it a documented ceiling.
- Strip comparison mentions of the target ("as in Arabidopsis", "like its Arabidopsis homologue", "unlike (in)
  Arabidopsis", "similar to Arabidopsis", "compared with Arabidopsis") before any "the sentence names the
  target" test; add bare `homolog(ue)?s?` to the homology markers.
- `fold_text`: apply `unicodedata.normalize("NFKC", …)` first (ligatures "ﬁ"/"ﬂ" defeat markers; N-1).

## 4. Support and decisions (R1-B6, R1-N1, R1-N5)

- **R1-B6 (bug):** when a quarantined lead is lifted, the admitted node's `gene`, `tissue`, `conditions`,
  `species_text`, `hypothesis` (and display `substrate_names`/`product_names`) come from the supporting
  claim(s), never from the lead. Extend the A3 lift test: lead gene "MpGH3A", tissue "flowers" -> the lifted node
  shows the supporting claim's values.
- **R1-N1 (structural, do it):** judge a paper's claims only after ALL of that paper's extraction batches in the
  round have been registered (register every batch's claims into the store first, so the F-6 same-paper union is
  complete; then judge each distinct stored claim/identity once). A same-paper merge then can never downgrade an
  already-admitted reaction within a paper. Keep cross-paper behaviour as in repair 1. Test: batch A
  (loss-of-function in-planta sentence) + batch B of the same paper (an inference sentence for the same claim) ->
  never admitted, in either batch order, and `summary`/`report` consistent.
- **R1-N5:** (a) the replay harness counts only outcomes without `claim_verdict` in `decision.detail`; (b) build
  non-supporting rows from `store.siblings()` with per-claim paper_id and species_state.
- Qualify `evidence_policy.md` item 4 for the remaining starting-graph edge.

## 5. Tests and proof

- Every probe the round-1 reviewers marked admitted (red team p1a/p1b families neg, h, ap, iv, ho, cmp, gp;
  bio N/V/H/T blocks; code R1-B1..B5 lists) becomes a `test_repair2_745b_*` parametrised row asserting NOT
  target_species_supported, plus the controls they marked correct. These must FAIL at `fd53a301` (they were
  accepted there) — list them in `C:/t/bt/m745-b/g9_manifest_repair2.json` with `fails_at_fd53a301`.
- All existing T-rows, the full replay, the round-0 and round-1 regression rows must still pass.
- Add one honest note to `known_limitations.md`: the remaining risk class is lexical (paraphrase) and the design
  now fails closed — a paraphrase that slips a veto can only cost a lead unless it also forms a complete
  loss-of-function sentence with the claim's own gene and an effect phrase; list what positive credit now
  requires; list the recall cost of removing 5(a).
- Keep `ARTIFACT_SCHEMA_VERSION` 1.3.0 unless you add persisted fields (then 1.4.0 + artifacts.md).

Run the full mining suite at your new tip. Final report: per item, the T-row table, the replay, suite count,
manifest path, and every deliberately moved test with its exact delta.
