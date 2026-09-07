# ORCH-724 unseen pilot — manual PWML review package

Run: `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425`

> `release_ready` is a runtime fact, not a biological verdict. **`review_required` is not a failure** — a `review_required` PWML is a successful product output if it is biologically useful.


Label each leg: **PASS** · **PASS WITH MINOR OMISSIONS** · **MAJOR BIOLOGICAL ERROR** · **INVALID OR UNUSABLE PWML**


Minor omissions: ordinary cofactors, currency metabolites, non-defining regulators, ancillary proteins. Major errors: fabricated reaction, wrong defining substrate/product, missing major pathway branch, wrong organism, repeated major enzyme error, unusable PWML.


---

## PMC11172790 / research

| field | value |
|---|---|
| requested scope | pyoverdine biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC11172790/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC11172790/research/final_mapped.json |
| organisms | Pseudomonas aeruginosa |
| enzymes | PvdD, PvdI, PvdJ, PvdL |
| reactions / rows | 3 / 12 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 84) |
| rows with NO provenance | **0** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | PvdL reaction | myristic acid, L-glutamate, L-tyrosine, L-2,4-diaminobutyrate | three-amino-acid product | PvdL |  |  |
| 2 | pyoverdine backbone elongation and cyclization | three-amino-acid product, L-Serine, L-Arginine, L-hydroxyornithine, L-Lysine, L-Threonine | pyoverdine backbone | PvdI, PvdJ, PvdD |  |  |
| 3 | pyoverdine maturation in periplasm | acylated ferribactin | pyoverdine, myristic acid | — |  |  |
| 4 | PvdE-mediated acylated ferribactin export into the periplasm | — | — | — |  |  |
| 5 | PvdL interacts with PvdD | — | — | — |  |  |
| 6 | PvdA interacts with PvdL | — | — | — |  |  |
| 7 | PvdA interacts with PvdI | — | — | — |  |  |
| 8 | PvdA interacts with PvdJ | — | — | — |  |  |
| 9 | PvdA interacts with PvdD | — | — | — |  |  |
| 10 | PvdL co-localizes with PvdI | — | — | — |  |  |
| 11 | PvdL co-localizes with PvdJ | — | — | — |  |  |
| 12 | PvdL co-localizes with PvdD | — | — | — |  |  |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC11172790 / strict

| field | value |
|---|---|
| requested scope | pyoverdine biosynthesis |
| PWML | **none generated** |
| release status | diagnostic_only |
| semantic evaluation | failed |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC11172790/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC11172790/strict/final_mapped.json |
| organisms | Pseudomonas aeruginosa |
| enzymes | (none) |
| reactions / rows | 0 / 3 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 63) |
| rows with NO provenance | **0** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | acylated ferribactin export | — | — | — |  |  |
| 2 | PvdA interacts with PvdI | — | — | — |  |  |
| 3 | PvdA interacts with PvdJ | — | — | — |  |  |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC11405693 / research

| field | value |
|---|---|
| requested scope | phenylpropanoid biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC11405693/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC11405693/research/final_mapped.json |
| organisms | Arabidopsis thaliana |
| enzymes | 4-Coumarate:CoA ligase, PAL, cinnamate 4-hydroxylase |
| reactions / rows | 6 / 7 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 142) |
| rows with NO provenance | **0** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | PAL reaction | phenylalanine | cinnamic acid | PAL |  |  |
| 2 | C4H reaction | cinnamic acid | p-coumaric acid | cinnamate 4-hydroxylase |  |  |
| 3 | 4CL reaction | p-coumaric acid, CoA-SH | p-coumaroyl-CoA | 4-Coumarate:CoA ligase |  |  |
| 4 | PAL reaction | phenylalanine | cinnamic acid | PAL |  |  |
| 5 | C4H reaction | cinnamic acid | p-coumaric acid | cinnamate 4-hydroxylase |  |  |
| 6 | 4CL reaction | p-coumaric acid | p-coumaroyl-CoA | 4-Coumarate:CoA ligase |  |  |
| 7 | Piperonylic acid inactivates trans-cinnamate 4-hydroxylase | — | — | — |  |  |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC11405693 / strict

| field | value |
|---|---|
| requested scope | phenylpropanoid biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC11405693/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC11405693/strict/final_mapped.json |
| organisms | Arabidopsis thaliana |
| enzymes | 4-coumarate:CoA ligase complex, AtCYP73A5, Cinnamate 4-hydroxylase, PAL |
| reactions / rows | 6 / 7 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 163) |
| rows with NO provenance | **0** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | PAL deamination | phenylalanine | cinnamic acid | PAL |  |  |
| 2 | C4H hydroxylation | cinnamic acid | p-coumaric acid | AtCYP73A5 |  |  |
| 3 | 4CL activation | p-coumaric acid, CoA-SH | p-coumaroyl-CoA | 4-coumarate:CoA ligase complex |  |  |
| 4 | PAL reaction | phenylalanine | cinnamic acid | PAL |  |  |
| 5 | C4H reaction | cinnamic acid | p-coumaric acid | Cinnamate 4-hydroxylase |  |  |
| 6 | 4CL reaction | p-coumaric acid | p-coumaroyl-CoA | 4-coumarate:CoA ligase complex |  |  |
| 7 | piperonylic acid inhibits C4H | — | — | — |  |  |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC12071552 / research

| field | value |
|---|---|
| requested scope | wall teichoic acid D-alanylation |
| PWML | **none generated** |
| release status | diagnostic_only |
| semantic evaluation | not_evaluated |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12071552/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12071552/research/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 36) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC12071552 / strict

| field | value |
|---|---|
| requested scope | wall teichoic acid D-alanylation |
| PWML | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12071552/strict/pathway.review_required.pwml |
| release status | review_required |
| semantic evaluation | failed |
| completeness / missing anchors | 0.4 / wall teichoic acid (WTA), lipoteichoic acid (LTA), ATP, dlt operon products, DltC, DltB |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12071552/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12071552/strict/final_mapped.json |
| organisms | Staphylococcus aureus, Staphylococcus aureus (MRSA N315) |
| enzymes | DltA complex |
| reactions / rows | 2 / 4 |
| graph valid | True (issues 0) |
| QA | ok=True errors=0 warnings=4 |
| RAG added rows | 0 (accepted 0 / rejected 30) |
| rows with NO provenance | **0** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | DltA-catalyzed D-Alanine adenylation | D-Alanine, Adenosine triphosphate | Pyrophosphate | DltA complex |  |  |
| 2 | DltA-catalyzed D-Serine adenylation | D-Serine, Adenosine triphosphate | Pyrophosphate | DltA complex |  |  |
| 3 | D-Alanine transport from cytoplasmic state to cell wall state | — | — | — |  |  |
| 4 | D-Serine transport from cytoplasmic state to cell wall state | — | — | — |  |  |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC12326985 / research

| field | value |
|---|---|
| requested scope | siderophore biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12326985/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12326985/research/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 0) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC12326985 / strict

| field | value |
|---|---|
| requested scope | siderophore biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12326985/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12326985/strict/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 0) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC12376012 / research

| field | value |
|---|---|
| requested scope | sphingolipid biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12376012/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12376012/research/final_mapped.json |
| organisms | Homo sapiens |
| enzymes | KDSR, Nogo B, ORMDL3, SPT/ORMDL complex, SPTLC1, SPTLC2, SPTssa, ceramidases, ceramide kinase, ceramide synthase, dihydroceramide Δ4-desaturase 1, galactosylceramide synthase, glucosylceramide synthase, sphingomyelin synthase, sphingomyelinase, sphingomyelinases, sphingosine 1-phosphate lyase, sphingosine kinase, sphingosine kinase-1, sphingosine-1-phosphate (S1P) lyase, sphingosine-1-phosphate (S1P) phosphatase, sphingosine-1-phosphate phosphatases |
| reactions / rows | 19 / 22 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 1 / rejected 200) |
| rows with NO provenance | **1** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | SPT reaction | L-serine, palmitoyl-CoA | 3-ketodihydrosphingosine (3-KDS) | SPT/ORMDL complex, Nogo B, SPTLC1, SPTLC2, SPTssa, ORMDL3 |  |  |
| 2 | KDSR reaction | 3-ketodihydrosphingosine (3-KDS) | dihydrosphingosine (DHS) | KDSR |  |  |
| 3 | Ceramide synthase reaction | dihydrosphingosine (DHS) | dihydroceramide | ceramide synthase |  |  |
| 4 | Dihydroceramide desaturase reaction | dihydroceramide | ceramide | dihydroceramide Δ4-desaturase 1 |  |  |
| 5 | Sphingomyelin synthase reaction | ceramide | sphingomyelin | sphingomyelin synthase |  |  |
| 6 | Glucosylceramide synthesis | ceramide | glucosylceramide | glucosylceramide synthase |  |  |
| 7 | Galactosylceramide synthesis | ceramide | galactosylceramide | galactosylceramide synthase |  |  |
| 8 | Ceramide kinase reaction | ceramide | ceramide-1-phosphate | ceramide kinase |  |  |
| 9 | Sphingosine kinase reaction | sphingosine | sphingosine-1-phosphate | sphingosine kinase |  |  |
| 10 | S1P lyase reaction | sphingosine-1-phosphate | ethanolamine phosphate, hexadecenal | sphingosine-1-phosphate (S1P) lyase, sphingosine 1-phosphate lyase |  |  |
| 11 | Ceramidase reaction | ceramide | sphingosine, fatty acid | ceramidases |  |  |
| 12 | Sphingosine-1-phosphate phosphatase reaction | sphingosine-1-phosphate | sphingosine | sphingosine-1-phosphate (S1P) phosphatase, sphingosine-1-phosphate phosphatases |  |  |
| 13 | Sphingomyelinase reaction | sphingomyelin | ceramide | sphingomyelinase, sphingomyelinases |  |  |
| 14 | Glycosphingolipid hydrolysis (glucosylceramide) | glucosylceramide | ceramide | — |  |  |
| 15 | Ceramide synthase acylation | dihydrosphingosine (DHS), fatty acyl-CoA | dihydroceramide | ceramide synthase |  |  |
| 16 | Sphingosine phosphorylation | sphingosine | sphingosine-1-phosphate | sphingosine kinase-1 |  |  |
| 17 | S1P dephosphorylation | sphingosine-1-phosphate | sphingosine | sphingosine-1-phosphate phosphatases |  |  |
| 18 | S1P lyase degradation | sphingosine-1-phosphate | ethanolamine phosphate, hexadecenal | sphingosine 1-phosphate lyase |  |  |
| 19 | Glycosphingolipid hydrolysis in lysosome | glycosphingolipids | ceramide | — |  |  |
| 20 | CERT-mediated ceramide transport | — | — | — |  |  |
| 21 | Vesicular ceramide transport to trans-Golgi | — | — | — |  |  |
| 22 | SPTLC2 contains pyridoxyl phosphate | — | — | — |  | **YES** |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC12376012 / strict

| field | value |
|---|---|
| requested scope | sphingolipid biosynthesis |
| PWML | **none generated** |
| release status | diagnostic_only |
| semantic evaluation | failed |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12376012/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12376012/strict/final_mapped.json |
| organisms | Homo sapiens, Arabidopsis thaliana |
| enzymes | 3-ketodihydrosphingosine reductase, S1P phosphatase complex, Serine Palmitoyltransferase complex, ceramidase, ceramide synthase, dihydroceramide Δ4-desaturase 1, sphingomyelin synthase 1, sphingomyelinase, sphingosine kinase |
| reactions / rows | 15 / 16 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 1 / rejected 200) |
| rows with NO provenance | **0** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | SPT reaction | L-serine, palmitoyl-CoA | 3-Dehydrosphinganine | Serine Palmitoyltransferase complex |  |  |
| 2 | KDSR reaction | 3-Dehydrosphinganine | Sphinganine | 3-ketodihydrosphingosine reductase |  |  |
| 3 | Ceramide synthase reaction | Sphinganine | ceramide | — |  |  |
| 4 | serine palmitoyltransferase reaction | palmitoyl-CoA, serine | 3-Dehydrosphinganine | Serine Palmitoyltransferase complex |  |  |
| 5 | ceramide synthase reaction | Sphinganine | dihydroceramide | ceramide synthase |  |  |
| 6 | dihydroceramide desaturase reaction | dihydroceramide | ceramide | dihydroceramide Δ4-desaturase 1 |  |  |
| 7 | Sphingomyelin synthase reaction | ceramide | sphingomyelin | — |  |  |
| 8 | Glucosylceramide synthase reaction | ceramide | glucosylceramide | — |  |  |
| 9 | Sphingosine kinase reaction | sphingosine | sphingosine-1-phosphate | sphingosine kinase |  |  |
| 10 | Sphingomyelin hydrolysis | sphingomyelin | ceramide | — |  |  |
| 11 | Glucosylceramide hydrolysis | glucosylceramide | ceramide | — |  |  |
| 12 | sphingomyelin synthase 1 reaction | ceramide, phosphorylcholine | sphingomyelin | sphingomyelin synthase 1 |  |  |
| 13 | ceramidase reaction | ceramide | sphingosine, fatty acid | ceramidase |  |  |
| 14 | sphingomyelinase reaction | sphingomyelin | ceramide, phosphorylcholine | sphingomyelinase |  |  |
| 15 | S1P phosphatase reaction | sphingosine-1-phosphate | sphingosine | S1P phosphatase complex |  |  |
| 16 | CERT-mediated ceramide transport from ER state to Golgi state | — | — | — |  |  |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC12542839 / research

| field | value |
|---|---|
| requested scope | riboflavin biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12542839/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12542839/research/final_mapped.json |
| organisms | Bacillus subtilis |
| enzymes | RibA, RibB, RibFC, RibG, RibH, RibM, RibR |
| reactions / rows | 11 / 14 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 200) |
| rows with NO provenance | **0** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | GTP cyclohydrolase II reaction | GTP | 2,5-diamino-6-ribosylamino-4(3H)-pyrimidinone-5'-phosphate (DARPP) | RibA |  |  |
| 2 | DARPP deaminase reaction | 2,5-diamino-6-ribosylamino-4(3H)-pyrimidinone-5'-phosphate (DARPP) | 5-amino-6-ribitylamino-2,4(1H,3H)-pyrimidinedione-5'-phosphate (ArPP) | RibG |  |  |
| 3 | ArPP dephosphorylation | 5-amino-6-ribitylamino-2,4(1H,3H)-pyrimidinedione-5'-phosphate (ArPP) | 5-amino-6-ribitylamino-2,4(1H,3H)-pyrimidinedione (ArP) | — |  |  |
| 4 | DHBP synthase reaction | ribulose-5-phosphate (Ru5P) | 3,4-dihydroxy-2-butanone-4-phosphate (DHBP) | RibA |  |  |
| 5 | DRL synthase reaction | 5-amino-6-ribitylamino-2,4(1H,3H)-pyrimidinedione (ArP), 3,4-dihydroxy-2-butanone-4-phosphate (DHBP) | 6,7-dimethyl-8-ribityllumazine (DRL) | RibH |  |  |
| 6 | riboflavin synthase reaction | 6,7-dimethyl-8-ribityllumazine (DRL) | riboflavin, 5-amino-6-ribitylamino-2,4(1H,3H)-pyrimidinedione-5'-phosphate (ArPP) | RibB |  |  |
| 7 | riboflavin kinase reaction | riboflavin | FMN | RibFC, RibM, RibR |  |  |
| 8 | FAD synthetase reaction | FMN, Adenosine triphosphate | FAD | RibFC |  |  |
| 9 | FAD synthetase reaction | FMN | FAD | RibFC |  |  |
| 10 | RibFC riboflavin kinase reaction | riboflavin, Adenosine triphosphate | FMN, Adenosine diphosphate | RibFC, RibR |  |  |
| 11 | RibFC FAD synthetase reaction | FMN, Adenosine triphosphate | FAD, Pyrophosphate | RibFC |  |  |
| 12 | FMN binding to FMN riboswitch | — | — | — |  |  |
| 13 | FMN riboswitch binding regulates transcription | — | — | — |  |  |
| 14 | ribFMN riboswitch regulation | — | — | — |  |  |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC12542839 / strict

| field | value |
|---|---|
| requested scope | riboflavin biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12542839/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC12542839/strict/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 0) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC13017326 / research

| field | value |
|---|---|
| requested scope | anthocyanin biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC13017326/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC13017326/research/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 0) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC13017326 / strict

| field | value |
|---|---|
| requested scope | anthocyanin biosynthesis |
| PWML | **none generated** |
| release status | None |
| semantic evaluation | None |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC13017326/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC13017326/strict/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 0) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC3480714 / research

| field | value |
|---|---|
| requested scope | cobalamin biosynthesis |
| PWML | **none generated** |
| release status | diagnostic_only |
| semantic evaluation | not_evaluated |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC3480714/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC3480714/research/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 73) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC3480714 / strict

| field | value |
|---|---|
| requested scope | cobalamin biosynthesis |
| PWML | **none generated** |
| release status | diagnostic_only |
| semantic evaluation | not_evaluated |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC3480714/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC3480714/strict/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 200) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC7232280 / research

| field | value |
|---|---|
| requested scope | molybdenum cofactor biosynthesis |
| PWML | **none generated** |
| release status | diagnostic_only |
| semantic evaluation | not_evaluated |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC7232280/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC7232280/research/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 200) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC7232280 / strict

| field | value |
|---|---|
| requested scope | molybdenum cofactor biosynthesis |
| PWML | **none generated** |
| release status | diagnostic_only |
| semantic evaluation | failed |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC7232280/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC7232280/strict/final_mapped.json |
| organisms | Neurospora crassa, Arabidopsis thaliana |
| enzymes | MPT synthase complex, NIT-7A, NIT-7B, NIT-9E, NIT-9G |
| reactions / rows | 5 / 6 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 200) |
| rows with NO provenance | **0** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | NIT-7A conversion of GTP to 3',8-cH2 GTP | GTP | 3',8-cH2 GTP | NIT-7A |  |  |
| 2 | NIT-7B conversion of 3',8-cH2 GTP to cPMP | 3',8-cH2 GTP | cPMP | NIT-7B |  |  |
| 3 | MPT synthase conversion of cPMP to MPT | cPMP | MPT | MPT synthase complex |  |  |
| 4 | NIT-9G adenylation of MPT to MPT-AMP | MPT | MPT-AMP | NIT-9G |  |  |
| 5 | NIT-9E conversion of MPT-AMP to Moco | MPT-AMP | Moco | NIT-9E |  |  |
| 6 | cPMP export from mitochondrial state to cytoplasmic state | — | — | — |  |  |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC8510960 / research

| field | value |
|---|---|
| requested scope | terpenoid indole alkaloid biosynthesis |
| PWML | **none generated** |
| release status | diagnostic_only |
| semantic evaluation | not_evaluated |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC8510960/research/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC8510960/research/final_mapped.json |
| organisms | (none) |
| enzymes | (none) |
| reactions / rows | 0 / 0 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 0) |
| rows with NO provenance | **0** |

**Reactions**

_none retained_


**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**


---

## PMC8510960 / strict

| field | value |
|---|---|
| requested scope | terpenoid indole alkaloid biosynthesis |
| PWML | **none generated** |
| release status | diagnostic_only |
| semantic evaluation | failed |
| completeness / missing anchors | None / none |
| source text | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC8510960/strict/stage1_payload.json |
| final payload | C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC8510960/strict/final_mapped.json |
| organisms | Catharanthus roseus, Arabidopsis thaliana |
| enzymes | G10H, PRX1 complex, SGD, STR, TDC complex |
| reactions / rows | 5 / 7 |
| graph valid | None (issues None) |
| QA | ok=None errors=0 warnings=0 |
| RAG added rows | 0 (accepted 0 / rejected 0) |
| rows with NO provenance | **0** |

**Reactions**

| # | name | substrates | products | enzymes | RAG | no prov |
|---|---|---|---|---|---|---|
| 1 | TDC reaction | tryptophan | tryptamine | TDC complex |  |  |
| 2 | G10H reaction | geraniol | 10-hydroxygeraniol | G10H |  |  |
| 3 | STR reaction | secologanin, tryptamine | strictosidine | STR |  |  |
| 4 | SGD reaction | strictosidine | strictosidine aglycone | SGD |  |  |
| 5 | Catharanthine-vindoline coupling | catharanthine, vindoline | anhydrovinblastine | PRX1 complex |  |  |
| 6 | Strictosidine export from vacuole to cytosol | — | — | — |  |  |
| 7 | Catharanthine secretion to leaf surface | — | — | — |  |  |

**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / `MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`

**Notes:**
