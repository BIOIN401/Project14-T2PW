# 07b KEGG licence and access, and other databases a public T2PW server would call

Retrieved 2026-11-10 by plain GET (generic browser-style client, about 25 requests, 2 s spacing). Raw pages cached in C:/t/ev-oa/kegg/. Labels: L = quoted from the page. G = my inference. This is NOT legal advice; the conclusions marked "risk" are for the product owner.

## 1. NOT ESTABLISHED

1. Whether a T2PW public server may call rest.kegg.jp on behalf of users. The rule text exists (section 2) but the status of a non-profit research web server that is free to the public is not spelled out; it is a judgement on "academic user" and "providing services". Not a legal conclusion I can make.
2. Whether KEGG compound/gene IDs inside a user-downloaded KGML-style file are "redistribution of KEGG data". No KEGG page addresses identifiers-only output. Not established either way. Only Kanehisa Labs / Pathway Solutions can answer; their licence forms refuse direct email (inquiry form only).
3. The text of the "Non-academic use of KEGG" guideline. legal.html names it, but the link in the cached page does not resolve to a separate page (the three guessed URLs returned 404). The commercial position is taken from legal.html and the Pathway Solutions pages instead.
4. The Service Provider Agreement and End User Agreement themselves (PDFs/forms behind the request form). Not fetched; the exact clauses on third-party users, derived files and identifiers are unseen.
5. Whether "3 calls per second" is the only throttle, and how "blocked" is applied. Only the one sentence below exists.
6. PubChem terms: pubchem.ncbi.nlm.nih.gov served a reCAPTCHA challenge page to a plain client, so no PubChem sentence was obtained from PubChem. (NCBI-wide policy below is the nearest source.)
7. UniProt: the help/license page is a JavaScript app; I used the licence file on UniProt's FTP instead (same licence statement, different URL).
8. Ensembl: www.ensembl.org returned 403 to curl; the text below is from the archive mirror reached by redirect (jun2026.archive.ensembl.org). Ensembl REST rate limits not checked.

## 2. KEGG quotes (L)

(a) Academic REST API use. https://www.kegg.jp/kegg/rest/ (retrieved 2026-11-10):
- "KEGG API is a REST-style Application Programming Interface to the KEGG database resource."
- "Restriction: KEGG API at rest.kegg.jp is made available only for academic use by academic users. Non-academic users are requested to follow the guidline shown in Non-academic use of KEGG."
- "Please limit your API calls up to 3 times per second. Otherwise, your access will be blocked."

https://www.kegg.jp/kegg/legal.html (page says "Last updated: October 1, 2024"):
- "KEGG is an original database product, copyright Kanehisa Laboratories."
- "Academic users may freely use the KEGG website at https://www.kegg.jp/ or its mirror site at GenomeNet https://www.genome.jp/kegg/."
- "Academic users who utilize KEGG for providing services are requested to obtain an academic service provider license, which is included in the KEGG FTP academic subscription."
- "Non-academic users must understand that KEGG is not a public database, nor is it a publicly funded database. Non-academic use of KEGG requires a commercial license. Please contact Pathway Solutions, which operates KEGG FTP sites and a KEGG mirror website, for license agreements."

https://www.kegg.jp/kegg/download/ (FTP):
- "The KEGG FTP site for academic users is available to subscribers only."
- "The weekly updated FTP site contains the entire set of KEGG data" (includes a kgml directory).

(b) Service providers. https://www.pathway.jp/en/academic.html:
- "The KEGG FTP Academic Subscription is a paid service managed by Pathway Solutions for those academic users who wish to bulk-download KEGG data and/or to provide outside services using KEGG data (see Academic use of KEGG)."
- "Service Provider -- The service provider type is a form of academic service provider license given to a research group who develops and provides software tools, web services, etc. utilizing KEGG."
- "Requests made by a non-research group, such as a library or an IT service department, are not accepted."
- Subscription is invoice-based; fee is on a request form (not fetched, behind the form).

Commercial. https://www.pathway.jp/en/licensing.html:
- "Please honor the request made by Kanehisa Laboratories (see Non-academic use of KEGG) and obtain a license for any commercial use of KEGG. Specifically, please do not use the KEGG website unless you have an End User License Agreement with the web option (EULA type 2)."
- Service Provider License: "Includes rights to provide services to authorized end users (commercial end users with valid end user licenses or academic end users)".
- "The terms and conditions of the End User License Agreement (EULA) and the Service Provider License Agreement (SPLA) are non-negotiable."

FAQ https://www.pathway.jp/en/licensing_faq.html:
- "These sites are made freely available following the guideline of Kanehisa Laboratories, which makes a request of academic use only but does not enforce any restriction."
- "The web service is meant for browsing data and not for obtaining data."

(c) Figures. https://www.kegg.jp/feedback/copyright.html: "Please use the form below to request copyright permission of using KEGG pathway maps and other images as figures in academic publications." (This concerns map images, not IDs.)

KGML. https://www.kegg.jp/kegg/xml/: "The KGML file may be obtained from the 'Download KGML' link for each pathway map or by the get method of KEGG API." KGML is described as the "exchange format of the KEGG pathway maps"; it carries KEGG map layouts, so redistributing KEGG-generated KGML files is the most clearly covered-by-copyright use.

## 3. What the REST API offers (L; https://www.kegg.jp/kegg/rest/keggapi.html)

Operations: "info | list | find | get | conv | link | ddi". Base: https://rest.kegg.jp/<operation>/...
- find: "/find/compound/C7H10O5/formula", "/find/genes/shiga+toxin" (keyword search of "entry identifier and associated name fields"); also exact_mass, mol_weight. So name-to-ID lookup (find compound) is the same API, same terms (G: no separate terms exist on the page).
- list: "/list/pathway", "/list/compound", up to 10 ids for definitions of `/list/<dbentries>`.
- get: entries, and options "aaseq | ntseq | mol | kcf | image | conf | kgml | json"; "/get/hsa00600/kgml" "retrieves the kgml file of a pathway map"; input limited to "one pathway entry with the image or kgml option".
- conv: outside ids to/from KEGG ids. Outside db for compounds: "compound | glycan | drug" with "pubchem | chebi"; for genes "ncbi-geneid | ncbi-proteinid | uniprot". Example: /conv/compound/chebi.
- link: cross-references, e.g. compound to reaction to ko to pathway; "reaction" and "enzyme" included.
- Bulk list calls (`/list/compound`, `/conv/compound/chebi`) are single requests that return the whole table, so a local one-off cache of the table is possible within 3 req/s, but see risk R4 on reuse.

## 4. Other databases (one quoted line each, L)

| DB | Quote | URL |
|---|---|---|
| UniProt | "We have chosen to apply the Creative Commons Attribution 4.0 International (CC BY 4.0) License ... to all copyrightable parts of our databases." | https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/LICENSE |
| ChEBI | "The data on this website is available under the Creative Commons License ( CC BY 4.0 ), and governed by EMBL-EBI's terms of use" | https://www.ebi.ac.uk/chebi/ |
| Rhea | "Rhea is available under the Creative Commons Attribution 4.0 International (CC BY 4.0) License. This means that you are free to copy, distribute, display and make commercial use of the database in all legislations, provided you credit Rhea." | https://www.rhea-db.org/help/license-disclaimer |
| NCBI | "NCBI itself places no restrictions on the use or distribution of the data contained therein." (Submitters may impose limits on some records.) | https://www.ncbi.nlm.nih.gov/home/about/policies/ |
| Ensembl | "Ensembl imposes no restrictions on access to, or use of, the data provided and the software used to analyse and present it." | https://www.ensembl.org/info/about/legal/disclaimer.html (via jun2026.archive.ensembl.org) |
| PubChem | NOT OBTAINED (CAPTCHA). NCBI-wide policy above is the closest source (PubChem is NCBI-hosted); PubChem's own statement unverified. | https://pubchem.ncbi.nlm.nih.gov/ |

EMBL-EBI general terms (https://www.ebi.ac.uk/about/terms-of-use): "EMBL-EBI expects attribution (e.g. in publications, services or products) for the use of any of its Data Resources and Tools in accordance with good scientific practice." Applies to ChEBI, Rhea (Rhea is SIB/EBI), and UniProt data via EBI. Attribution means a visible credit line and citation on the server and in output file headers.

## 5. Risk assessment (product-owner call; G unless quoted)

R1. Server calls rest.kegg.jp live for each user request (name to id via find/conv). Risk: HIGH. The page says the API is "made available only for academic use by academic users". A public server serves users who are not verified academic, and calls originate from our server, not the user. The "academic service provider license" sentence points to the paid FTP Service Provider route for exactly this. Throttle (3/s) would be shared across all users and the server IP could be blocked.

R2. Server uses a local, pre-built KEGG id table (from a one-off API pull, or FTP subscription) to resolve names to ids. Risk: MEDIUM with FTP Service Provider subscription (explicit right to "provide services ... utilizing KEGG"), HIGH if built by bulk-scraping the free API (the FAQ says the web service "is meant for browsing data and not for obtaining data" and the FTP is the sanctioned bulk route).

R3. Emitting a KGML-format file whose content is only T2PW's own pathway (our reactions, compound ids, gene ids), with no KEGG map layout, coordinates, names or images. Risk: LOW to MEDIUM. A KGML file is a schema; the schema is public (DTD). Ids like C00001 are KEGG accessions. Whether accessions alone are protectable "KEGG data" is not addressed on the pages (NOT ESTABLISHED 2). Risk rises if the ids were obtained from the API for our users, rises further if names, KO assignments or map layout are copied from KEGG.

R4. Emitting KEGG-sourced content (get .../kgml, KEGG names, KO/EC assignments, map coordinates) in user files. Risk: HIGH. Copyright statement "KEGG is an original database product, copyright Kanehisa Laboratories" and the copyright-permission form cover maps and images; redistribution of KGML is the clearest case.

R5. User supplies KEGG ids (pasted in the paper, in the input, or typed by the user), server only passes them through. Risk: LOW (G: server does not obtain them from KEGG).

R6. ChEBI/UniProt/Rhea/Ensembl/NCBI calls and redistribution of their ids and cross-references. Risk: LOW. Open licences with attribution (CC BY 4.0 for UniProt, ChEBI, Rhea; no restriction for NCBI, Ensembl). Keep attribution lines and honour third-party-constraint notes. Note: ChEBI/UniProt cross-reference columns that list KEGG ids are themselves KEGG-derived; whether a CC BY database may pass on KEGG accessions under CC BY is NOT ESTABLISHED (G: EBI/UniProt redistribute them openly, which suggests they consider accessions linkable, but this is not KEGG's statement).

## 6. KEGG-independent identifiers a KGML export could use

- Compounds: ChEBI id (CC BY 4.0), PubChem CID. The API itself says conv supports "pubchem | chebi" mapping, so KEGG-to-ChEBI is a published relationship. KGML <entry name="cpd:C00001"> expects KEGG form, so strict KGML consumers (Cytoscape KEGGscape, pathview, KEGGgraph) will not resolve ChEBI. A KGML-like file could carry `name="chebi:15377"` or a `<graphics>`-only entry; consumers may reject unknown prefixes (G, untested).
- Proteins/genes: UniProt accession (CC BY 4.0), NCBI GeneID, Ensembl id.
- Reactions: Rhea id (CC BY 4.0); MetaCyc id (licence NOT checked in this task).
- Offline mapping: build a local name/ChEBI/UniProt/Rhea table, no KEGG calls. KEGG ids then appear only when (i) the user supplies them, or (ii) the owner buys the Service Provider licence. ChEBI itself exposes KEGG COMPOUND cross-references; reuse of those is covered by ChEBI's CC BY 4.0 but see R6 caveat.

## 7. WHAT TO DO

1. Owner decision, no code needed: choose one of (A) KGML export uses KEGG ids only when user-supplied, default ids ChEBI/UniProt; (B) apply for the KEGG FTP Academic Service Provider licence (request form; paid, invoice, signed by a researcher; entity must be an academic research group); (C) skip KEGG export. A is the lowest risk and no cost; B is the route KEGG's own page names for "providing services".
2. Do not call rest.kegg.jp from the public server until the owner has a written answer from Kanehisa Labs / Pathway Solutions (inquiry form; they do not accept direct email). Question to ask: "May a free, academic-hosted web server that converts papers to pathway files look up KEGG compound/gene ids via the REST API per user request, and may the resulting files contain those ids (and no map layout)?" Record the reply in DECISIONS.md (owner only).
3. If no answer by the time the NAR letter of intent is due, ship option A, and label KEGG as "export format only, ids user-supplied or ChEBI/UniProt", with no KEGG map content ever embedded.
4. Add attribution lines to every export header and to the About page for UniProt, ChEBI, Rhea, NCBI, Ensembl (the CC BY ones require credit).
5. Re-fetch PubChem's terms from a normal browser (CAPTCHA blocked the check) before depending on it; confirm MetaCyc terms if MetaCyc ids are wanted.
6. Test, offline, whether Cytoscape/KEGGgraph accept a KGML with non-KEGG name prefixes before promising a "KGML export" at all. This is a format question, not a licence one.
