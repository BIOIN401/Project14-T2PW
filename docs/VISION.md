# T2PW product vision: next phase

This document holds the product owner's (PO's) direction for the next phase of T2PW. It was written by the
orchestrator from the PO's own words, which are quoted in the last section. Every agent that plans or builds
product features reads it. The PO approves changes to it; agents do not edit it on their own.

Related documents:
- `docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md` defines biological correctness and outranks every benchmark.
- `docs/webserver/PLAN.md` covers the server and its deadlines.
- `docs/webserver/DECISIONS.md` holds the rulings, D-W28 onwards for this phase.

Status: **under discussion**, started 2026-11-17. The open points are in the last sections.

---

## 1. The goal

A user gives T2PW a paper, or a request about a pathway. T2PW returns a pathway that is:
1. **Correct.** The right reactions, enzymes, transports, interactions and compartments, with correct
   database identifiers.
2. **Faithful.** It is true to the paper. It is also consistent with the rest of the peer-reviewed literature
   and the curated databases.
3. **Attractive.** It reads clearly as a pathway, both in our own rendering and in PathWhiz.
4. **Usable in every format** the user needs: PWML for PathWhiz, SBML, SBGN, and the web diagram.

The public server must be live and working for the NAR Web Server Issue letter of intent (Dec 15, 2026), and
tested and ready for the paper (around Feb 15, 2027).

## 2. One big payload, many small outputs

```
paper(s) ──► 1. EXTRACT everything ──► 2. GROUND ids ──► 3. EXTEND with other papers/DBs
                                                              │
                                                              ▼
                                                  BIG PAYLOAD (provenance on every element)
                                                              │
                                     4. SCOPE (set by the LLM from the request, adjustable later)
                                                              │
                                                              ▼
                                                     scoped view of the payload
                                                              │
                                     5. GENERATE ON DEMAND: web diagram, PWML, SBML, SBGN, ...
```

1. **Extract for recall.** Get every piece of information the paper has: reactions, enzymes, transports,
   interactions and regulation, compartments, cofactors and regulators. Each one is tied to its supporting
   quote. At this stage it is better to collect too much than to miss something.
2. **Ground.** Give every entity the correct identifier: compounds, proteins and complexes (UniProt and the
   like), cell components (GO) and cell types (CL).
3. **Extend.** Look outside the paper to other **peer-reviewed** sources, abstract or full text, and to the
   curated databases (PathBank, Reactome, WikiPathways). Use them to fill gaps and to connect the pathway
   into a cohesive whole: the steps before and after, missing enzymes, compartments, transports.
   - Every element records where it came from: *this paper*, *another paper* (with its citation and quote),
     or *a database*.
   - Outside information never enters without that label.
4. **Scope.** The managing LLM reads the user's request and decides the depth the user wants. The output is
   a view cut from the big payload.
   - **Default scope: the paper plus the immediate context** needed to make it a connected, readable pathway.
   - Other levels: paper only, and the full pathway.
   - **The scope can change after the output exists.** For example, a user asks "can you make this more
     detailed?" and the LLM takes a larger part of the payload that already exists. The pipeline does not
     run again.
5. **Generate on demand.** We do not build every file format up front. The web view shows the pathway, and a
   file is generated when the user asks to download it. Every format is generated from the same scoped view,
   so the formats can never disagree.

The final payload built in 2026-11 is the first version of this big payload. The RAG and paper-fetch code
already exist (`docs/pwml_recovery_sprint/MASTER_PLAN.md` § 2). The new parts are:
- extension from outside sources, with provenance on every element;
- the scope layer, including re-scoping after the output exists;
- file generation on demand.

## 3. How we check it (now, during development)

These checks belong to development and testing. In production the system will not call external validators
on every job; we will rely on the system once it is proven.

- **File validity.** Each format goes through real tools:
  - libSBML and the SBML online validator for SBML;
  - the libSBGN schema and schematron for SBGN;
  - a BioPAX validator, if we add BioPAX;
  - a real PathWhiz import for PWML (our patched PathWhiz, `t2pw/*` branches).
- **Consistency across formats.** Every format of one pathway contains the same reactions, entities and
  compartments.
- **Correctness and faithfulness.**
  - The paper-only view is checked against the paper.
  - The extended view is checked against the curated databases and the cited papers.
- **Attractiveness.**
  - Layout metrics: overlaps, edge crossings, steps against reading order, label fit, duplicated compounds,
    and how far each enzyme sits from its reaction.
  - Screenshots in PathWhiz and in our own renderer, which the orchestrator itself looks at and shows to
    the PO as a gallery.
  - For SBML and SBGN, a standard third-party viewer to confirm they draw well outside our tools.
- **The orchestrator checks real outputs itself.** It imports pathways into the local PathWhiz harness,
  looks at them, and shows the PO the results. Passing tests is not enough.

## 4. The benchmark (for the paper)

- **Main benchmark: paper only.** What a paper covers is what we score against. If a paper covers only part of
  the TCA cycle, the benchmark for that paper is only that part.
- **A smaller full-view set.** A few pathways are scored against the full curated pathway: how much of
  everything known about the pathway did the software recover?

## 5. Speed and cost

The PO wants lower time and cost per job.
- **Lab vLLM.** The lab owns a vLLM server on an A6000 GPU. T2PW will be wired to it as an LLM provider and
  tested against the current provider on speed, cost and quality, A/B on identical inputs.
- **Server.** The PO has access to the server that will host T2PW, so deployment work can start now.
- The three Stage-1 draws already run concurrently (2026-11).

## 6. Standing rules that still apply

- **No regression.** Text→PWML output never gets worse. Anything that cannot be shown to cause no loss
  is deferred.
- **The product contract outranks every test or benchmark.**
- **PathWhiz changes are stacked `t2pw/*` branches.** They are pushed and never merged by us; the PO raises
  the pull requests.
- **Never read, print or commit secrets** (`.env`). Never commit caches.
- **Every test or pipeline run goes through the bounded wrapper.** One heavy job at a time.

## 7. Open points (to settle with the PO)

- **Provenance in PathWhiz.** Should outside-sourced elements be drawn differently there, or marked only in
  our own view?
- **Re-scoping.** How does the user ask for it (chat, a control, or both), and how far beyond "full pathway"
  can it go?
- **Limits per job.** How many outside papers per pathway, and the time and cost budget per job (the server
  cap of 12 or 15 minutes is still open).
- **The lab vLLM.** Which model it serves, and its endpoint and access method. The PO supplies these; secrets
  go in `.env` only.
- **The server.** The host, access method, OS, domain and HTTPS, the read-only PathBank DB user, and SMTP.

## 8. The PO's words

2026-11-17, first message:
> "we want it to not only be correct but also look like an attractive pathway on both our internal rendering
> and the pathwhiz rendering, all the other file types also need to be checked so u can make sure of make use
> of software to check the output of the files. You can look up an SBML software and make sure that the SBML
> file output is correct as well. Of course, we want everything to look attractive, be accurate, and be
> faithful to the paper, but also to other papers as well. We need to look at the pathway as a whole and bring
> it in and relate it to other papers as well. In this sense, of course, there will be times where you have
> reactions that go way beyond the scope of the paper. In which case, I want to add something that maybe
> allows the user to adjust the scope themselves. Perhaps we can give an output constrained to the paper to
> some extent. The LLM should be able to take the prompt from the user and glean what kind of depth that they
> were actually looking for. And then make that as the depth. But during our extraction phase, we should really
> be trying to get as much information out of the paper. Synthesize it with the correct protein IDs and all of
> that stuff, all the transports, all the interactions, then look out into other papers, get that information,
> and create a cohesive pathway. And then use this to then create the smaller files that we then need. So we
> use this big payload to make our PWML files, all these other files. And then, of course, everything needs to
> look good as well. So we need to test and make sure our outputs are attractive looking."

2026-11-17, second message:
> "the user should be able to set the scope even afterwards as well. So they can get an output and be like, can
> you make this more detailed? And in this manner, the LLM that's managing this should be able to go back and
> take a larger piece of the payload that we have. And we can generate every file, but we don't need to do that
> right away. We can do that later. We can have the user request to download the file, and then we generate it.
> These checks are just right now for us. [...] like the validators like lib.sbml, that's just for now. Later
> [...] we can just have faith in our system. Yeah, I want you to check the actual outputs of the pathways
> yourself, because you have access to SPMDB. You can do that all yourself and then actually show me the
> results. [...] The default scope should be exactly that, the paper plus some immediate context. Outside
> papers, anything that is from a good source, like a proper peer-reviewed source. It doesn't matter if it's an
> abstract or the full text. We want to improve our time and cost per job. That's why I've gotten access to a
> VLLM that runs on an A6000 that the lab owns. So we can test on that. [...] I also have access to the server,
> and so we can also do that now as well. the benchmark should basically be on like the paper only mostly, but
> we also want to be able to do a full view. [...] main are going to be paper only, and then a few things go
> full view, how much did this software actually get from all the knowledge that humanity has, basically."
