"""C-125 seam flip table: STOP / PROCEED for a fixed corpus, base tree vs tip.

Orchestration evidence, not pipeline code. Round-1 review of C-125 reproduced
twelve seam flips the first implementation admitted by accident, so the flip set
is now measured rather than argued. Run it TWICE -- once in a tree exported at the
base SHA, once at the tip -- and join the two outputs::

    <py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label flip-base       --timeout 600 --json <report> -- <py> -u <this file>

Run it a THIRD time against the previous tip as well. Round 2 of this card closed
the round-1 flips with a rule that opened a new false-accept class, and only a
base / previous-tip / tip comparison shows a pair moving refuse -> accept between
two accepted versions.

Every PROCEED-at-tip that is not one of the two target rescues, a documented
narrowing, or a documented residual open-world case is a regression.
"""
from types import SimpleNamespace

from t2pw.batch import driver

FUM = "fumonisin biosynthesis"
SAP = "steroidal saponin biosynthesis"
SAP0 = (
    "steroidal saponin (polyphyllin) biosynthesis in Paris polyphylla, "
    "focusing on UGT-mediated 3-O-glucosylation"
)

CASES = [
    # the two target rescues
    (FUM, "fumonisin B1 biosynthesis"),
    (SAP, SAP0),
    # round-1 reviewer flips
    (FUM, "suppression of fumonisin B1 biosynthesis"),
    (FUM, "fumonisin B1 biosynthesis repression"),
    (FUM, "fumonisin B1 biosynthesis gene cluster silencing"),
    (FUM, "fumonisin B1 biosynthesis knockdown"),
    (FUM, "fumonisin B1 biosynthesis blockade"),
    (FUM, "fumonisin B1 biosynthesis deficiency"),
    (FUM, "fumonisin B1 biosynthesis drug screening"),
    (FUM, "antifungal agents that abolish fumonisin B1 biosynthesis"),
    (FUM, "non-fumonisin B1 biosynthesis"),
    (FUM, "loss of fumonisin B1 biosynthesis"),
    (FUM, "review of fumonisin B1 biosynthesis"),
    ("siderophore uptake", "siderophore efflux"),
    (FUM, "fumonisin B1 biosynthesis turnover and clearance"),
    # round-2 reviewer flips: compound and activity names a PREFIX test ate
    ("nonribosomal peptide biosynthesis", "antimicrobial peptide biosynthesis"),
    ("nonribosomal peptide biosynthesis", "antifungal peptide biosynthesis in Bacillus"),
    ("antifungal polyketide biosynthesis", "antitumor polyketide biosynthesis"),
    ("antimycin biosynthesis in Streptomyces", "antitumor biosynthesis in Streptomyces"),
    ("methyl nonanoate biosynthesis", "methyl nonadecanoate biosynthesis"),
    ("nonulosonic acid biosynthesis", "nonanoate acid biosynthesis"),
    (
        "nonribosomal peptide biosynthesis",
        "nonribosomal peptide (tyrocidine) biosynthesis in Brevibacillus brevis",
    ),
    # inflections closed in round 3
    ("heme biosynthesis", "heme biosynthesis inhibited"),
    ("heme biosynthesis", "defective heme biosynthesis"),
    ("heme biosynthesis", "heme biosynthesis dysregulation"),
    ("heme biosynthesis", "heme biosynthesis downregulated"),
    # the documented NON-MONOTONE consequence of the study-design family
    ("screen heme biosynthesis", "review heme biosynthesis"),
    ("review article biosynthesis", "survey article biosynthesis"),
    # neighbours of those, same shapes
    (FUM, "fumonisin B1 biosynthesis inhibitor screening"),
    (FUM, "attenuation of fumonisin B1 biosynthesis"),
    (FUM, "disruption of the fumonisin B1 biosynthesis gene cluster"),
    (FUM, "impaired fumonisin B1 biosynthesis"),
    (FUM, "ablation of fumonisin B1 biosynthesis"),
    (FUM, "antagonists of fumonisin B1 biosynthesis"),
    (FUM, "fumonisin B1 biosynthesis knockout mutants"),
    (FUM, "survey of fumonisin B1 biosynthesis"),
    (FUM, "downregulation of fumonisin B1 biosynthesis"),
    ("siderophore transport", "siderophore efflux"),
    ("siderophore efflux", "siderophore uptake"),
    ("siderophore uptake", "siderophore B uptake in Pseudomonas aeruginosa"),
    # genuine specializations that must still be rescued
    (FUM, "fumonisin B1 biosynthetic pathway"),
    ("fumonisins biosynthesis", "fumonisin B1 biosynthesis"),
    (FUM, "fumonisin B1 biosynthesis in Fusarium verticillioides"),
    (SAP, "steroidal saponin (polyphyllin) biosynthesis"),
    # must keep stopping
    ("steroidal glycoalkaloid biosynthesis", "potato solanidane glycoalkaloid biosynthesis"),
    (FUM, "fumonisin B1 degradation"),
    ("fumonisin degradation", "fumonisin B1 biosynthesis"),
    ("fumonisin B1 biosynthesis", FUM),
    ("lipid A biosynthesis", "cholesterol biosynthesis"),
    ("lipid A biosynthesis", "lipid biosynthesis"),
    # residual open-world exposure, documented
    (FUM, "mathematical modelling of fumonisin B1 biosynthesis"),
    (FUM, "evolutionary origin of fumonisin B1 biosynthesis"),
    (FUM, "in vitro reconstitution of fumonisin B1 biosynthesis"),
    (FUM, "prevention of fumonisin B1 biosynthesis"),
    (FUM, "editorial on fumonisin B1 biosynthesis"),
    (FUM, "fumonisin B1 biosynthesis eliminated by fungicide treatment"),
]


def decide(requested: str, observed: str) -> str:
    paper = {"paper_id": "PMC-X", "requested_pathway": requested, "requested_organism": ""}
    at = SimpleNamespace(session_state={"pathway_context": {"pathway_name": observed}})
    outcome = driver.RunOutcome(paper_id="PMC-X")
    return "STOP" if driver._reconcile_stage0_scope(at, paper, outcome) else "PROCEED"


for index, (requested, observed) in enumerate(CASES):
    print(f"{index:02d}\t{decide(requested, observed)}\t{requested}\t{observed}")
