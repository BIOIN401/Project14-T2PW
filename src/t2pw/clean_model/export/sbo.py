"""SBO terms the SBML exporter may write. Every id was checked against EBI OLS4 on 2026-11-10; the label is the OLS label."""

from __future__ import annotations

SBO = {
    "SBO:0000290": "physical compartment",
    "SBO:0000247": "simple chemical",
    "SBO:0000252": "polypeptide chain",
    "SBO:0000253": "non-covalent complex",
    "SBO:0000246": "information macromolecule",
    "SBO:0000285": "material entity of unspecified nature",
    "SBO:0000013": "catalyst",
    "SBO:0000019": "modifier",
    "SBO:0000020": "inhibitor",
    "SBO:0000459": "stimulator",
    "SBO:0000176": "biochemical reaction",
    "SBO:0000655": "transport reaction",
    "SBO:0000169": "inhibition",
    "SBO:0000170": "stimulation",
    "SBO:0000177": "non-covalent binding",
    "SBO:0000216": "phosphorylation",
    "SBO:0000330": "dephosphorylation",
    "SBO:0000224": "ubiquitination",
    "SBO:0000179": "degradation",
    "SBO:0000172": "catalysis",
    "SBO:0000168": "control",
    "SBO:0000526": "protein complex formation",
    "SBO:0000342": "molecular or genetic interaction",
}

ENTITY_SBO = {"compound": "SBO:0000247", "protein": "SBO:0000252", "complex": "SBO:0000253",
              "nucleic_acid": "SBO:0000246", "undeclared": "SBO:0000285"}

# vocab.KINDS -> SBO. Kinds with no specific term get the generic interaction term; no sign is read from the SBO term.
KIND_SBO = {
    "activation": "SBO:0000170", "inhibition": "SBO:0000169", "phosphorylation": "SBO:0000216",
    "dephosphorylation": "SBO:0000330", "binding": "SBO:0000177", "complex_formation": "SBO:0000526",
    "ubiquitination": "SBO:0000224", "degradation": "SBO:0000179", "catalysis": "SBO:0000172",
    "expression_regulation": "SBO:0000168", "modulation": "SBO:0000168",
}
GENERIC_INTERACTION = "SBO:0000342"

ROLE_SBO = {"catalyst": "SBO:0000013", "inhibitor": "SBO:0000020", "activator": "SBO:0000459",
            "stimulator": "SBO:0000459"}
