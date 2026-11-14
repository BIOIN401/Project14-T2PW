"""Final payload phase 3 (B2): map location / cell-type / tissue names onto PathWhiz vocabulary rows.

Used by the PWML writer only under T2PW_FINAL_PAYLOAD. The importer looks a vocabulary item up by name (and by ontology id when
one is written); a miss creates a row and a row needs an ontology id, so a name PathWhiz lacks fails the import. Three outcomes:

* ``exact``   the name is already a PathWhiz row (case-insensitive, as MySQL compares): written unchanged.
* ``mapped``  a reviewed synonym below names the same entity as an existing row: the row's name is written and the ontology id
              is left out, so the importer finds the row by name alone (the same way the 89 existing hits are found).
* ``mapped_by_id`` the data carries an ontology id (looked up online, D-W20a) that PathWhiz already holds under another name: the
              row's name is written without the id (a second row with that id would fail the unique-id validation).
* ``new_row`` / ``unmapped``  no row fits: the name is left unchanged; ``new_row`` when the data already carries an ontology id
              (PathWhiz can create the row), ``unmapped`` when it does not. No id is ever invented.

PATHWHIZ_ROWS is the name -> ontology id content of the production tables ``subcellular_locations`` (77), ``cell_types`` (43)
and ``tissues`` (53), taken from the harness seed ``vocab_inserts.sql`` (grep of the PathBank dump, 2026-11). Re-extract it when
production gains rows. Every SYNONYMS target must be a key of PATHWHIZ_ROWS (tests/test_final_payload_vocabulary.py).
Only the same biological entity is mapped; no "closest match" (a tissue is never mapped to a different organ, nor a cell line or a
subtype to its parent type).

Phase 5 additions: (B) a plural name whose singular is a PathWhiz row is mapped to it (same entity); (E) inner and outer membrane are
organism-aware. PathWhiz rows ``Inner Membrane`` / ``Outer Membrane`` are the bacterial cell membranes (98 % / 99 % of the 1285 / 1243
production states using them belong to prokaryotes; mitochondria have their own rows; evidence/final_payload_p5/ontoBE/). Their stored
GO ids (GO:0070258 "inner membrane pellicle complex", GO:0043165 "outer membrane assembly") are other terms, so the rows are identified
by name only. Called with ``organisms`` (a set of organism kinds of the states that use the location), the names ``inner membrane``,
``outer membrane`` and their ``(IM)`` / ``(OM)`` spellings map to those rows only when the organisms are exactly {"bacterium"}; for any other
or undetermined organism they are NOT written under a row name (the name would bind to the bacterial row at import). ``organisms=None``
(no context given) keeps the phase 3 behaviour.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple

PATHWHIZ_ROWS: Dict[str, Dict[str, str]] = {
    "subcellular-location": {
        'Cytosol': 'GO:0005829',
        'Mitochondrion': 'GO:0005739',
        'Mitochondrial Matrix': 'GO:0005759',
        'Peroxisome': 'GO:0005777',
        'Cytoplasm': 'GO:0005737',
        'Lysosome': 'GO:0005764',
        'Endoplasmic Reticulum Membrane': 'GO:0005789',
        'Smooth Endoplasmic Reticulum': 'GO:0005790',
        'Smooth Endoplasmic Reticulum Lumen': 'GO:0048238',
        'Cell Membrane': 'GO:0005886',
        'Extracellular Space': 'GO:0005615',
        'Mitochondrial Inner Membrane': 'GO:0005743',
        'Endoplasmic Reticulum': 'GO:0005783',
        'Mitochondrial Outer Membrane': 'GO:0005741',
        'Nucleus': 'GO:0005634',
        'Lysosomal Lumen': 'GO:0043202',
        'Nucleoplasm': 'GO:0005654',
        'Melanosome Membrane': 'GO:0033162',
        'Sarcoplasmic Reticulum': 'GO:0016529',
        'Endoplasmic Reticulum Lumen': 'GO:0005788',
        'Synapse': 'GO:0045202',
        'post-synaptic membrane': 'GO:0045211',
        'Extracellular Matrix': 'GO:0031012',
        'Mitochondrial Intermembrane Space': 'GO:0005758',
        'Golgi Apparatus': 'GO:0005794',
        'Golgi Apparatus Membrane': 'GO:0000139',
        'Peroxisome Membrane': 'GO:0005778',
        'Microvillus': 'GO:0005902',
        'GLUT4 storage compartment': 'GO:0032593',
        'Lysosomal Membrane': 'GO:0005765',
        'Periplasmic Space': 'GO:0005620',
        'Inner Membrane': 'GO:0070258',
        'Outer Membrane': 'GO:0043165',
        'Plant-Type Vacuole': 'GO:0000325',
        'Chloroplast': 'GO:0009507',
        'Membrane': 'GO:0016020',
        'Basolateral cell membrane': 'GO:0016323',
        'Apical cell membrane': 'GO:0016324',
        'Mitochondrial membrane': 'GO:0031966',
        'Periplasm': 'GO:0042597',
        'Transcriptional Repressor Complex': 'GO:0017053',
        'Nuclear Chromatin': 'GO:0000790',
        'Cell periphery': 'GO:0071944',
        'Intracellular': 'GO:0005622',
        'Endosome membrane': 'GO:0010008',
        'Cellular bud neck': 'GO:0005935',
        'Cellular bud tip': 'GO:0005934',
        'Cell cortex': 'GO:0005938',
        'Nuclear Envelope': 'GO:0005635',
        'Prospore membrane': 'GO:0005628',
        'Cellular bud neck split septin rings': 'GO:0032177',
        'Mating projection tip': 'GO:0043332',
        'Endoplasmic Reticulum Body': 'GO:0010168',
        'Endocytic Vesicle': 'GO:0030139',
        'Exocytic Vesicle': 'GO:0070382',
        'Basal Cell Membrane': 'GO:0009925',
        'Melanosome': 'GO:0042470',
        'Cell Wall': 'GO:0005618',
        'autophagosome': 'GO:0005776',
        'chloroplast outer membrane (intrinsic component)': 'GO:0031359',
        'Plastid': 'GO:0009536',
        'Plastid envelope': 'GO:0009526',
        'Apoplast': 'GO:0048046',
        'Sacroplasmic Reticulum Membrane': 'GO:0033017',
        'Presynaptic membrane': 'GO:0042734',
        'Vacuole': 'GO:0005773',
        'Exocytic Vesicle membrane': 'GO:0012506',
        'Nucleus Membrane': 'GO:0005640',
        'Inflammasome': 'GO:0097169',
        'Cytoskeleton': 'GO:0005856',
        'Ribosome': 'GO:0005840',
        'ciliary membrane': 'GO:0060170',
        'Lamellar body membrane': 'GO:0097232',
        'lamellar body': 'GO:0042599',
        'Rough Endoplasmic Reticulum Lumen': 'GO:0048237',
        'Multivesicular Body': 'GO:0005771',
        'phagocytic vesicle': 'GO:0045335',
    },
    "cell-type": {
        'Cell': 'CL:0000000',
        'Platelet': 'CL:0000233',
        'Neuron': 'CL:0000540',
        'Cardiomyocyte': 'CL:0000746',
        'Hepatocyte': 'CL:0000182',
        'Myocyte': 'CL:0000187',
        'Epithelial Cell': 'CL:0000066',
        'Beta cell': 'CL:0000639',
        'Pancreatic Beta Cell': 'CL:0000169',
        'Glial cell': 'CL:0000125',
        'Colorectal Cancer Cell': 'CL:0001064',
        'Astrocyte': 'CL:0000127',
        'Parietal Cell': 'CL:0000162',
        'Delta Cell': 'CL:0000173',
        'Enterochromaffin-like cell': 'CL:0000504',
        'fibroblast': 'CL:1248234',
        'Erythrocyte': 'CL:0000232',
        'Sertoli Cell': 'CL:0000216',
        'Leydig Cell': 'CL:0000178',
        'Theca Cell': 'CL:0000503',
        'Granulosa Cell': 'CL:0000501',
        'T Cell': 'CL:0000084',
        'Sensory Neuron': 'CL:0000101',
        'Endothelial cell': 'CL:0000115',
        'Plasma cell': 'CL:0000786',
        'Macrophage': 'CL:0000235',
        'Pneumocyte': 'CL:0000322',
        'Keratinocyte': 'CL:0000312',
        'Mesangial cells': 'CL:0000650',
        'Acinar Cell': 'CL:0000622',
        'Myometrium': 'CL:0002366',
        'Mast Cell': 'CL:0000097',
        'Gonadotroph cell': 'CL:0000437',
        'colon enteroendocrine cell': 'CL:0009042',
        'ciliated olfactory sensory neuron': 'CL:0000847',
        'type III taste bud cell': 'CL:0002285',
        'type I taste bud cell': 'CL:0002289',
        'type II taste cell': 'CL:0002286',
        'type II alveolar cell': 'CL:0002063',
        'Enterocyte': 'CL:0000584',
        'Adipocyte': 'CL:0000136',
        'mammary epithelial cell': 'CL:0002327',
        'Pancreatic A cell': 'CL:0000171',
    },
    "tissue": {
        'Liver': 'BTO:0000759',
        'Endothelium': 'BTO:0000393',
        'Sympathetic Nervous System': 'BTO:0001832',
        'Adrenal Medulla': 'BTO:0000049',
        'cardiocyte': 'BTO:0001539',
        'Kidney': 'BTO:0000671',
        'Nervous System': 'BTO:0001484',
        'Blood Vessel': 'BTO:0001102',
        'Muscle': 'BTO:0000887',
        'Renal Proximal Tubule': 'BTO:0001498',
        'Heart': 'BTO:0000562',
        'Genitalia': 'BTO:0000081',
        'Breast': 'BTO:0000149',
        'Eye': 'BTO:0000439',
        'Bone': 'BTO:0000140',
        'Skin': 'BTO:0001253',
        'Ear': 'BTO:0000368',
        'Pancreas': 'BTO:0000988',
        'Spleen': 'BTO:0001281',
        'Gastrointestinal Tract': 'BTO:0000511',
        'Hair': 'BTO:0001501',
        'Bladder': 'BTO:0000123',
        'Gall bladder': 'BTO:0000493',
        'Brain': 'BTO:0000142',
        'Intestine': 'BTO:0000648',
        'Lung': 'BTO:0000763',
        'Adrenal Gland': 'BTO:0000047',
        'Stomach': 'BTO:0001307',
        'Pituitary Gland': 'BTO:0001073',
        'Corpus Allatum': 'BTO:0000291',
        'Dermal tissue': 'BTO:0001310',
        'Adrenal Cortex': 'BTO:0000045',
        'nn': 'BTO:0000938',
        'Testis': 'BTO:0001363',
        'Ovary': 'BTO:0000975',
        'Adipose': 'BTO:0000441',
        'Blood-Brain Barrier': 'BTO:0006244',
        'Myenteric Plexus': 'BTO:0002436',
        'Bile': 'BTO:0000121',
        'Leaf': 'BTO:0000713',
        'Vagina': 'BTO:0000243',
        'Hypothalamus': 'BTO:0000614',
        'Seed': 'BTO:0001226',
        'Trachea': 'BTO:0001388',
        'Alveolus': 'BTO:0000060',
        'Spinal cord': 'BTO:0001279',
        'Salivary Gland': 'BTO:0001203',
        'Mouth': 'BTO:0001090',
        'Uterus': 'BTO:0001424',
        'Distal Colon': 'BTO:0000646',
        'Olfactory epithelium': 'BTO:0000108',
        'Taste Bud': 'BTO:0000989',
        'Adipocyte': 'BTO:0000443',
    },
}

# lower-case source name -> existing PathWhiz row name. One line of justification per entry.
SYNONYMS: Dict[str, Dict[str, str]] = {
    "subcellular-location": {
        "er": "Endoplasmic Reticulum",  # standard abbreviation of the organelle (GO:0005783)
        "endoplasmic_reticulum": "Endoplasmic Reticulum",  # same words, underscore spelling
        "golgi": "Golgi Apparatus",  # short form of the same organelle (GO:0005794)
        "mitochondria": "Mitochondrion",  # plural of the same organelle (GO:0005739)
        "plasma membrane": "Cell Membrane",  # PathWhiz row Cell Membrane is GO:0005886, the GO term "plasma membrane"
        "extracellular": "Extracellular Space",  # PathBank's name for the extracellular compartment (GO:0005615)
        "apical membrane": "Apical cell membrane",  # apical plasma membrane of a polarised cell (GO:0016324); reviewer to confirm
        "postsynaptic membrane": "post-synaptic membrane",  # hyphen spelling only (GO:0045211)
        "plasma_membrane": "Cell Membrane",  # underscore spelling of "plasma membrane" (see above)
    },
    "cell-type": {
        "neurons": "Neuron",  # plural of the same cell type (CL:0000540)
        "astrocytes": "Astrocyte",  # plural of the same cell type (CL:0000127)
        "leydig cells": "Leydig Cell",  # plural of the same cell type (CL:0000178)
        "type iii taste cell": "type III taste bud cell",  # same taste-receptor cell type (CL:0002285); reviewer to confirm
    },
    "tissue": {
        "taste buds": "Taste Bud",  # plural of the same structure (BTO:0000989)
    },
}
# Deliberately NOT mapped (a row looks near but is a different entity): "Type II cell" (taste cell or alveolar cell, ambiguous),
# "Type III cell(s)" (no context), erythroid (progenitors are not erythrocytes), central nervous system (Nervous System is wider),
# bone marrow (Bone is not marrow), young leaf (Leaf is wider), apical (no membrane stated), etioplast (Plastid is wider),
# synaptic vesicle / cleft (Synapse is wider), IM leaflets (Inner Membrane is the whole membrane), cell lines MG-63 and Saos-2.
# Phase 5: plurals of a row (granulosa cells) are mapped by ``singular_form``; the inner / outer membrane names by organism (above).

_ROW_KEYS: Dict[str, Dict[str, str]] = {k: {n.casefold(): n for n in rows} for k, rows in PATHWHIZ_ROWS.items()}
_ROW_BY_ID: Dict[str, Dict[str, str]] = {k: {v: n for n, v in rows.items()} for k, rows in PATHWHIZ_ROWS.items()}


# ---- phase 5 (E): organism-aware inner / outer membrane ----------------------------------------------------------------------
_MEMBRANE_ROW = {"inner membrane": "Inner Membrane", "outer membrane": "Outer Membrane"}
_MEMBRANE_NAME = re.compile(r"^(inner|outer) membrane(?: \((im|om)\))?$")
_ABBR_OF = {"inner": "im", "outer": "om"}
# ids that may accompany the name of the same bacterial compartment: the row's own stored id, GO cell outer membrane, GO outer membrane
_MEMBRANE_OWN_IDS = {"Inner Membrane": {"GO:0070258"}, "Outer Membrane": {"GO:0043165", "GO:0009279", "GO:0019867"}}

# Exact genus tables (no guessing). A genus in neither table, with the species classification "Prokaryote", counts as bacterial.
_BACTERIAL_GENERA = frozenset("""escherichia pseudomonas bacillus lactococcus listeria staphylococcus streptococcus streptomyces borrelia
borreliella aquifex campylobacter helicobacter mycobacterium salmonella klebsiella acinetobacter haemophilus vibrio yersinia shigella
clostridium clostridioides enterococcus lactobacillus corynebacterium neisseria legionella synechocystis synechococcus thermus agrobacterium
rhizobium sinorhizobium bacteroides chlamydia rickettsia treponema burkholderia bordetella brucella francisella xanthomonas zymomonas
geobacter shewanella caulobacter deinococcus thermotoga bifidobacterium enterobacter citrobacter serratia proteus erwinia pectobacterium
photorhabdus aeromonas azotobacter rhodobacter prochlorococcus nostoc anabaena cupriavidus ralstonia paracoccus akkermansia""".split())
_ARCHAEAL_GENERA = frozenset("""methanocaldococcus methanococcus methanosarcina methanobacterium methanothermobacter methanopyrus
halobacterium haloferax haloarcula sulfolobus saccharolobus thermococcus pyrococcus archaeoglobus thermoplasma picrophilus nitrosopumilus
aeropyrum pyrobaculum ignicoccus nanoarchaeum thermoproteus haloquadratum natronomonas methanobrevibacter""".split())
_ABBREVIATED = {"e. coli": "escherichia", "b. subtilis": "bacillus", "l. lactis": "lactococcus", "s. aureus": "staphylococcus"}


def organism_kind(name: Any, classification: Any = None) -> Optional[str]:
    """"bacterium" / "archaeon" / "eukaryote" for a species row, or None when it cannot be told. ``classification`` is the PathWhiz species
    column (Prokaryote / Eukaryote). The genus tables and the classification must agree; a contradiction gives None."""
    text = re.sub(r"\s+", " ", str(name or "")).strip().casefold()
    genus = _ABBREVIATED.get(" ".join(text.split()[:2])) or (text.split()[0] if text else "")
    table = "bacterium" if genus in _BACTERIAL_GENERA else "archaeon" if genus in _ARCHAEAL_GENERA else None
    cls = str(classification or "").strip().casefold()
    if cls == "eukaryote":
        return None if table else "eukaryote"
    if cls == "prokaryote":
        return table or ("bacterium" if text else None)  # PathWhiz "Prokaryote": bacterial unless the genus is a listed archaeon
    return table


def organism_kinds(species: Iterable[Tuple[Any, Any]]) -> frozenset:
    """The set of organism kinds of (name, classification) pairs; a species whose kind cannot be told contributes "unknown"."""
    return frozenset(organism_kind(n, c) or "unknown" for n, c in species)


def row_id(kind: str, row_name: Any) -> Optional[str]:
    """The ontology id of the PathWhiz row called ``row_name`` (case-insensitive), or None."""
    name = _ROW_KEYS.get(kind, {}).get(str(row_name or "").strip().casefold())
    return PATHWHIZ_ROWS[kind][name] if name else None


def membrane_row(name: Any) -> Optional[str]:
    """The PathWhiz bacterial membrane row a name stands for (inner membrane, outer membrane, with an optional (IM) / (OM)), else None."""
    m = _MEMBRANE_NAME.match(re.sub(r"\s+", " ", str(name or "")).strip().casefold())
    return _MEMBRANE_ROW[f"{m.group(1)} membrane"] if m and m.group(2) in (None, _ABBR_OF[m.group(1)]) else None


def singular_form(name: Any) -> Optional[str]:
    """The name with a plural -s cut from its last word (cells -> cell, vesicles -> vesicle), or None. No irregular plurals; words
    ending in ss / us / is are left alone."""
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    last = text.rsplit(" ", 1)[-1].casefold()
    return None if len(last) < 4 or not last.endswith("s") or last.endswith(("ss", "us", "is")) else text[:-1]


def map_vocabulary_name(kind: str, name: Any, ontology_id: Any, organisms: Optional[Iterable[str]] = None) -> Tuple[str, Optional[str], str]:
    """Return (name to write, ontology id to write, outcome). ``kind`` is a PWML tag: subcellular-location, cell-type, tissue.
    ``organisms``: organism kinds (``organism_kinds``) of the states that use a location; None = no context (phase 3 behaviour)."""
    text = str(name or "").strip()
    onto = str(ontology_id).strip() if ontology_id not in (None, "") else None
    key = text.casefold()
    if not text or kind not in PATHWHIZ_ROWS:
        return name, ontology_id, "unmapped"
    mem = membrane_row(text) if kind == "subcellular-location" and organisms is not None else None
    if mem is not None:
        if frozenset(organisms) == {"bacterium"} and (onto is None or onto in _MEMBRANE_OWN_IDS[mem]):
            return mem, None, "mapped"
        if onto in _ROW_BY_ID[kind]:  # the data carries the id of another row (a mitochondrial membrane, say): that row
            return _ROW_BY_ID[kind][onto], None, "mapped_by_id"
        return name, None, "unmapped"  # not bacterial or undetermined: the name equals a bacterial row's name, so no id is written either
    if key in _ROW_KEYS[kind]:
        return name, ontology_id, "exact"
    target = SYNONYMS[kind].get(key)
    if target is None and (sing := singular_form(text)) is not None:  # B: the plural of a row name or of a reviewed synonym
        target = _ROW_KEYS[kind].get(sing.casefold()) or SYNONYMS[kind].get(sing.casefold())
    if target is not None and onto in (None, PATHWHIZ_ROWS[kind][target]):
        return target, None, "mapped"  # an id that disagrees with the row means the entity is not the same: leave it
    if onto in _ROW_BY_ID[kind]:  # an id found online that PathWhiz already holds under another name: same term, use its row
        return _ROW_BY_ID[kind][onto], None, "mapped_by_id"
    return name, ontology_id, "new_row" if onto else "unmapped"
