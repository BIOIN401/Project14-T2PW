"""Merge the manual screenshot review and the per-pathway problem list into a1/per_pathway.json (run after a1_check.py)."""
import json

P = 'C:/t/corr-assess/a1/per_pathway.json'
d = json.load(open(P, encoding='utf-8'))

# (flow_readable, labels_readable, shared_compounds_redrawn, note)  -- from looking at a1/shots/<idx>.png
SHOT = {
 7:  ('no', 'no (enzyme/compound text ~3.6 px at fit 0.24; overflows boxes)', 'yes, 7 compounds drawn 2-4x (protoporphyrinogen IX, Mg-Proto IX, Pchlide, Glu-1-semialdehyde)', 'two boxes both labelled "plastid"; chain split into 5 rows; nucleus box holds only regulator ellipses with red inhibition ends'),
 128:('partly', 'label "Bacillus subtilis cytoplasm" cut off', 'yes, pantetheine drawn 4x, cysteine 2x', 'chain breaks into 4 rows; long transport lines run across the whole cell box; reaction without enzyme drawn as bare arrow'),
 9:  ('no', 'tiny at fit 0.29', 'yes, isochorismate/DHNA-CoA/DHNA drawn 2-3x', 'same 9-reaction chain drawn in two boxes; 5 boxes for a 2-compartment cell; chorismate<->isochorismate pair far right of lower row of the KF147 box'),
 116:('partly', 'ok at 100%, tiny at fit 0.27', 'yes, L-serine 2x, palmitoyl-CoA 2x, ceramide 6x', 'two boxes both "ER"; long diagonal/vertical edge from Golgi ceramide to the ER chain; SPT duplicate drawn as parallel stack'),
 5:  ('no', 'tiny at fit 0.23', 'yes, DHNA 5x', 'chain drawn twice (Lactococcus lactis intracellular / intracellular); 5 boxes (cells, intracellular x2, extracellular x2)'),
 134:('no', 'label "human carnitine biosynthesis" cut off; compound names overflow', 'yes, glycine 3x, 4-trimethylaminobutyraldehyde 7x', 'compartment box named after the pathway; 6 reactions with no stated compartment drawn in "mitochondrial"; isozyme enzyme ellipses without edges'),
 135:('no (chain end on top, start at bottom)', 'ok', 'yes, noreugenin 3x, visamminol 3x, peucenin 2x', 'single unlabelled box; MgCl2 drawn as a substrate; SdPT/SdUGT stacks converge on one enzyme'),
 78: ('partly', 'ok', 'yes, enterobactin 4x, 2,3-DHB 3x', 'unlabelled empty second box; long diagonal transport edges; EntB/E/F/D drawn as 4 ellipses on one stack'),
 125:('no (8 reactions collapse onto 4 enzyme nodes)', 'ok', 'yes, SAM 5x, bergaptol 2x', 'AgOMT1 node carries 4 reactions: substrate->product pairing unreadable'),
 129:('partly', 'ok', 'yes, 5-amino-6-ribitylamino-pyrimidinedione(+5-P) 4x, D-ribulose 5-phosphate 2x', 'rows run top-down but earlier steps sit below later ones; orphan IMP/HFP ellipses'),
 138:('partly', 'label cut off ("Streptomyces tubercidicu")', 'yes, compound 3 / compound 5 2x', 'whole biosynthesis drawn inside a box labelled "extracellular"; second row runs backwards; empty unlabelled box'),
 120:('yes', 'ok', 'yes, androstenedione 3x, testosterone 4x', 'same androstenedione->testosterone step drawn in 3 boxes (Leydig cells of testis / testis / adult Leydig cells)'),
 148:('yes (single row of 7)', 'compound names overflow their boxes', 'no', 'canvas 4240x350 so fit-to-window scale is tiny; otherwise the cleanest result'),
 19: ('no (bottom-up)', 'ok', 'yes, chanoclavine-I aldehyde 4x', 'EasA node carries 3 reactions; EasH/EasG/EasF/EasC enzyme ellipses with no edges; NADPH wires loop'),
 144:('partly', 'label cut off', 'yes, liquiritigenin 2x, naringenin 2x', 'two boxes with identical label "endoplasmic reticulum of A. membranaceus"; AmIFS node carries 3 reactions'),
 126:('no', 'ok', 'yes, benzaldehyde 4x, benzyl alcohol 3x', '4 boxes (cytosolic, Petunia hybrida cytosol, peroxisomal, peroxisome); stacks of isozyme ellipses without edges'),
 122:('partly', 'label cut off', 'yes, farnesyl pyrophosphate 2x', 'two ER boxes ("endoplasmic reticulum", "Candida albicans endoplasmic reticulum"); Dpp2/Dpp3 ellipses without edges'),
 121:('yes', 'ok', 'yes, CDP-4-keto-6-deoxy-d-glucose 2x, CDP-4-keto-6-deoxy-l-altrose 2x', 'one-reaction rows stacked vertically; reaction order top-down is not the biochemical order'),
 142:('no (bottom-up)', 'label overflows box', 'yes, (S)-2-acetolactate 2x', 'IlvI at the bottom feeds IlvC above it; order inverted'),
 114:('partly', 'ok', 'yes, D-alanine/D-serine 3x each', 'two long orthogonal edges across the cell-wall box; second reaction has products only in the other box'),
 14: ('yes', 'label overflows box', 'no', 'single reaction; fine'),
 10: ('no', 'ok', 'yes, 5-HT 4x', 'no reactions; 6 boxes; long edges cross the whole canvas and each other'),
 58: ('partly', 'ok', 'yes, enterobactin 4x, 2,3-DHB 3x', 'EntA step drawn below the EntB/E/F/D step it feeds; long edges to the extracellular box'),
 12: ('no', 'ok', 'yes, diosgenin 4x, UDP-glucose 4x', '4 isozyme reactions with identical substrates/products; every substrate redrawn and converging'),
 150:('no', 'labels cut off', 'yes, 5-HT 5x', '8 boxes for near-identical locations; long edges through boxes; mostly transports'),
 68: ('yes', 'label cut off ("cytoplasmic leaflet of IM")', 'no', 'readable; two empty/nearly empty compartment boxes'),
}
NULL_BAD = {116, 134, 138, 144}   # default state is an organelle or "extracellular"
DB_TRANSPORT_DROP = {128: 'duplicate transporter element (TcyJKLMN listed twice)', 78: 'both ends __auto_state__', 138: 'both ends __auto_state__',
                     10: 'same-state / __auto_state__ ends', 150: 'same-state / __auto_state__ ends', 68: 'new subcellular locations rejected by empty harness DB (Ontology blank)'}
for k, v in d.items():
    i = int(k)
    pr = []
    imp = v.get('import') or {}
    if v['reactions']['missing_in_pwml']:
        pr.append({'class': 'data/extraction (strict quarantine at export)', 'sev': 'S1', 'what': 'payload reaction absent from PWML: %s' % v['reactions']['missing_in_pwml']})
    if v['states']['null_state_reactions_payload']:
        pr.append({'class': 'writer (ir.py default_bs_key)', 'sev': 'S1' if i in NULL_BAD else 'S3',
                   'what': '%d reaction(s) with no stated compartment placed in biological_states[0]' % v['states']['null_state_reactions_payload']})
    if v['payload_dups']['exact_dup'] or v['payload_dups']['reverse_pairs_any_enzyme'] or v['payload_dups']['same_inout_diff_enzyme']:
        pr.append({'class': 'data/extraction', 'sev': 'S2', 'what': 'duplicate reactions in payload: %s' % v['payload_dups']})
    if v['compartment_boxes']['dup_labels']:
        pr.append({'class': 'data/extraction', 'sev': 'S2', 'what': 'two boxes with the same label %s' % v['compartment_boxes']['dup_labels']})
    if any(len(m) > 1 for m in v['states']['synonym_groups_used'].values()):
        pr.append({'class': 'data/extraction', 'sev': 'S2', 'what': 'near-synonym compartments used: %s' % {g: m for g, m in v['states']['synonym_groups_used'].items() if len(m) > 1}})
    if v['drawing']['shared_redrawn']:
        pr.append({'class': 'writer (shared-intermediate reuse only for adjacent reactions)', 'sev': 'S2',
                   'what': '%d shared compounds redrawn; %d drawn components vs %d logical' % (v['drawing']['shared_redrawn'], v['drawing']['drawn_components'], v['drawing']['logical_components'])})
    if v['flow']['right_to_left_pairs']:
        pr.append({'class': 'layout', 'sev': 'S2', 'what': '%d of %d producer->consumer pairs drawn against reading order' % (v['flow']['right_to_left_pairs'], v['flow']['precedence_pairs'])})
    if v['canvas_use']['fit_scale_1600x900'] < 0.3:
        pr.append({'class': 'layout (+PathWhiz fixed font)', 'sev': 'S3', 'what': 'fit-to-window scale %s: text unreadable' % v['canvas_use']['fit_scale_1600x900']})
    if v['noncatalyst_modifiers']['written_as_reaction_enzyme']:
        pr.append({'class': 'writer (ir.py enzyme_sources)', 'sev': 'S2', 'what': 'activator/inhibitor written as reaction enzyme: %s' % v['noncatalyst_modifiers']['examples']})
    if i in DB_TRANSPORT_DROP:
        pr.append({'class': 'PathWhiz importer' if i in (68,) else 'data/extraction (writer passes it through)', 'sev': 'S1', 'what': 'transport not created in PathWhiz: %s' % DB_TRANSPORT_DROP[i]})
    if i == 7:
        pr.append({'class': 'writer/data (nucleic acids without nucleotide type)', 'sev': 'S1', 'what': 'PathWhiz importer rejected 7 nucleic acids, 4 nucleic-acid locations and 2 interactions'})
    if v['interactions']['payload'] > v['interactions']['pwml']:
        pr.append({'class': 'writer/IR (silent drop)', 'sev': 'S1', 'what': 'interactions payload %d -> PWML %d' % (v['interactions']['payload'], v['interactions']['pwml'])})
    if i == 134:
        pr.append({'class': 'writer (species id collision)', 'sev': 'S1', 'what': 'Homo sapiens and Mus musculus both written with species id 1; PathWhiz stores the pathway as Mus musculus'})
    if i == 114:
        pr.append({'class': 'writer (species id collision)', 'sev': 'S3', 'what': 'two S. aureus species rows share id 257; strain name lost'})
    if i == 58:
        pr.append({'class': 'writer / PathWhiz exporter', 'sev': 'S3', 'what': 'Fe3+ uses visualization template 299; PathWhiz SBGN export fails (HTTP 500, "Template not found!")'})
    v['problems'] = pr
    s = SHOT[i]
    v['screenshot_review'] = {'file': 'C:/t/corr-assess/a1/shots/%d.png' % i, 'flow_in_pathway_order': s[0], 'labels_readable': s[1],
                              'shared_compounds_redrawn': s[2], 'note': s[3]}
json.dump(d, open(P, 'w', encoding='utf-8'), indent=1)
print('finalized', len(d))
