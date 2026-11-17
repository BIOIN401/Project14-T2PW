import json
F = []
def f(sev, layer, pathways, evidence, fix, changes, risk, title):
    F.append(dict(rank=len(F) + 1, severity=sev, title=title, layer=layer, pathways=pathways, evidence=evidence, proposed_fix=fix, changes_content=changes, regression_risk=risk))

f("S1 content lost", "PWML writer (D-104 withhold) + PathWhiz importer", "7,128,78,58,68 (13 kept in 7,9,150)",
  "Stated interactions: payload 54 = IR 54 -> PWML 13 -> PathWhiz DB 11 (a1 import also drops 2 in idx 7: 'Interaction right elements wrong length' for nucleic-acid endpoints). 41 of 54 withheld (bare protein / nucleic-acid endpoints: 7:21, 68:13, 58:4, 128:2, 78:1). Web diagram draws all 54, clean-model SBML keeps all 54 (t2pw:interactions annotation), PathWhiz SBML/SBGN/DB show 11. Evidence: a3/interactions_by_stage.json, compare.json (ir|pwml itn 2/7 pathways agree).",
  "Wrap bare-protein/nucleic-acid interaction endpoints in a generated single-member complex (already done for enzymes, single_protein_pathwhiz_wrapper) instead of withholding; set nucleotide type on nucleic acids. Until then label the web interactions 'not in PathWhiz export'.",
  True, "medium: D-104 is a deliberate decision, touches ir.py ~2414 and golden-seam tests", "Interactions: 80 percent never reach PathWhiz formats")
f("S1 content wrong", "IR / PWML writer (ir.py ~2174)", "7 (2 reactions), 9 (1 reaction)",
  "4 non-catalyst modifiers in 3 of 196 reactions become enzymes: idx 7 'singlet oxygen generation from Pchlide' (payload enzymes none; modifiers GluTR, FC1 role activator, evidence 'responsive to singlet oxygen signaling') -> PWML/PathWhiz/web show GluTR and FC1 as the enzymes producing singlet oxygen; idx 7 Mg-chelatase: GUN4 activator -> enzyme; idx 9 MenB step: MenD (role inhibitor, rate-equation evidence) -> co-catalyst, SBGN draws it as OR catalyst with MenB. Clean-model SBML keeps SBO:0000020 inhibitor vs SBO:0000013 catalyst. Evidence: a3/modifiers_as_enzymes.json, p_ir_aligned.json. Contract s.5: exporters must not reinterpret roles.",
  "Do not promote activator/inhibitor to enzyme; carry as modifier-role (or interaction) and, where PathWhiz has no modifier slot, drop with a report entry. For the zero-enzyme reaction leave enzyme-less.",
  True, "low in volume (2 percent of reactions) but changes PWML; needs golden-test update", "Regulators written as catalysts")
f("S1 content lost", "Pipeline quarantine (strict) -> every PathWhiz-side format", "116",
  "'ceramide synthase reaction' (Sphinganine -> dihydroceramide, the step that makes ceramide) is in final_payload (15 reactions) but quarantined with reason reaction_enzyme_must_be_protein_complex (enzyme is a bare protein, the other 235 catalysts survive); PWML/PathWhiz/SBGN/web have 14, clean-model SBML has 15 so formats disagree. Evidence: base/116/outputs__quarantine_report.json, compare.json ours_sbml|pwml rxn.",
  "Wrap the bare-protein enzyme in a generated complex like the sibling reactions rather than quarantining; or at least surface 'omitted_from_pwml' in the web UI.",
  True, "medium: quarantine policy, pipeline/ is outside this track", "A central reaction silently disappears")
f("S1 content wrong", "Extraction / mapping (shows in all formats)", "116",
  "protein 'S1P phosphatase' carries gene_name HM13 and UniProt Q8TCT9; PathWhiz exports it as 'Signal peptide peptidase' (HM13) - S1P phosphatase is SGPP1. Every format shows the wrong identity under a paper-correct complex.",
  "Mapping bug upstream; not an exporter issue.", True, "n/a (biology change, outside track)", "Wrong protein identity (S1P phosphatase = HM13)")
f("S2 misleading", "PWML writer (direction) + PathWhiz SBML", "all with reactions (196 reactions); stated reversible in 9, 5, 129, 142, 68",
  "Payload has no direction field. IR writes direction '>' for 196/196, PWML 'Right' 196/196, PathWhiz SBML reversible=false 230/230. 7 payload reactions carry reversibility wording in their evidence (e.g. idx 9/5 'MenF catalyzes a reversible reaction', 129 HFP isomerase reversibility, 142 VALTA). Clean-model SBML is honest: reversible=false placeholder + t2pw:directionStated=false. Contract s.1: never invent directionality.",
  "Carry a stated reversibility into the payload/IR; otherwise write PathWhiz 'unknown/both' if its schema allows, and do not claim irreversible.",
  True, "medium", "Every reaction asserted one-way")
f("S2 misleading", "PathWhiz SBGN exporter (+ unresolved Unknown protein)", "25 SBGN files; worst 5 (13 of 16), 7, 125, 148",
  "PathWhiz SBGN draws proteins, not complexes. 82 of 188 SBGN reactions (44 percent) have an enzyme glyph labelled 'Unknown' (86 enzyme complexes in the DBs wrap the placeholder protein 9659) while SBML/PWML name the complex (MenA, MenI, TubA...). Evidence: a3/protein_names.json, render 9_pw_sbgn.png.",
  "PathWhiz side: label by complex name when the protein is the placeholder; our side: resolve the protein before export or give the placeholder the complex name.",
  False, "low (labels only)", "Enzymes shown as 'Unknown' in SBGN")
f("S2 misleading", "PathWhiz SBML exporter", "all 13 pathways with transports (23 transports)",
  "PathWhiz SBML writes each transport as X -> X with the same species id in one compartment: 23 of 23 transports are no-ops for any SBML tool (compartments only in the reaction name text). Clean-model SBML uses one species per (entity, location), so 26 of 30 transports go between two compartments; the other 4 (idx 78, 138, 10, 150) have no stated from/to and sit in one compartment.",
  "PathWhiz: species per (entity, biological state). Ours: see next finding for the 4.", False, "n/a for us", "PathWhiz SBML transports are no-ops")
f("S2 misleading", "IR / PWML writer", "78, 138, 10 (one each), 150",
  "Transports without stated from/to locations are written with both ends in __auto_state__ (same state). PathWhiz rejects them ('Biological state must have unique biological states', 'Transport elements is too short'); web draws no membrane; clean-model SBML places both species in one compartment. 7 of 30 transports never reached PathWhiz (3 same-state, 128 duplicate elements, 5/150/68 duplicate or merged).",
  "Do not emit a transport without two distinct states; keep it as an unlocated transport note.", False, "low", "Same-state transports are invalid or invisible")
f("S2 misleading", "PathWhiz SBML exporter", "all 26",
  "529 of 529 species carry initialAmount=1 (unit Mole, hasOnlySubstanceUnits), none stated by any paper; reversible=false on 230/230; 219 reaction names contain a literal '&rarr;' (double-escaped). Clean-model SBML states no amounts (libSBML warning 80601 x695, intentional) and is valid. Evidence: a3/pw_format_facts.json.",
  "PathWhiz exporter: drop initialAmount, fix escaping; for our own SBML nothing to do.", False, "n/a", "PathWhiz SBML invents amounts and mangles reaction names")
f("S2 misleading", "PathWhiz SBGN exporter", "25 SBGN files",
  "Every compartment glyph is labelled 'Simple' (compartment names never shown), process glyphs unlabelled (all reactions anonymous), map element has no id (XSD error in 25/25 after 0.2 to 0.3 upconversion), schematron: arcs target the process glyph not its port (pd10102 297, pd10105 247, pd10141 221), logic arcs pd10125/pd10142 91 each. 58 SBGN export fails with HTTP 500 (Fe3+ template 299). momapy loads 25/25. Renders: a3/render/*_pw_sbgn.png.",
  "PathWhiz exporter fixes (labels from membrane names, port references, map id).", False, "n/a", "SBGN loses compartment names and fails schematron")
f("S2 misleading", "Upstream duplicates / PathWhiz SBML+SBGN", "9, 7, 5, 134, 116",
  "Payload holds exact duplicate reactions (idx 9: 19 reactions, 10 distinct substrate->product keys; SBML/SBGN list 19 visualizations over 11 DB reactions). PathWhiz DB merges them (195 to 176 in the sample) but SBML and SBGN draw one per visualization, so a reader sees chorismate -> isochorismate twice (render 9_pw_sbgn.png) and in the web diagram (A2).",
  "Merge identical reactions (same L,R,compartment) in the writer, keep all enzymes (A2 merge_dups mock).", False, "low", "Duplicate reactions drawn twice")
f("S2 misleading", "PathWhiz importer / exporters", "60 protein names across 17 pathways; compounds in 78,126,142,58,150,128",
  "Paper names are replaced: proteins by UniProt long names (MenF -> 'isochorismate synthase', KDSR -> '3-ketodihydrosphingosine reductase'); compounds by DB names (pyruvate -> Pyruvic acid, H2O -> Water, H+ -> Hydrogen Ion) already in the IR; PathWhiz SBGN uses short names ('Mg++', 'H', 'atp') where SBML uses full names. Complex names survive in SBML only. Evidence: a3/p_ir_aligned.json, protein_names.json.",
  "Keep paper label as synonym/tooltip in PWML; cosmetic otherwise.", False, "low", "Names differ between formats")
f("S2 misleading", "PWML writer", "7",
  "7 of 12 nucleic acids and the 2 interactions that use them are lost at PathWhiz import ('Nucleotide type can't be blank / not included in the list'); SBML shows 3 nucleic acids, SBGN none. Cross-reference A1.",
  "Write a valid nucleotide type or omit with a report line.", True, "low", "Nucleic acids dropped at import")
f("S3 cosmetic", "Renderers / SBML structure", "all",
  "Neither SBML has a layout, so momapy cannot render SBML at all (layout=None, loads 52/52); SBMLNetwork autolayout gives huge canvases with tiny labels, bare protein species as floating nodes in both SBMLs, and the clean-model SBML interactions (inside t2pw annotation) are not drawn, so DHNA-inhibits-MenD is invisible in a generic SBML viewer.",
  "Do not promise a drawn SBML; the web diagram and PathWhiz SBGN are the visuals.", False, "none", "SBML has no usable picture")
json.dump(F, open("C:/t/corr-assess/a3/findings.json", "w", encoding="utf-8"), indent=1)
print(len(F))
