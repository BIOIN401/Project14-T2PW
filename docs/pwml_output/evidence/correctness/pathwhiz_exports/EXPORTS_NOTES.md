PathWhiz exports for the 26 sample pathways (A1). Source: PWML C:/t/corr-assess/replay/on_v2/pwml/<idx>.pwml imported into the local harness (t2pw-pwi), one pathway at a time (always PW000001).
<idx>.sbml / <idx>.sbgn / <idx>.pwml = /live/view/PW000001.<fmt>; <idx>.db.json = DB snapshot (row counts of all non-empty tables + reaction/transport/interaction rows with names).
Complete: sbml 26/26, pwml 26/26, db.json 26/26. sbgn 25/26: 58.sbgn is HTTP 500 (error page kept as 58.sbgn.FAILED_500.html; IllegalArgumentException "Template not found!" at compound_location.rb:175, Fe3+ uses template 299).
db.json names: the mysql client returned non-ASCII characters as "?" (e.g. alpha-ketoisovalerate); match on ASCII only.
SBML/SBGN list one reaction per reaction-visualization (plus transports and interactions), so counts exceed DB reactions when PathWhiz merged identical reactions (idx 9: 19 visualizations, 11 reactions).
Importer messages and created-vs-expected counts: C:/t/corr-assess/a1/import_results.jsonl.
