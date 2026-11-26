#!/bin/bash
# usage: round.sh OUTNAME  -- render the comparison set with the vis-pw4 worktree renderer into C:/t/vis-pw4-work/OUTNAME
W=C:/t/vis-pw4-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
REF=C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations/pwml_markups
S=C:/t/vis-pw1-work/cc_show/final2
export T2PW_STRUCTURE_CACHE=${T2PW_STRUCTURE_CACHE:-$W/structcache2}
TREE=${2:-C:/t/vis-pw4}
[ "$1" = r0 ] && TREE=$W/base   # baseline: the merged VIS-3 renderer (019c521f)
echo "TREE $TREE"
mkdir -p $W/$1 && cp $TREE/src/t2pw/render/pathbank.py $W/$1/pathbank.py.snapshot
$PY -I $W/rend.py $TREE $W/$1 png \
  $REF/PW000001.pwml $REF/PW000017.pwml $REF/PW000779.pwml $REF/PW012911.pwml $REF/PW127049.pwml \
  ent-item6-final2=C:/t/vis-pw1-work/cc_item6/final2/ent-item6.pwml \
  ent-item6-base=C:/t/vis-pw1-work/cc_item6/base/ent-item6.pwml \
  $S/chlorophyll.pwml $S/lipid-a.pwml $S/ceramide-sphingolipid.pwml $S/coenzyme-a.pwml $S/menaquinone.pwml \
  $S/furochromone.pwml $S/sour-taste.pwml $S/enterobactin.pwml
echo ROUND_DONE
