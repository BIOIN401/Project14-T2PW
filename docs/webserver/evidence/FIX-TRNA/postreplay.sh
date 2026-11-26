#!/bin/bash
# FIX-TRNA decisive post-pipeline replay, flag off then on, one lock hold.
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONPATH=C:/t/fix-trna/src
S=C:/t/vis-runs/scratch/fixtrna
cd C:/t/fix-trna
for f in off on; do
  echo "== $f"; git -C C:/t/fix-trna checkout -q -- data/enrichment_cache.json data/id_mapping_cache.json
  timeout 3600 $PY -u -X utf8 C:/t/corr-assess/scripts/live_launch.py --script $S/postreplay.py $f $S/postreplay-$f.json > $S/post-$f.log 2>&1
  echo "EXIT_$f=${PIPESTATUS[0]}"
done
git -C C:/t/fix-trna checkout -q -- data/enrichment_cache.json data/id_mapping_cache.json
echo POSTREPLAY_DONE
