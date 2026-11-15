#!/bin/bash
S=C:/t/fp/docs/pwml_output/evidence/final_payload_p2/scripts
for n in off5 on5 baseline; do bash $S/import_loop.sh $n; done
echo ALLDONE
