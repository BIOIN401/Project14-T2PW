#!/bin/bash
S=C:/t/fp/docs/pwml_output/evidence/final_payload_p5/measure/scripts
L=C:/t/p5scratch/measure
bash $S/import_one.sh m03 C:/t/pwpr/h03 $L/list_on59.txt
bash $S/import_one.sh m04 C:/t/pwpr/h04 $L/list_on59.txt
bash $S/import_one.sh c03 C:/t/pwpr/h03 $L/list_controls.txt
bash $S/import_one.sh c04 C:/t/pwpr/h04 $L/list_controls.txt
echo IMPORTALLDONE
