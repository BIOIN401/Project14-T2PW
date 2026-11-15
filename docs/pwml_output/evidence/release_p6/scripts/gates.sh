#!/bin/bash
cd C:/t/rel
R=C:/t/rel-reports
G=$R/run.sh
bash $G smoke 1500 -- $(cat C:/t/integ-reports/smoke_files.txt)
bash $G golden 900 -- tests/test_batch_driver_seam_golden.py tests/test_c011_freeze_seam_golden_equivalence.py tests/test_streamlit_stage8_export_contract.py tests/test_streamlit_quarantine_boundary.py
bash $G eq_run_pathway 900 -- tests/test_equivalence_run_pathway.py
bash $G eq_app_vs_service 900 -- tests/test_equivalence_app_vs_service.py
bash $G eq_default_path 900 -- tests/test_equivalence_default_path.py
F=$(ls tests/test_final_payload*.py tests/test_pwml*.py tests/test_process_normalizer*.py tests/test_c121_autostate_lifecycle.py tests/test_transport_sides.py tests/test_pwml_member_side.py | sort -u)
echo "$F" > $R/focused_files.txt
bash $G focused_default 1800 -- $F
bash $G focused_v1 1800 T2PW_PWML_LAYOUT=v1 -- $F
echo ALLDONE > $R/gates.done
