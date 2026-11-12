# usage: run1.sh NAME MODE PMID PATHWAY ORG
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
E=C:/t/fp/docs/pwml_output/evidence/final_payload_p3/live
cd C:/t/fp
export PYTHONPATH=C:/t/fp/src T2PW_FINAL_PAYLOAD=1
unset T2PW_OFFLINE_CURATOR
$PY docs/pwml_recovery_sprint/evidence/bounded_run.py --label fp-live-$1 --timeout 3000 --heavy-lock fp --json $E/$1.wrapper.json -- \
 $PY $E/drive.py "$@" > $E/$1.stdout.txt 2>&1
echo exit=$?
