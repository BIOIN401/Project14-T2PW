# usage: run1.sh NAME MODE PMID PATHWAY ORG   (retries the heavy lock on exit 95)
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
E=C:/t/fp/docs/pwml_output/evidence/final_payload_p5/liveG
cd C:/t/fp
export PYTHONPATH=C:/t/fp/src T2PW_FINAL_PAYLOAD=1
unset T2PW_OFFLINE_CURATOR
n=0
while [ $n -lt 7200 ]; do
  n=$((n+1))
  $PY docs/pwml_recovery_sprint/evidence/bounded_run.py --label fp5-liveG-$1 --timeout 4500 --heavy-lock fp5-liveG-$1 --json $E/$1.wrapper.json -- \
   $PY $E/drive.py "$@" > $E/$1.stdout.txt 2>&1
  rc=$?
  if [ $rc -ne 95 ]; then break; fi
  sleep 2
done
echo exit=$rc tries=$n
