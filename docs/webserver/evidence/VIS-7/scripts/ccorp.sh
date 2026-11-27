#!/bin/bash
# usage: ccorp.sh LABEL : content check (rule-based + strict) of corpus replay LABEL vs base
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw7-work
$PY -I -X utf8 $W/scripts/content_check.py $W/replay/base/pwml $W/replay/$1/pwml $W/metrics/corp_$1.json
$PY -I -X utf8 $W/scripts/content_check.py $W/replay/base/pwml $W/replay/$1/pwml $W/metrics/corps_$1.json strict
