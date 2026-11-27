#!/bin/bash
# usage: ccorp.sh LABEL : content check of corpus replay LABEL: vs base (rule-based, strict) and vs VIS-7 (strict)
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw8-work
$PY -I -X utf8 $W/scripts/content_check.py $W/replay/base/pwml $W/replay/$1/pwml $W/metrics/corp_$1.json
$PY -I -X utf8 $W/scripts/content_check.py $W/replay/base/pwml $W/replay/$1/pwml $W/metrics/corps_$1.json strict
$PY -I -X utf8 $W/scripts/content_check.py $W/replay/v7/pwml $W/replay/$1/pwml $W/metrics/corp7s_$1.json strict
