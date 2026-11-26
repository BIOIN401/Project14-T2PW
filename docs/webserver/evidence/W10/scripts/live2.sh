#!/bin/bash
# W10 live check 2 (after the full-colour SVG and duplicate-choice fixes): Edge only, no LLM, no idconv.
W=C:/t/ws-search-work; S=$W/shots; PORT=8811; BASE=http://127.0.0.1:$PORT
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
$PY -I $W/live_app.py > $W/logs/live2_server.log 2>&1 &
SPID=$(cat /proc/$!/winpid)
echo "server winpid $SPID"
for i in $(seq 1 60); do curl -s -o /dev/null $BASE/ && break; sleep 1; done
shot() { timeout 90 "$EDGE" --headless --disable-gpu --user-data-dir=$W/edgeprof --hide-scrollbars --window-size=1280,1800 --screenshot=$S/$1.png "$2" > /dev/null 2>&1; echo "$1 $(stat -c %s $S/$1.png 2>/dev/null) $2"; }
shot 11-tca-fullcolour "$BASE/search?q=TCA+cycle"
shot 12-glycolysis-yeast-dedup "$BASE/search?q=glycolysis+in+Saccharomyces+cerevisiae"
shot 13-pathbank-alanine-fullcolour "$BASE/pathbank/PW000001"
shot 14-enterobactin-ecoli "$BASE/search?q=enterobactin+biosynthesis+E.+coli"
taskkill //PID $SPID //T //F
sleep 1
echo "survivors: $(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'live_app\.py' }).Count" | tr -d '\r')"
echo "LIVE2_DONE"
