#!/bin/bash
# W10 live check driver: start the site (own PID), query 7 prompts, screenshot with headless Edge, stop own PIDs.
W=C:/t/ws-search-work; S=$W/shots; PORT=8811; BASE=http://127.0.0.1:$PORT
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
mkdir -p $S $W/live
$PY -I $W/live_app.py > $W/logs/live_server.log 2>&1 &
SPID=$(cat /proc/$!/winpid)  # the Windows PID: taskkill does not know the MSYS one
echo "server pid $SPID"
for i in $(seq 1 60); do curl -s -o /dev/null $BASE/ && break; sleep 1; done
enc() { $PY -I -c "import sys,urllib.parse;print(urllib.parse.quote_plus(sys.argv[1]))" "$1"; }
shot() { name=$1; url=$2
  code=$(curl -s -o $S/$name.html -w "%{http_code}" "$url"); echo "$name $code $url"
  timeout 90 "$EDGE" --headless --disable-gpu --user-data-dir=$W/edgeprof --hide-scrollbars --window-size=1280,1800 --screenshot=$S/$name.png "$url" > /dev/null 2>&1
  ls -la $S/$name.png 2>/dev/null | awk '{print "  png", $5}'; }
shot 00-front "$BASE/"
shot 01-tca "$BASE/search?q=$(enc 'TCA cycle')"
shot 02-glycolysis-yeast "$BASE/search?q=$(enc 'glycolysis in Saccharomyces cerevisiae')"
shot 03-enterobactin "$BASE/search?q=$(enc 'enterobactin biosynthesis')"
shot 04-pmid "$BASE/search?q=$(enc 'PMID 23637638')"
shot 05-doi "$BASE/search?q=$(enc '10.1093/nar/gkad1041')"
PARA="The enterobactin biosynthetic pathway of Escherichia coli converts chorismate into the siderophore enterobactin. EntC converts chorismate to isochorismate. EntB hydrolyses isochorismate to 2,3-dihydro-2,3-dihydroxybenzoate, which EntA oxidises to 2,3-dihydroxybenzoate. EntE adenylates 2,3-dihydroxybenzoate and loads it onto the aryl carrier domain of EntB. EntF activates L-serine, and the EntB, EntE and EntF assembly line condenses three 2,3-dihydroxybenzoyl-serine units into the cyclic trilactone enterobactin."
shot 06-pasted "$BASE/search?q=$(enc "$PARA")"
shot 07-sentence-llm "$BASE/search?q=$(enc "how does baker's yeast make ergosterol?")"
shot 08-pathbank-alanine "$BASE/pathbank/PW000001"
# one queued job from the pasted text (no worker runs: nothing calls the LLM)
loc=$(curl -s -o /dev/null -w "%{redirect_url}" -X POST --data-urlencode "kind=text" --data-urlencode "text=$PARA" --data-urlencode "name=Enterobactin biosynthesis" --data-urlencode "organism_1=Escherichia coli" --data-urlencode "consent=1" $BASE/search/run)
echo "run redirect: ${loc:+/job/<sid>}"
[ -n "$loc" ] && shot 09-queued-job "$loc"
taskkill //PID $SPID //T //F
sleep 1
tasklist //FI "PID eq $SPID" | grep -c python || true
echo "LIVE_DONE"
