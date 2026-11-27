#!/bin/bash
# VIS-9 chain in one heavy hold: A/B replay + content checks, focused tests, suite2, PathWhiz import of the 9 + PW000779.
GB="C:/Program Files/Git/bin/bash.exe"; W=C:/t/vis-pw9-work
echo "=== ab1"; "$GB" $W/scripts/ab1.sh h1
echo "=== focused"; "$GB" $W/scripts/focused.sh f1
echo "=== suite2"; "$GB" C:/t/vis-runs/suite2.sh C:/t/vis-pw9 vis9-suite2-s1
echo "=== pwab"; "$GB" $W/scripts/pwab9.sh h1 p1
echo "CHAIN1_DONE"
