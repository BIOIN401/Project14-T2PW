#!/bin/bash
# usage: svgshot.sh OUTDIR SVG... -- screenshot reference SVGs scaled to 1600 px wide with headless Edge
OUT=$1; shift
mkdir -p $OUT/edge-profile
E="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
for s in "$@"; do
  n=$(basename $s .svg)
  h=$OUT/$n.html
  echo "<html><body style='margin:0;background:#fff'><img src='file:///$s' style='width:1600px'></body></html>" > $h
  "$E" --headless --disable-gpu --user-data-dir=$OUT/edge-profile --screenshot=$OUT/$n.png --window-size=1600,1500 "file:///$h" 2>/dev/null
  echo "shot $n rc=$?"
done
