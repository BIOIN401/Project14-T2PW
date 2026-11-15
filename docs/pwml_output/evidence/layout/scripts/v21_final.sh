#!/bin/bash
# v2.1 final pass, one bounded job holding the heavy lock: measurements + focused tests, then the harness (TAG=v21b).
B="C:/Program Files/Git/usr/bin/bash.exe"; S=C:/t/lay/docs/pwml_output/evidence/layout/scripts
"$B" $S/v21_measure.sh
TAG=v21b "$B" $S/harness_tag.sh
