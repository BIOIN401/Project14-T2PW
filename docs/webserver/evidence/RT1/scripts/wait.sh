#!/bin/bash
# wait up to 540 s for the shared heavy lock folder to disappear
for i in $(seq 1 18); do [ -d C:/t/heavylock ] || { echo clear; exit 0; }; sleep 30; done
echo STILL_HELD; exit 1
