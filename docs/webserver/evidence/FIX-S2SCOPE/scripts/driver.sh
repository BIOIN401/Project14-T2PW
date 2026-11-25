#!/bin/bash
# FIX-S2SCOPE A/B driver (from acc-runs/scripts/driver5g.sh): enterobactin x 3 runs x 2 arms, interleaved
# off1 on1 off2 on2 off3 on3, 5-min gap. Resume-safe (.end skips); low RAM (<5 GB) creates PAUSE and stops.
A=C:/t/acc-runs
L=$A/logs-s2scope
mkdir -p $L $A/legs-s2scope $A/legs-s2scope_partial
for r in 1 2 3; do
  for arm in off on; do
    G=$arm$r-PMC12096016
    if [ -e $L/$G.end ]; then continue; fi
    if [ -e $A/PAUSE ]; then echo "$(date -Is) STOPPED_ON_PAUSE before $G" | tee -a $L/mem.log; exit 0; fi
    mem=$(powershell.exe -NoProfile -Command '$o=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Objects).Processes; $f=[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1); "$o $f"' | tr -d '\r')
    echo "$(date -Is) before $G kernel_process_objects/free_GB: $mem" >> $L/mem.log
    free=${mem#* }
    if [ "${free%.*}" -lt 5 ]; then touch $A/PAUSE; echo "$(date -Is) PAUSED_LOW_RAM before $G (free ${free} GB)" | tee -a $L/mem.log; exit 0; fi
    if [ -e $L/$G.start ] && [ -d $A/legs-s2scope/$G ]; then
      mv $A/legs-s2scope/$G $A/legs-s2scope_partial/$G-$(date +%s)
      mv $L/$G.start $L/$G.start.interrupted-$(date +%s)
    fi
    echo "$G" > $L/inflight.txt
    if [ $arm = off ]; then
      "C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/scratch/s2scope/leg.sh $G
    else
      "C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/scratch/s2scope/leg.sh $G T2PW_STAGE2_RESPECT_SCOPE=1
    fi
    rm -f $L/inflight.txt
    sleep 300
  done
done
echo DRIVER_S2SCOPE_DONE
