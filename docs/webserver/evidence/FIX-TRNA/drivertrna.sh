#!/bin/bash
# FIX-TRNA live A/B: chlorophyll PMC5130987, Gemma, 3 legs flag-off (off) and 3 flag-on (on), interleaved.
# Both arms: T2PW_REACTION_VERIFIER=1 (what the server runs). on adds T2PW_AUDIT_NA_DECLARED=1.
# Resume-safe: a leg with an .end file is skipped. Stops on PAUSE (never creates it). 5-min gap after each leg.
A=C:/t/acc-runs
L4=$A/logs-trna
mkdir -p $L4 $A/legs-trna
p=PMC5130987
for r in 1 2 3; do
  for arm in off on; do
    L=$arm$r-$p
    if [ -e $L4/$L.end ]; then continue; fi
    if [ -e $A/PAUSE ]; then echo "$(date -Is) STOPPED_ON_PAUSE before $L"; exit 0; fi
    mem=$(powershell.exe -NoProfile -Command '$o=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Objects).Processes; $f=[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1); "$o $f"' | tr -d '\r')
    echo "$(date -Is) before $L kernel_process_objects/free_GB: $mem" >> $L4/mem.log
    free=${mem#* }
    if [ "${free%.*}" -lt 5 ]; then
      echo "$(date -Is) LOW_RAM before $L (free ${free} GB), stopping (PAUSE not created)" | tee -a $L4/mem.log
      exit 0
    fi
    if [ -e $L4/$L.start ] && [ -d $A/legs-trna/$L ]; then
      mv $A/legs-trna/$L $A/legs-trna/partial-$L-$(date +%s)
      mv $L4/$L.start $L4/$L.start.interrupted-$(date +%s)
    fi
    echo "$L" > $L4/inflight.txt
    if [ $arm = off ]; then
      "C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/scratch/fixtrna/legtrna.sh $L $p T2PW_REACTION_VERIFIER=1
    else
      "C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/scratch/fixtrna/legtrna.sh $L $p T2PW_REACTION_VERIFIER=1 T2PW_AUDIT_NA_DECLARED=1
    fi
    rm -f $L4/inflight.txt
    sleep 300
  done
done
echo DRIVERTRNA_DONE
