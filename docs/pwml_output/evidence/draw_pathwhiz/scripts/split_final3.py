"""Split final3.sh: keep tests/replay/content check/suite (Docker stopped: RAM under the 4 GB floor with it up);
the PathWhiz A/B moves to final4.sh, run in a second hold with Docker started again."""
W = "C:/t/lane-pwdraw-work/scripts/"
s = open(W + "final3.sh", encoding="utf-8").read()
cut = s.index("$PY -I -X utf8 $W/scripts/prep_pwab.py base $L")
head, tail = s[:cut], s[cut:]
head = head.replace("# focused tests, the 97 + 8 replays (after5), content check (97 and 8), suite2, then the PathWhiz A/B:\n"
                    "# base arm into pw/base2 with --resume (same base files as before, so only the new F3 files import),\n"
                    "# after arm fresh into pw/after5; then the stack goes down (volumes kept).\n",
                    "# focused tests, the 97 + 8 replays (after5), content check (97 and 8), suite2. The PathWhiz A/B is final4.sh\n"
                    "# (Docker had to be stopped: free RAM 3 GB, under the 4 GB gate floor, with it up).\n")
open(W + "final3.sh", "w", encoding="utf-8", newline="\n").write(head + 'echo "FINAL3_DONE"\n')
four = ('#!/bin/bash\n'
        '# Review round 1 PathWhiz A/B (second hold, Docker started again): base arm into pw/base2 with --resume (same base\n'
        '# files as before, so only the new F3 files import), after arm (replay after5 = HEAD) fresh into pw/after5,\n'
        '# then the stack goes down (volumes kept).\n'
        'W=C:/t/lane-pwdraw-work\n'
        'PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe\n'
        'L=after5\n'
        'echo "HEAD $(git -C C:/t/lane-pwdraw rev-parse HEAD)"\n' + tail.replace('echo "FINAL3_DONE"', 'echo "FINAL4_DONE"'))
open(W + "final4.sh", "w", encoding="utf-8", newline="\n").write(four)
print("split")
