"""A/B file set per arm under unique names; list + screenshot lists.
Files: 8 showcase replays + V7b idx 7,10,14,116,126,128 + review round 1 (F3): idx 11 and 134 (Mitochondria Membrane 84)
and 114 (Cell Wall Membrane 304). Screenshots: the 8 showcases, 134 and 114. The after arm also gets one probe file
(extra_missing-template: the furochromone showcase with its membrane template id set to 99999, which PathWhiz does not
have); pw_ab.py leaves "extra_" files out of the A/B.
usage: prep_pwab.py BASE_LABEL AFTER_LABEL"""
import os
import shutil
import subprocess
import sys

W = "C:/t/lane-pwdraw-work"
SLUGS = ["menaquinone", "enterobactin", "lipid-a", "ceramide-sphingolipid", "chlorophyll", "coenzyme-a", "sour-taste", "furochromone"]
V7B = [7, 10, 14, 116, 126, 128, 11, 134, 114]
V7B_SHOTS = {134, 114}
for arm, label in (("base", sys.argv[1]), ("after", sys.argv[2])):
    d = f"{W}/pwab/{arm}"
    os.makedirs(d, exist_ok=True)
    shows, paths = [], []
    for s in SLUGS:
        p = f"{d}/{arm}_show_{s}.pwml"
        shutil.copyfile(f"{W}/show/{label}/{s}/pathway.pwml", p)
        shows.append(p)
    for i in V7B:
        p = f"{d}/{arm}_v7b_{i}.pwml"
        shutil.copyfile(f"{W}/replay/{label}/pwml/{i}.pwml", p)
        (shows if i in V7B_SHOTS else paths).append(p)
    if arm == "after":
        p = f"{d}/{arm}_extra_missing-template.pwml"
        subprocess.run([sys.executable, "-I", f"{W}/scripts/make_missing_template.py",
                        f"{W}/show/{label}/furochromone/pathway.pwml", p], check=True)
        shows.append(p)
    # LF only: the shell reads the screenshot list with $(cat ...), and a CR would stay on every path
    open(f"{W}/pwab/{arm}_list.txt", "w", newline="\n").write("\n".join(paths) + "\n")
    open(f"{W}/pwab/{arm}_shots.txt", "w", newline="\n").write("\n".join(shows) + "\n")
    print(arm, len(shows), len(paths))
