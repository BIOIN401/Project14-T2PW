"""VIS-1 A/B file set per arm (copy of the DRAW-PATHWHIZ prep without the missing-template probe): the 8 showcase replays
+ V7b idx 7, 10, 14, 116, 126, 128, 11, 134, 114 = 17 files. Screenshots: the 8 showcases, 134 and 114.
usage: prep_pwab.py BASE_LABEL AFTER_LABEL"""
import os
import shutil
import sys

W = "C:/t/vis-pw1-work"
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
    open(f"{W}/pwab/{arm}_list.txt", "w", newline="\n").write("\n".join(paths) + "\n")
    open(f"{W}/pwab/{arm}_shots.txt", "w", newline="\n").write("\n".join(shows) + "\n")
    print(arm, len(shows), len(paths))
