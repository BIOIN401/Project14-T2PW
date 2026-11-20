"""A/B file set per arm (8 showcase replays + V7b idx 7,10,14,116,126,128) under unique names; list + screenshot lists.
usage: prep_pwab.py BASE_LABEL AFTER_LABEL"""
import os
import shutil
import sys

W = "C:/t/lane-pwdraw-work"
SLUGS = ["menaquinone", "enterobactin", "lipid-a", "ceramide-sphingolipid", "chlorophyll", "coenzyme-a", "sour-taste", "furochromone"]
V7B = [7, 10, 14, 116, 126, 128]
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
        paths.append(p)
    # LF only: the shell reads the screenshot list with $(cat ...), and a CR would stay on every path
    open(f"{W}/pwab/{arm}_list.txt", "w", newline="\n").write("\n".join(paths) + "\n")
    open(f"{W}/pwab/{arm}_shots.txt", "w", newline="\n").write("\n".join(shows) + "\n")
    print(arm, len(shows), len(paths))
