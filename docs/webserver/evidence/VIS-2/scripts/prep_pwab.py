"""VIS-2 A/B after arm: the same 17 files as VIS-1 (8 showcases + V7b idx 7, 10, 14, 116, 126, 128, 11, 134, 114) from
the VIS-2 replay LABEL. The base arm is VIS-1's own after arm (C:/t/vis-pw1-work/pw/after_final2, 49ab4da8 = this
branch's base). Screenshots: enterobactin (bacterium), ceramide (eukaryote), chlorophyll (plant), lipid A.
usage: prep_pwab.py LABEL"""
import os
import shutil
import sys

W = "C:/t/vis-pw2-work"
L = sys.argv[1]
SLUGS = ["menaquinone", "enterobactin", "lipid-a", "ceramide-sphingolipid", "chlorophyll", "coenzyme-a", "sour-taste", "furochromone"]
SHOTS = {"enterobactin", "ceramide-sphingolipid", "chlorophyll", "lipid-a"}
V7B = [7, 10, 14, 116, 126, 128, 11, 134, 114]
d = f"{W}/pwab/after_{L}"
os.makedirs(d, exist_ok=True)
shows, paths = [], []
for s in SLUGS:
    p = f"{d}/after_show_{s}.pwml"
    shutil.copyfile(f"{W}/show/{L}/{s}/pathway.pwml", p)
    (shows if s in SHOTS else paths).append(p)
for i in V7B:
    p = f"{d}/after_v7b_{i}.pwml"
    shutil.copyfile(f"{W}/replay/{L}/pwml/{i}.pwml", p)
    paths.append(p)
open(f"{W}/pwab/after_{L}_list.txt", "w", newline="\n").write("\n".join(paths) + "\n")
open(f"{W}/pwab/after_{L}_shots.txt", "w", newline="\n").write("\n".join(shows) + "\n")
print("after", len(shows), len(paths))
