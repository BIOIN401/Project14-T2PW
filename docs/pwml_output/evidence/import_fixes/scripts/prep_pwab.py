"""Copy the A/B file set (8 showcase replays + V7b idx 7,10,14,116,126,128) of each arm under unique names, write list files."""
import os, shutil
W = "C:/t/pwml-fix-work"
SLUGS = ["menaquinone", "enterobactin", "lipid-a", "ceramide-sphingolipid", "chlorophyll", "coenzyme-a", "sour-taste", "furochromone"]
V7B = [7, 10, 14, 116, 126, 128]
for arm, show, rep in (("base", "base2", "base2"), ("after", "after", "after")):
    d = f"{W}/pwab/{arm}"
    os.makedirs(d, exist_ok=True)
    paths = []
    for s in SLUGS:
        p = f"{d}/{arm}_show_{s}.pwml"
        shutil.copy(f"{W}/show/{show}/{s}/pathway.pwml", p)
        paths.append(p)
    for i in V7B:
        p = f"{d}/{arm}_v7b_{i}.pwml"
        shutil.copy(f"{W}/replay/{rep}/pwml/{i}.pwml", p)
        paths.append(p)
    open(f"{W}/pwab/{arm}_list.txt", "w").write("\n".join(paths) + "\n")
    print(arm, len(paths))
