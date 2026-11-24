"""Copy the 8 showcase PWMLs of a replay label to cc_show/<label>/<slug>.pwml (content_check wants one folder).
usage: cc_copy.py LABEL"""
import os
import shutil
import sys

W = "C:/t/lane-pwdraw-work"
L = sys.argv[1]
os.makedirs(f"{W}/cc_show/{L}", exist_ok=True)
for s in ["menaquinone", "enterobactin", "lipid-a", "ceramide-sphingolipid", "chlorophyll", "coenzyme-a", "sour-taste", "furochromone"]:
    shutil.copyfile(f"{W}/show/{L}/{s}/pathway.pwml", f"{W}/cc_show/{L}/{s}.pwml")
print("copied", L)
