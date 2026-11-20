"""Replace the 'A/B NOT RUN' paragraph of DRAW-PATHWHIZ.md section 6 with the A/B results (run after the reboot)."""
t = "C:/t/lane-pwdraw/docs/pwml_output/DRAW-PATHWHIZ.md"
s = open(t, encoding="utf-8").read()
a = s.index("**PathWhiz A/B on `23ffd20c`: NOT RUN")
b = s.index("then `pw_ab.py pw/base2 pw/after5 OUT.json` and `compare_page.py` (point it at `after5`).")
b = b + len("then `pw_ab.py pw/base2 pw/after5 OUT.json` and `compare_page.py` (point it at `after5`).")
rows = open("C:/t/lane-pwdraw-work/metrics/ab_rows_after5.md", encoding="utf-8").read().strip()
new = f"""**PathWhiz A/B on `23ffd20c`** (run after the PC reboot, 14 GB free; `final4.sh` through the queue, holder `pwdraw`,
log `final4.log`, BW_EXIT 0, 0 survivors; project `t2pw-pwdraw` only). The first attempt was blocked by memory: with
Docker up the machine sat under the queue's 4 GB floor (41 GB committed on 31 GB), so the A/B waited for the reboot.
Base arm `pw/base2` (`fd814b8e` files; the 14 imported before were reused with `--resume`, the 3 F3 files imported
now); after arm `pw/after5` (replay `after5` = `23ffd20c`, every file imported fresh). `pathwhiz_ab_23ffd20c.json`.

| 17 files | base | after |
|---|---|---|
| imports successful | 17/17 | 17/17 |
| compounds / proteins / complexes / reactions | 210 / 159 / 168 / 167 | same |
| transports / transport transporters / interactions | 32 / 18 / 39 | same |
| compound locations / nucleic acids / states / labels | 322 / 20 / 67 / 70 | same |
| importer messages | 89 | **89** |

No created count and no message count changed in any file. The files the coordinator asked for (the 6 that F1
changed, then the 3 F3 files):

{rows}

What PathWhiz draws (screenshots `pw/after5/screenshots/`, showcases side by side in `C:/t/lane-pwdraw-work/compare5/`):
idx 134 shows the mitochondrion as two red Mitochondria Membrane lines, with the two cytoplasm-kind states
("cytosolic", "human carnitine biosynthesis") as separate cells beside it, as F1 intends; idx 114 shows the green Cell
Wall band around the cyan cell, with the transporter DltD on the cell membrane.

**Missing template probe** (`after_extra_missing-template.pwml`: the furochromone showcase with membrane template
99999). The import succeeds with **0 messages** and creates the membrane row (`membrane_visualizations: 1`); the viewer
logs **no console error** and draws the membrane with the Plasma Membrane template's look (`data-default-color
#4BC8F2`, 50-unit band; the template's own default colour). This **corrects** the source-based guess above ("the
browser throws"): the viewer falls back to the default. PathWhiz's SBGN export (`sbgn_instance`, which reads the
template's description) was not exercised. Every id T2PW writes exists, so this only matters for hand-edited files."""
s = s[:a] + new + s[b:]
open(t, "w", encoding="utf-8", newline="\n").write(s)
print("updated")
