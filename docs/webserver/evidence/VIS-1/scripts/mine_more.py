"""Second pass over mined_edges.json: shared reaction points, tangent vs side, offsets."""
import collections, json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
rows = [r for r in d["rows"] if r["proc"] == "reaction"]
c = collections.Counter
print("member side -> member dir:", c(f'{r["member_side"]}{r["member_dir"]}' for r in rows).most_common(10))
print("hub side -> hub dir:", c(f'{r["hub_side"]}{r["hub_dir"]}' for r in rows).most_common(10))
print("member offset (|off|<=0.05):", sum(1 for r in rows if isinstance(r["member_off"], (int, float)) and abs(r["member_off"]) <= 0.05), "of", len(rows))
print("hub offset (|off|<=0.05):", sum(1 for r in rows if isinstance(r["hub_off"], (int, float)) and abs(r["hub_off"]) <= 0.05), "of", len(rows))
by = collections.defaultdict(list)
for r in rows:
    by[r["rx"]].append(r)
same_l = same_r = n_l = n_r = opp = nopp = 0
for rx, rs in by.items():
    for side in ("Left", "Right"):
        pts = {tuple(r["hub_pt"]) for r in rs if r["side"] == side}
        if len([r for r in rs if r["side"] == side]) > 1:
            if side == "Left":
                n_l += 1; same_l += len(pts) == 1
            else:
                n_r += 1; same_r += len(pts) == 1
    ls = {r["hub_side"] for r in rs if r["side"] == "Left" and not r["currency"]}
    rs_ = {r["hub_side"] for r in rs if r["side"] == "Right" and not r["currency"]}
    if len(ls) == 1 and len(rs_) == 1:
        nopp += 1
        a, b = ls.pop(), rs_.pop()
        opp += {a, b} in ({"L", "R"}, {"T", "B"})
print(f"reactions with >1 substrate edges sharing one hub point: {same_l}/{n_l}; products: {same_r}/{n_r}")
print(f"main substrate and main product on opposite hub sides: {opp}/{nopp}")
# side compound position relative to the hub point: perpendicular offset and along-shaft offset
off = []
for r in rows:
    if r["currency"]:
        off.append((r["side"], r["hub_side"], r["dx"], r["dy"]))
alongs, perps = [], []
for side, hs, dx, dy in off:
    if hs in ("L", "R"):
        alongs.append(abs(dx)); perps.append(abs(dy))
    elif hs in ("T", "B"):
        alongs.append(abs(dy)); perps.append(abs(dx))
import statistics as st
print("side compound: hub->member |along shaft| median", st.median(alongs), "quartiles", st.quantiles(alongs, n=4))
print("side compound: hub->member |perpendicular| median", st.median(perps), "quartiles", st.quantiles(perps, n=4))
# main chain step: distance between a main substrate member point and the hub point along the shaft
ms = [(abs(r["dx"]) if r["hub_side"] in "LR" else abs(r["dy"])) for r in rows if not r["currency"] and r["hub_side"] in ("L", "R", "T", "B")]
print("main member -> hub along shaft: median", st.median(ms), st.quantiles(ms, n=4))
straight = sum(1 for r in rows if not r["currency"] and (r["dx"] == 0 or r["dy"] == 0))
print("main edges perfectly straight (dx or dy 0):", straight, "of", sum(1 for r in rows if not r["currency"]))
near = sum(1 for r in rows if not r["currency"] and (abs(r["dx"]) <= 5 or abs(r["dy"]) <= 5))
print("main edges within 5 of straight:", near)
print("sizes of side-compound members:", c(f'{r["w"]}x{r["h"]}' for r in rows if r["currency"]).most_common(6))
