"""Canvas size stats of a PWML set: median width, height, area, worst aspect."""
import sys, re, statistics, glob
ws, hs = [], []
for p in glob.glob(sys.argv[1] + "/*.pwml"):
    s = open(p, encoding="utf-8").read()
    m = re.search(r"<pathway-visualization>.*?<height[^>]*>(\d+)</height>\s*<width[^>]*>(\d+)</width>", s, re.S)
    if m:
        hs.append(int(m.group(1))); ws.append(int(m.group(2)))
ar = [w / h for w, h in zip(ws, hs)]
print(f"n={len(ws)} median w={statistics.median(ws):.0f} h={statistics.median(hs):.0f} area={statistics.median([w*h for w,h in zip(ws,hs)])/1e6:.1f}M "
      f"max w={max(ws)} aspect>3: {sum(a > 3 for a in ar)} aspect<0.5: {sum(a < 0.5 for a in ar)}")
