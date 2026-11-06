import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
dumps = re.split(r"(?m)^Timeout \(0:05:00\)!\s*$", txt)
print("header:", dumps[0].strip()[:200])
for i, d in enumerate(dumps[1:], 1):
    # find thread that contains run_pathway / streamlit script thread
    threads = re.split(r"(?m)^(?:Thread|Current thread) ", d)
    out = []
    for t in threads:
        if "streamlit_app.py" in t:
            frames = re.findall(r'File "([^"]+)", line (\d+) in (\S+)', t)
            tw = [f for f in frames if "t2pw" in f[0].replace("\\", "/") and "streamlit_app" not in f[0]]
            lib = frames[0]
            out.append("  top=%s:%s %s | t2pw frames(top3)=%s" % (lib[0].split("\\")[-1], lib[1], lib[2],
                       [f[0].split("\\")[-1] + ":" + f[1] for f in tw[:3]]))
            lines = [f[0].split("\\")[-1] + ":" + f[1] for f in frames if "site-packages" not in f[0] and "t2pw" in f[0]]
    print("dump", i, "\n".join(out))
