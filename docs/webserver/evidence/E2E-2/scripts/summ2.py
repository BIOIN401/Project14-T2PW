import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
parts = re.split(r"(?m)^Timeout \(0:05:00\)!\s*$", txt)
import datetime
t0 = datetime.datetime(2026,11,5,20,51,8)
for i, d in enumerate(parts[1:], 1):
    print("== dump %d  ~%s" % (i, (t0+datetime.timedelta(seconds=300*i)).strftime("%H:%M:%S")))
    threads = re.split(r"(?m)^(?=Thread |Current thread )", d)
    for t in threads:
        if "streamlit_app.py" not in t: continue
        frames = re.findall(r'File "([^"]+)", line (\d+) in (\S+)', t)
        sp = [(f[0].replace("\\","/").split("/")[-1], f[1], f[2]) for f in frames]
        print("  lib top3:", [ "%s:%s %s"%x for x in sp[:3]])
        pk = [x for x in sp if any(k in x[0] for k in ("urllib3","requests","pymysql","mysql","httpx","openai","socket.py","ssl.py","http"))]
        print("  net libs:", sorted(set(x[0] for x in pk)))
        t2 = [ "%s:%s %s"%x for x,f in zip(sp,frames) if "/t2pw/" in f[0].replace("\\","/") and "streamlit_app" not in f[0] and "driver" not in f[0] and "worker" not in f[0] and "bridge" not in f[0]]
        print("  t2pw:", t2[:8])
