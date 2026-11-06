import os, sys, time
if os.environ.get("T2PW_E2E_STACKS") and not any("bounded_run" in a for a in sys.argv):
    import faulthandler
    _f = open("C:/t/e2e2/stacks/%d.txt" % os.getpid(), "w", buffering=1, encoding="utf-8")
    _f.write("HEADER pid=%d argv=%r time=%s\n" % (os.getpid(), sys.argv, time.strftime("%Y-%m-%d %H:%M:%S")))
    _f.flush()
    faulthandler.dump_traceback_later(300, repeat=True, file=_f)
