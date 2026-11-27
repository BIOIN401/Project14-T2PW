import runpy, sys
sys.path.insert(0, "C:/t/vis-pw9/src")
sys.argv = ["refresh_pwc_id_skip", "--snapshot", "C:/t/vis-pw9/docs/webserver/evidence/VIS-9/db/pb_mismatch.json",
            "--out", "C:/t/vis-pw9/data/pwc_id_skip.json"]
runpy.run_module("t2pw.pwml.refresh_pwc_id_skip", run_name="__main__")
