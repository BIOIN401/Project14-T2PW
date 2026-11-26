import sys, runpy
sys.path.insert(0, "C:/t/ws-search/src")
sys.argv = ["export_pathbank_index.py", "--env-file", "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.env", "--out", "C:/t/ws-search-work/index/pathbank_index.sqlite"]
runpy.run_path("C:/t/ws-search/scripts/webserver/export_pathbank_index.py", run_name="__main__")
