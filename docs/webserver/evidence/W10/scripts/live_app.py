"""W10 live check: serve the worktree's site on 127.0.0.1:8811 with the real PathBank index and the local mirror."""
import os
import sys

sys.path.insert(0, "C:/t/ws-search/src")
from dotenv import load_dotenv

load_dotenv("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.env", override=False)  # LLM + NCBI settings; never printed
os.environ["T2PW_PATHBANK_INDEX"] = "C:/t/ws-search-work/index/pathbank_index.sqlite"
os.environ["T2PW_PATHBANK_FILES"] = "C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations"
import uvicorn

from t2pw.web.app import create_app

app = create_app("C:/t/ws-search-work/live/jobs.sqlite", "C:/t/ws-search-work/live/work", "", rate_limit=50)
uvicorn.run(app, host="127.0.0.1", port=8811, log_level="info")
