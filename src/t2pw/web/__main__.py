"""``python -m t2pw.web --db jobs.sqlite --work-root work --base-url https://host``.

Serves the public site with uvicorn. Start the worker separately
(``python -m t2pw.web.worker --db ... --work-root ...``) with the SAME db and work root.
Behind a reverse proxy set ``T2PW_TRUSTED_PROXIES`` and ``T2PW_RATE_KEY_SECRET``.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m t2pw.web")
    p.add_argument("--db", default=os.environ.get("T2PW_DB", "t2pw_jobs.sqlite"))
    p.add_argument("--work-root", default=os.environ.get("T2PW_WORK_ROOT", "t2pw_work"))
    p.add_argument("--base-url", default=os.environ.get("T2PW_BASE_URL", ""))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)

    import uvicorn

    from t2pw.web.app import create_app

    app = create_app(a.db, a.work_root, a.base_url,
                     trusted_proxies=os.environ.get("T2PW_TRUSTED_PROXIES"))
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
