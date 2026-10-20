"""Suite-wide isolation from the operator's ``.env`` draw count (ORCH-754a, finding 4).

``t2pw.llm.client`` runs ``load_dotenv(override=True)`` when it is imported, so a
``T2PW_EXTRACTION_DRAWS=3`` line in the primary checkout's ``.env`` beats both the code
default and any shell value. Every Stage-1 call in the suite then issues three draws:
``tests/test_early_failure_replay.py`` failed 24 cases there ("the replay made 4 provider
call(s) but the case scripted only 3") and passed in a checkout with no ``.env``.

So every test starts with the variable UNSET -- the environment the suite was written
against, which means one draw. A test that exercises ensembling sets it itself
(``monkeypatch.setenv``), as ``tests/test_stage_one_ensemble.py`` does.

The pin also has to survive ``.env`` being loaded again MID-test, which happens whenever
the first import of ``t2pw.llm.client`` (or ``t2pw.config.ensure_dotenv_loaded``) falls
inside a test body: both bind ``dotenv.load_dotenv`` at that moment, so it is wrapped to
leave this one variable exactly as the test had it. ``client.py``'s override semantics are
unchanged for every other variable.
"""

from __future__ import annotations

import os

import dotenv
import pytest

#: ``t2pw.pipeline.stage_one_ensemble.ENV_VAR``, spelled out so this file imports nothing
#: from ``src`` (``tests/test_stage_one_ensemble.py`` pins the two equal).
DRAWS_ENV_VAR = "T2PW_EXTRACTION_DRAWS"


@pytest.fixture(autouse=True)
def _one_extraction_draw_unless_a_test_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DRAWS_ENV_VAR, raising=False)
    real = dotenv.load_dotenv

    def load_dotenv_keeping_the_draw_count(*args, **kwargs):  # type: ignore[no-untyped-def]
        before = os.environ.get(DRAWS_ENV_VAR)
        try:
            return real(*args, **kwargs)
        finally:
            if before is None:
                os.environ.pop(DRAWS_ENV_VAR, None)
            else:
                os.environ[DRAWS_ENV_VAR] = before

    monkeypatch.setattr(dotenv, "load_dotenv", load_dotenv_keeping_the_draw_count)
