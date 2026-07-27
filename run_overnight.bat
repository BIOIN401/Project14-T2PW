@echo off
REM ---------------------------------------------------------------------------
REM Double-click this to run the overnight T2PW batch.
REM
REM It resumes by itself: if the newest run in runs\ is recent (within 24h) and
REM incomplete it is continued and its finished paper+mode pairs are skipped, so
REM re-running after a reboot or a Ctrl+C is always safe. An older incomplete run
REM is left alone and a new one is started. Pass --fresh to force a new run.
REM
REM There is also a whole-night ceiling (--deadline, 10 hours by default): once it
REM passes, no further paper is started, the summary is written, and the run stays
REM resumable so the remainder finishes the next time this is launched.
REM
REM PAUSE at the end is deliberate: a double-click opens its own window, and
REM without it the window would vanish with the result in it.
REM ---------------------------------------------------------------------------
setlocal

REM cd to the repo root (the folder holding this .bat) so relative paths work.
cd /d "%~dp0"

echo ===========================================================================
echo T2PW overnight batch run
echo repo: %CD%
echo ===========================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe not found.
    echo Create the virtualenv first, then run this again.
    echo.
    pause
    exit /b 2
)

REM Any arguments given to this .bat are forwarded (e.g. --fresh, --limit 5).
.venv\Scripts\python.exe scripts\batch_run.py %*
set EXITCODE=%ERRORLEVEL%

echo.
echo ===========================================================================
if "%EXITCODE%"=="0" (
    echo DONE -- every paper+mode run passed.
) else (
    echo DONE -- exit code %EXITCODE%: something did not pass.
    echo Read SUMMARY.txt and failures_by_code.txt in the newest runs\ folder.
)
echo ===========================================================================
echo.

pause
exit /b %EXITCODE%
