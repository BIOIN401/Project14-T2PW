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
REM
REM EXIT CODES (kept in step with scripts\batch_run.py and docs\batch_runner.md
REM section 2d). 3 is handled separately below and that separation is
REM load-bearing, not cosmetic: on a preflight refusal NO run directory is
REM created, so the generic "read SUMMARY.txt in the newest runs\ folder" line
REM would send the operator to the PREVIOUS night's summary and invite them to
REM debug a pipeline that never ran. That is the exact confusion -- a failed
REM night that looks like a night that happened -- the preflight was added to
REM prevent, and this window's last line is what the morning actually reads.
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
) else if "%EXITCODE%"=="3" (
    echo STOPPED BEFORE STARTING -- exit code 3: preflight failed.
    echo NOTHING was fetched and NO run folder was created, so there is no new
    echo SUMMARY.txt to read and nothing in runs\ changed. The message above
    echo names what could not be imported and the exact command to rerun with.
) else (
    echo DONE -- exit code %EXITCODE%: something did not pass.
    echo Read SUMMARY.txt and failures_by_code.txt in the newest runs\ folder.
)
echo ===========================================================================
echo.

pause
exit /b %EXITCODE%
