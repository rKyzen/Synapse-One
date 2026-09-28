@echo off
REM Synapse One - portable launcher (CMD).
REM One command to start the backend from a clean machine:
REM   - rebuilds a broken/relocated .venv, installs the package if missing
REM   - checks/installs AI models for this machine's hardware tier
REM   - starts uvicorn on 127.0.0.1:8000
REM Usage:  start.bat [--reload] [extra uvicorn args...]

setlocal enabledelayedexpansion
cd /d "%~dp0"
set "ROOT=%CD%"
set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"

call bootstrap.bat
if errorlevel 1 exit /b 1

REM =========================================================================
REM  Pre-Release Phase: Interactive Hardware Tier Selection
REM =========================================================================
echo.
echo =====================================================================
echo  Synapse One [Pre-Release] - Select Hardware Tier
echo =====================================================================
echo   1) Tier 1 (Lightweight : Gemma 3 1B / Qwen 2.5 1.5B / Qwen 3 1.7B)
echo   2) Tier 2 (Mid-Range   : Gemma 3 4B / Qwen 2.5 Coder 7B / DeepSeek 7B)
echo   3) Tier 3 (High-End    : Gemma 3 12B / Qwen 2.5 Coder 14B / DeepSeek 14B)
echo   4) Tier 3+ (Workstation: Qwen 2.5 Coder 32B / DeepSeek 32B)
echo   5) Auto-Detect (Detect hardware profile automatically)
echo =====================================================================
set "TIER_CHOICE="
set /p TIER_CHOICE="Select Tier to work on [1-5, Default=5]: "

if "%TIER_CHOICE%"=="1" (
    set "SYNAPSE_HARDWARE_TIER=tier1"
    set "SYNAPSE_TIER=tier1"
    echo   [OK] Selected: Tier 1 ^(Lightweight^)
) else if "%TIER_CHOICE%"=="2" (
    set "SYNAPSE_HARDWARE_TIER=tier2"
    set "SYNAPSE_TIER=tier2"
    echo   [OK] Selected: Tier 2 ^(Mid-Range^)
) else if "%TIER_CHOICE%"=="3" (
    set "SYNAPSE_HARDWARE_TIER=tier3"
    set "SYNAPSE_TIER=tier3"
    echo   [OK] Selected: Tier 3 ^(High-End^)
) else if "%TIER_CHOICE%"=="4" (
    set "SYNAPSE_HARDWARE_TIER=tier3_plus"
    set "SYNAPSE_TIER=tier3_plus"
    echo   [OK] Selected: Tier 3+ ^(Workstation^)
) else (
    set "SYNAPSE_HARDWARE_TIER="
    set "SYNAPSE_TIER="
    echo   [OK] Selected: Auto-Detect
)

REM --- Skip automatic model installation in pre-release mode ----------------
set "SYNAPSE_SKIP_MODEL_SETUP=1"

:start_server
echo.
echo Starting Synapse One backend ...
echo   Swagger UI : http://127.0.0.1:8000/docs
echo   Status     : http://127.0.0.1:8000/status
echo   Press CTRL+C to stop.
echo.

if "%~1"=="--reload" (
    "%VENV_PY%" -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000 --reload %2 %3 %4 %5 %6 %7 %8 %9
) else (
    "%VENV_PY%" -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000 %1 %2 %3 %4 %5 %6 %7 %8 %9
)
endlocal
