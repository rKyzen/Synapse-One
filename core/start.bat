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

REM --- Adaptive model installation (onboarding gate) -----------------------
REM First launch (and every launch with missing models): detect, prompt,
REM run SetupModels.bat, then continue startup. Skip with
REM SYNAPSE_SKIP_MODEL_SETUP=1.
if "%SYNAPSE_SKIP_MODEL_SETUP%"=="1" goto :start_server
if exist "%~dp0..\SetupModels.bat" (
    echo Checking AI models ...
    call "%~dp0..\SetupModels.bat" --check
    set "MODEL_STATUS=!errorlevel!"
    if not "!MODEL_STATUS!"=="0" (
        echo.
        echo Synapse needs to install AI models before first use.
        if "!MODEL_STATUS!"=="2" (
            echo   Ollama is not installed. SetupModels.bat will install it from the bundled OllamaSetup.exe ^(or download it^).
        ) else (
            echo   Models for this machine's hardware tier are missing and will be downloaded.
        )
        set /p CHOICE="Proceed with model installation? [Y/n] "
        if /i not "!CHOICE!"=="n" (
            call "%~dp0..\SetupModels.bat" --yes
            if errorlevel 1 (
                echo [WARN] Model installation did not fully complete - starting anyway, some features may be limited.
            )
        ) else (
            echo [WARN] Skipping model installation - starting anyway, AI features may be limited.
        )
    )
)

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
