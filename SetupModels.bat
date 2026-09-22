@echo off
setlocal enabledelayedexpansion

REM =============================================================
REM  Synapse One - Adaptive Model Setup (SetupModels.bat)
REM
REM  Hardware-aware Ollama installation. All model recommendations,
REM  hardware thresholds, and download URLs are read from models.json
REM  (the single source of truth) via setup_models.ps1.
REM
REM  Usage:
REM    SetupModels.bat            interactive install (pauses at the end)
REM    SetupModels.bat --yes      install and exit without pausing
REM    SetupModels.bat --check    report only; no downloads
REM       exit codes: 0 = ready | 2 = Ollama missing | 3 = models missing
REM =============================================================

set "ROOT=%~dp0"

if /i "%~1"=="--check" goto :check_only

echo =========================================
echo  Synapse One - Adaptive Model Setup
echo =========================================
echo.

REM ---------------------------------------------
REM 1. Ollama: installed? install if missing.
REM ---------------------------------------------
echo Checking Ollama installation...
where ollama >nul 2>&1
if errorlevel 1 goto :install_ollama
echo   [OK] Ollama found.
goto :ollama_ready

:install_ollama
echo   [MISSING] Ollama is not installed.

REM 1a. Prefer a local installer next to this script; otherwise download it.
set "OLLAMA_INSTALLER="
if exist "%ROOT%OllamaSetup.exe" set "OLLAMA_INSTALLER=%ROOT%OllamaSetup.exe"
if defined OLLAMA_INSTALLER (
    echo   [OK] Using local installer: %OLLAMA_INSTALLER%
) else (
    echo   Downloading Ollama installer ...
    for /f "usebackq delims=" %%L in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%ROOT%setup_models.ps1' -Status" ^| findstr /B "INSTALL_URL="`) do set "%%L"
    if not defined INSTALL_URL set "INSTALL_URL=https://ollama.com/download/OllamaSetup.exe"
    curl -fSL -o "%TEMP%\OllamaSetup.exe" "%INSTALL_URL%"
    if errorlevel 1 (
        echo   [FAIL] Download failed. Install Ollama manually from https://ollama.com
        exit /b 1
    )
    set "OLLAMA_INSTALLER=%TEMP%\OllamaSetup.exe"
)

echo   Installing silently ...
"%OLLAMA_INSTALLER%" /S
if errorlevel 1 (
    echo   [FAIL] Ollama installer failed. Install Ollama manually from https://ollama.com
    exit /b 1
)

REM 1b. Verify the installation actually landed (PATH or known install dir).
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
where ollama >nul 2>&1
if errorlevel 1 (
    if not exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        echo   [FAIL] Ollama installation could not be verified. Install Ollama manually from https://ollama.com
        exit /b 1
    )
)
echo   [OK] Ollama installed.

:ollama_ready

REM make sure the Ollama service is running
tasklist | findstr /I "ollama.exe" >nul 2>&1
if errorlevel 1 (
    echo   Starting Ollama service ...
    start "" /B ollama serve >nul 2>&1
)

echo   Waiting for Ollama service ...
set /a TRIES=0
:wait_loop
curl -s -o nul "http://localhost:11434/api/tags" >nul 2>&1
if not errorlevel 1 goto :service_up
set /a TRIES+=1
if !TRIES! geq 90 (
    echo   [FAIL] Ollama service did not become ready.
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto :wait_loop
:service_up
echo   [OK] Ollama service is ready.
echo.

REM ---------------------------------------------
REM 2. Detect hardware and pick the tier.
REM ---------------------------------------------
echo Checking hardware ...
for /f "usebackq delims=" %%L in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%ROOT%setup_models.ps1' -Status"`) do (
    for /f "tokens=1,* delims==" %%A in ("%%L") do set "%%A=%%B"
)
echo Tier detected: %TIER% (%TIER_NAME%)
echo   Available RAM : %RAM_AVAILABLE_GB% GB (Total: %RAM_TOTAL_GB% GB)
echo   VRAM          : %VRAM_GB% GB
echo   CPU           : %CPU%
echo.

REM ---------------------------------------------
REM 3+4. Check installed models, pull only missing.
REM ---------------------------------------------
echo Checking installed models ...
set "INSTALLED_LIST="
set "INSTALLED_COUNT=0"
set "FAILED_LIST="
set "FAILED_COUNT=0"

for %%M in (%MODELS%) do (
    echo !PRESENT! | findstr /C:"%%M" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   Skipping %%M - already installed
    ) else (
        echo   Installing %%M ...
        ollama pull %%M
        if errorlevel 1 (
            echo   [FAIL] %%M could not be installed.
            set "FAILED_LIST=!FAILED_LIST! %%M"
            set /a FAILED_COUNT+=1
        ) else (
            echo   [OK] %%M installed.
            set "INSTALLED_LIST=!INSTALLED_LIST! %%M"
            set /a INSTALLED_COUNT+=1
        )
    )
)

REM ---------------------------------------------
REM 5. Summary
REM ---------------------------------------------
echo.
echo =========================================
echo  Summary
echo    Tier             : %TIER% (%TIER_NAME%)
echo    Installed models : !INSTALLED_COUNT! !INSTALLED_LIST!
echo    Skipped models   : %PRESENT_COUNT% %PRESENT%
echo    Failed downloads : !FAILED_COUNT! !FAILED_LIST!
echo =========================================

if !FAILED_COUNT! gtr 0 (
    echo Result: some models failed to install - see list above.
    if /i not "%~1"=="--yes" pause
    exit /b 1
)
echo Result: OK - all required models for this machine are available.
if /i not "%~1"=="--yes" pause
exit /b 0

REM ---------------------------------------------
REM Check-only mode: report, never download.
REM ---------------------------------------------
:check_only
echo Checking hardware ...
for /f "usebackq delims=" %%L in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%ROOT%setup_models.ps1' -Status"`) do (
    for /f "tokens=1,* delims==" %%A in ("%%L") do set "%%A=%%B"
)
echo Tier detected: %TIER% (%TIER_NAME%)
echo   Available RAM : %RAM_AVAILABLE_GB% GB (Total: %RAM_TOTAL_GB% GB)
echo   VRAM          : %VRAM_GB% GB
echo   CPU           : %CPU%
echo.
echo Checking installed models ...

for %%M in (%MODELS%) do (
    echo !PRESENT! | findstr /C:"%%M" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   Skipping %%M - already installed
    ) else (
        echo   [MISSING] %%M
    )
)
echo.
if not "!OLLAMA!"=="YES" (
    echo Result: Ollama is not installed. Run SetupModels.bat to install it.
    exit /b 2
)
if !MISSING_COUNT! gtr 0 (
    echo Result: %MISSING_COUNT% required models missing. Run SetupModels.bat to install them.
    exit /b 3
)
echo Result: OK - all required models for this machine are present.
exit /b 0


