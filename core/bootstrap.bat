@echo off
REM Synapse One - portable one-time setup (CMD).
REM Creates a fresh virtual environment and installs the package with the
REM dev + server extras. Safe from any location: paths are relative to here.
REM
REM Portable by design:
REM   - locates a real Python interpreter (py launcher first, PATH next)
REM   - rebuilds a .venv that is broken or bound to another machine
REM   - every path is quoted and resolved from %~dp0
REM Usage:  bootstrap.bat

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "ROOT=%CD%"
set "VENV_DIR=%ROOT%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo == Synapse One setup ==
echo Project root : %ROOT%
echo.

REM ---- 1. find a working Python interpreter -----------------------------
set "PYTHON="
for /f "delims=" %%p in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PYTHON=%%p"
if not defined PYTHON (
    for /f "delims=" %%p in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PYTHON=%%p"
)
if not defined PYTHON (
    echo [ERROR] No Python 3 found. Install Python 3.11+ from https://www.python.org/downloads/ and retry.
    exit /b 1
)

"%PYTHON%" -c "import sys;sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Python found is below 3.11: "%PYTHON%" ^(recommended 3.11+; trying anyway^).
) else (
    echo Python     : "%PYTHON%"
)
"%PYTHON%" --version
echo Virtual env  : %VENV_DIR%
echo.

REM ---- 2. verify an existing .venv actually runs ------------------------
REM Rebuild when the venv is broken OR was created by a different Python
REM version than the one we just selected (stale/relocated environments).
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys;print('.'.join(map(str, sys.version_info[:2])))" >"%TEMP%\_syn_venv_ver.txt" 2>nul
    set /p VENV_VER=<"%TEMP%\_syn_venv_ver.txt"
    "%PYTHON%" -c "import sys;print('.'.join(map(str, sys.version_info[:2])))" >"%TEMP%\_syn_py_ver.txt" 2>nul
    set /p PY_VER=<"%TEMP%\_syn_py_ver.txt"
    del "%TEMP%\_syn_venv_ver.txt" "%TEMP%\_syn_py_ver.txt" >nul 2>&1
    "%VENV_PY%" --version >nul 2>&1
    if errorlevel 1 goto :rebuild_venv
    if not "!VENV_VER!"=="!PY_VER!" (
        echo [WARN] .venv was built with Python !VENV_VER!, but !PY_VER! is available now - rebuilding it.
        goto :rebuild_venv
    )
    goto :check_install
)

:rebuild_venv
echo [WARN] .venv is broken, relocated, or built for a different Python version - rebuilding it.
rmdir /s /q "%VENV_DIR%"

REM ---- 3. create the venv if missing ------------------------------------
echo Creating virtual environment (.venv) ...
"%PYTHON%" -m venv "%VENV_DIR%"
if not exist "%VENV_PY%" (
    echo [ERROR] Failed to create .venv from "%PYTHON%".
    exit /b 1
)
goto :install

:check_install
echo Virtual environment found.
"%VENV_PY%" -c "import synapse" >nul 2>&1
if errorlevel 1 goto :install
echo synapse already installed - nothing to do.
goto :done

:install
echo Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip
echo Installing synapse-core (editable) with dev + server extras ...
"%VENV_PY%" -m pip install -e ".[dev,server]"
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your internet connection and retry.
    exit /b 1
)

:done
echo.
echo Setup complete.
echo   Start the backend :  start.bat
echo   Activate manually :  .venv\Scripts\activate.bat
endlocal
