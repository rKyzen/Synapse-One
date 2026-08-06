@echo off
REM Synapse One - portable one-time setup (CMD).
REM Creates a fresh virtual environment and installs the package with the
REM dev + server extras. Safe from any location: paths are relative to here.
REM Usage:  bootstrap.bat

setlocal
cd /d "%~dp0"

set "VENV_DIR=%CD%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo == Synapse One setup ==
echo Project root : %CD%
echo Virtual env  : %VENV_DIR%
echo.

if exist "%VENV_PY%" goto :check_install

echo Creating virtual environment (.venv) ...
python -m venv "%VENV_DIR%"
if not exist "%VENV_PY%" (
    echo [ERROR] Failed to create .venv. Is Python 3.11+ installed and on PATH?
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
