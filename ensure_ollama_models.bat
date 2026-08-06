@echo off
setlocal enabledelayedexpansion

echo =========================================
echo  Synapse One - Ollama Model Checker
echo =========================================
echo.

REM Required models list
set "REQUIRED_MODELS=llama3.1:8b qwen3:4b nomic-embed-text:latest qwen2.5:7b llama3.2:latest qwen2.5-coder:7b qwen2.5vl:7b"

REM -----------------------------------------
REM 1. Check / install Ollama
REM -----------------------------------------
echo [1/3] Checking Ollama installation...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo       Ollama not found. Installing...
    echo       Downloading installer...
    powershell -Command "irm https://ollama.com/download/OllamaSetup.exe -OutFile $env:TEMP\OllamaSetup.exe; Start-Process -Wait -FilePath $env:TEMP\OllamaSetup.exe -ArgumentList '/S'"
    if %errorlevel% neq 0 (
        echo       [FAIL] Ollama installation failed. Install manually from https://ollama.com
        exit /b 1
    )
    echo       [OK] Ollama installed.
    REM Refresh PATH
    set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
) else (
    echo       [OK] Ollama found.
)

REM -----------------------------------------
REM 2. Ensure Ollama service is running
REM -----------------------------------------
echo.
echo [2/3] Starting Ollama service...
tasklist | findstr /I "ollama.exe" >nul 2>&1
if %errorlevel% neq 0 (
    echo       Starting ollama serve in background...
    start /B ollama serve >nul 2>&1
    echo       Waiting for service to be ready...
    timeout /t 5 /nobreak >nul
) else (
    echo       [OK] Ollama service already running.
)

REM -----------------------------------------
REM 3. Check / install required models
REM -----------------------------------------
echo.
echo [3/3] Checking required models...
set "MISSING="
set "ALL_OK=1"

for %%M in (%REQUIRED_MODELS%) do (
    ollama list | findstr /R /C:"^%%M " >nul 2>&1
    if !errorlevel! equ 0 (
        echo       [FOUND] %%M
    ) else (
        echo       [MISSING] %%M - pulling...
        ollama pull %%M
        if !errorlevel! neq 0 (
            echo       [FAIL] Failed to pull %%M
            set "ALL_OK=0"
        ) else (
            echo       [OK] Pulled %%M
        )
        set "MISSING=!MISSING! %%M"
    )
)

echo =========================================
if %ALL_OK% equ 1 (
    echo RESULT: PASS - All required models are available.
    echo =========================================
) else (
    echo RESULT: FAIL - Some models could not be installed.
    echo Missing/failed: %MISSING%
    echo =========================================
)

echo.
pause