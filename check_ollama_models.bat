@echo off
setlocal enabledelayedexpansion

echo Checking for Ollama installation...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] Ollama is not installed or not in PATH.
    echo        Install from https://ollama.com
    exit /b 1
)

echo [OK] Ollama found.
echo.

echo Querying installed models...
set "MISSING="
set "FOUND="

for %%M in (
    "llama3.1:8b"
    "qwen3:4b"
    "nomic-embed-text:latest"
    "qwen2.5:7b"
    "llama3.2:latest"
    "qwen2.5-coder:7b"
    "qwen2.5vl:7b"
) do (
    ollama list | findstr /R /C:"^%%~M " >nul 2>&1
    if !errorlevel! equ 0 (
        echo [FOUND] %%~M
        set "FOUND=!FOUND! %%~M"
    ) else (
        echo [MISSING] %%~M
        set "MISSING=!MISSING! %%~M"
    )
)

echo.
if defined MISSING (
    echo =========================================
    echo Missing models: %MISSING%
    echo.
    echo To install them, run:
    for %%M in (%MISSING%) do echo   ollama pull %%M
    exit /b 1
) else (
    echo All required models are present.
    exit /b 0
)