# Synapse One - portable launcher (PowerShell).
#
# One command to start the backend from a clean machine:
#   - creates the virtual environment if missing
#   - installs the package if missing
#   - starts uvicorn on 127.0.0.1:8000
#
# No hardcoded paths: everything resolves relative to this file.
#
# Usage:
#   .\start.ps1
#   .\start.ps1 --reload        (auto-restart on code changes, for development)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
Push-Location $Root
try {
    & (Join-Path $Root "bootstrap.ps1")
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # uvicorn writes its normal INFO logs to stderr. Under PowerShell 5.1 a
    # "Stop" preference would turn that stderr output into a fatal error and
    # kill the still-running server, so relax it before launching.
    $ErrorActionPreference = "Continue"

    Write-Host ""
    Write-Host "Starting Synapse One backend ..."
    Write-Host "  Swagger UI : http://127.0.0.1:8000/docs"
    Write-Host "  Status     : http://127.0.0.1:8000/status"
    Write-Host "  Press CTRL+C to stop." 
    Write-Host ""

    if ($args -contains "--reload") {
        & .\.venv\Scripts\python.exe -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000 --reload
    } else {
        & .\.venv\Scripts\python.exe -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000
    }
} finally {
    Pop-Location
}
