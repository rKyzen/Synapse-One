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
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
Push-Location $Root
try {
    & (Join-Path $Root "bootstrap.ps1")
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # --- Adaptive model installation (onboarding gate) ---------------------
    # First launch (and every launch with missing models): detect, prompt,
    # run SetupModels.bat, then continue startup. Skip with
    # $env:SYNAPSE_SKIP_MODEL_SETUP = "1".
    $SetupBat = Join-Path (Split-Path $Root -Parent) "SetupModels.bat"
    if ($env:SYNAPSE_SKIP_MODEL_SETUP -ne "1" -and (Test-Path -LiteralPath $SetupBat)) {
        Write-Host "Checking AI models ..."
        & $SetupBat --check
        $modelStatus = $LASTEXITCODE
        if ($modelStatus -ne 0) {
            Write-Host ""
            Write-Host "Synapse needs to install AI models before first use." -ForegroundColor Yellow
            if ($modelStatus -eq 2) {
                Write-Host "  Ollama is not installed. SetupModels.bat will install it from the bundled OllamaSetup.exe (or download it)."
            } else {
                Write-Host "  Models for this machine's hardware tier are missing and will be downloaded."
            }
            $choice = Read-Host "Proceed with model installation? [Y/n]"
            if ($choice -notmatch "^n") {
                & $SetupBat
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[WARN] Model installation did not fully complete - starting anyway (some features may be limited)." -ForegroundColor Yellow
                }
            } else {
                Write-Host "[WARN] Skipping model installation - starting anyway (AI features may be limited)." -ForegroundColor Yellow
            }
        }
    }

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
        & $VenvPy -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000 --reload
    } else {
        & $VenvPy -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000
    }
} finally {
    Pop-Location
}
