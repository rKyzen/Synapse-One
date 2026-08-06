# Synapse One - portable one-time setup (PowerShell).
#
# Creates a fresh virtual environment and installs the package with the
# dev + server extras. Safe to run from ANY location: every path is resolved
# relative to this script, never hardcoded. Run it again any time to repair
# a broken environment.
#
# Usage:
#   .\bootstrap.ps1

$ErrorActionPreference = "Stop"

$Root    = $PSScriptRoot
$VenvDir = Join-Path $Root ".venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"

# Run every relative reference (like ".", for pip) from this script's folder.
Push-Location $Root

Write-Host "== Synapse One setup =="
Write-Host "Project root : $Root"
Write-Host "Virtual env  : $VenvDir"

# 1. Create the virtual environment if it does not exist yet.
$Fresh = $false
if (-not (Test-Path -LiteralPath $VenvPy)) {
    Write-Host ""
    Write-Host "Creating virtual environment (.venv) ..."
    python -m venv $VenvDir
    if (-not (Test-Path -LiteralPath $VenvPy)) {
        Write-Host "[ERROR] Failed to create .venv. Is Python 3.11+ installed and on PATH?" -ForegroundColor Red
        exit 1
    }
    $Fresh = $true
} else {
    Write-Host ""
    Write-Host "Virtual environment found."
}

# 2. Make sure pip is current.
Write-Host "Upgrading pip ..."
& $VenvPy -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] pip upgrade failed." -ForegroundColor Red
    exit 1
}

# 3. Install the package (editable) if it is not importable.
if (-not $Fresh) {
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $VenvPy -c "import synapse" 2>$null | Out-Null
    $ErrorActionPreference = $prevEAP
    if ($LASTEXITCODE -ne 0) { $Fresh = $true }
}
if ($Fresh) {
    Write-Host "Installing synapse-core (editable) with dev + server extras ..."
    & $VenvPy -m pip install -e ".[dev,server]"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] pip install failed. Check your internet connection and retry." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "synapse already installed - nothing to do."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Start the backend :  .\start.ps1"
Write-Host "  Activate manually :  .\.venv\Scripts\Activate.ps1"

Pop-Location