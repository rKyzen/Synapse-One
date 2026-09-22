# Synapse One - portable one-time setup (PowerShell).
#
# Creates a fresh virtual environment and installs the package with the
# dev + server extras. Safe to run from ANY location: every path is resolved
# relative to this script, never hardcoded. Run it again any time to repair
# a broken environment.
#
# Portable by design:
#   - locates a real Python interpreter (py launcher first, PATH next)
#   - rebuilds a .venv that is broken or bound to another machine
#   - every path is quoted and resolved from $PSScriptRoot
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

# 1. Find a working Python interpreter (py launcher first, PATH next).
$Python = $null
foreach ($probe in @(@("py", "-3"), @("python"))) {
    try {
        $exe = & $probe[0] @($probe[1..($probe.Count - 1)]) -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) {
            $Python = ($exe | Select-Object -Last 1).Trim()
            break
        }
    } catch {
        # try the next candidate
    }
}
if (-not $Python) {
    Write-Host "[ERROR] No Python 3 found. Install Python 3.11+ from https://www.python.org/downloads/ and retry." -ForegroundColor Red
    exit 1
}

# Check version >= 3.11 (recommended; keep going with a warning otherwise).
$verOk = $true
try {
    & $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) { $verOk = $false }
} catch { $verOk = $false }
if ($verOk) {
    Write-Host "Python     : $Python"
} else {
    Write-Host "[WARN] Python found is below 3.11: $Python (recommended 3.11+; trying anyway)." -ForegroundColor Yellow
}
& $Python --version
Write-Host "Virtual env  : $VenvDir"

# 2. Verify an existing .venv actually runs AND matches the selected Python
#    version; rebuild it when broken, relocated, or stale (built for another
#    Python version than what is available now).
$VenvPyVersion = $null
$SelectedVersion = $null
if (Test-Path -LiteralPath $VenvPy) {
    try {
        $VenvPyVersion = (& $VenvPy -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null | Select-Object -Last 1)
        $SelectedVersion = (& $Python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null | Select-Object -Last 1)
    } catch { $VenvPyVersion = $null }
    if ($VenvPyVersion -ne $SelectedVersion) {
        $reason = if ($null -eq $VenvPyVersion) { "broken or relocated" } else { "built for Python $VenvPyVersion (have $SelectedVersion)" }
        Write-Host "[WARN] .venv is $reason - rebuilding it." -ForegroundColor Yellow
        Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 3. Create the venv if it does not exist yet.
$Fresh = $false
if (-not (Test-Path -LiteralPath $VenvPy)) {
    Write-Host ""
    Write-Host "Creating virtual environment (.venv) ..."
    & $Python -m venv $VenvDir
    if (-not (Test-Path -LiteralPath $VenvPy)) {
        Write-Host "[ERROR] Failed to create .venv from $Python." -ForegroundColor Red
        exit 1
    }
    $Fresh = $true
} else {
    Write-Host ""
    Write-Host "Virtual environment found."
}

# 4. Make sure pip is current.
Write-Host "Upgrading pip ..."
& $VenvPy -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] pip upgrade failed." -ForegroundColor Red
    exit 1
}

# 5. Install the package (editable) if it is not importable.
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
