param(
    [switch]$Status,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$ConfigPath = Join-Path $PSScriptRoot "models.json"

function Get-Config {
    if (-not (Test-Path -LiteralPath $ConfigPath)) {
        throw "models.json not found next to setup_models.ps1: $ConfigPath"
    }
    return Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
}

function Get-TotalRamGb {
    $bytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    return [math]::Round($bytes / 1GB, 1)
}

function Get-CpuName {
    return (Get-CimInstance Win32_Processor).Name
}

function Get-TierInfo {
    param($Config, $RamGb, $CpuName)
    $ramOk = $RamGb -ge [double]$Config.hardware.tier2_min_ram_gb
    $cpuOk = $false
    foreach ($pattern in $Config.hardware.tier2_min_cpu_patterns) {
        if ($CpuName -match $pattern) { $cpuOk = $true; break }
    }
    if ($ramOk -and $cpuOk) {
        return @{ tier = "tier2"; name = $Config.tiers.tier2.display }
    }
    return @{ tier = "tier1"; name = $Config.tiers.tier1.display }
}

function Normalize-Id {
    param($Id)
    $id = $Id.Trim().ToLower()
    if ($id -eq "latest" -or $id -eq "max") { $id = "latest" }
    $id = $id -replace ':(latest|max)$', ''
    return $id
}

function Get-InstalledModels {
    try {
        $raw = & ollama list 2>$null
    } catch {
        return @()
    }
    if ($null -eq $raw -or $raw.Count -lt 2) { return @() }
    $names = @()
    for ($i = 1; $i -lt $raw.Count; $i++) {
        $parts = ($raw[$i] -split "\s+", 2)
        if ($parts.Count -ge 1 -and $parts[0]) {
            $names += (Normalize-Id $parts[0])
        }
    }
    return ($names | Sort-Object -Unique)
}

function Get-Status {
    $config = Get-Config
    $ramGb = Get-TotalRamGb
    $cpuName = Get-CpuName
    $tier = Get-TierInfo $config $ramGb $cpuName

    $modelIds = @()
    foreach ($m in $config.tiers.($tier.tier).models) { $modelIds += $m.id }
    $installed = @(Get-InstalledModels)
    $present = @()
    foreach ($id in $modelIds) {
        $norm = Normalize-Id $id
        if ($installed -contains $norm) { $present += $norm }
    }

    $ollamaOk = $false
    try { $null = Get-Command ollama -ErrorAction Stop; $ollamaOk = $true } catch { $ollamaOk = $false }

    Write-Output "OLLAMA=$(if ($ollamaOk) { 'YES' } else { 'NO' })"
    Write-Output "INSTALL_URL=$($config.ollama.install_url)"
    Write-Output "TIER=$($tier.tier)"
    Write-Output "TIER_NAME=$($tier.name)"
    Write-Output "RAM_GB=$ramGb"
    Write-Output "CPU=$cpuName"
    Write-Output "MODELS=$($modelIds -join ' ')"
    Write-Output "PRESENT=$($present -join ' ')"
    Write-Output "PRESENT_COUNT=$($present.Count)"
    Write-Output "MISSING_COUNT=$($modelIds.Count - $present.Count)"
}

if ($Status) {
    Get-Status
    exit 0
}

if ($Check) {
    $config = Get-Config
    $ollamaOk = $false
    try { $null = Get-Command ollama -ErrorAction Stop; $ollamaOk = $true } catch { $ollamaOk = $false }
    if (-not $ollamaOk) {
        Write-Output "Ollama is not installed. Run SetupModels.bat to install it."
        exit 2
    }
    $status = @(Get-Status)
    $missing = ($status | Where-Object { $_ -like "MISSING_COUNT=*" }) -replace "MISSING_COUNT=", ""
    if ([int]$missing -gt 0) {
        Write-Output "$missing required models missing. Run SetupModels.bat to install them."
        exit 3
    }
    Write-Output "All required models for this machine are present."
    exit 0
}

Write-Host "Usage: setup_models.ps1 -Status | -Check"
exit 1
