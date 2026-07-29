[CmdletBinding()]
param(
    [Parameter()]
    [string]$Python = "python",

    [Parameter()]
    [switch]$Development
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if ((Test-Path -LiteralPath $cargoBin) -and ($env:Path -notlike "*$cargoBin*")) {
    $env:Path = "$cargoBin;$env:Path"
}

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "Rust Cargo is unavailable. Install Rust 1.82 or newer with rustup."
}

$pythonEnvironment = (& $Python -c "import sys; print(sys.prefix)").Trim()
if (($LASTEXITCODE -ne 0) -or (-not $pythonEnvironment)) {
    throw "Could not determine the selected Python environment."
}
$env:VIRTUAL_ENV = $pythonEnvironment

$maturinArguments = @(
    "-m",
    "maturin",
    "develop"
)
if (-not $Development) {
    $maturinArguments += "--release"
}

Push-Location (Join-Path $projectRoot "native\python-cpu")
try {
    & $Python @maturinArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Maturin native CPU build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Installed _renewable_huber_native_cpu into the selected Python environment."
