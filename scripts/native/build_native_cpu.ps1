[CmdletBinding()]
param(
    [Parameter()]
    [string]$Python = "python",

    [Parameter()]
    [switch]$Development,

    [Parameter()]
    [string]$OutputDirectory,

    [Parameter()]
    [switch]$NoInstall
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
    "build"
)
if (-not $Development) {
    $maturinArguments += "--release"
}
$wheelDirectory = if ($OutputDirectory) {
    if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    }
    else {
        [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
    }
}
else {
    Join-Path $projectRoot "build\native-cpu-wheel"
}
New-Item -ItemType Directory -Path $wheelDirectory -Force | Out-Null
$maturinArguments += @("--out", $wheelDirectory)

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

$wheel = Get-ChildItem -LiteralPath $wheelDirectory -Filter "renewable_huber_native_cpu-*.whl" |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if (-not $wheel) {
    throw "Maturin did not produce a renewable-huber-native-cpu wheel."
}
if (-not $NoInstall) {
    & $Python -m pip install --disable-pip-version-check --force-reinstall --no-deps $wheel.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install the native CPU wheel."
    }
    Write-Host "Installed _renewable_huber_native_cpu into the selected Python environment."
}

Write-Host "Built native CPU wheel: $($wheel.FullName)"
