[CmdletBinding()]
param(
    [Parameter()]
    [string]$Python = "python",

    [Parameter()]
    [switch]$Development
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Visual Studio Installer (vswhere.exe) was not found."
}

$installationPath = & $vswhere `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installationPath) {
    throw "Visual Studio 2022 C++ Build Tools were not found."
}

$vsDevCmd = Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path -LiteralPath $vsDevCmd)) {
    throw "VsDevCmd.bat was not found at $vsDevCmd."
}

# Import the x64 compiler environment from VsDevCmd into this PowerShell
# process so CMake, nvcc, Cargo, and Maturin all observe the same toolchain.
$compilerEnvironment = & cmd.exe /d /s /c `
    "`"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
if ($LASTEXITCODE -ne 0) {
    throw "Visual Studio x64 environment initialization failed."
}
foreach ($line in $compilerEnvironment) {
    if ($line -match "^([^=]+)=(.*)$") {
        # The Codex/CI launcher can supply case-variant environment keys.
        # PowerShell's Env: provider rejects those duplicates while the .NET
        # process API safely applies the Visual Studio environment in place.
        [Environment]::SetEnvironmentVariable(
            $Matches[1],
            $Matches[2],
            [EnvironmentVariableTarget]::Process
        )
    }
}

$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if ((Test-Path -LiteralPath $cargoBin) -and ($env:Path -notlike "*$cargoBin*")) {
    $env:Path = "$cargoBin;$env:Path"
}

foreach ($command in ("cargo", "cmake", "ninja", "nvcc")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required native build command '$command' is unavailable."
    }
}

# The Rust `cmake` crate otherwise selects the Visual Studio generator on
# Windows. Ninja avoids MSBuild file-tracker state and is also the generator
# required by this documented developer build.
if (-not [Environment]::GetEnvironmentVariable("CMAKE_GENERATOR", "Process")) {
    [Environment]::SetEnvironmentVariable(
        "CMAKE_GENERATOR",
        "Ninja",
        [EnvironmentVariableTarget]::Process
    )
}

$pythonEnvironment = (& $Python -c "import sys; print(sys.prefix)").Trim()
if (($LASTEXITCODE -ne 0) -or (-not $pythonEnvironment)) {
    throw "Could not determine the selected Python environment."
}
# `maturin develop` targets VIRTUAL_ENV. Point it at the interpreter selected
# by -Python instead of any unrelated shell environment that may be active.
$env:VIRTUAL_ENV = $pythonEnvironment

$maturinArguments = @(
    "-m",
    "maturin",
    "develop",
    "--manifest-path",
    "crates/rh-python-cuda/Cargo.toml",
    "--features",
    "cuda"
)
if (-not $Development) {
    $maturinArguments += "--release"
}

Push-Location (Join-Path $projectRoot "native")
try {
    & $Python @maturinArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Maturin native CUDA build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Installed _renewable_huber_native_cuda for the renewable_huber._native_cuda shim."
