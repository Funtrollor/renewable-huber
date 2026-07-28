param(
    [string]$Python = "python",
    [string]$OutputDirectory = "artifacts/nsight",
    [int]$Samples = 100000,
    [int]$Features = 90,
    [int]$BatchSize = 32768,
    [ValidateSet("float32", "float64")]
    [string]$DType = "float32",
    [ValidateSet("none", "l1")]
    [string]$Penalty = "none",
    [int]$LaunchCount = 50
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$profiler = Get-Command ncu -ErrorAction SilentlyContinue
if ($null -eq $profiler) {
    throw "Nsight Compute CLI (ncu) is not available on PATH."
}

$outputPath = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$reportPrefix = Join-Path $outputPath "native-core-p0-compute"
$metadataPath = Join-Path $outputPath "native-core-p0-compute.json"
$workload = Join-Path $PSScriptRoot "profile_cuda_update.py"

& $profiler.Source `
    --target-processes all `
    --set basic `
    --launch-count $LaunchCount `
    --force-overwrite `
    --export $reportPrefix `
    $Python $workload `
    --samples $Samples `
    --features $Features `
    --batch-size $BatchSize `
    --dtype $DType `
    --penalty $Penalty `
    --input-location device `
    --warmup 1 `
    --repeats 1 `
    --metadata-output $metadataPath

if ($LASTEXITCODE -ne 0) {
    throw "Nsight Compute exited with code $LASTEXITCODE."
}

Write-Host "Nsight Compute report: $reportPrefix.ncu-rep"
Write-Host "Metadata: $metadataPath"
