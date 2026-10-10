param(
    [string]$Python = "python",
    # Empty selects a directory named after the configuration under
    # artifacts/nsight-compute, so two configurations never overwrite each other.
    [string]$OutputDirectory = "",
    [int]$Samples = 100000,
    [int]$Features = 90,
    [int]$BatchSize = 32768,
    [ValidateSet("float32", "float64")]
    [string]$DType = "float32",
    [ValidateSet("none", "l1")]
    [string]$Penalty = "none",
    [ValidateSet("cupy", "native_cuda")]
    [string]$Engine = "cupy",
    # Empty keeps the historical defaults: host NumPy input for native_cuda,
    # resident CuPy input for cupy. "device" hands native_cuda CuPy arrays
    # through its DLPack path.
    [ValidateSet("", "host", "device")]
    [string]$InputLocation = "",
    [int]$LaunchCount = 50,
    [switch]$CudaGraphs,
    [switch]$CudaFastMath
)

$ErrorActionPreference = "Stop"
if (($CudaGraphs -or $CudaFastMath) -and ($Engine -ne "native_cuda")) {
    throw "CUDA tuning switches require Engine='native_cuda'."
}
if ($CudaFastMath -and ($DType -ne "float32")) {
    throw "CudaFastMath requires DType='float32'."
}
if ($InputLocation -eq "") {
    $InputLocation = if ($Engine -eq "native_cuda") { "host" } else { "device" }
}
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$profiler = Get-Command ncu -ErrorAction SilentlyContinue
if ($null -eq $profiler) {
    throw "Nsight Compute CLI (ncu) is not available on PATH."
}

# Every input that changes the workload is part of the name, matching
# run_nsight_systems.ps1, so reports cannot overwrite one another.
$engineLabel = if ($Engine -eq "native_cuda") { "native" } else { "cupy" }
$profileName = "$engineLabel-$DType-$Penalty-$InputLocation-${Samples}x${Features}-b$BatchSize"
if ($CudaGraphs) { $profileName += "-graphs" }
if ($CudaFastMath) { $profileName += "-fastmath" }
if ($OutputDirectory -eq "") {
    $OutputDirectory = Join-Path "artifacts/nsight-compute" $profileName
}
$outputPath = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$reportPrefix = Join-Path $outputPath "$profileName-compute"
$metadataPath = Join-Path $outputPath "$profileName-compute.json"
$workload = Join-Path $PSScriptRoot "profile_cuda_update.py"
$tuningArguments = @()
if ($CudaGraphs) { $tuningArguments += "--cuda-graphs" }
if ($CudaFastMath) { $tuningArguments += "--cuda-fast-math" }

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
    --engine $Engine `
    --input-location $InputLocation `
    --warmup 1 `
    --repeats 1 `
    @tuningArguments `
    --metadata-output $metadataPath

if ($LASTEXITCODE -ne 0) {
    throw "Nsight Compute exited with code $LASTEXITCODE."
}

Write-Host "Nsight Compute report: $reportPrefix.ncu-rep"
Write-Host "Metadata: $metadataPath"
