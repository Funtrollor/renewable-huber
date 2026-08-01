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
    [ValidateSet("cupy", "native_cuda")]
    [string]$Engine = "cupy",
    [int]$LaunchCount = 50,
    [switch]$CudaGraphs,
    [switch]$CudaFastMath
)

$ErrorActionPreference = "Stop"
if (($Engine -eq "native_cuda") -and ($Penalty -ne "none")) {
    throw "The P2 native CUDA engine supports penalty='none' only."
}
if (($CudaGraphs -or $CudaFastMath) -and ($Engine -ne "native_cuda")) {
    throw "CUDA tuning switches require Engine='native_cuda'."
}
if ($CudaFastMath -and ($DType -ne "float32")) {
    throw "CudaFastMath requires DType='float32'."
}
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$profiler = Get-Command ncu -ErrorAction SilentlyContinue
if ($null -eq $profiler) {
    throw "Nsight Compute CLI (ncu) is not available on PATH."
}

$outputPath = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$profileName = if ($CudaGraphs -or $CudaFastMath) {
    "native-core-p4-tuned"
} elseif ($Engine -eq "native_cuda") {
    "native-core-p2-native"
} else {
    "native-core-p0-cupy"
}
$inputLocation = if ($Engine -eq "native_cuda") { "host" } else { "device" }
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
    --input-location $inputLocation `
    --warmup 1 `
    --repeats 1 `
    @tuningArguments `
    --metadata-output $metadataPath

if ($LASTEXITCODE -ne 0) {
    throw "Nsight Compute exited with code $LASTEXITCODE."
}

Write-Host "Nsight Compute report: $reportPrefix.ncu-rep"
Write-Host "Metadata: $metadataPath"
