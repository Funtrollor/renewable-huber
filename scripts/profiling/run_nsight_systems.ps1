param(
    [string]$Python = "python",
    # Empty selects a directory named after the configuration under
    # artifacts/nsight, so two configurations never overwrite each other.
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
$profiler = Get-Command nsys -ErrorAction SilentlyContinue
if ($null -eq $profiler) {
    $installedProfiler = Get-ChildItem `
        -Path "C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe" `
        -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if ($null -eq $installedProfiler) {
        throw "Nsight Systems CLI (nsys) is not installed or available on PATH."
    }
    $profilerPath = $installedProfiler.FullName
} else {
    $profilerPath = $profiler.Source
}

# Every input that changes the workload is part of the name, so reports from
# different configurations cannot overwrite one another.
$engineLabel = if ($Engine -eq "native_cuda") { "native" } else { "cupy" }
$profileName = "$engineLabel-$DType-$Penalty-$InputLocation-${Samples}x${Features}-b$BatchSize"
if ($CudaGraphs) { $profileName += "-graphs" }
if ($CudaFastMath) { $profileName += "-fastmath" }
if ($OutputDirectory -eq "") {
    $OutputDirectory = Join-Path "artifacts/nsight" $profileName
}
$outputPath = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$reportPrefix = Join-Path $outputPath "$profileName-systems"
$metadataPath = Join-Path $outputPath "$profileName-systems.json"
$summaryPath = Join-Path $outputPath "$profileName-summary.json"
$workload = Join-Path $PSScriptRoot "profile_cuda_update.py"
$summarizer = Join-Path $PSScriptRoot "summarize_nsys_sqlite.py"
$tuningArguments = @()
if ($CudaGraphs) { $tuningArguments += "--cuda-graphs" }
if ($CudaFastMath) { $tuningArguments += "--cuda-fast-math" }

& $profilerPath profile `
    --trace=cuda,nvtx,cublas,cusolver `
    --stats=true `
    --force-overwrite=true `
    --output=$reportPrefix `
    $Python $workload `
    --samples $Samples `
    --features $Features `
    --batch-size $BatchSize `
    --dtype $DType `
    --penalty $Penalty `
    --engine $Engine `
    --input-location $InputLocation `
    --warmup 2 `
    --repeats 3 `
    --phase-ranges `
    @tuningArguments `
    --metadata-output $metadataPath

if ($LASTEXITCODE -ne 0) {
    throw "Nsight Systems exited with code $LASTEXITCODE."
}

& $Python $summarizer "$reportPrefix.sqlite" `
    --metadata $metadataPath `
    --output $summaryPath
if ($LASTEXITCODE -ne 0) {
    throw "Nsight summary exited with code $LASTEXITCODE."
}

Write-Host "Nsight Systems report: $reportPrefix.nsys-rep"
Write-Host "Metadata: $metadataPath"
Write-Host "Summary: $summaryPath"
