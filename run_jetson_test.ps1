<#
.SYNOPSIS
    Runs SYCL tests and benchmarks on NVIDIA Jetson (192.168.8.66).

.DESCRIPTION
    Synchronizes workspace sources to the Jetson target, executes build and
    automated test suites (GPU sm_87 / CPU), and optionally fetches standalone
    Linux ARM64 release artifacts.

.PARAMETER TargetHost
    SSH hostname or IP address of NVIDIA Jetson (default: "192.168.8.66").

.PARAMETER Device
    Compute device: "gpu", "cpu", or "auto" (default: "gpu").

.PARAMETER Mode
    Execution mode: "test", "bench", "all", or "build" (default: "test").

.PARAMETER RemoteDir
    Destination directory on Jetson (default: "~/AmplitudeCallibrationSYCL").

.PARAMETER FetchArtifacts
    If specified, packages standalone ARM64 distribution on Jetson and downloads to ./artifacts.

.PARAMETER Clean
    Clean remote build cache before compiling.

.PARAMETER NoSync
    Skip source synchronization and execute directly on existing remote directory.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$TargetHost = "192.168.8.66",

    [Parameter(Position = 1)]
    [ValidateSet("gpu", "cpu", "auto")]
    [string]$Device = "gpu",

    [Parameter(Position = 2)]
    [ValidateSet("test", "bench", "all", "build")]
    [string]$Mode = "test",

    [string]$RemoteDir = "~/AmplitudeCallibrationSYCL",
    [switch]$FetchArtifacts,
    [switch]$Clean,
    [switch]$NoSync
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   NVIDIA Jetson Remote SYCL Test Runner                         " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Target:           $TargetHost" -ForegroundColor Yellow
Write-Host "Compute Device:   $Device" -ForegroundColor Yellow
Write-Host "Mode:             $Mode" -ForegroundColor Yellow
Write-Host "Remote Dir:       $RemoteDir" -ForegroundColor Yellow
Write-Host ""

# 1. Connectivity check
Write-Host "[1/4] Checking SSH connection to $TargetHost..." -ForegroundColor Cyan
$sshProbe = ssh -o BatchMode=yes -o ConnectTimeout=5 $TargetHost "echo OK" 2>&1
if ($LASTEXITCODE -ne 0 -or $sshProbe -notmatch "OK") {
    Write-Error "Failed to connect to Jetson at $TargetHost via SSH. Ensure host is reachable and SSH keys are configured."
    exit 1
}
Write-Host "      Connection established successfully [OK]" -ForegroundColor Green

# 2. Source synchronization
$WorkspaceRoot = (Get-Item $PSScriptRoot).FullName
if (-not $NoSync) {
    Write-Host "[2/4] Synchronizing workspace to Jetson..." -ForegroundColor Cyan
    ssh $TargetHost "mkdir -p $RemoteDir/scripts $RemoteDir/include $RemoteDir/src $RemoteDir/examples $RemoteDir/AmplitudeCallibrationSYCL $RemoteDir/artifacts"

    Push-Location $WorkspaceRoot
    $tempTar = Join-Path $env:TEMP "jetson_sync_$((Get-Date).Ticks).tar.gz"
    try {
        & tar.exe --exclude="*.flr" --exclude="*.dcm" --exclude="*.fit" --exclude="*.INL" --exclude="build" --exclude="x64" --exclude="artifacts" --exclude=".git" -czf "$tempTar" CMakeLists.txt README.md LICENSE include src examples AmplitudeCallibrationSYCL scripts
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $tempTar)) {
            Write-Error "Failed to create source synchronization archive."
            exit 1
        }
        scp "$tempTar" "$TargetHost`:/tmp/src_sync.tar.gz"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to upload source synchronization archive via SCP."
            exit 1
        }
        ssh $TargetHost "tar -xzf /tmp/src_sync.tar.gz -C $RemoteDir && rm -f /tmp/src_sync.tar.gz && chmod +x $RemoteDir/scripts/*.sh"
        Write-Host "      Workspace synchronized successfully [OK]" -ForegroundColor Green
    }
    finally {
        if (Test-Path $tempTar) {
            Remove-Item $tempTar -Force -ErrorAction SilentlyContinue
        }
        Pop-Location
    }
} else {
    Write-Host "[2/4] Skipping source synchronization (-NoSync specified)." -ForegroundColor Gray
}

# 3. Build & Execute Remote Test Suite
Write-Host "[3/4] Executing test suite on Jetson ($TargetHost)..." -ForegroundColor Cyan
$flags = @()
if ($Clean) { $flags += "--clean" }
if ($FetchArtifacts) { $flags += "--package" }

switch ($Mode) {
    "test"  { $flags += "--test --device $Device" }
    "bench" { $flags += "--bench" }
    "all"   { $flags += "--all" }
    "build" { $flags += "--build-only" }
}

$remoteCmd = "bash $RemoteDir/scripts/run_jetson_tests.sh $($flags -join ' ')"
Write-Host "      Remote command: $remoteCmd" -ForegroundColor Gray
Write-Host "-----------------------------------------------------------------" -ForegroundColor DarkGray

ssh -t $TargetHost $remoteCmd
$testExitCode = $LASTEXITCODE

Write-Host "-----------------------------------------------------------------" -ForegroundColor DarkGray

# 4. Fetch artifacts if requested
if ($FetchArtifacts) {
    Write-Host "[4/4] Fetching ARM64 distribution tarball..." -ForegroundColor Cyan
    $localArtifactsDir = Join-Path $WorkspaceRoot "artifacts"
    if (-not (Test-Path $localArtifactsDir)) {
        New-Item -ItemType Directory -Path $localArtifactsDir | Out-Null
    }
    scp "$TargetHost`:$RemoteDir/artifacts/sycl-imageprocess-linux-arm64.tar.gz" "$localArtifactsDir/"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      Downloaded artifacts to $localArtifactsDir\sycl-imageprocess-linux-arm64.tar.gz [OK]" -ForegroundColor Green
    } else {
        Write-Warning "Failed to download ARM64 artifacts from Jetson."
    }
} else {
    Write-Host "[4/4] Skipped artifact fetch (use -FetchArtifacts to download ARM64 package)." -ForegroundColor Gray
}

Write-Host ""
if ($testExitCode -eq 0) {
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host "   NVIDIA JETSON SUITE EXECUTION: ALL PASSED [OK]                " -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Green
} else {
    Write-Host "=================================================================" -ForegroundColor Red
    Write-Host "   NVIDIA JETSON SUITE EXECUTION FAILED WITH CODE $testExitCode  " -ForegroundColor Red
    Write-Host "=================================================================" -ForegroundColor Red
}

exit $testExitCode
