param(
    [string]$Vivado = "",
    [int]$RramTrialsPerCode = 1000,
    [switch]$SkipRramInt4Model,
    [switch]$RequireRramInt4Pl
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-Vivado {
    param([string]$Requested)

    if ($Requested) {
        if (Test-Path -LiteralPath $Requested) { return (Resolve-Path -LiteralPath $Requested).Path }
        throw "Requested Vivado executable was not found: $Requested"
    }

    $cmd = Get-Command vivado -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $candidates = @(
        "C:\Xilinx\Vivado\*\bin\vivado.bat",
        "C:\Xilinx\Vivado\*\bin\vivado.exe",
        "C:\Xilinx\*\Vivado\*\bin\vivado.bat",
        "C:\Xilinx\*\Vivado\*\bin\vivado.exe",
        "C:\Vivado\*\bin\vivado.bat",
        "C:\Vivado\*\bin\vivado.exe",
        "C:\AMD\Vivado\*\bin\vivado.bat",
        "C:\AMD\Vivado\*\bin\vivado.exe"
    )

    foreach ($pattern in $candidates) {
        $match = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($match) { return $match.FullName }
    }

    throw "Vivado was not found. Pass -Vivado C:\path\to\vivado.bat or add Vivado to PATH."
}

$VivadoExe = Find-Vivado $Vivado
Write-Host "Using Vivado: $VivadoExe"
Write-Host "RRAM INT4 model: $(if ($SkipRramInt4Model) { 'SKIP' } else { 'RUN' })"

$PythonExe = $null
if (-not $SkipRramInt4Model) {
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCmd) {
        throw "Python was not found on PATH. Install Python or run with -SkipRramInt4Model."
    }
    $PythonExe = $PythonCmd.Source
}

$OriginalPath = $env:PATH
try {
    # Isolate Vivado from legacy third-party DLL directories on the host PATH.
    $env:PATH = @(
        (Join-Path $env:SystemRoot "System32"),
        $env:SystemRoot,
        (Join-Path $env:SystemRoot "System32\Wbem")
    ) -join ";"

    & $VivadoExe -mode batch -nojournal -nolog -source (Join-Path $Repo "scripts\program_shared_pe_overlay.tcl")
    if ($LASTEXITCODE -ne 0) {
        throw "Vivado programming failed with exit code $LASTEXITCODE"
    }

    $HwhPath = Join-Path $Repo "deploy\tcam_yolo_shared_pe\overlay\zcu106_axi_x2_mvm.hwh"
    $RramPlPresent = $false
    if (Test-Path -LiteralPath $HwhPath) {
        $HwhText = Get-Content -LiteralPath $HwhPath -Raw
        $RramPlPresent = $HwhText -match "rram_int4_accuracy"
    }
    if ($RramPlPresent) {
        Write-Host "RRAM INT4 PL block: PRESENT in programmed overlay metadata at 0xA0060000"
    } else {
        Write-Host "RRAM INT4 PL block: NOT PRESENT in current overlay metadata; rebuild bitstream to synthesize it into PL."
        if ($RequireRramInt4Pl) {
            throw "RRAM INT4 PL block is required but not present in the programmed overlay."
        }
    }

    if (-not $SkipRramInt4Model) {
        $RramModel = Join-Path $Repo "scripts\model_rram_int4_programming_accuracy.py"
        if (-not (Test-Path -LiteralPath $RramModel)) {
            throw "RRAM INT4 model script was not found: $RramModel"
        }

        Write-Host ""
        Write-Host "Running RRAM INT4 programming-accuracy diagnostic model..."
        & $PythonExe $RramModel --trials-per-code $RramTrialsPerCode --sample-records 0
        if ($LASTEXITCODE -ne 0) {
            throw "RRAM INT4 programming-accuracy model failed with exit code $LASTEXITCODE"
        }
    }
} finally {
    $env:PATH = $OriginalPath
}
