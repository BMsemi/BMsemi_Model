param(
    [string]$PythonExe = "python",
    [string]$Model = "",
    [string]$Output = "C:\fpga_build\x2_compiler_regression",
    [switch]$ValidateAgainstLiteRT
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $PSCommandPath
$x2Root = Split-Path -Parent $scriptRoot
$workspaceRoot = Split-Path -Parent $x2Root

if (-not $Model) {
    $Model = Join-Path $workspaceRoot "YOLO26n-TFLite-FPGA-Partner-Project\yolo26n_int8.tflite"
}
$Model = [System.IO.Path]::GetFullPath($Model)
$Output = [System.IO.Path]::GetFullPath($Output)

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "TFLite model not found: $Model"
}

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $Arguments"
    }
}

Push-Location $workspaceRoot
try {
    Invoke-CheckedPython -m x2_zcu106.x2compiler.test_compiler

    Push-Location (Join-Path $x2Root "runtime")
    try {
        Invoke-CheckedPython test_x2_runtime.py
    }
    finally {
        Pop-Location
    }

    $program = Join-Path $Output "first_conv"
    Invoke-CheckedPython -m x2_zcu106.x2compiler.compile_tflite `
        --model $Model `
        --op-index 2 `
        --activation-max-abs 1.0 `
        --output $program

    $ireeOptCommand = Get-Command iree-opt -ErrorAction SilentlyContinue
    $ireeOptPath = if ($ireeOptCommand) { $ireeOptCommand.Source } else { "" }
    if (-not $ireeOptPath) {
        $pythonCommand = Get-Command $PythonExe -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $siblingIreeOpt = Join-Path (Split-Path -Parent $pythonCommand.Source) "iree-opt.exe"
            if (Test-Path -LiteralPath $siblingIreeOpt -PathType Leaf) {
                $ireeOptPath = $siblingIreeOpt
            }
        }
    }
    if ($ireeOptPath) {
        & $ireeOptPath --allow-unregistered-dialect `
            (Join-Path $program "program.mlir") -o NUL
        if ($LASTEXITCODE -ne 0) {
            throw "iree-opt rejected the generated MLIR."
        }
    }
    else {
        Write-Warning "iree-opt is not on PATH; skipping the independent MLIR parse."
    }

    if ($ValidateAgainstLiteRT) {
        Invoke-CheckedPython -m x2_zcu106.x2compiler.validate_tflite_layer `
            --model $Model `
            --program $program `
            --output (Join-Path $program "validation.json")
    }

    Write-Output "X2_COMPILER_REGRESSION_PASS: $program"
}
finally {
    Pop-Location
}
