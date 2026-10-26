param(
    [string]$Vivado = ""
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

$OriginalPath = $env:PATH
$BuildDrive = @("V", "U", "T", "S") |
    Where-Object { -not (Test-Path -LiteralPath ("{0}:\" -f $_)) } |
    Select-Object -First 1
if (-not $BuildDrive) {
    throw "No free drive letter was available for Vivado's Windows path-length workaround."
}

& subst.exe ("{0}:" -f $BuildDrive) $Repo
if ($LASTEXITCODE -ne 0) {
    throw "Could not create the temporary Vivado drive mapping."
}
$MappedRepo = "{0}:\" -f $BuildDrive

try {
    # Isolate Vivado from legacy third-party DLL directories on the host PATH.
    $env:PATH = @(
        (Join-Path $env:SystemRoot "System32"),
        $env:SystemRoot,
        (Join-Path $env:SystemRoot "System32\Wbem")
    ) -join ";"

    & $VivadoExe -mode batch -nojournal -nolog -source (Join-Path $MappedRepo "scripts\build_zcu106.tcl")
    if ($LASTEXITCODE -ne 0) {
        throw "Vivado build failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:PATH = $OriginalPath
    & subst.exe ("{0}:" -f $BuildDrive) /D | Out-Null
}

Write-Host "Generated:"
Write-Host "  $Repo\deploy\tcam_rag_zcu106.bit"
Write-Host "  $Repo\deploy\tcam_rag_zcu106.xsa"
