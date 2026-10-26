param(
    [string]$Source = "x2_zcu106/scripts/build_axi_x2_mvm_bd.tcl",
    [ValidateSet("vivado", "xvlog", "xelab", "xsim")]
    [string]$Tool = "vivado",
    [string]$ToolWorkingDirectory = "",
    [string]$ToolArgumentsJson = "",
    [string]$ToolArgumentsBase64 = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ToolArguments = @()
)

$ErrorActionPreference = "Stop"

$root = "C:/Vivado/2022.2"
$env:XILINX_VIVADO = $root
$env:RDI_DATADIR = "$root/data"
$env:RDI_LIBDIR = "$root/lib/win64.o"
$env:RDI_BINDIR = "$root/bin"
$env:RDI_BINROOT = "$root/bin"
$env:RDI_APPROOT = $root
$env:HDI_APPROOT = $root
$env:RDI_BASEROOT = "C:/Vivado"
$env:RDI_PLATFORM = "win64"
$env:RDI_OPT_EXT = ".o"
$env:RDI_PROG = "vivado"
$env:RDI_PROGNAME = "vivado"
$env:TCL_LIBRARY = "$root/tps/tcl/tcl8.5"
$env:RT_LIBPATH = "$root/scripts/rt/data"
$env:RT_TCL_PATH = "$root/scripts/rt/base_tcl/tcl"
$env:SYNTH_COMMON = $env:RT_LIBPATH
$env:AXI_X2_MVM_JOBS = "1"
$env:PATH = "$root/gnuwin/bin;$root/gnu/microblaze/nt/bin;$root/bin;$root/bin/unwrapped/win64.o;$root/tps/win64/java-cef-95.0.4638.69/bin/lib/win64;$root/tps/win64/javafx-sdk-11.0.2/lib;$root/tps/win64/javafx-sdk-11.0.2/bin;$root/lib/win64.o;$root/tps/win64/jre11.0.11_9/bin/server;$root/tps/win64/jre11.0.11_9/bin;C:/Vitis/2022.2/bin;C:/Vitis_HLS/2022.2/bin;$root/tps/mingw/6.2.0/win64.o/nt/bin;$root/tps/mingw/6.2.0/win64.o/nt/libexec/gcc/x86_64-w64-mingw32/6.2.0;$root/tps/win64/python-3.8.3;$root/tps/win64/python-3.8.3/bin;$root/tps/win64/python-3.8.3/lib;$root/tps/win64/python-3.8.3/lib/site-packages;" + $env:PATH

if ($ToolWorkingDirectory) {
    Push-Location $ToolWorkingDirectory
}

try {
    if ($ToolArgumentsBase64) {
        $ToolArgumentsJson = [Text.Encoding]::UTF8.GetString(
            [Convert]::FromBase64String($ToolArgumentsBase64)
        )
        $ToolArguments = [string[]]($ToolArgumentsJson | ConvertFrom-Json)
    } elseif ($ToolArgumentsJson) {
        $ToolArguments = [string[]]($ToolArgumentsJson | ConvertFrom-Json)
    }
    if ($Tool -eq "vivado") {
        & "$root/bin/unwrapped/win64.o/vivado.exe" -mode batch -source $Source -notrace @ToolArguments
    } else {
        & "$root/bin/unwrapped/win64.o/$Tool.exe" @ToolArguments
    }
    $toolExitCode = $LASTEXITCODE
} finally {
    if ($ToolWorkingDirectory) {
        Pop-Location
    }
}

exit $toolExitCode
