param(
    [string]$Port = "COM17",
    [string]$BoardIPAddress = "192.168.137.99",
    [int]$PrefixLength = 24,
    [string]$InterfaceName = "eth0"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SerialExec = Join-Path $ScriptDir "serial_exec_zcu106.ps1"

$commands = @(
    "echo BOARD_BEFORE; hostname -I; ip -4 addr show $InterfaceName",
    "sudo ip link set $InterfaceName up",
    "sudo ip addr replace $BoardIPAddress/$PrefixLength dev $InterfaceName",
    "sudo systemctl restart ssh || sudo service ssh restart || sudo /etc/init.d/ssh restart",
    "echo BOARD_AFTER; hostname -I; ip -4 addr show $InterfaceName; (systemctl is-active ssh || service ssh status) 2>&1"
)

& $SerialExec -Port $Port -Command $commands -ReadSeconds 4
if (-not $?) {
    throw "Serial board network repair failed."
}

Write-Host "Configured board candidate SSH target: xilinx@$BoardIPAddress"
