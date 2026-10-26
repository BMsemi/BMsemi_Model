param(
    [string]$InterfaceAlias = "Ethernet",
    [string]$BoardIPAddress = "192.168.137.99",
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

$adapter = Get-NetAdapter -Name $InterfaceAlias -ErrorAction Stop
$hostAddress = Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 -ErrorAction Stop |
    Where-Object { $_.IPAddress -notmatch '^169\.254\.' } |
    Select-Object -First 1

if (-not $hostAddress) {
    throw "No usable IPv4 address is configured on $InterfaceAlias."
}

$destination = "$BoardIPAddress/32"
$route = Get-NetRoute -DestinationPrefix $destination -InterfaceAlias $InterfaceAlias -ErrorAction SilentlyContinue

Write-Host "Adapter: $($adapter.Name) / $($adapter.InterfaceDescription)"
Write-Host "Status:  $($adapter.Status), LinkSpeed: $($adapter.LinkSpeed)"
Write-Host "Windows Ethernet IP: $($hostAddress.IPAddress)/$($hostAddress.PrefixLength)"
Write-Host "Target board host route: $destination through $InterfaceAlias"

if (-not $Apply) {
    Write-Host "DRY RUN: no route changed. Add -Apply from an Administrator PowerShell to install the route."
    Write-Host "Temporary workaround without admin:"
    Write-Host ".\scripts\run_remote_zcu106.ps1 -BoardHost xilinx@$BoardIPAddress -BindAddress $($hostAddress.IPAddress)"
    exit 0
}

if (-not (Test-IsAdmin)) {
    throw "Administrator privileges are required to add the Windows host route for $destination."
}

if ($route) {
    Set-NetRoute -DestinationPrefix $destination -InterfaceAlias $InterfaceAlias -RouteMetric 1 -Confirm:$false
    Write-Host "Updated existing host route for $destination."
} else {
    New-NetRoute -DestinationPrefix $destination -InterfaceAlias $InterfaceAlias -NextHop 0.0.0.0 -RouteMetric 1 -PolicyStore ActiveStore | Out-Null
    Write-Host "Added host route for $destination."
}

Get-NetRoute -DestinationPrefix $destination -InterfaceAlias $InterfaceAlias |
    Format-Table ifIndex,InterfaceAlias,DestinationPrefix,NextHop,RouteMetric,InterfaceMetric -AutoSize
