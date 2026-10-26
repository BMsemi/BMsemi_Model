param(
    [string]$InterfaceAlias = "Ethernet",
    [string]$HostIPAddress = "192.168.2.1",
    [int]$PrefixLength = 24,
    [string]$BoardIPAddress = "192.168.2.99",
    [switch]$Apply,
    [switch]$NoElevate
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port = 22,
        [int]$TimeoutMs = 1000
    )
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $iar = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            return $false
        }
        $client.EndConnect($iar)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

$adapter = Get-NetAdapter -Name $InterfaceAlias -ErrorAction Stop
$existing = Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -eq $HostIPAddress }

Write-Host "Adapter: $($adapter.Name) / $($adapter.InterfaceDescription)"
Write-Host "Status:  $($adapter.Status), LinkSpeed: $($adapter.LinkSpeed)"
Write-Host "Target Windows secondary IP: $HostIPAddress/$PrefixLength"
Write-Host "Target board IP: $BoardIPAddress"

if (-not $Apply) {
    Write-Host "DRY RUN: no network settings changed. Add -Apply to configure the adapter."
    Write-Host "After apply, test with: ssh xilinx@$BoardIPAddress"
    exit 0
}

if (-not (Test-IsAdmin)) {
    if ($NoElevate) {
        throw "Administrator privileges are required to add $HostIPAddress to $InterfaceAlias."
    }
    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`"",
        "-InterfaceAlias", "`"$InterfaceAlias`"",
        "-HostIPAddress", $HostIPAddress,
        "-PrefixLength", $PrefixLength,
        "-BoardIPAddress", $BoardIPAddress,
        "-Apply",
        "-NoElevate"
    )
    Write-Host "Requesting administrator elevation to update the Ethernet adapter..."
    Start-Process -FilePath "powershell.exe" -ArgumentList $args -Verb RunAs -Wait
    exit $LASTEXITCODE
}

if (-not $existing) {
    New-NetIPAddress `
        -InterfaceAlias $InterfaceAlias `
        -IPAddress $HostIPAddress `
        -PrefixLength $PrefixLength `
        -SkipAsSource $true | Out-Null
    Write-Host "Added $HostIPAddress/$PrefixLength to $InterfaceAlias."
} else {
    Write-Host "$HostIPAddress is already configured on $InterfaceAlias."
}

Start-Sleep -Seconds 1
if (Test-Connection -ComputerName $BoardIPAddress -Count 1 -Quiet) {
    Write-Host "Ping OK: $BoardIPAddress"
} else {
    Write-Host "Ping failed: $BoardIPAddress"
}

if (Test-TcpPort -HostName $BoardIPAddress -Port 22 -TimeoutMs 1500) {
    Write-Host "SSH OK: xilinx@$BoardIPAddress"
} else {
    Write-Host "SSH port 22 is not reachable at $BoardIPAddress."
}
