param(
    [string]$BoardUser = "xilinx",
    [switch]$DeepScan,
    [string]$BindAddress = $env:ZCU106_BIND_ADDRESS,
    [int]$ConnectTimeoutMs = 400
)

$ErrorActionPreference = "Stop"

function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port = 22,
        [int]$TimeoutMs = 400,
        [string]$LocalAddress = ""
    )
    if ($LocalAddress) {
        $localEndPoint = [System.Net.IPEndPoint]::new(
            [System.Net.IPAddress]::Parse($LocalAddress),
            0
        )
        $client = [System.Net.Sockets.TcpClient]::new($localEndPoint)
    } else {
        $client = [System.Net.Sockets.TcpClient]::new()
    }
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

function Get-ArpCandidates {
    $ips = @()
    $arpOutput = & arp -a 2>$null
    foreach ($line in $arpOutput) {
        if ($line -match '^\s*(\d{1,3}(?:\.\d{1,3}){3})\s+([0-9a-fA-F-]{17})\s+dynamic') {
            $ip = $matches[1]
            if ($ip -notmatch '\.255$' -and $ip -notmatch '^224\.') {
                $ips += $ip
            }
        }
    }
    return $ips | Select-Object -Unique
}

function Get-LocalSubnetCandidates {
    param([switch]$Deep)
    $suffixes = if ($Deep) { 1..254 } else { @(1, 2, 3, 10, 20, 50, 99, 100, 101, 102, 137) }
    $candidates = @()
    $addresses = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -notmatch '^127\.' -and
            $_.IPAddress -notmatch '^169\.254\.' -and
            $_.PrefixLength -le 24
        }
    foreach ($addr in $addresses) {
        $parts = $addr.IPAddress.Split(".")
        if ($parts.Count -ne 4) { continue }
        $prefix = "$($parts[0]).$($parts[1]).$($parts[2])"
        foreach ($suffix in $suffixes) {
            $candidate = "$prefix.$suffix"
            if ($candidate -ne $addr.IPAddress) {
                $candidates += $candidate
            }
        }
    }
    return $candidates | Select-Object -Unique
}

$candidates = @("pynq", "pynq.local", "zcu106", "zcu106.local")
$candidates += Get-ArpCandidates
$candidates += Get-LocalSubnetCandidates -Deep:$DeepScan
$candidates = $candidates | Where-Object { $_ } | Select-Object -Unique

if ($BindAddress) {
    Write-Host "Scanning $($candidates.Count) SSH candidates bound to $BindAddress..."
} else {
    Write-Host "Scanning $($candidates.Count) SSH candidates..."
}
foreach ($candidate in $candidates) {
    if (Test-TcpPort -HostName $candidate -Port 22 -TimeoutMs $ConnectTimeoutMs -LocalAddress $BindAddress) {
        $boardHost = "$BoardUser@$candidate"
        Write-Host "FOUND_ZCU106_SSH=$boardHost"
        exit 0
    }
}

Write-Host "No ZCU106/PYNQ SSH target found."
exit 1
