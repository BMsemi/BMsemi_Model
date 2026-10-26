param(
    [string]$BoardHost = $env:ZCU106_HOST,
    [string]$BoardUser = "xilinx",
    [string]$BoardDir = "/home/xilinx/tcam_rag_zcu106",
    [string]$Image = "deploy/tcam_yolo_shared_pe/bus.jpg",
    [string]$RemotePython = "python3",
    [switch]$SkipCopy,
    [switch]$SkipInstall,
    [switch]$NoSudo,
    [switch]$DryRun,
    [switch]$DeepScan,
    [string]$BindAddress = $env:ZCU106_BIND_ADDRESS,
    [int]$ConnectTimeoutMs = 400
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Assert-Tool {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH."
    }
}

function Quote-Remote {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

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

    $suffixes = if ($Deep) {
        1..254
    } else {
        @(1, 2, 3, 10, 20, 50, 99, 100, 101, 102, 137)
    }

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

function Find-Zcu106BoardHost {
    param(
        [string]$User = "xilinx",
        [switch]$Deep,
        [int]$TimeoutMs = 400,
        [string]$LocalAddress = ""
    )

    $candidates = @(
        "pynq",
        "pynq.local",
        "zcu106",
        "zcu106.local"
    )
    $candidates += Get-ArpCandidates
    $candidates += Get-LocalSubnetCandidates -Deep:$Deep
    $candidates = $candidates | Where-Object { $_ } | Select-Object -Unique

    if ($LocalAddress) {
        Write-Host "Searching for ZCU106/PYNQ SSH on $($candidates.Count) candidates bound to $LocalAddress..."
    } else {
        Write-Host "Searching for ZCU106/PYNQ SSH on $($candidates.Count) candidates..."
    }
    foreach ($candidate in $candidates) {
        if (Test-TcpPort -HostName $candidate -Port 22 -TimeoutMs $TimeoutMs -LocalAddress $LocalAddress) {
            $resolved = if ($candidate -match '@') { $candidate } else { "$User@$candidate" }
            Write-Host "Found SSH candidate: $resolved"
            return $resolved
        }
    }
    return $null
}

function Invoke-Remote {
    param([string]$Command)
    $sshOptions = @("-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
    if ($BindAddress) {
        $sshOptions += @("-o", "BindAddress=$BindAddress")
    }
    & ssh @sshOptions $BoardHost $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE`: $Command"
    }
}

Assert-Tool ssh
Assert-Tool scp
Assert-Tool tar

if (-not $BoardHost) {
    $BoardHost = Find-Zcu106BoardHost -User $BoardUser -Deep:$DeepScan -TimeoutMs $ConnectTimeoutMs -LocalAddress $BindAddress
}

if (-not $BoardHost) {
    throw (
        "Could not discover a reachable ZCU106 SSH target. " +
        "Connect the board Ethernet to this PC/network, boot Linux/PYNQ, ensure sshd is running, " +
        "or pass -BoardHost ${BoardUser}@<board-ip>. Use -DeepScan for a wider subnet scan."
    )
}

$RemoteArchive = "/tmp/tcam_rag_zcu106_remote.tar"

$InstallReqs = if ($SkipInstall) { "0" } else { "1" }
$NoSudoEnv = if ($NoSudo) { "1" } else { "0" }
$RemoteScript = $BoardDir.TrimEnd("/") + "/scripts/run_remote_zcu106_board.sh"
$RemoteEnvironment = @(
    "PYTHON_BIN=$(Quote-Remote $RemotePython)",
    "IMAGE=$(Quote-Remote $Image)",
    "INSTALL_REQS=$InstallReqs",
    "NO_SUDO=$NoSudoEnv"
) -join " "
$RemoteCommand = "cd $(Quote-Remote $BoardDir) && $RemoteEnvironment bash $(Quote-Remote $RemoteScript)"

if ($DryRun) {
    Write-Host "BoardHost: $BoardHost"
    Write-Host "BoardDir: $BoardDir"
    if ($BindAddress) {
        Write-Host "BindAddress: $BindAddress"
    }
    Write-Host "Remote archive: $RemoteArchive"
    Write-Host "Remote command:"
    Write-Host $RemoteCommand
    exit 0
}

if (-not $SkipCopy) {
    $Archive = Join-Path ([System.IO.Path]::GetTempPath()) (
        "tcam_rag_zcu106_remote_{0}.tar" -f ([guid]::NewGuid().ToString("N"))
    )

    Push-Location $Repo
    try {
        & tar -cf $Archive `
            run_tcam_yolo.py `
            README.md `
            integrated `
            legacy_neuromorphic_core `
            deploy `
            xforge_zcu106 `
            scripts/run_remote_zcu106_board.sh
        if ($LASTEXITCODE -ne 0) {
            throw "Local archive creation failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    Invoke-Remote ("mkdir -p " + (Quote-Remote $BoardDir))
    $scpOptions = @()
    if ($BindAddress) {
        $scpOptions += @("-o", "BindAddress=$BindAddress")
    }
    & scp @scpOptions $Archive "${BoardHost}:$RemoteArchive"
    if ($LASTEXITCODE -ne 0) {
        throw "SCP archive upload failed with exit code $LASTEXITCODE"
    }

    Invoke-Remote (
        "tar -xf " + (Quote-Remote $RemoteArchive) +
        " -C " + (Quote-Remote $BoardDir) +
        " && chmod +x " + (Quote-Remote ($BoardDir.TrimEnd("/") + "/scripts/run_remote_zcu106_board.sh"))
    )
}

Invoke-Remote $RemoteCommand
