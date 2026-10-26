param(
    [string]$Port = "COM17",
    [int]$BaudRate = 115200,
    [string[]]$Command,
    [int]$ReadSeconds = 2
)

$ErrorActionPreference = "Stop"

if (-not $Command -or $Command.Count -eq 0) {
    throw "Provide one or more commands with -Command."
}

$sp = [System.IO.Ports.SerialPort]::new($Port, $BaudRate, "None", 8, "One")
$sp.ReadTimeout = 250
$sp.WriteTimeout = 500
$sp.DtrEnable = $true
$sp.RtsEnable = $true

function Read-Serial {
    param([int]$Seconds = 2)
    $bytes = [System.Collections.Generic.List[byte]]::new()
    $deadline = (Get-Date).AddSeconds($Seconds)
    $buf = New-Object byte[] 4096
    while ((Get-Date) -lt $deadline) {
        $count = $sp.BytesToRead
        if ($count -gt 0) {
            $read = $sp.Read($buf, 0, [Math]::Min($count, $buf.Length))
            for ($i = 0; $i -lt $read; $i++) {
                $bytes.Add($buf[$i])
            }
        }
        Start-Sleep -Milliseconds 100
    }
    if ($bytes.Count -eq 0) {
        return ""
    }
    return [System.Text.Encoding]::ASCII.GetString($bytes.ToArray())
}

try {
    $sp.Open()
    Start-Sleep -Milliseconds 250
    $sp.Write("`r`n")
    [void](Read-Serial -Seconds 1)

    foreach ($cmd in $Command) {
        Write-Output "===== COMMAND: $cmd ====="
        $sp.Write($cmd + "`r`n")
        $output = Read-Serial -Seconds $ReadSeconds
        if ($output) {
            Write-Output $output
        }
    }
} finally {
    if ($sp.IsOpen) {
        $sp.Close()
    }
}
