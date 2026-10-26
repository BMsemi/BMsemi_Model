param(
    [Parameter(Mandatory = $true)]
    [string]$LocalPath,
    [Parameter(Mandatory = $true)]
    [string]$RemotePath,
    [string]$Port = "COM17",
    [int]$BaudRate = 115200,
    [int]$LineDelayMs = 3,
    [int]$ReadSeconds = 8
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LocalPath)) {
    throw "Local file not found: $LocalPath"
}

function Quote-Sh {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Read-Serial {
    param(
        [System.IO.Ports.SerialPort]$Serial,
        [int]$Seconds = 2
    )
    $bytes = [System.Collections.Generic.List[byte]]::new()
    $deadline = (Get-Date).AddSeconds($Seconds)
    $buf = New-Object byte[] 4096
    while ((Get-Date) -lt $deadline) {
        $count = $Serial.BytesToRead
        if ($count -gt 0) {
            $read = $Serial.Read($buf, 0, [Math]::Min($count, $buf.Length))
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

function Send-Line {
    param(
        [System.IO.Ports.SerialPort]$Serial,
        [string]$Line
    )
    $Serial.Write($Line + "`r`n")
    if ($LineDelayMs -gt 0) {
        Start-Sleep -Milliseconds $LineDelayMs
    }
}

$remoteDir = $RemotePath -replace '/[^/]+$', ''
if (-not $remoteDir -or $remoteDir -eq $RemotePath) {
    $remoteDir = "."
}
$remoteB64 = "$RemotePath.b64"
$localHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $LocalPath).Hash.ToLowerInvariant()
$bytes = [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $LocalPath).Path)
$base64 = [Convert]::ToBase64String($bytes)
$lineCount = [Math]::Ceiling($base64.Length / 76)

$sp = [System.IO.Ports.SerialPort]::new($Port, $BaudRate, "None", 8, "One")
$sp.ReadTimeout = 250
$sp.WriteTimeout = 1000
$sp.DtrEnable = $true
$sp.RtsEnable = $true

try {
    $sp.Open()
    Start-Sleep -Milliseconds 250
    $sp.Write("`r`n")
    [void](Read-Serial -Serial $sp -Seconds 1)

    Send-Line $sp "stty -echo"
    Send-Line $sp ("mkdir -p " + (Quote-Sh $remoteDir))
    Send-Line $sp ("rm -f " + (Quote-Sh $remoteB64) + " " + (Quote-Sh $RemotePath))
    Send-Line $sp ("cat > " + (Quote-Sh $remoteB64))
    Start-Sleep -Seconds 1

    Write-Output "Uploading $LocalPath to $RemotePath over $Port"
    Write-Output "Bytes: $($bytes.Length), base64 lines: $lineCount, sha256: $localHash"

    for ($offset = 0; $offset -lt $base64.Length; $offset += 76) {
        $count = [Math]::Min(76, $base64.Length - $offset)
        Send-Line $sp $base64.Substring($offset, $count)
        if (($offset / 76) % 500 -eq 0) {
            Write-Output ("sent_lines={0}/{1}" -f [int]($offset / 76), $lineCount)
        }
    }

    $sp.Write([byte[]](4), 0, 1)
    Start-Sleep -Seconds 1
    Send-Line $sp (
        "base64 -d " + (Quote-Sh $remoteB64) + " > " + (Quote-Sh $RemotePath) +
        " && rm -f " + (Quote-Sh $remoteB64) +
        " && sha256sum " + (Quote-Sh $RemotePath) +
        " && ls -l " + (Quote-Sh $RemotePath) +
        " ; stty echo"
    )
    $output = Read-Serial -Serial $sp -Seconds $ReadSeconds
    if ($output) {
        Write-Output $output
    }
    if ($output -and ($output.ToLowerInvariant().Contains($localHash))) {
        Write-Output "UPLOAD_SHA256_OK"
    } else {
        Write-Warning "Could not confirm remote SHA256 in serial output. Check the output above."
    }
} finally {
    if ($sp.IsOpen) {
        try {
            $sp.Write("stty echo`r`n")
        } catch {
        }
        $sp.Close()
    }
}
