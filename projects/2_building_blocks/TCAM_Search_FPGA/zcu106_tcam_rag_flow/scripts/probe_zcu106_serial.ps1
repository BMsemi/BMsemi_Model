param(
    [string[]]$Ports = @("COM14", "COM15", "COM16", "COM17"),
    [int]$BaudRate = 115200,
    [int]$ReadSeconds = 3
)

$ErrorActionPreference = "Continue"

foreach ($portName in $Ports) {
    Write-Output "===== $portName ====="
    $sp = $null
    try {
        $sp = [System.IO.Ports.SerialPort]::new($portName, $BaudRate, "None", 8, "One")
        $sp.ReadTimeout = 250
        $sp.WriteTimeout = 250
        $sp.DtrEnable = $true
        $sp.RtsEnable = $true
        $sp.Open()
        Start-Sleep -Milliseconds 250
        $sp.Write("`r`n")
        Start-Sleep -Milliseconds 500

        $bytes = [System.Collections.Generic.List[byte]]::new()
        $deadline = (Get-Date).AddSeconds($ReadSeconds)
        while ((Get-Date) -lt $deadline) {
            $count = $sp.BytesToRead
            if ($count -gt 0) {
                $buf = New-Object byte[] ([Math]::Min($count, 4096))
                $read = $sp.Read($buf, 0, $buf.Length)
                for ($i = 0; $i -lt $read; $i++) {
                    $bytes.Add($buf[$i])
                }
            }
            Start-Sleep -Milliseconds 100
        }

        if ($bytes.Count -gt 0) {
            [System.Text.Encoding]::ASCII.GetString($bytes.ToArray())
        } else {
            "no serial output"
        }
    } catch {
        "ERROR: $($_.Exception.Message)"
    } finally {
        if ($sp -and $sp.IsOpen) {
            $sp.Close()
        }
    }
}
