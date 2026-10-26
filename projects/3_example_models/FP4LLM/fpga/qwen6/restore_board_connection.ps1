param([switch]$Apply)

$ErrorActionPreference = 'Stop'
$adapterName = 'Ethernet'
$proxyFacingIP = '169.254.228.200'
$boardFacingIP = '192.168.137.137'
$boardIP = '192.168.137.99'
$destination = "$boardIP/32"

$adapter = Get-NetAdapter -Name $adapterName
if ($adapter.Status -ne 'Up') { throw 'Connect the board-facing Ethernet cable first.' }
$addresses = @(Get-NetIPAddress -InterfaceAlias $adapterName -AddressFamily IPv4)
$unexpected = @($addresses | Where-Object { $_.IPAddress -notin @($proxyFacingIP, $boardFacingIP) })
if ($unexpected.Count) { throw 'Ethernet addresses have changed. Inspect them before applying this saved-profile repair.' }
$gateways = @(Get-NetRoute -InterfaceAlias $adapterName -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue)
if ($gateways.Count) { throw 'Ethernet now has a default gateway. Inspect it before changing address configuration.' }

Write-Host "Preserve proxy-facing address: $proxyFacingIP/16"
Write-Host "Restore board-facing address: $boardFacingIP/24"
Write-Host "Route $destination directly through $adapterName, overriding the broader Tailscale route."
Write-Host 'Changes use the active configuration only and last until reboot.'
if (-not $Apply) {
    Write-Host 'PREVIEW ONLY: no settings changed. Run this file with -Apply in Administrator PowerShell.'
    exit 0
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Open PowerShell as Administrator, then run this script with -Apply. No settings have been changed.'
}

# Make the existing link-local address explicit before adding the second subnet.
# Simply adding the board IP disables DHCP and may discard the proxy-facing APIPA address.
& netsh interface ipv4 set address "name=$adapterName" source=static "address=$proxyFacingIP" mask=255.255.0.0 gateway=none store=active
if ($LASTEXITCODE -ne 0) { throw 'Could not preserve the proxy-facing address; stopped.' }
& netsh interface ipv4 add address "name=$adapterName" "address=$boardFacingIP" mask=255.255.255.0 store=active
if ($LASTEXITCODE -ne 0) { throw 'Could not add the board-facing address; the proxy address remains configured.' }

$route = Get-NetRoute -InterfaceAlias $adapterName -DestinationPrefix $destination -PolicyStore ActiveStore -ErrorAction SilentlyContinue
if ($route) {
    if (@($route | Where-Object { $_.NextHop -ne '0.0.0.0' }).Count) {
        throw 'An unexpected existing board host route needs inspection.'
    }
    Set-NetRoute -InterfaceAlias $adapterName -DestinationPrefix $destination -PolicyStore ActiveStore -RouteMetric 1 -Confirm:$false
} else {
    New-NetRoute -InterfaceAlias $adapterName -DestinationPrefix $destination -NextHop 0.0.0.0 -RouteMetric 1 -PolicyStore ActiveStore | Out-Null
}

Start-Sleep -Seconds 2
$client = [System.Net.Sockets.TcpClient]::new()
try {
    $client.Client.Bind([System.Net.IPEndPoint]::new([System.Net.IPAddress]::Parse($boardFacingIP), 0))
    $pending = $client.BeginConnect($boardIP, 22, $null, $null)
    if (-not $pending.AsyncWaitHandle.WaitOne(3000)) { throw 'SSH connection timed out.' }
    $client.EndConnect($pending)
    Write-Host 'SUCCESS: board SSH is reachable.'
    Write-Host ('ssh -b ' + $boardFacingIP + ' -i "' + $env:USERPROFILE + '\.ssh\codex_zcu106_ed25519" xilinx@' + $boardIP)
} catch {
    Write-Warning "PC network repair completed, but board SSH is unavailable: $($_.Exception.Message)"
    Write-Host 'Check board power, its Linux boot, Ethernet cable, and actual IP through the COM17 serial console at 115200 baud.'
} finally {
    $client.Close()
}
