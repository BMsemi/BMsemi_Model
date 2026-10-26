$viewerRoot = $PSScriptRoot
$viewerAddress = 'http://127.0.0.1:63462/'
try {
    $viewerResponse = Invoke-WebRequest -Uri $viewerAddress -TimeoutSec 2 -UseBasicParsing
    if ($viewerResponse.StatusCode -eq 200) { Write-Output $viewerAddress; return }
} catch { }
$viewerPython = (Get-Command python -ErrorAction Stop).Source
Start-Process -FilePath $viewerPython -ArgumentList @('-u', '-m', 'http.server', '63462', '--bind', '127.0.0.1', '--directory', ('"' + $viewerRoot + '"')) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $viewerRoot 'server.log') -RedirectStandardError (Join-Path $viewerRoot 'server-error.log') | Out-Null
Write-Output $viewerAddress
