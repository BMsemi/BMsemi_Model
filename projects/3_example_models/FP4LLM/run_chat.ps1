$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& "$env:USERPROFILE\.venvs\llmqwen\Scripts\python.exe" -u -B chat_server.py @args
