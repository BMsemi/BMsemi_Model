param(
    [string]$PythonPath = 'C:\Users\elesamj\.venvs\llmqwen\Scripts\python.exe',
    [string]$ModelPath = 'D:\hf_models\Qwen--Qwen3-1.7B',
    [switch]$ResumeClean
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$outputDirectory = Join-Path $PSScriptRoot 'results\qwen_noise_comparison'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$runArguments = @(
    '-B', 'qwen_frontend.py', '--model', $ModelPath, '--local-files-only',
    '--plan', 'all:crossbar', '--crossbar-config', 'configs/qwen_crossbar_noise.json',
    '--raw-prompt', '--prompt', 'Hello world', '--reference-text', 'Hello world',
    '--max-new-tokens', '1', '--threads', '4', '--out-dir', $outputDirectory
)
if ($ResumeClean) { $runArguments += '--resume-clean' }
& $PythonPath @runArguments *> (Join-Path $outputDirectory 'run.log')
exit $LASTEXITCODE
