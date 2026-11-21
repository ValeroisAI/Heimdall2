[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $BenchPath,
    [Parameter(Mandatory)] [string[]] $ModelPath,
    [Parameter(Mandatory)] [string] $OutputDirectory,
    [ValidatePattern('^Vulkan\d+$')] [string] $Device = 'Vulkan0',
    [ValidateRange(0, 2147483000)] [int[]] $Depth = @(0, 4096, 32768, 65536),
    [ValidateRange(1, 100)] [int] $Repetitions = 5,
    [ValidateRange(1, 1024)] [int] $Threads = 8,
    [ValidateSet('f16', 'q8_0', 'q4_0')] [string] $CacheKeyType = 'f16',
    [ValidateSet('f16', 'q8_0', 'q4_0')] [string] $CacheValueType = 'f16'
)

$ErrorActionPreference = 'Stop'
$bench = (Resolve-Path -LiteralPath $BenchPath).ProviderPath
$models = @($ModelPath | ForEach-Object { (Resolve-Path -LiteralPath $_).ProviderPath })
$depths = @($Depth | Select-Object -Unique)
if ($env:BIFROST_RET -and ($depths | Where-Object { $_ -gt 0 })) {
    throw 'Depth tests restore sequence state without Bifrost retrieval summaries. Clear BIFROST_RET or use a fresh-prefill test.'
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Select a new output directory to preserve previous results: $OutputDirectory"
}

# In Windows PowerShell 5.1 native stderr also appears as PowerShell error records.
$ErrorActionPreference = 'Continue'
$devices = & $bench --list-devices 2>&1 | ForEach-Object { $_.ToString() }
$deviceExit = $LASTEXITCODE
$help = & $bench --help 2>&1 | ForEach-Object { $_.ToString() }
$helpExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
$deviceLine = $devices | Where-Object { $_ -match "^\s*${Device}:" } | Select-Object -First 1
if ($deviceExit -ne 0 -or -not $deviceLine) {
    throw "Vulkan device $Device not available.`n$($devices -join "`n")"
}
if ($helpExit -ne 0 -or ($help -join "`n") -notmatch '--n-depth') {
    throw 'llama-bench must support context-depth tests.'
}
$flashHelp = $help | Where-Object { $_ -match '--flash-attn\s' } | Select-Object -First 1
$flashAttention = if ($flashHelp -match '\bon\b') { 'on' } else { '1' }

$output = (New-Item -ItemType Directory -Path $OutputDirectory).FullName
$stderrPath = Join-Path $output 'llama-bench.stderr.log'
$jsonPath = Join-Path $output 'llama-bench.json'
$devices | Set-Content -LiteralPath (Join-Path $output 'devices.txt') -Encoding UTF8
$benchArgs = @('-dev', $Device, '-ngl', '999', '-sm', 'none', '-fa', $flashAttention,
    '-p', '512', '-n', '64', '-d', ($depths -join ','), '-b', '2048', '-ub', '512',
    '-r', "$Repetitions", '-t', "$Threads", '-ctk', $CacheKeyType, '-ctv', $CacheValueType, '-o', 'json', '-v')
foreach ($model in $models) {
    $benchArgs += @('-m', $model)
}

Write-Host "Benchmarking $deviceLine"
$runtimeDlls = Get-ChildItem -File -LiteralPath (Split-Path -Parent $bench) -Filter '*.dll' |
    Where-Object { $_.Name -match '^(lib)?(llama|ggml)([-_.]|$)' }
$metadata = [ordered]@{
    started_utc = [DateTime]::UtcNow.ToString('o')
    device = $deviceLine.Trim()
    windows_build = [Environment]::OSVersion.Version.ToString()
    cpu = (Get-CimInstance Win32_Processor).Name
    display_drivers = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
    binary = Get-FileHash -LiteralPath $bench
    runtime_dlls = @($runtimeDlls | ForEach-Object { Get-FileHash -LiteralPath $_.FullName })
    models = @(Get-FileHash -LiteralPath $models)
    bifrost_retrieval = $env:BIFROST_RET
    args = $benchArgs
}
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'metadata.json') -Encoding UTF8

$ErrorActionPreference = 'Continue'
& $bench @benchArgs 2> $stderrPath | Set-Content -LiteralPath $jsonPath -Encoding UTF8
$benchExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($benchExit -ne 0) {
    throw "llama-bench exited with $benchExit. See $stderrPath"
}

$rows = Get-Content -Raw -LiteralPath $jsonPath | ConvertFrom-Json
$expectedCases = foreach ($model in $models) {
    foreach ($contextDepth in $depths) {
        "$model|$contextDepth|512|0"
        "$model|$contextDepth|0|64"
    }
}
$actualCases = @($rows | ForEach-Object { "$($_.model_filename)|$($_.n_depth)|$($_.n_prompt)|$($_.n_gen)" })
if (Compare-Object ($expectedCases | Sort-Object) ($actualCases | Sort-Object)) {
    throw "Unexpected test matrix. See $jsonPath and $stderrPath"
}

# Check model layer placement in the log. Op-level fallback still needs profiling.
$offloadLines = @(Select-String -LiteralPath $stderrPath -Pattern 'offloaded (\d+)/(\d+) layers to GPU')
if (-not $offloadLines -or ($offloadLines | Where-Object {
    $_.Matches[0].Groups[1].Value -ne $_.Matches[0].Groups[2].Value
})) {
    throw "Could not verify full GPU layer offload. See $stderrPath"
}
$rows | Select-Object @{ Name = 'model'; Expression = { Split-Path -Leaf $_.model_filename } },
    n_depth, n_prompt, n_gen, avg_ts, stddev_ts | Format-Table -AutoSize
Write-Host "Raw results and logs saved to $output"
