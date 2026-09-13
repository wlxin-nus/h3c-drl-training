[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('sz_air_ppo','mz_air_ppo','mz_air_mappo','mz_hydro_ppo','mz_hydro_mappo')]
    [string]$Task,
    [ValidateSet(42,1337,2026)][int]$Seed = 1337,
    [ValidateSet('smoke','full')][string]$Mode = 'full',
    [string]$Endpoint = 'http://127.0.0.1:8000',
    [string]$Device = 'cpu',
    [ValidateRange(1,128)][int]$WorkerCapacity = 12,
    [ValidateRange(0,86400)][double]$LeaseWaitSeconds = 0,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$arguments = @('-m','drl_multiseed.cli','evaluate','--task',$Task,'--seed',$Seed,
    '--mode',$Mode,'--endpoint',$Endpoint,'--device',$Device,
    '--output-root',(Join-Path $root 'runs'),
    '--worker-capacity',$WorkerCapacity,'--lease-wait-seconds',$LeaseWaitSeconds)
if ($Force) { $arguments += '--force' }
Push-Location $root
try { python @arguments }
finally { Pop-Location }
