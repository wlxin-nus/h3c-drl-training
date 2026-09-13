[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('sz_air_ppo','mz_air_ppo','mz_air_mappo','mz_hydro_ppo','mz_hydro_mappo')]
    [string]$Task,
    [ValidateSet(42,1337,2026)][int]$Seed = 1337,
    [ValidateSet('smoke','full')][string]$Mode = 'full',
    [switch]$Resume,
    [switch]$ContinueUntilConverged,
    [switch]$AllowHostMigration,
    [ValidateSet('online','offline','disabled')][string]$WandbMode = 'online',
    [string]$Endpoint = 'http://127.0.0.1:8000',
    [string]$Device = 'auto',
    [ValidateRange(5,256)][int]$WorkerCapacity = 12,
    [ValidateRange(0,100)][int]$HttpAutoResumeAttempts = 12,
    [ValidateRange(1,300)][double]$HttpResumeBackoffSeconds = 15,
    [ValidateRange(30,3600)][double]$HttpHealthTimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$arguments = @(
    '-m', 'drl_multiseed.cli', 'train',
    '--task', $Task, '--seed', $Seed, '--mode', $Mode,
    '--endpoint', $Endpoint, '--wandb-mode', $WandbMode,
    '--device', $Device, '--output-root', (Join-Path $root 'runs'),
    '--worker-capacity', $WorkerCapacity,
    '--http-auto-resume-attempts', $HttpAutoResumeAttempts,
    '--http-resume-backoff-seconds', $HttpResumeBackoffSeconds,
    '--http-health-timeout-seconds', $HttpHealthTimeoutSeconds
)
if ($Resume) { $arguments += '--resume' }
if ($ContinueUntilConverged) { $arguments += '--continue-until-converged' }
if ($AllowHostMigration) { $arguments += '--allow-host-migration' }

Push-Location $root
try { python @arguments }
finally { Pop-Location }
