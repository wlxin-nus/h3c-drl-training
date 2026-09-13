[CmdletBinding()]
param(
    [ValidateSet(42,1337,2026)][int]$Seed = 1337,
    [ValidateSet('smoke','full')][string]$Mode = 'full',
    [ValidateSet(1,2)][int]$MaxParallel = 1,
    [ValidateRange(0,2)][int]$GpuSlots = 1,
    [ValidateRange(1,16)][int]$ThreadsPerTask = 4,
    [switch]$Resume,
    [switch]$ContinueUntilConverged,
    [switch]$AllowHostMigration,
    [ValidateSet('online','offline','disabled')][string]$WandbMode = 'online',
    [string]$Endpoint = 'http://127.0.0.1:8000',
    [ValidateRange(0,100)][int]$HttpAutoResumeAttempts = 12,
    [ValidateRange(1,300)][double]$HttpResumeBackoffSeconds = 15,
    [ValidateRange(30,3600)][double]$HttpHealthTimeoutSeconds = 600,
    [string[]]$Tasks
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$arguments = @(
    '-m', 'drl_multiseed.cli', 'suite', '--seed', $Seed, '--mode', $Mode,
    '--max-parallel', $MaxParallel, '--gpu-slots', $GpuSlots,
    '--threads-per-task', $ThreadsPerTask, '--endpoint', $Endpoint,
    '--wandb-mode', $WandbMode, '--worker-capacity', '12',
    '--http-auto-resume-attempts', $HttpAutoResumeAttempts,
    '--http-resume-backoff-seconds', $HttpResumeBackoffSeconds,
    '--http-health-timeout-seconds', $HttpHealthTimeoutSeconds,
    '--output-root', (Join-Path $root 'outputs_refine_v2')
)
if ($Resume) { $arguments += '--resume' }
if ($ContinueUntilConverged) { $arguments += '--continue-until-converged' }
if ($AllowHostMigration) { $arguments += '--allow-host-migration' }
if ($Tasks) { $arguments += '--tasks'; $arguments += $Tasks }

Push-Location $root
try { python @arguments }
finally { Pop-Location }
