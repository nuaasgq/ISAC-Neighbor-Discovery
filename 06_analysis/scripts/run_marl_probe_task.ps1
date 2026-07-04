param(
  [Parameter(Mandatory = $true)][string]$Workdir,
  [Parameter(Mandatory = $true)][string]$Output,
  [Parameter(Mandatory = $true)][string]$Variant,
  [Parameter(Mandatory = $true)][int]$Seed,
  [int]$BcEpisodes = 40,
  [int]$RlEpisodes = 0,
  [int]$EvalEpisodes = 5,
  [int]$Slots = 80,
  [int]$NodeCount = 10,
  [int]$AzimuthCells = 12,
  [int]$ElevationCells = 6,
  [double]$CommunicationRange = 1200.0,
  [double]$SensingRange = 1200.0,
  [string]$EnvProtocol = "isac_structured_marl",
  [string]$ExpertProtocol = "improved_rl_isac",
  [int]$HiddenDim = 64,
  [double]$LearningRate = 0.001,
  [double]$RuleResidualScale = 1.0
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Workdir
$env:PYTHONDONTWRITEBYTECODE = "1"

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$stdout = "$Output.stdout.log"
$stderr = "$Output.stderr.log"
$status = "$Output.status.txt"

$probeArgs = @(
  "05_simulation/run_actor_critic_imitation_probe.py",
  "--config", "05_simulation/configs/mvp.yaml",
  "--bc-episodes", [string]$BcEpisodes,
  "--rl-episodes", [string]$RlEpisodes,
  "--eval-episodes", [string]$EvalEpisodes,
  "--eval-both",
  "--slots", [string]$Slots,
  "--node-count", [string]$NodeCount,
  "--azimuth-cells", [string]$AzimuthCells,
  "--elevation-cells", [string]$ElevationCells,
  "--communication-range", [string]$CommunicationRange,
  "--sensing-range", [string]$SensingRange,
  "--false-alarm-rate", "0",
  "--miss-detection-rate", "0",
  "--angular-cell-offset-std", "0",
  "--sensing-period-slots", "1",
  "--env-protocol", $EnvProtocol,
  "--hidden-dim", [string]$HiddenDim,
  "--learning-rate", [string]$LearningRate,
  "--expert-protocol", $ExpertProtocol,
  "--output", $Output,
  "--seed", [string]$Seed
)

switch ($Variant) {
  "flat" {}
  "mask" {
    $probeArgs += "--candidate-mask"
  }
  "mask_score" {
    $probeArgs += @("--candidate-mask", "--candidate-score")
  }
  "mask_score_topo_rule" {
    $probeArgs += @(
      "--candidate-mask",
      "--candidate-score",
      "--topology-deficit",
      "--rule-residual",
      "--rule-residual-scale",
      [string]$RuleResidualScale
    )
  }
  default {
    throw "Unsupported Variant '$Variant'."
  }
}

"started $(Get-Date -Format s) variant=$Variant seed=$Seed" | Set-Content -Encoding UTF8 $status
& python @probeArgs 1> $stdout 2> $stderr
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  "failed $(Get-Date -Format s) exit=$exitCode variant=$Variant seed=$Seed" | Set-Content -Encoding UTF8 $status
  exit $exitCode
}
"completed $(Get-Date -Format s) variant=$Variant seed=$Seed" | Set-Content -Encoding UTF8 $status
