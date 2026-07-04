param(
    [string]$TrainedConfig = "05_simulation/results_raw/round7_long_cem_train_n10_b10_600slot/best_config.yaml",
    [string]$OutputRoot = "05_simulation/results_raw",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function New-UniqueOutputPath {
    param([string]$BasePath)
    if (-not (Test-Path -LiteralPath $BasePath)) {
        return $BasePath
    }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    return "${BasePath}_${stamp}"
}

if (-not (Test-Path -LiteralPath $TrainedConfig)) {
    Write-Output "trained_config_missing=$TrainedConfig"
    Write-Output "Run the long CEM training first, then rerun this script."
    exit 0
}

New-Item -ItemType Directory -Force -Path "tmp" | Out-Null

$common = @(
    "--config", "05_simulation/configs/paper_transfer_train_n10_b10_singlehop.yaml",
    "--trained-config", $TrainedConfig,
    "--train-node-count", "10",
    "--train-beamwidth-deg", "10"
)

$jobs = @(
    @{
        Name = "round7_n100_multimobility_600slot"
        Args = @(
            "05_simulation/run_transfer_sweep.py"
        ) + $common + @(
            "--output", (Join-Path $OutputRoot "round7_n100_multimobility_600slot"),
            "--node-counts", "100",
            "--beamwidth-degs", "10,15",
            "--mobilities", "gauss_markov,random_walk,random_direction,random_waypoint",
            "--seeds", "20290704,20291713,20292722",
            "--episodes-per-seed", "1",
            "--slots", "600",
            "--slot-metric-period", "0",
            "--area-scale", "density",
            "--range-mode", "singlehop",
            "--protocols", "uniform_random,improved_rl_no_isac,ablation_isac_one_slot_delay,improved_rl_isac"
        )
    },
    @{
        Name = "round7_scale_beam_grid_light"
        Args = @(
            "05_simulation/run_transfer_sweep.py"
        ) + $common + @(
            "--output", (Join-Path $OutputRoot "round7_scale_beam_grid_light"),
            "--node-counts", "10,20,50,100",
            "--beamwidth-degs", "3,5,10,15,30",
            "--mobilities", "gauss_markov",
            "--seeds", "20290704,20291713",
            "--episodes-per-seed", "1",
            "--slots", "600",
            "--slot-metric-period", "0",
            "--area-scale", "density",
            "--range-mode", "singlehop",
            "--protocols", "uniform_random,improved_rl_no_isac,improved_rl_isac"
        )
    },
    @{
        Name = "round7_error_profiles_light"
        Args = @(
            "05_simulation/run_transfer_sweep.py"
        ) + $common + @(
            "--output", (Join-Path $OutputRoot "round7_error_profiles_light"),
            "--node-counts", "100",
            "--beamwidth-degs", "10",
            "--mobilities", "gauss_markov",
            "--seeds", "20290704,20291713,20292722",
            "--episodes-per-seed", "1",
            "--slots", "600",
            "--slot-metric-period", "0",
            "--area-scale", "density",
            "--range-mode", "singlehop",
            "--protocols", "improved_rl_no_isac,ablation_isac_one_slot_delay,improved_rl_isac",
            "--isac-error-profiles", "0:0:0;0.01:0.05:0.5;0.05:0.15:1.0;0.1:0.3:1.5"
        )
    }
)

foreach ($job in $jobs) {
    $rawOutputArgIndex = [Array]::IndexOf($job.Args, "--output")
    $baseOutput = $job.Args[$rawOutputArgIndex + 1]
    $output = New-UniqueOutputPath -BasePath $baseOutput
    $job.Args[$rawOutputArgIndex + 1] = $output
    $stdout = "tmp/$($job.Name).out.log"
    $stderr = "tmp/$($job.Name).err.log"
    if ($DryRun) {
        Write-Output "dry_run=$($job.Name) output=$output args=$($job.Args -join ' ')"
        continue
    }
    $process = Start-Process -FilePath "python" -ArgumentList $job.Args -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-Output "$($job.Name),$($process.Id),$output,$stdout,$stderr"
}
