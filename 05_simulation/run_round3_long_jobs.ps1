$ErrorActionPreference = "Stop"

$jobs = @(
    @{
        Name = "round3_range_rs_ratio"
        Args = @(
            "05_simulation/run_transfer_sweep.py",
            "--config", "05_simulation/configs/paper_transfer_train_n10_b10_singlehop.yaml",
            "--trained-config", "06_analysis/paper_tables/round2_transfer/training/best_config.yaml",
            "--output", "05_simulation/results_raw/round3_range_rs_ratio",
            "--node-counts", "50,100",
            "--beamwidth-degs", "10,15",
            "--mobilities", "gauss_markov",
            "--seeds", "20290704,20291713,20292722",
            "--episodes-per-seed", "1",
            "--slots", "600",
            "--slot-metric-period", "0",
            "--area-scale", "density",
            "--range-mode", "singlehop",
            "--communication-range-ratios", "1.05",
            "--sensing-to-comm-ratios", "0.5,0.75,1.0,1.25",
            "--train-node-count", "10",
            "--train-beamwidth-deg", "10"
        )
    },
    @{
        Name = "round3_ablation"
        Args = @(
            "05_simulation/run_transfer_sweep.py",
            "--config", "05_simulation/configs/paper_transfer_train_n10_b10_singlehop.yaml",
            "--trained-config", "06_analysis/paper_tables/round2_transfer/training/best_config.yaml",
            "--output", "05_simulation/results_raw/round3_ablation",
            "--node-counts", "10,50,100",
            "--beamwidth-degs", "10,15",
            "--mobilities", "gauss_markov",
            "--seeds", "20290704,20291713,20292722",
            "--episodes-per-seed", "1",
            "--slots", "600",
            "--slot-metric-period", "0",
            "--area-scale", "density",
            "--range-mode", "singlehop",
            "--protocols", "uniform_random,improved_rl_no_isac,improved_rl_isac,ablation_isac_no_candidate_set,ablation_isac_no_beam_lock,ablation_isac_no_topology",
            "--train-node-count", "10",
            "--train-beamwidth-deg", "10"
        )
    },
    @{
        Name = "round3_error_robustness"
        Args = @(
            "05_simulation/run_transfer_sweep.py",
            "--config", "05_simulation/configs/paper_transfer_train_n10_b10_singlehop.yaml",
            "--trained-config", "06_analysis/paper_tables/round2_transfer/training/best_config.yaml",
            "--output", "05_simulation/results_raw/round3_error_robustness",
            "--node-counts", "50,100",
            "--beamwidth-degs", "10",
            "--mobilities", "gauss_markov",
            "--seeds", "20290704,20291713",
            "--episodes-per-seed", "1",
            "--slots", "400",
            "--slot-metric-period", "0",
            "--area-scale", "density",
            "--range-mode", "singlehop",
            "--protocols", "improved_rl_no_isac,improved_rl_isac",
            "--false-alarm-rates", "0,0.005,0.01",
            "--miss-detection-rates", "0,0.02,0.05",
            "--angular-cell-offsets", "0",
            "--train-node-count", "10",
            "--train-beamwidth-deg", "10"
        )
    }
)

New-Item -ItemType Directory -Force -Path "tmp" | Out-Null

foreach ($job in $jobs) {
    $stdout = "tmp/$($job.Name).out.log"
    $stderr = "tmp/$($job.Name).err.log"
    foreach ($log in @($stdout, $stderr)) {
        if (Test-Path $log) {
            Remove-Item -LiteralPath $log -Force
        }
    }
    $output = "05_simulation/results_raw/$($job.Name)"
    if (Test-Path $output) {
        Remove-Item -LiteralPath $output -Recurse -Force
    }
    $process = Start-Process -FilePath "python" -ArgumentList $job.Args -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-Output "$($job.Name),$($process.Id),$stdout,$stderr"
}
