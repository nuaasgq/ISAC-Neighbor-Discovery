from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import SimulationConfig, load_config  # noqa: E402
from isac_nd_sim.marl_env import CANDIDATE_SOURCES, REWARD_VERSIONS, MarlNeighborDiscoveryEnv  # noqa: E402
from isac_nd_sim.neural_contention_actor_critic import (  # noqa: E402
    AdaptiveGatedContentionGraphActorCritic,
    BalancedTopologyGatedContentionGraphActorCritic,
    ContentionGraphActorCritic,
    GatedContentionGraphActorCritic,
    TopologyAdaptiveGatedContentionGraphActorCritic,
)
from isac_nd_sim.neural_scalegraph_beam_actor_critic import ScaleGraphBeamActorCritic  # noqa: E402
from isac_nd_sim.neural_recurrent_contention_actor_critic import (  # noqa: E402
    RecurrentContentionGraphActorCritic,
)
from isac_nd_sim.neural_shared_actor_critic import SharedBeamActorCritic  # noqa: E402
from isac_nd_sim.phy_sensing import SENSING_MEASUREMENT_MODES  # noqa: E402
from isac_nd_sim.simulator import Action, MODE_IDLE, MODE_RX, MODE_SENSE, MODE_TX  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained shared MARL policy under transfer settings.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="05_simulation/configs/twc_canonical_n10_b10.yaml")
    parser.add_argument("--output", default="05_simulation/results_raw/marl_eval")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--node-count", type=int, default=None)
    parser.add_argument("--area-size-m", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--azimuth-cells", type=int, default=None)
    parser.add_argument("--elevation-cells", type=int, default=None)
    parser.add_argument("--communication-range", type=float, default=None)
    parser.add_argument("--sensing-range", type=float, default=None)
    parser.add_argument("--false-alarm-rate", type=float, default=None)
    parser.add_argument("--miss-detection-rate", type=float, default=None)
    parser.add_argument("--angular-cell-offset-std", type=float, default=None)
    parser.add_argument("--sensing-period-slots", type=int, default=None)
    parser.add_argument("--sensing-measurement-mode", choices=SENSING_MEASUREMENT_MODES, default=None)
    parser.add_argument("--mobility-model", default=None)
    parser.add_argument("--env-protocol", default=None)
    parser.add_argument(
        "--candidate-source",
        choices=CANDIDATE_SOURCES,
        default=None,
        help="Source used to build MARL candidate_mask/candidate_score observations. Defaults to checkpoint value.",
    )
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions.")
    parser.add_argument("--eval-both", action="store_true", help="Run deterministic and stochastic evaluation.")
    parser.add_argument(
        "--beam-executor",
        choices=[
            "policy",
            "uniform_random",
            "local_candidate_random",
            "candidate_score_proportional",
            "rule_candidate",
            "wang_candidate_random",
        ],
        default="policy",
        help=(
            "Beam execution rule. 'policy' uses neural beam actions; "
            "'uniform_random' samples uniformly from every beam without using local memory; "
            "'local_candidate_random' samples uniformly from the same local candidate mask seen by the actor; "
            "'rule_candidate' keeps neural mode/gate actions but executes the "
            "beam through the local ISAC candidate/memory rule; "
            "'wang_candidate_random' keeps neural mode/gate actions but samples "
            "beams uniformly from the Wang sensing-table Flag=1 set."
        ),
    )
    parser.add_argument(
        "--mode-executor",
        choices=["policy", "uniform_tx_rx", "rule_protocol"],
        default="policy",
        help=(
            "Mode execution rule. 'policy' uses neural mode actions; "
            "'uniform_tx_rx' samples TX/RX with equal probability; "
            "'rule_protocol' uses the simulator's local protocol role rule while "
            "preserving the neural access-gate action."
        ),
    )
    parser.add_argument(
        "--policy-ablation",
        choices=["trained", "random_weights", "zero_weights"],
        default="trained",
        help="Evaluate the trained checkpoint, a randomly initialized policy, or a zero-weight rule-only policy.",
    )
    parser.add_argument("--ablation-label", default=None, help="Optional label recorded in the evaluation manifest.")
    parser.add_argument("--ablation-seed", type=int, default=None, help="Seed used for random/zero-weight ablations.")
    parser.add_argument("--disable-candidate-mask", action="store_true", help="Disable hard ISAC candidate beam masking at evaluation time.")
    parser.add_argument("--disable-candidate-score", action="store_true", help="Zero ISAC candidate confidence features at evaluation time.")
    parser.add_argument("--disable-topology-deficit", action="store_true", help="Zero topology-deficit features at evaluation time.")
    parser.add_argument("--disable-rule-residual", action="store_true", help="Disable handcrafted rule residual logits at evaluation time.")
    parser.add_argument(
        "--rendezvous-observation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override local sensing-derived rendezvous observations. "
            "Defaults to the resolved training checkpoint setting."
        ),
    )
    parser.add_argument(
        "--disable-contention-mode-prior",
        action="store_true",
        help="Disable the hand-coded contention/topology mode-logit prior in contention networks during evaluation.",
    )
    parser.add_argument(
        "--disable-rendezvous-adapter",
        action="store_true",
        help="Zero the trained ISAC evidence adapter while retaining the checkpoint's base network.",
    )
    parser.add_argument(
        "--forbid-sense",
        action="store_true",
        help="Disable standalone SENSE actions during evaluation. Defaults to the checkpoint setting when present.",
    )
    parser.add_argument(
        "--allow-idle",
        action="store_true",
        help="Allow IDLE even if the checkpoint used the TX/RX-only contract.",
    )
    parser.add_argument("--eval-rule-residual-scale", type=float, default=None, help="Override rule residual scale at evaluation time.")
    parser.add_argument("--reward-version", choices=REWARD_VERSIONS, default=None)
    parser.add_argument(
        "--mode-temperature",
        type=float,
        default=1.0,
        help="Temperature for stochastic mode sampling. Values >1 increase randomized role selection.",
    )
    parser.add_argument(
        "--beam-temperature",
        type=float,
        default=1.0,
        help="Temperature for stochastic beam sampling.",
    )
    parser.add_argument(
        "--gate-temperature",
        type=float,
        default=1.0,
        help="Temperature for stochastic access-gate sampling when the checkpoint has a gate head.",
    )
    parser.add_argument("--seed", type=int, default=30260705)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--resource-log-period", type=int, default=500)
    parser.add_argument("--max-rss-mb", type=float, default=12000.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=92.0)
    parser.add_argument(
        "--full-step-info",
        action="store_true",
        help="Collect per-slot metrics and rich step info during evaluation. Disabled by default for exact fast eval.",
    )
    parser.add_argument(
        "--target-status-diagnostics",
        action="store_true",
        help="Classify selected beams with offline true topology; disabled by default because it is expensive.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume from an existing partial eval_episode_metrics.csv in the output directory.",
    )
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for run_marl_evaluate.py") from exc

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    ensure_resource_args(args)
    checkpoint = load_checkpoint(args.checkpoint, torch)
    train_args = checkpoint.get("args", {})
    if args.rendezvous_observation is None and train_args.get("rendezvous_observation") is not None:
        args.rendezvous_observation = bool(train_args["rendezvous_observation"])
    cfg = override_config(load_config(args.config), args)
    checkpoint_feature_flags = checkpoint_feature_flags_from_args(train_args, checkpoint.get("feature_flags"))
    feature_flags = apply_eval_feature_overrides(checkpoint_feature_flags, args)
    hidden_dim = int(train_args.get("hidden_dim", 128))
    train_network = str(train_args.get("network", "shared"))
    if "allow_standalone_sense" in train_args:
        forbid_sense = bool(args.forbid_sense or not train_args.get("allow_standalone_sense", False))
    else:
        forbid_sense = bool(args.forbid_sense or train_args.get("forbid_sense", False))
    forbid_idle = bool(
        "allow_idle" in train_args
        and not train_args.get("allow_idle", False)
        and not args.allow_idle
    )
    reward_version = str(args.reward_version or train_args.get("reward_version", "legacy"))
    ablation_seed = int(args.ablation_seed) if args.ablation_seed is not None else int(args.seed) + 9173
    if str(args.policy_ablation) in {"random_weights", "zero_weights"}:
        torch.manual_seed(ablation_seed)
        np.random.seed(ablation_seed % (2**32 - 1))
    rule_residual_scale = (
        float(args.eval_rule_residual_scale)
        if args.eval_rule_residual_scale is not None
        else float(train_args.get("rule_residual_scale", 1.0))
    )
    if "contention_mode_prior" in train_args:
        checkpoint_contention_mode_prior = bool(train_args.get("contention_mode_prior", False))
    else:
        checkpoint_contention_mode_prior = not bool(train_args.get("disable_contention_mode_prior", False))
    use_contention_mode_prior = checkpoint_contention_mode_prior and not bool(args.disable_contention_mode_prior)
    checkpoint_rendezvous_adapter = bool(train_args.get("rendezvous_adapter", False))
    action_contract = str(
        checkpoint.get("action_contract", train_args.get("action_contract", "joint_role_beam"))
    )
    policy = build_policy(
        train_network,
        cfg.n_beams,
        hidden_dim=hidden_dim,
        device="cpu",
        use_candidate_mask=feature_flags["candidate_mask"],
        use_candidate_score=feature_flags["candidate_score"],
        use_topology_deficit=feature_flags["topology_deficit"],
        use_rule_residual=feature_flags["rule_residual"],
        rule_residual_scale=rule_residual_scale,
        use_contention_mode_prior=use_contention_mode_prior,
        use_rendezvous_adapter=checkpoint_rendezvous_adapter,
        use_residual_measurement_features=bool(train_args.get("residual_measurement_features", False)),
        measurement_feature_set=train_args.get("measurement_feature_set"),
        use_measurement_prediction_head=float(
            train_args.get("measurement_prediction_aux_coef", 0.0)
        )
        > 0.0,
        role_probability_floor=float(train_args.get("role_probability_floor", 0.0)),
        beam_uniform_mixture=float(train_args.get("beam_uniform_mixture", 0.0)),
        disabled_modes=disabled_modes_from_flags(forbid_sense, forbid_idle),
        action_contract=action_contract,
        azimuth_cells=int(cfg.azimuth_cells),
        elevation_cells=int(cfg.elevation_cells),
        use_candidate_score_prior=bool(train_args.get("candidate_score_prior", False)),
        candidate_score_prior_power=float(train_args.get("candidate_score_prior_power", 1.0)),
        use_bounded_score_residual=bool(train_args.get("bounded_score_residual", False)),
        score_residual_max_logit=float(train_args.get("score_residual_max_logit", 2.0)),
        use_decoupled_role_tower=bool(train_args.get("decoupled_role_tower", False)),
        role_factorization=str(train_args.get("role_factorization", "independent")),
    )
    checkpoint_loaded = str(args.policy_ablation) == "trained"
    if checkpoint_loaded:
        policy_state = migrate_policy_state_dict(
            checkpoint["policy_state_dict"],
            policy.model.state_dict(),
        )
        policy.model.load_state_dict(policy_state)
    elif str(args.policy_ablation) == "zero_weights":
        zero_policy_weights(policy, torch)
    if bool(args.disable_rendezvous_adapter):
        zero_rendezvous_adapter(policy, torch)
    policy.eval()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env_protocol = str(
        args.env_protocol
        or checkpoint.get("env_protocol")
        or train_args.get("resolved_env_protocol")
        or train_args.get("env_protocol")
        or inferred_env_protocol(train_args)
    )
    candidate_source = str(args.candidate_source or train_args.get("candidate_source", "default"))
    args._evaluation_fingerprint = evaluation_fingerprint(
        args, cfg, feature_flags, env_protocol, candidate_source, reward_version
    )
    resource_rows: list[dict[str, Any]] = []
    eval_rows = evaluate_policy(
        cfg,
        policy,
        torch,
        args,
        env_protocol,
        candidate_source,
        reward_version,
        progress_dir=output,
        resource_rows=resource_rows,
    )
    write_rows(output / "eval_episode_metrics.csv", eval_rows)
    write_rows(output / "resource_log.csv", resource_rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "marl_transfer_evaluation",
        "checkpoint": str(args.checkpoint),
        "train_algorithm": str(train_args.get("algorithm", checkpoint.get("algorithm", "unknown"))),
        "train_network": train_network,
        "train_reward_version": str(train_args.get("reward_version", "legacy")),
        "eval_reward_version": reward_version,
        "config": str(args.config),
        "output": str(args.output),
        "eval_episodes": int(args.eval_episodes),
        "slots_per_episode": int(cfg.slots_per_episode),
        "node_count": int(cfg.n_nodes),
        "area_size_m": [float(value) for value in cfg.area_size_m],
        "area_diagonal_m": math.sqrt(sum(float(value) ** 2 for value in cfg.area_size_m)),
        "beam_count": int(cfg.n_beams),
        "azimuth_cells": int(cfg.azimuth_cells),
        "elevation_cells": int(cfg.elevation_cells),
        "communication_range_m": float(cfg.communication_range_m),
        "sensing_range_m": float(cfg.sensing_range_m),
        "sensing_measurement": {
            "mode": cfg.sensing_measurement_mode,
            "identity_exposed_before_handshake": False,
            "common_protocol_interface": True,
        },
        "communication_phy": {
            "model": cfg.communication_phy_model,
            "carrier_frequency_hz": cfg.communication_carrier_frequency_hz,
            "bandwidth_hz": cfg.communication_bandwidth_hz,
            "tx_power_w": cfg.communication_tx_power_w,
            "noise_figure_db": cfg.communication_noise_figure_db,
            "path_loss_exponent": cfg.communication_path_loss_exponent,
            "shadowing_std_db": cfg.communication_shadowing_std_db,
            "rician_k_db": cfg.communication_rician_k_db,
            "sinr_threshold_db": cfg.communication_sinr_threshold_db,
            "sidelobe_gain_db": cfg.communication_sidelobe_gain_db,
            "antenna_gain_mode": cfg.communication_antenna_gain_mode,
            "fixed_main_lobe_gain_db": cfg.communication_fixed_main_lobe_gain_db,
            "shared_waveform_power_enabled": cfg.shared_waveform_power_enabled,
            "channel_seed_policy": "scenario_seed_only",
        },
        "rendezvous_observation_enabled": bool(cfg.rendezvous_observation_enabled),
        "env_protocol": env_protocol,
        "candidate_source": candidate_source,
        "forbid_sense": forbid_sense,
        "forbid_idle": forbid_idle,
        "disabled_modes": list(disabled_modes_from_flags(forbid_sense, forbid_idle)),
        "policy_ablation": str(args.policy_ablation),
        "ablation_label": str(args.ablation_label or args.policy_ablation),
        "ablation_seed": ablation_seed,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_feature_flags": checkpoint_feature_flags,
        "feature_flags": feature_flags,
        "rule_residual_scale": rule_residual_scale,
        "checkpoint_contention_mode_prior": checkpoint_contention_mode_prior,
        "contention_mode_prior": use_contention_mode_prior,
        "checkpoint_rendezvous_adapter": checkpoint_rendezvous_adapter,
        "residual_measurement_features": bool(train_args.get("residual_measurement_features", False)),
        "measurement_feature_set": str(
            train_args.get(
                "measurement_feature_set",
                "residual" if train_args.get("residual_measurement_features", False) else "none",
            )
        ),
        "measurement_prediction_aux_coef": float(
            train_args.get("measurement_prediction_aux_coef", 0.0)
        ),
        "stochastic_support": {
            "role_probability_floor": float(train_args.get("role_probability_floor", 0.0)),
            "beam_uniform_mixture": float(train_args.get("beam_uniform_mixture", 0.0)),
        },
        "rendezvous_adapter_disabled": bool(args.disable_rendezvous_adapter),
        "training_contract_version": str(checkpoint.get("training_contract_version", "legacy")),
        "evaluation_fingerprint": str(args._evaluation_fingerprint),
        "deterministic": bool(args.deterministic),
        "stochastic": bool(args.stochastic),
        "eval_both": bool(args.eval_both),
        "mode_temperature": float(args.mode_temperature),
        "beam_temperature": float(args.beam_temperature),
        "gate_temperature": float(args.gate_temperature),
        "policy_rng_seed_policy": "torch_and_numpy_episode_seed",
        "beam_executor": str(args.beam_executor),
        "mode_executor": str(args.mode_executor),
        "resource_limits": {
            "max_rss_mb": float(args.max_rss_mb),
            "max_system_memory_percent": float(args.max_system_memory_percent),
        },
        "resume": {
            "enabled": not bool(args.no_resume),
            "existing_rows_used": int(getattr(args, "_resume_existing_rows", 0)),
        },
        "fast_eval": {
            "collect_slot_metrics": bool(getattr(args, "full_step_info", False)),
            "rich_info": bool(getattr(args, "full_step_info", False)),
            "target_status_diagnostics": bool(getattr(args, "target_status_diagnostics", False)),
        },
        "final_eval": eval_rows[-1] if eval_rows else {},
        "files": ["eval_episode_metrics.csv", "resource_log.csv", "manifest.json"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def checkpoint_feature_flags_from_args(
    train_args: dict[str, Any], stored_flags: dict[str, Any] | None = None
) -> dict[str, bool]:
    if stored_flags is not None:
        return {
            name: bool(stored_flags.get(name, False))
            for name in ("candidate_mask", "candidate_score", "topology_deficit", "rule_residual")
        }
    feature_flags = {
        "candidate_mask": bool(train_args.get("candidate_mask", False)),
        "candidate_score": bool(train_args.get("candidate_score", False)),
        "topology_deficit": bool(train_args.get("topology_deficit", False)),
        "rule_residual": bool(train_args.get("rule_residual", False)),
    }
    if str(train_args.get("algorithm", "")) == "isac_mappo" and not bool(train_args.get("disable_isac_features", False)):
        feature_flags = {
            "candidate_mask": True,
            "candidate_score": True,
            "topology_deficit": True,
            "rule_residual": True,
        }
    return feature_flags


def apply_eval_feature_overrides(feature_flags: dict[str, bool], args: argparse.Namespace) -> dict[str, bool]:
    resolved = dict(feature_flags)
    if bool(getattr(args, "disable_candidate_mask", False)):
        resolved["candidate_mask"] = False
    if bool(getattr(args, "disable_candidate_score", False)):
        resolved["candidate_score"] = False
    if bool(getattr(args, "disable_topology_deficit", False)):
        resolved["topology_deficit"] = False
    if bool(getattr(args, "disable_rule_residual", False)):
        resolved["rule_residual"] = False
    return resolved


def zero_policy_weights(policy: Any, torch_module: Any) -> None:
    with torch_module.no_grad():
        for parameter in policy.model.parameters():
            parameter.zero_()


def zero_rendezvous_adapter(policy: Any, torch_module: Any) -> None:
    with torch_module.no_grad():
        for name, parameter in policy.model.named_parameters():
            if name.startswith("rendezvous_") and "adapter" in name:
                parameter.zero_()


def load_checkpoint(path: str | Path, torch_module: Any) -> dict[str, Any]:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def migrate_policy_state_dict(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
) -> dict[str, Any]:
    """Migrate derived buffers and pre-direction recurrent beam encoders."""

    import torch

    migrated = {
        key: value
        for key, value in source_state.items()
        if not key.endswith("beam_center_directions")
    }
    weight_key = "beam_linear.weight"
    if weight_key in migrated and weight_key in target_state:
        source_weight = migrated[weight_key]
        target_weight = target_state[weight_key]
        if (
            source_weight.ndim == 2
            and target_weight.ndim == 2
            and source_weight.shape[0] == target_weight.shape[0]
            and source_weight.shape[1] + 3 == target_weight.shape[1]
        ):
            migrated[weight_key] = torch.cat(
                [
                    source_weight,
                    torch.zeros(
                        (source_weight.shape[0], 3),
                        dtype=source_weight.dtype,
                        device=source_weight.device,
                    ),
                ],
                dim=1,
            )
    return migrated


def ensure_resource_args(args: argparse.Namespace) -> None:
    defaults = {
        "area_size_m": None,
        "full_step_info": False,
        "target_status_diagnostics": False,
        "no_resume": False,
        "resource_log_period": 500,
        "max_rss_mb": 12000.0,
        "max_system_memory_percent": 92.0,
        "policy_ablation": "trained",
        "ablation_label": None,
        "ablation_seed": None,
        "disable_candidate_mask": False,
        "disable_candidate_score": False,
        "disable_topology_deficit": False,
        "disable_rule_residual": False,
        "disable_contention_mode_prior": False,
        "disable_rendezvous_adapter": False,
        "rendezvous_observation": None,
        "eval_rule_residual_scale": None,
        "beam_executor": "policy",
        "mode_executor": "policy",
        "candidate_source": None,
        "forbid_sense": False,
        "allow_idle": False,
        "mode_temperature": 1.0,
        "beam_temperature": 1.0,
        "gate_temperature": 1.0,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    for name in ("mode_temperature", "beam_temperature", "gate_temperature"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")


def build_policy(
    network: str, *args: Any, **kwargs: Any
) -> (
    SharedBeamActorCritic
    | ScaleGraphBeamActorCritic
    | ContentionGraphActorCritic
    | GatedContentionGraphActorCritic
    | AdaptiveGatedContentionGraphActorCritic
    | TopologyAdaptiveGatedContentionGraphActorCritic
    | RecurrentContentionGraphActorCritic
):
    use_contention_mode_prior = bool(kwargs.pop("use_contention_mode_prior", True))
    use_rendezvous_adapter = bool(kwargs.pop("use_rendezvous_adapter", False))
    use_residual_measurement_features = bool(kwargs.pop("use_residual_measurement_features", False))
    measurement_feature_set = kwargs.pop("measurement_feature_set", None)
    use_measurement_prediction_head = bool(kwargs.pop("use_measurement_prediction_head", False))
    role_probability_floor = float(kwargs.pop("role_probability_floor", 0.0))
    beam_uniform_mixture = float(kwargs.pop("beam_uniform_mixture", 0.0))
    action_contract = str(kwargs.pop("action_contract", "joint_role_beam"))
    azimuth_cells = int(kwargs.pop("azimuth_cells", int(args[0]) if args else 1))
    elevation_cells = int(kwargs.pop("elevation_cells", 1))
    use_candidate_score_prior = bool(kwargs.pop("use_candidate_score_prior", False))
    candidate_score_prior_power = float(kwargs.pop("candidate_score_prior_power", 1.0))
    use_bounded_score_residual = bool(kwargs.pop("use_bounded_score_residual", False))
    score_residual_max_logit = float(kwargs.pop("score_residual_max_logit", 2.0))
    use_decoupled_role_tower = bool(kwargs.pop("use_decoupled_role_tower", False))
    role_factorization = str(kwargs.pop("role_factorization", "independent"))
    if str(network) == "shared":
        return SharedBeamActorCritic(*args, **kwargs)
    if str(network) == "scalegraph_beam":
        return ScaleGraphBeamActorCritic(*args, **kwargs)
    kwargs["use_contention_mode_prior"] = use_contention_mode_prior
    kwargs["use_rendezvous_adapter"] = use_rendezvous_adapter
    kwargs["use_residual_measurement_features"] = use_residual_measurement_features
    kwargs["role_probability_floor"] = role_probability_floor
    kwargs["beam_uniform_mixture"] = beam_uniform_mixture
    kwargs["action_contract"] = action_contract
    if str(network) == "contention_shared":
        return ContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "recurrent_contention_shared":
        kwargs["role_factorization"] = role_factorization
        kwargs["azimuth_cells"] = azimuth_cells
        kwargs["elevation_cells"] = elevation_cells
        kwargs["use_candidate_score_prior"] = use_candidate_score_prior
        kwargs["candidate_score_prior_power"] = candidate_score_prior_power
        kwargs["use_bounded_score_residual"] = use_bounded_score_residual
        kwargs["score_residual_max_logit"] = score_residual_max_logit
        kwargs["use_decoupled_role_tower"] = use_decoupled_role_tower
        kwargs["measurement_feature_set"] = measurement_feature_set
        kwargs["use_measurement_prediction_head"] = use_measurement_prediction_head
        return RecurrentContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "gated_contention_shared":
        return GatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "adaptive_gated_contention_shared":
        return AdaptiveGatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "topology_adaptive_gated_contention_shared":
        return TopologyAdaptiveGatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "balanced_topology_gated_contention_shared":
        return BalancedTopologyGatedContentionGraphActorCritic(*args, **kwargs)
    raise ValueError(f"Unsupported network in checkpoint: {network}")


def override_config(config: SimulationConfig, args: argparse.Namespace) -> SimulationConfig:
    replacements: dict[str, Any] = {
        "slots_per_episode": int(args.slots),
        "episodes": int(args.eval_episodes),
        "seed": int(args.seed),
    }
    optional_fields = {
        "node_count": "n_nodes",
        "azimuth_cells": "azimuth_cells",
        "elevation_cells": "elevation_cells",
        "communication_range": "communication_range_m",
        "sensing_range": "sensing_range_m",
        "false_alarm_rate": "false_alarm_rate",
        "miss_detection_rate": "miss_detection_rate",
        "angular_cell_offset_std": "angular_cell_offset_std",
        "sensing_period_slots": "sensing_period_slots",
        "sensing_measurement_mode": "sensing_measurement_mode",
    }
    for arg_name, field_name in optional_fields.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            replacements[field_name] = value
    rendezvous_observation = getattr(args, "rendezvous_observation", None)
    if rendezvous_observation is not None:
        replacements["rendezvous_observation_enabled"] = bool(rendezvous_observation)
    mobility = dict(config.mobility)
    if args.mobility_model is not None:
        mobility["model"] = str(args.mobility_model)
    replacements["mobility"] = mobility
    if args.area_size_m is not None:
        replacements["area_size_m"] = tuple(float(value) for value in args.area_size_m)
    return replace(config, **replacements)


def inferred_env_protocol(train_args: dict[str, Any]) -> str:
    if bool(train_args.get("disable_isac_features", False)):
        return "structured_marl_no_isac"
    if "allow_standalone_sense" in train_args or "contention_mode_prior" in train_args:
        return "improved_rl_isac_tables"
    return "isac_structured_marl"


def disabled_modes_from_flags(forbid_sense: bool, forbid_idle: bool = False) -> tuple[str, ...]:
    disabled = []
    if bool(forbid_sense):
        disabled.append(MODE_SENSE)
    if bool(forbid_idle):
        disabled.append(MODE_IDLE)
    return tuple(disabled)


def disabled_modes_from_flag(forbid_sense: bool) -> tuple[str, ...]:
    """Backward-compatible helper for historical analysis scripts."""

    return disabled_modes_from_flags(forbid_sense, False)


def evaluation_fingerprint(
    args: argparse.Namespace,
    cfg: SimulationConfig,
    feature_flags: dict[str, bool],
    env_protocol: str,
    candidate_source: str,
    reward_version: str,
) -> str:
    checkpoint_digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    payload = {
        "checkpoint_sha256": checkpoint_digest,
        "config": cfg.__dict__,
        "feature_flags": feature_flags,
        "env_protocol": env_protocol,
        "candidate_source": candidate_source,
        "reward_version": reward_version,
        "policy_ablation": str(args.policy_ablation),
        "ablation_seed": int(args.ablation_seed) if args.ablation_seed is not None else None,
        "eval_episodes": int(args.eval_episodes),
        "deterministic": bool(args.deterministic),
        "stochastic": bool(args.stochastic),
        "eval_both": bool(args.eval_both),
        "mode_temperature": float(args.mode_temperature),
        "beam_temperature": float(args.beam_temperature),
        "gate_temperature": float(args.gate_temperature),
        "beam_executor": str(args.beam_executor),
        "mode_executor": str(args.mode_executor),
        "target_status_diagnostics": bool(getattr(args, "target_status_diagnostics", False)),
        "disable_rendezvous_adapter": bool(args.disable_rendezvous_adapter),
        "allow_idle": bool(args.allow_idle),
        "seed": int(args.seed),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_policy(
    cfg: SimulationConfig,
    policy: SharedBeamActorCritic,
    torch_module: Any,
    args: argparse.Namespace,
    env_protocol: str,
    candidate_source: str,
    reward_version: str,
    progress_dir: Path | None = None,
    resource_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = (
        existing_eval_rows(progress_dir, str(args._evaluation_fingerprint))
        if progress_dir is not None and not bool(args.no_resume)
        else []
    )
    args._resume_existing_rows = len(rows)
    completed = {
        (str(row.get("phase", "")), int(row.get("eval_episode", -1)))
        for row in rows
        if str(row.get("eval_episode", "")).strip() != ""
    }
    if bool(args.eval_both):
        eval_modes = (False, True)
    elif bool(args.stochastic):
        eval_modes = (True,)
    else:
        eval_modes = (False,)
    with torch_module.no_grad():
        for mode_index, use_stochastic in enumerate(eval_modes):
            phase = "eval_stochastic" if use_stochastic else "eval_deterministic"
            for episode in range(int(args.eval_episodes)):
                if (phase, int(episode)) in completed:
                    continue
                seed = int(args.seed) + 10_000 * mode_index + episode
                torch_module.manual_seed(seed)
                np.random.seed(seed % (2**32 - 1))
                env = MarlNeighborDiscoveryEnv(
                    cfg,
                    seed=seed,
                    protocol=env_protocol,
                    reward_version=reward_version,
                    candidate_source=candidate_source,
                    collect_slot_metrics=bool(getattr(args, "full_step_info", False)),
                    rich_info=bool(getattr(args, "full_step_info", False)),
                    collect_target_status_metrics=bool(getattr(args, "target_status_diagnostics", False)),
                )
                observations, _ = env.reset(seed=seed)
                if hasattr(policy, "reset_recurrent_state"):
                    policy.reset_recurrent_state(env.n_agents)
                role_rng = np.random.default_rng(seed + 777)
                rewards = []
                truncated = False
                slot = 0
                while not truncated:
                    step = policy.act(
                        observations,
                        deterministic=not use_stochastic,
                        mode_temperature=float(args.mode_temperature),
                        beam_temperature=float(args.beam_temperature),
                        gate_temperature=float(args.gate_temperature),
                        role_rng=(role_rng if getattr(policy, "action_contract", "") == "beam_only_fixed_role" else None),
                    )
                    executed_actions = apply_action_executor(step.actions, env, args)
                    observations, reward, _terminated, truncated, _info = env.step(executed_actions)
                    rewards.append(torch_module.as_tensor(reward, dtype=torch_module.float32))
                    slot += 1
                    if int(args.resource_log_period) > 0 and slot % int(args.resource_log_period) == 0:
                        snapshot = resource_snapshot()
                        snapshot.update(
                            {
                                "phase": "eval_stochastic" if use_stochastic else "eval_deterministic",
                                "eval_episode": episode,
                                "slot": slot,
                                "seed": seed,
                                "node_count": int(cfg.n_nodes),
                                "beam_count": int(cfg.n_beams),
                                "slots_per_episode": int(cfg.slots_per_episode),
                            }
                        )
                        if resource_rows is not None:
                            resource_rows.append(snapshot)
                            if progress_dir is not None:
                                write_rows(progress_dir / "resource_log.csv", resource_rows)
                        enforce_resource_limits(snapshot, args)
                rewards_tensor = torch_module.stack(rewards)
                summary = env._sim.summarize(episode).as_dict()
                summary.update(env.access_gate_summary())
                row = {
                    "phase": phase,
                    "eval_episode": episode,
                    "seed": seed,
                    "env_protocol": env_protocol,
                    "candidate_source": candidate_source,
                    "episode_return_sum": float(rewards_tensor.sum().item()),
                    "episode_return_mean_per_agent": float(rewards_tensor.sum(dim=0).mean().item()),
                    "step_reward_mean": float(rewards_tensor.mean().item()),
                }
                row.update(summary)
                rows.append(row)
                if progress_dir is not None:
                    write_rows(progress_dir / "eval_episode_metrics.csv", rows)
                    progress = {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "evaluation_fingerprint": str(args._evaluation_fingerprint),
                        "completed_rows": len(rows),
                        "resumed_existing_rows": int(getattr(args, "_resume_existing_rows", 0)),
                        "eval_episodes": int(args.eval_episodes),
                        "mode": "stochastic" if use_stochastic else "deterministic",
                        "eval_episode": episode,
                        "slots_per_episode": int(cfg.slots_per_episode),
                        "node_count": int(cfg.n_nodes),
                        "beam_count": int(cfg.n_beams),
                        "latest": row,
                    }
                    (progress_dir / "progress.json").write_text(
                        json.dumps(progress, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                print(
                    json.dumps(
                        {
                            "phase": "eval_progress",
                            "completed_rows": len(rows),
                            "mode": "stochastic" if use_stochastic else "deterministic",
                            "eval_episode": episode,
                            "discovery_rate": row.get("discovery_rate"),
                            "lambda2": row.get("lambda2"),
                            "collision_count": row.get("collision_count"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return rows


def apply_action_executor(actions: list[Action], env: MarlNeighborDiscoveryEnv, args: argparse.Namespace) -> list[Action]:
    """Apply optional rule-based execution constraints to neural MARL actions.

    The helper uses only simulator-local public memory: belief, success/failure
    counters, discovered degree, and protocol RNG. It does not inspect hidden
    neighbor positions or true adjacency.
    """

    beam_executor = str(getattr(args, "beam_executor", "policy"))
    mode_executor = str(getattr(args, "mode_executor", "policy"))
    if beam_executor == "policy" and mode_executor == "policy":
        return actions

    rewritten: list[Action] = []
    for node, action in enumerate(actions):
        mode = action.mode
        beam = action.beam
        if mode_executor == "uniform_tx_rx":
            mode = uniform_tx_rx_mode(env)
        elif mode_executor == "rule_protocol":
            mode = env._sim.select_mode(node, env._slot)
        if mode == MODE_IDLE:
            beam = 0
        elif beam_executor == "uniform_random":
            beam = uniform_random_beam(env, mode)
        elif beam_executor == "local_candidate_random":
            beam = local_candidate_random_beam(env, node, mode)
        elif beam_executor == "candidate_score_proportional":
            beam = local_candidate_score_proportional_beam(env, node, mode)
        elif beam_executor == "rule_candidate":
            beam = rule_candidate_beam(env, node, env._slot, mode)
        elif beam_executor == "wang_candidate_random":
            beam = wang_candidate_random_beam(env, node, mode)
        rewritten.append(Action(mode, int(beam), action.access_gate))
    return rewritten


def uniform_tx_rx_mode(env: MarlNeighborDiscoveryEnv) -> str:
    return MODE_TX if float(env._sim.rng.random()) < 0.5 else MODE_RX


def uniform_random_beam(env: MarlNeighborDiscoveryEnv, mode: str) -> int:
    """Sample from all beams without consulting observations or protocol memory."""

    if mode == MODE_IDLE:
        return 0
    return int(env._sim.rng.integers(0, env.n_beams))


def local_candidate_random_beam(env: MarlNeighborDiscoveryEnv, node: int, mode: str) -> int:
    """Sample from the actor-visible local mask without using hidden topology."""

    if mode == MODE_IDLE:
        return 0
    candidate = env._candidate_features_for(int(node))
    active = np.flatnonzero(np.asarray(candidate["mask"], dtype=float) > 0.5)
    if active.size == 0:
        active = np.arange(env.n_beams, dtype=int)
    return int(env._sim.rng.choice(active))


def local_candidate_score_proportional_beam(
    env: MarlNeighborDiscoveryEnv,
    node: int,
    mode: str,
) -> int:
    """Sample a beam from actor-visible local candidate scores only."""

    if mode == MODE_IDLE:
        return 0
    candidate = env._candidate_features_for(int(node))
    active = np.flatnonzero(np.asarray(candidate["mask"], dtype=float) > 0.5)
    if active.size == 0:
        active = np.arange(env.n_beams, dtype=int)
    scores = np.maximum(0.0, np.asarray(candidate["score"], dtype=float)[active])
    if float(scores.sum()) <= 0.0:
        return int(env._sim.rng.choice(active))
    return int(env._sim.rng.choice(active, p=scores / scores.sum()))


def rule_candidate_beam(env: MarlNeighborDiscoveryEnv, node: int, slot: int, mode: str) -> int:
    if mode == MODE_IDLE:
        return 0
    if "isac" in env.protocol:
        candidate = env._sim.isac_candidate_cycle_beam(node, slot)
        if candidate is not None:
            return int(candidate)
        return int(env._sim.memory_guided_beam(node, use_isac=True, topology=True))
    return int(env._sim.memory_guided_beam(node, use_isac=False, topology=True))


def wang_candidate_random_beam(env: MarlNeighborDiscoveryEnv, node: int, mode: str) -> int:
    if mode == MODE_IDLE:
        return 0
    return int(env._sim.wang2025_table_beam(node))


def resource_snapshot() -> dict[str, Any]:
    try:
        import psutil
    except ImportError:
        return windows_resource_snapshot()
    process = psutil.Process()
    memory = psutil.virtual_memory()
    return {
        "rss_mb": process.memory_info().rss / (1024.0 * 1024.0),
        "process_cpu_percent": process.cpu_percent(interval=None),
        "system_memory_percent": memory.percent,
        "system_available_mb": memory.available / (1024.0 * 1024.0),
    }


def windows_resource_snapshot() -> dict[str, Any]:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return empty_resource_snapshot()
    if not hasattr(ctypes, "windll"):
        return empty_resource_snapshot()

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        memory = MemoryStatusEx()
        memory.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return empty_resource_snapshot()
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            rss_mb = current_process_rss_mb_fallback()
        else:
            rss_mb = counters.WorkingSetSize / (1024.0 * 1024.0)
        available_mb = memory.ullAvailPhys / (1024.0 * 1024.0)
        total_mb = max(memory.ullTotalPhys / (1024.0 * 1024.0), 1e-9)
        return {
            "rss_mb": rss_mb,
            "process_cpu_percent": "",
            "system_memory_percent": 100.0 * (1.0 - available_mb / total_mb),
            "system_available_mb": available_mb,
        }
    except Exception:
        return empty_resource_snapshot()


def current_process_rss_mb_fallback() -> float | str:
    try:
        import subprocess

        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-Process -Id {os.getpid()}).WorkingSet64",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = completed.stdout.strip()
        return float(value) / (1024.0 * 1024.0) if value else ""
    except Exception:
        return ""


def empty_resource_snapshot() -> dict[str, Any]:
    return {
        "rss_mb": "",
        "process_cpu_percent": "",
        "system_memory_percent": "",
        "system_available_mb": "",
    }


def enforce_resource_limits(snapshot: dict[str, Any], args: argparse.Namespace) -> None:
    rss = snapshot.get("rss_mb")
    memory_percent = snapshot.get("system_memory_percent")
    if isinstance(rss, (int, float)) and rss > float(args.max_rss_mb):
        raise RuntimeError(f"RSS memory limit exceeded: {rss:.1f} MB > {float(args.max_rss_mb):.1f} MB")
    if isinstance(memory_percent, (int, float)) and memory_percent > float(args.max_system_memory_percent):
        raise RuntimeError(
            f"System memory limit exceeded: {memory_percent:.1f}% > {float(args.max_system_memory_percent):.1f}%"
        )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def existing_eval_rows(progress_dir: Path, expected_fingerprint: str) -> list[dict[str, Any]]:
    data_path = progress_dir / "eval_episode_metrics.csv"
    if not data_path.exists() or data_path.stat().st_size == 0:
        return []
    fingerprint = None
    for metadata_name in ("progress.json", "manifest.json"):
        metadata_path = progress_dir / metadata_name
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fingerprint = metadata.get("evaluation_fingerprint")
        if fingerprint:
            break
    if fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Existing evaluation rows do not match this checkpoint/config. "
            "Use a new output directory or pass --no-resume."
        )
    try:
        with data_path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def main() -> None:
    print(json.dumps(run_evaluation(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
