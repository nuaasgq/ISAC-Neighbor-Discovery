from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import SimulationConfig, load_config  # noqa: E402
from isac_nd_sim.centralized_graph_critic import CentralizedGraphCritic  # noqa: E402
from isac_nd_sim.neural_contention_actor_critic import (  # noqa: E402
    ACTION_CONTRACTS,
    AdaptiveGatedContentionGraphActorCritic,
    BalancedTopologyGatedContentionGraphActorCritic,
    ContentionGraphActorCritic,
    GatedContentionGraphActorCritic,
    TopologyAdaptiveGatedContentionGraphActorCritic,
)
from isac_nd_sim.marl_env import (  # noqa: E402
    ACCESS_GATE_TO_INDEX,
    CANDIDATE_SOURCES,
    MODE_NAMES,
    MODE_TO_INDEX,
    REWARD_VERSIONS,
    MarlNeighborDiscoveryEnv,
)
from isac_nd_sim.neural_scalegraph_beam_actor_critic import ScaleGraphBeamActorCritic  # noqa: E402
from isac_nd_sim.neural_recurrent_contention_actor_critic import (  # noqa: E402
    RecurrentContentionGraphActorCritic,
)
from isac_nd_sim.neural_shared_actor_critic import SharedBeamActorCritic  # noqa: E402
from isac_nd_sim.phy_sensing import SENSING_MEASUREMENT_MODES  # noqa: E402
from isac_nd_sim.simulator import (  # noqa: E402
    ACCESS_AGGRESSIVE,
    ACCESS_BACKOFF,
    ACCESS_NORMAL,
    MODE_RX,
    MODE_SENSE,
    MODE_TX,
    Action,
    beam_matches,
)


CLEAN_CTDE_CONTRACT_VERSION = "clean_local_ctde_v1"
CLEAN_CTDE_RESIDUAL_CONTRACT_VERSION = "clean_local_ctde_residual_v2"
BEAM_ONLY_ACTION_CONTRACTS = {
    "beam_only_fixed_role",
    "beam_only_complementary_role",
}


def is_beam_only_action_contract(action_contract: str) -> bool:
    return str(action_contract) in BEAM_ONLY_ACTION_CONTRACTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a real slot-level MARL policy for ISAC-assisted UAV neighbor discovery. "
            "The script logs per-step rewards, per-episode returns, resource usage, and held-out evaluation."
        )
    )
    parser.add_argument("--config", default="05_simulation/configs/twc_canonical_n10_b10.yaml")
    parser.add_argument("--output", default="05_simulation/results_raw/marl_training")
    parser.add_argument("--algorithm", choices=["ippo", "mappo", "isac_mappo"], default="isac_mappo")
    parser.add_argument(
        "--action-contract",
        choices=(*ACTION_CONTRACTS, "beam_only_complementary_role"),
        default="joint_role_beam",
    )
    parser.add_argument(
        "--network",
        choices=[
            "shared",
            "scalegraph_beam",
            "contention_shared",
            "recurrent_contention_shared",
            "gated_contention_shared",
            "adaptive_gated_contention_shared",
            "topology_adaptive_gated_contention_shared",
            "balanced_topology_gated_contention_shared",
        ],
        default="contention_shared",
    )
    parser.add_argument("--reward-version", choices=REWARD_VERSIONS, default="discovery_first")
    parser.add_argument(
        "--local-potential-shaping-coef",
        type=float,
        default=0.0,
        help="Potential-based shaping from local candidate count and score entropy.",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--stochastic-eval", action="store_true", help="Sample from the policy during evaluation.")
    parser.add_argument("--eval-both", action="store_true", help="Run both deterministic and stochastic evaluations.")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--flush-interval-episodes", type=int, default=1)
    parser.add_argument(
        "--training-scenario-mode", choices=("varying", "fixed"), default="varying"
    )
    parser.add_argument(
        "--evaluation-scenario-mode", choices=("held_out", "fixed"), default="held_out"
    )
    parser.add_argument("--terminate-on-full-discovery", action="store_true")
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
    parser.add_argument("--spatial-dimensions", type=int, choices=(2, 3), default=None)
    parser.add_argument("--env-protocol", default=None)
    parser.add_argument(
        "--candidate-source",
        choices=CANDIDATE_SOURCES,
        default="default",
        help="Source used to build MARL candidate_mask/candidate_score observations.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--critic-network", choices=("pooled", "mpnn"), default="pooled")
    parser.add_argument("--critic-hidden-dim", type=int, default=None)
    parser.add_argument("--return-scope", choices=("team", "per_agent"), default="team")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--advantage-estimator", choices=("mc", "gae"), default="mc")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--separate-action-loss",
        action="store_true",
        help="Train mode, beam, and gate action factors with separate PPO losses.",
    )
    parser.add_argument("--beam-loss-coef", type=float, default=1.0)
    parser.add_argument(
        "--beam-isac-feedback-coef",
        type=float,
        default=0.0,
        help="Add local post-action ISAC occupancy feedback to beam-only PPO credit.",
    )
    parser.add_argument("--gate-loss-coef", type=float, default=0.25)
    parser.add_argument(
        "--role-balance-coef",
        type=float,
        default=0.0,
        help="Training-only penalty on each slot's mean learned TX probability.",
    )
    parser.add_argument(
        "--beam-rank-aux-coef",
        type=float,
        default=0.0,
        help="Auxiliary coefficient for fitting beam logits to local candidate-score rankings.",
    )
    parser.add_argument("--beam-rank-temperature", type=float, default=4.0)
    parser.add_argument(
        "--rendezvous-beam-aux-coef",
        type=float,
        default=0.0,
        help="Training-only auxiliary coefficient for predicting the locally reprojected ISAC rendezvous beam.",
    )
    parser.add_argument(
        "--rendezvous-role-aux-coef",
        type=float,
        default=0.0,
        help="Training-only auxiliary coefficient for predicting the complementary TX/RX rendezvous role.",
    )
    parser.add_argument(
        "--rendezvous-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the zero-initialized learned ISAC evidence adapter in contention networks.",
    )
    parser.add_argument("--rendezvous-adapter-learning-rate", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--expert-bc-weight",
        type=float,
        default=0.0,
        help="Auxiliary behavior-cloning weight from a local rule expert. Zero disables expert guidance.",
    )
    parser.add_argument(
        "--expert-protocol",
        default="collision_aware_isac",
        help="Local simulator protocol used as the behavior-cloning expert when --expert-bc-weight > 0.",
    )
    parser.add_argument("--candidate-mask", action="store_true", help="Use local ISAC candidate masks in beam sampling.")
    parser.add_argument(
        "--candidate-score-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize a recurrent residual policy from the local candidate-score distribution.",
    )
    parser.add_argument(
        "--candidate-score-prior-power",
        type=float,
        default=1.0,
        help="Initial positive exponent applied to the local candidate-score policy.",
    )
    parser.add_argument(
        "--bounded-score-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bound recurrent logit corrections around the local candidate-score prior.",
    )
    parser.add_argument("--score-residual-max-logit", type=float, default=2.0)
    parser.add_argument(
        "--decoupled-role-tower",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a local role tower with no trainable parameters shared with the recurrent beam tower.",
    )
    parser.add_argument(
        "--role-factorization",
        choices=("independent", "beam_conditioned"),
        default="independent",
    )
    parser.add_argument(
        "--candidate-score",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use local ISAC candidate scores in beam-token features (enabled by default for ISAC-MAPPO).",
    )
    parser.add_argument(
        "--rendezvous-observation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Expose local sensing-derived rendezvous beam/role observations. "
            "Defaults to the selected YAML configuration."
        ),
    )
    parser.add_argument("--topology-deficit", action="store_true", help="Use local discovered-degree deficit token.")
    parser.add_argument(
        "--residual-measurement-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add local target-count, uncertainty, interaction, and residual-opportunity fields to beam tokens.",
    )
    parser.add_argument("--role-probability-floor", type=float, default=0.0)
    parser.add_argument("--beam-uniform-mixture", type=float, default=0.0)
    parser.add_argument("--rule-residual", action="store_true", help="Use local rule logits and beam priors as residual policy logits.")
    parser.add_argument("--rule-residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--contention-mode-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to the hand-coded contention/topology mode-logit prior.",
    )
    parser.add_argument("--disable-contention-mode-prior", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable-isac-features", action="store_true", help="Disable all ISAC/structured feature flags.")
    parser.add_argument(
        "--clean-ctde",
        action="store_true",
        help=(
            "Enforce a decentralized-observation CTDE contract: actors may use local ISAC candidate processing "
            "and exchanged tables, while global truth and pair-derived action guidance remain critic-only/forbidden."
        ),
    )
    parser.add_argument(
        "--forbid-sense",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-standalone-sense",
        action="store_true",
        help="Opt in to standalone SENSE actions; the default single-RF model senses only during TX.",
    )
    parser.add_argument(
        "--allow-idle",
        action="store_true",
        help="Opt in to IDLE; the default neighbor-discovery action space contains only TX/RX and beam selection.",
    )
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--step-log-period", type=int, default=1)
    parser.add_argument("--resource-log-period", type=int, default=25)
    parser.add_argument("--max-rss-mb", type=float, default=12000.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=92.0)
    return parser.parse_args()


class CentralizedPooledCritic:
    """Scale-invariant centralized state-value critic for CTDE training."""

    def __new__(cls, input_dim: int, hidden_dim: int, torch_module: Any):
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, features: Any) -> Any:
                return self.net(features).squeeze(-1)

        return Module()


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for run_marl_training.py") from exc

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    validate_args(args)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = override_config(load_config(args.config), args)
    if str(getattr(args, "action_contract", "joint_role_beam")) == "beam_only_complementary_role":
        if int(cfg.n_nodes) != 2:
            raise ValueError("beam_only_complementary_role is a two-node diagnostic contract only.")
        if str(args.network) != "recurrent_contention_shared":
            raise ValueError("beam_only_complementary_role requires recurrent_contention_shared.")
    args.rendezvous_observation = bool(cfg.rendezvous_observation_enabled)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    feature_flags = resolved_feature_flags(args)
    env_protocol = resolved_env_protocol(args)
    policy = build_policy(
        str(getattr(args, "network", "shared")),
        cfg.n_beams,
        hidden_dim=int(args.hidden_dim),
        device="cpu",
        use_candidate_mask=feature_flags["candidate_mask"],
        use_candidate_score=feature_flags["candidate_score"],
        use_topology_deficit=feature_flags["topology_deficit"],
        use_rule_residual=feature_flags["rule_residual"],
        rule_residual_scale=float(args.rule_residual_scale),
        use_contention_mode_prior=contention_mode_prior_enabled(args),
        use_rendezvous_adapter=bool(getattr(args, "rendezvous_adapter", False)),
        use_residual_measurement_features=bool(getattr(args, "residual_measurement_features", False)),
        role_probability_floor=float(getattr(args, "role_probability_floor", 0.0)),
        beam_uniform_mixture=float(getattr(args, "beam_uniform_mixture", 0.0)),
        disabled_modes=disabled_modes_from_args(args),
        action_contract=str(getattr(args, "action_contract", "joint_role_beam")),
        azimuth_cells=int(cfg.azimuth_cells),
        elevation_cells=int(cfg.elevation_cells),
        use_candidate_score_prior=bool(getattr(args, "candidate_score_prior", False)),
        candidate_score_prior_power=float(getattr(args, "candidate_score_prior_power", 1.0)),
        use_bounded_score_residual=bool(getattr(args, "bounded_score_residual", False)),
        score_residual_max_logit=float(getattr(args, "score_residual_max_logit", 2.0)),
        use_decoupled_role_tower=bool(getattr(args, "decoupled_role_tower", False)),
        role_factorization=str(getattr(args, "role_factorization", "independent")),
    )
    setattr(policy, "_expert_bc_weight_cache", float(getattr(args, "expert_bc_weight", 0.0)))
    centralized = str(args.algorithm) in {"mappo", "isac_mappo"}
    critic = None
    adapter_params = [
        parameter
        for name, parameter in policy.model.named_parameters()
        if name.startswith("rendezvous_") and "adapter" in name
    ]
    adapter_param_ids = {id(parameter) for parameter in adapter_params}
    params = [parameter for parameter in policy.parameters() if id(parameter) not in adapter_param_ids]
    if centralized:
        critic_hidden_dim = int(getattr(args, "critic_hidden_dim", None) or args.hidden_dim)
        if str(getattr(args, "critic_network", "pooled")) == "mpnn":
            node_dim, edge_dim, global_dim = central_graph_feature_dims()
            critic = CentralizedGraphCritic(
                node_feature_dim=node_dim,
                edge_feature_dim=edge_dim,
                global_feature_dim=global_dim,
                hidden_dim=critic_hidden_dim,
            )
        else:
            critic = CentralizedPooledCritic(central_feature_dim(), critic_hidden_dim, torch)
        params += list(critic.parameters())
    optimizer_groups: list[dict[str, Any]] = [{"params": params, "lr": float(args.learning_rate)}]
    if adapter_params:
        optimizer_groups.append(
            {
                "params": adapter_params,
                "lr": float(getattr(args, "rendezvous_adapter_learning_rate", 0.03)),
            }
        )
    optimizer = torch.optim.Adam(optimizer_groups)

    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    global_step = 0
    fixed_role_rng = np.random.default_rng(int(args.seed) + 19)

    for episode in range(int(args.episodes)):
        scenario_seed = (
            int(args.seed)
            if str(getattr(args, "training_scenario_mode", "varying")) == "fixed"
            else int(args.seed) + episode
        )
        trajectory = collect_trajectory(
            cfg=cfg,
            policy=policy,
            torch_module=torch,
            seed=scenario_seed,
            episode=episode,
            env_protocol=env_protocol,
            global_step_start=global_step,
            step_log_period=int(args.step_log_period),
            args=args,
            resource_rows=resource_rows,
            fixed_role_rng=fixed_role_rng,
        )
        global_step += int(trajectory["slots"])
        losses = update_policy(
            trajectory=trajectory,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            torch_module=torch,
            functional=F,
            args=args,
            centralized=centralized,
        )
        row = build_episode_row(trajectory, losses, episode, global_step, args, cfg)
        row.update(rendezvous_adapter_diagnostics(policy))
        episode_rows.append(row)
        step_rows.extend(trajectory["step_rows"])
        if should_checkpoint(episode + 1, int(args.checkpoint_interval)):
            save_checkpoint(output / f"checkpoint_ep{episode + 1:05d}.pt", policy, critic, optimizer, args, cfg, episode + 1, torch)
        if should_checkpoint(episode + 1, int(args.eval_interval)):
            eval_rows.extend(
            evaluate_policy(
                cfg=cfg,
                policy=policy,
                torch_module=torch,
                args=args,
                env_protocol=env_protocol,
                start_episode=episode + 1,
                seed_start=int(args.seed) + 1_000_000 + 1000 * (episode + 1),
                stochastic_eval=bool(args.stochastic_eval),
            )
            )
        if (episode + 1) % int(getattr(args, "flush_interval_episodes", 1)) == 0:
            flush_outputs(output, step_rows, episode_rows, eval_rows, resource_rows)

    if int(args.eval_episodes) > 0:
        eval_rows.extend(
            evaluate_policy(
                cfg=cfg,
                policy=policy,
                torch_module=torch,
                args=args,
                env_protocol=env_protocol,
                start_episode=int(args.episodes),
                seed_start=int(args.seed) + 2_000_000,
                stochastic_eval=bool(args.stochastic_eval),
            )
        )
    save_checkpoint(output / "final_model.pt", policy, critic, optimizer, args, cfg, int(args.episodes), torch)
    manifest = build_manifest(args, cfg, feature_flags, env_protocol, centralized, episode_rows, eval_rows)
    manifest.update(
        {
            "actor_parameter_count": int(sum(parameter.numel() for parameter in policy.parameters())),
            "critic_parameter_count": int(
                sum(parameter.numel() for parameter in critic.parameters()) if critic is not None else 0
            ),
            "return_scope": str(getattr(args, "return_scope", "team")),
            "gradient_clipping_scope": "actor_and_critic_separate",
            "kl_estimator": "mean_exp_logratio_minus_one_minus_logratio_v1",
            "tracked_worktree_dirty": bool(git_worktree_dirty()),
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    flush_outputs(output, step_rows, episode_rows, eval_rows, resource_rows)
    return manifest


def validate_args(args: argparse.Namespace) -> None:
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive.")
    if int(args.slots) <= 0:
        raise ValueError("--slots must be positive.")
    if int(args.ppo_epochs) <= 0:
        raise ValueError("--ppo-epochs must be positive.")
    if int(getattr(args, "flush_interval_episodes", 1)) <= 0:
        raise ValueError("--flush-interval-episodes must be positive.")
    if float(args.max_rss_mb) <= 0.0:
        raise ValueError("--max-rss-mb must be positive.")
    if str(getattr(args, "return_scope", "team")) == "per_agent" and str(
        getattr(args, "critic_network", "pooled")
    ) != "mpnn":
        raise ValueError("--return-scope per_agent requires --critic-network mpnn.")
    if bool(getattr(args, "candidate_score_prior", False)):
        if str(getattr(args, "network", "")) != "recurrent_contention_shared":
            raise ValueError("--candidate-score-prior requires --network recurrent_contention_shared.")
        if getattr(args, "candidate_score", None) is False:
            raise ValueError("--candidate-score-prior requires candidate-score observations.")
    if float(getattr(args, "candidate_score_prior_power", 1.0)) <= 0.0:
        raise ValueError("--candidate-score-prior-power must be positive.")
    if bool(getattr(args, "bounded_score_residual", False)) and not bool(
        getattr(args, "candidate_score_prior", False)
    ):
        raise ValueError("--bounded-score-residual requires --candidate-score-prior.")
    if float(getattr(args, "score_residual_max_logit", 2.0)) <= 0.0:
        raise ValueError("--score-residual-max-logit must be positive.")
    if bool(getattr(args, "decoupled_role_tower", False)) and str(
        getattr(args, "action_contract", "joint_role_beam")
    ) != "joint_role_beam":
        raise ValueError("--decoupled-role-tower requires --action-contract joint_role_beam.")
    if str(getattr(args, "role_factorization", "independent")) == "beam_conditioned":
        if str(getattr(args, "network", "")) != "recurrent_contention_shared":
            raise ValueError("--role-factorization beam_conditioned requires recurrent_contention_shared.")
        if str(getattr(args, "action_contract", "")) != "joint_role_beam":
            raise ValueError("--role-factorization beam_conditioned requires joint_role_beam.")
        if not bool(getattr(args, "decoupled_role_tower", False)):
            raise ValueError("--role-factorization beam_conditioned requires --decoupled-role-tower.")
    if not 0.0 <= float(getattr(args, "gae_lambda", 0.95)) <= 1.0:
        raise ValueError("--gae-lambda must be in [0, 1].")
    if float(getattr(args, "local_potential_shaping_coef", 0.0)) < 0.0:
        raise ValueError("--local-potential-shaping-coef must be nonnegative.")
    if float(getattr(args, "expert_bc_weight", 0.0)) < 0.0:
        raise ValueError("--expert-bc-weight must be nonnegative.")
    if float(getattr(args, "beam_loss_coef", 1.0)) < 0.0:
        raise ValueError("--beam-loss-coef must be nonnegative.")
    if float(getattr(args, "beam_isac_feedback_coef", 0.0)) < 0.0:
        raise ValueError("--beam-isac-feedback-coef must be nonnegative.")
    if float(getattr(args, "beam_isac_feedback_coef", 0.0)) > 0.0 and not bool(
        getattr(args, "separate_action_loss", False)
    ):
        raise ValueError("--beam-isac-feedback-coef requires --separate-action-loss.")
    if float(getattr(args, "gate_loss_coef", 0.25)) < 0.0:
        raise ValueError("--gate-loss-coef must be nonnegative.")
    if float(getattr(args, "role_balance_coef", 0.0)) < 0.0:
        raise ValueError("--role-balance-coef must be nonnegative.")
    if float(getattr(args, "beam_rank_aux_coef", 0.0)) < 0.0:
        raise ValueError("--beam-rank-aux-coef must be nonnegative.")
    if float(getattr(args, "beam_rank_temperature", 4.0)) <= 0.0:
        raise ValueError("--beam-rank-temperature must be positive.")
    if float(getattr(args, "rendezvous_beam_aux_coef", 0.0)) < 0.0:
        raise ValueError("--rendezvous-beam-aux-coef must be nonnegative.")
    if float(getattr(args, "rendezvous_role_aux_coef", 0.0)) < 0.0:
        raise ValueError("--rendezvous-role-aux-coef must be nonnegative.")
    if float(getattr(args, "rendezvous_adapter_learning_rate", 0.03)) <= 0.0:
        raise ValueError("--rendezvous-adapter-learning-rate must be positive.")
    role_floor = float(getattr(args, "role_probability_floor", 0.0))
    beam_mixture = float(getattr(args, "beam_uniform_mixture", 0.0))
    if not 0.0 <= role_floor < 0.5:
        raise ValueError("--role-probability-floor must be in [0, 0.5).")
    if not 0.0 <= beam_mixture <= 1.0:
        raise ValueError("--beam-uniform-mixture must be in [0, 1].")
    if role_floor > 0.0 and (bool(getattr(args, "allow_standalone_sense", False)) or bool(getattr(args, "allow_idle", False))):
        raise ValueError("--role-probability-floor requires the TX/RX-only action contract.")
    if (role_floor > 0.0 or beam_mixture > 0.0) and str(getattr(args, "network", "")) in {
        "shared",
        "scalegraph_beam",
    }:
        raise ValueError("stochastic support constraints require a contention network.")
    if bool(getattr(args, "rendezvous_adapter", False)) and str(getattr(args, "network", "")) in {
        "shared",
        "scalegraph_beam",
    }:
        raise ValueError("--rendezvous-adapter requires a contention network.")
    if is_beam_only_action_contract(
        str(getattr(args, "action_contract", "joint_role_beam"))
    ):
        if str(getattr(args, "algorithm", "")) not in {"mappo", "isac_mappo"}:
            raise ValueError("Beam-only MAPPO requires centralized training.")
        if str(getattr(args, "network", "")) not in {
            "contention_shared",
            "recurrent_contention_shared",
        }:
            raise ValueError(
                "Beam-only MAPPO requires --network contention_shared or recurrent_contention_shared."
            )
        if role_floor > 0.0:
            raise ValueError("Beam-only MAPPO uses nonlearned roles, not a role floor.")
        if bool(getattr(args, "allow_standalone_sense", False)) or bool(
            getattr(args, "allow_idle", False)
        ):
            raise ValueError("Beam-only MAPPO supports only fixed TX/RX plus beam selection.")
        forbidden_beam_only = {
            "expert_bc_weight": "behavior cloning",
            "beam_rank_aux_coef": "beam-rank auxiliary target",
            "rendezvous_beam_aux_coef": "rendezvous beam auxiliary target",
            "rendezvous_role_aux_coef": "rendezvous role auxiliary target",
        }
        enabled = [
            label
            for field, label in forbidden_beam_only.items()
            if float(getattr(args, field, 0.0)) > 0.0
        ]
        if bool(getattr(args, "separate_action_loss", False)):
            enabled.append("separate role/beam loss")
        if enabled:
            raise ValueError("Beam-only MAPPO forbids: " + ", ".join(enabled))
        forbidden_flags = {
            "rule_residual": "rule residual",
            "contention_mode_prior": "contention mode prior",
            "rendezvous_adapter": "rendezvous adapter",
        }
        enabled_flags = [
            label for field, label in forbidden_flags.items() if bool(getattr(args, field, False))
        ]
        if enabled_flags:
            raise ValueError("Beam-only MAPPO forbids: " + ", ".join(enabled_flags))
    if bool(getattr(args, "separate_action_loss", False)) and str(getattr(args, "network", "")) == "scalegraph_beam":
        raise ValueError("--separate-action-loss is not implemented for scalegraph_beam.")
    violations = clean_ctde_violations(args)
    if violations:
        raise ValueError("--clean-ctde forbids: " + ", ".join(violations))


def clean_ctde_violations(args: argparse.Namespace) -> list[str]:
    if not bool(getattr(args, "clean_ctde", False)):
        return []
    violations: list[str] = []
    if str(getattr(args, "algorithm", "")) not in {"mappo", "isac_mappo"}:
        violations.append("non-centralized algorithm")
    if str(getattr(args, "network", "shared")) not in {
        "shared",
        "scalegraph_beam",
        "contention_shared",
        "recurrent_contention_shared",
    }:
        violations.append("rule-gated network")
    forbidden_flags = {
        "rule_residual": "rule residual",
        "contention_mode_prior": "contention mode prior",
        "rendezvous_adapter": "rendezvous adapter",
    }
    for field, label in forbidden_flags.items():
        if bool(getattr(args, field, False)):
            violations.append(label)
    rendezvous_observation = getattr(args, "rendezvous_observation", None)
    if rendezvous_observation is True:
        violations.append("rendezvous observation")
    forbidden_coefficients = {
        "expert_bc_weight": "behavior cloning",
        "beam_rank_aux_coef": "beam-ranking action target",
        "rendezvous_beam_aux_coef": "rendezvous beam target",
        "rendezvous_role_aux_coef": "rendezvous role target",
    }
    for field, label in forbidden_coefficients.items():
        if float(getattr(args, field, 0.0)) > 0.0:
            violations.append(label)
    return violations


def disabled_modes_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    disabled_modes: list[str] = []
    if hasattr(args, "allow_standalone_sense"):
        disable_sense = not bool(args.allow_standalone_sense)
    else:
        disable_sense = bool(getattr(args, "forbid_sense", False))
    disable_sense = disable_sense or bool(getattr(args, "forbid_sense", False))
    if disable_sense:
        disabled_modes.append(MODE_SENSE)
    if hasattr(args, "allow_idle") and not bool(args.allow_idle):
        disabled_modes.append("idle")
    return tuple(disabled_modes)


def contention_mode_prior_enabled(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "clean_ctde", False)):
        return False
    enabled = bool(getattr(args, "contention_mode_prior", False))
    return enabled and not bool(getattr(args, "disable_contention_mode_prior", False))


def override_config(config: SimulationConfig, args: argparse.Namespace) -> SimulationConfig:
    replacements: dict[str, Any] = {
        "slots_per_episode": int(args.slots),
        "episodes": int(args.episodes),
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
    area_size = getattr(args, "area_size_m", None)
    if area_size is not None:
        replacements["area_size_m"] = tuple(float(value) for value in area_size)
    rendezvous_observation = getattr(args, "rendezvous_observation", None)
    if bool(getattr(args, "clean_ctde", False)):
        replacements["rendezvous_observation_enabled"] = False
    elif rendezvous_observation is not None:
        replacements["rendezvous_observation_enabled"] = bool(rendezvous_observation)
    mobility = dict(config.mobility)
    if args.mobility_model is not None:
        mobility["model"] = str(args.mobility_model)
    spatial_dims = getattr(args, "spatial_dimensions", None)
    if spatial_dims is not None:
        mobility["spatial_dimensions"] = int(spatial_dims)
    replacements["mobility"] = mobility
    return replace(config, **replacements)


def resolved_feature_flags(args: argparse.Namespace) -> dict[str, bool]:
    if bool(getattr(args, "clean_ctde", False)):
        is_isac = str(getattr(args, "algorithm", "")) == "isac_mappo"
        candidate_score = getattr(args, "candidate_score", None)
        return {
            "candidate_mask": bool(getattr(args, "candidate_mask", False)),
            "candidate_score": is_isac if candidate_score is None else bool(candidate_score),
            "topology_deficit": bool(getattr(args, "topology_deficit", False)),
            "rule_residual": False,
        }
    if bool(args.disable_isac_features):
        return {
            "candidate_mask": False,
            "candidate_score": False,
            "topology_deficit": bool(args.topology_deficit),
            "rule_residual": False,
        }
    if str(args.algorithm) == "isac_mappo":
        return {
            "candidate_mask": bool(args.candidate_mask),
            "candidate_score": True if args.candidate_score is None else bool(args.candidate_score),
            "topology_deficit": bool(args.topology_deficit),
            "rule_residual": bool(args.rule_residual),
        }
    return {
        "candidate_mask": bool(args.candidate_mask),
        "candidate_score": bool(args.candidate_score) if args.candidate_score is not None else False,
        "topology_deficit": bool(args.topology_deficit),
        "rule_residual": bool(args.rule_residual),
    }


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
    use_contention_mode_prior = bool(kwargs.pop("use_contention_mode_prior", False))
    use_rendezvous_adapter = bool(kwargs.pop("use_rendezvous_adapter", False))
    use_residual_measurement_features = bool(kwargs.pop("use_residual_measurement_features", False))
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
        if action_contract != "joint_role_beam":
            raise ValueError("beam_only_fixed_role requires a contention network.")
        return SharedBeamActorCritic(*args, **kwargs)
    if str(network) == "scalegraph_beam":
        if action_contract != "joint_role_beam":
            raise ValueError("beam_only_fixed_role requires a contention network.")
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
        return RecurrentContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "gated_contention_shared":
        return GatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "adaptive_gated_contention_shared":
        return AdaptiveGatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "topology_adaptive_gated_contention_shared":
        return TopologyAdaptiveGatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "balanced_topology_gated_contention_shared":
        return BalancedTopologyGatedContentionGraphActorCritic(*args, **kwargs)
    raise ValueError(f"Unsupported network: {network}")


def resolved_env_protocol(args: argparse.Namespace) -> str:
    if args.env_protocol:
        return str(args.env_protocol)
    if bool(args.disable_isac_features):
        return "structured_marl_no_isac"
    return "improved_rl_isac_tables"


def collect_trajectory(
    cfg: SimulationConfig,
    policy: SharedBeamActorCritic,
    torch_module: Any,
    seed: int,
    episode: int,
    env_protocol: str,
    global_step_start: int,
    step_log_period: int,
    args: argparse.Namespace,
    resource_rows: list[dict[str, Any]],
    fixed_role_rng: np.random.Generator,
) -> dict[str, Any]:
    env = MarlNeighborDiscoveryEnv(
        cfg,
        seed=seed,
        protocol=env_protocol,
        reward_version=str(getattr(args, "reward_version", "legacy")),
        candidate_source=str(getattr(args, "candidate_source", "default")),
    )
    observations, _ = env.reset(seed=seed)
    if hasattr(policy, "reset_recurrent_state"):
        policy.reset_recurrent_state(env.n_agents)
    old_log_probs = []
    old_mode_log_probs = []
    old_beam_log_probs = []
    old_gate_log_probs = []
    active_beam_masks = []
    rewards = []
    potential_shaping_rewards = []
    beam_isac_feedback_rows = []
    observations_by_step = []
    actions_by_step = []
    expert_actions_by_step = []
    central_features = []
    central_graph_rows: list[dict[str, np.ndarray]] = []
    step_rows = []
    rendezvous_totals = empty_rendezvous_action_diagnostics()
    cumulative_reward = 0.0
    role_trace = hashlib.blake2b(digest_size=12)
    truncated = False
    policy.train()

    while not truncated:
        slot = len(rewards)
        state = env.training_state()
        observations_by_step.append(copy_observations(observations))
        central_features.append(central_state_features(state, cfg))
        central_graph_rows.append(central_graph_features(state, cfg))
        if float(getattr(args, "expert_bc_weight", 0.0)) > 0.0:
            expert_actions_by_step.append(
                expert_actions_for_env(env, str(getattr(args, "expert_protocol", "collision_aware_isac")))
            )
        else:
            expert_actions_by_step.append([])
        with torch_module.no_grad():
            step = policy.act(
                observations,
                deterministic=False,
                role_rng=(
                    fixed_role_rng
                    if str(getattr(args, "action_contract", "joint_role_beam"))
                    == "beam_only_fixed_role"
                    else None
                ),
            )
        action_diagnostics = beam_selection_diagnostics(observations_by_step[-1], step.actions)
        role_trace.update(bytes(1 if action.mode == MODE_TX else 0 for action in step.actions))
        action_diagnostics.update(rendezvous_pair_diagnostics(env, observations_by_step[-1], step.actions))
        accumulate_rendezvous_action_diagnostics(rendezvous_totals, action_diagnostics)
        next_observations, reward, _terminated, truncated, info = env.step(step.actions)
        if (
            bool(getattr(args, "terminate_on_full_discovery", False))
            and len(env._sim.first_true_slot) > 0
            and len(env._sim.discovered_edges) >= len(env._sim.first_true_slot)
        ):
            truncated = True
        shaping = local_potential_shaping_reward(
            observations,
            next_observations,
            gamma=float(args.gamma),
            terminal=bool(truncated),
            coefficient=float(getattr(args, "local_potential_shaping_coef", 0.0)),
        )
        reward = np.asarray(reward, dtype=np.float32) + shaping
        beam_isac_feedback = local_beam_isac_feedback(step.actions, next_observations)
        observations = next_observations
        old_log_probs.append(step.log_probs.detach().cpu())
        old_mode_log_probs.append(component_or_zeros(step.mode_log_probs, step.log_probs).detach().cpu())
        old_beam_log_probs.append(component_or_zeros(step.beam_log_probs, step.log_probs).detach().cpu())
        old_gate_log_probs.append(component_or_zeros(step.gate_log_probs, step.log_probs).detach().cpu())
        active_beam_masks.append(active_mask_tensor(step, step.actions, torch_module).detach().cpu())
        reward_tensor = torch_module.as_tensor(reward, dtype=torch_module.float32)
        shaping_tensor = torch_module.as_tensor(shaping, dtype=torch_module.float32)
        rewards.append(reward_tensor)
        potential_shaping_rewards.append(shaping_tensor)
        beam_isac_feedback_rows.append(
            torch_module.as_tensor(beam_isac_feedback, dtype=torch_module.float32)
        )
        actions_by_step.append(step.actions)
        cumulative_reward += float(reward_tensor.sum().item())
        global_step = global_step_start + slot + 1

        if step_log_period > 0 and slot % step_log_period == 0:
            true_edges = max(1, int(state["true_edges"].shape[0]))
            row = {
                "episode": episode,
                "slot": slot,
                "training_step": global_step,
                "seed": seed,
                "algorithm": str(args.algorithm),
                "env_protocol": env_protocol,
                "reward_sum": float(reward_tensor.sum().item()),
                "reward_mean": float(reward_tensor.mean().item()),
                "potential_shaping_reward_mean": float(shaping_tensor.mean().item()),
                "beam_isac_feedback_mean": float(np.mean(beam_isac_feedback)),
                "reward_std_across_agents": float(reward_tensor.std(unbiased=False).item()),
                "reward_min_agent": float(reward_tensor.min().item()),
                "reward_max_agent": float(reward_tensor.max().item()),
                "positive_reward_agents": int((reward_tensor > 0.0).sum().item()),
                "episode_cumulative_reward": cumulative_reward,
                "new_edges_count": int(info["new_edges_count"]),
                "discovered_edges": int(info["discovered_edges_count"]),
                "active_discovered_edges": int(info.get("active_discovered_edges_count", info["discovered_edges_count"])),
                "true_edges": true_edges,
                "discovery_rate": int(info["discovered_edges_count"]) / true_edges,
                "empty_scan_ratio": float(info["empty_scan_ratio"]),
                "handshake_attempts": int(info.get("handshake_attempts", 0)),
                "handshake_successes": int(info.get("handshake_successes", 0)),
                "forward_decode_failures": int(info.get("forward_decode_failures", 0)),
                "ack_decode_failures": int(info.get("ack_decode_failures", 0)),
                "interference_limited_failures": int(info.get("interference_limited_failures", 0)),
                "phy_outage_failures": int(info.get("phy_outage_failures", 0)),
                "mean_handshake_sinr_db": float(info.get("mean_handshake_sinr_db", 0.0)),
                "collision_count": int(info["collision_count"]),
                "lambda2": float(info["lambda2"]),
                "knowledge_lambda2": float(info.get("knowledge_lambda2", info["lambda2"])),
                "lcc_ratio": float(info["lcc_ratio"]),
                "scan_actions": int(info["scan_actions"]),
                "tx_actions": int(info["tx_actions"]),
                "rx_actions": int(info["rx_actions"]),
                "sense_actions": int(info["sense_actions"]),
                "idle_actions": int(info["idle_actions"]),
                "access_gate_backoff_count": int(info.get("access_gate_backoff_count", 0)),
                "access_gate_normal_count": int(info.get("access_gate_normal_count", 0)),
                "access_gate_aggressive_count": int(info.get("access_gate_aggressive_count", 0)),
                "access_gate_backoff_ratio": float(info.get("access_gate_backoff_ratio", 0.0)),
                "access_gate_normal_ratio": float(info.get("access_gate_normal_ratio", 0.0)),
                "access_gate_aggressive_ratio": float(info.get("access_gate_aggressive_ratio", 0.0)),
            }
            row.update(action_diagnostics)
            step_rows.append(row)
        if int(args.resource_log_period) > 0 and global_step % int(args.resource_log_period) == 0:
            snapshot = resource_snapshot()
            snapshot.update({"episode": episode, "slot": slot, "training_step": global_step})
            resource_rows.append(snapshot)
            enforce_resource_limits(snapshot, args)

    summary = env._sim.summarize(episode).as_dict()
    summary.update(env.access_gate_summary())
    summary.update(summarize_rendezvous_action_diagnostics(rendezvous_totals))
    summary["role_sequence_hash"] = role_trace.hexdigest()
    replay_error = 0.0
    if hasattr(policy, "evaluate_action_sequence"):
        with torch_module.no_grad():
            replay = policy.evaluate_action_sequence(observations_by_step, actions_by_step)
        old = torch_module.stack([row.to(policy.device) for row in old_log_probs])
        replay_error = float(torch_module.max(torch_module.abs(replay["log_probs"] - old)).item())
        if replay_error > 1.0e-6:
            raise RuntimeError(
                f"Recurrent rollout/replay log-prob mismatch: {replay_error:.3e} exceeds 1e-6."
            )
    return {
        "episode": episode,
        "seed": seed,
        "slots": len(rewards),
        "observations": observations_by_step,
        "actions": actions_by_step,
        "expert_actions": expert_actions_by_step,
        "old_log_probs": torch_module.stack([row.to(policy.device) for row in old_log_probs]),
        "old_mode_log_probs": torch_module.stack([row.to(policy.device) for row in old_mode_log_probs]),
        "old_beam_log_probs": torch_module.stack([row.to(policy.device) for row in old_beam_log_probs]),
        "old_gate_log_probs": torch_module.stack([row.to(policy.device) for row in old_gate_log_probs]),
        "active_beam_mask": torch_module.stack([row.to(policy.device) for row in active_beam_masks]),
        "rewards": torch_module.stack(rewards).to(policy.device),
        "potential_shaping_rewards": torch_module.stack(potential_shaping_rewards).to(policy.device),
        "beam_isac_feedback": torch_module.stack(beam_isac_feedback_rows).to(policy.device),
        "central_features": torch_module.as_tensor(
            np.stack(central_features), dtype=torch_module.float32, device=policy.device
        ),
        "central_graph": {
            key: torch_module.as_tensor(
                np.stack([row[key] for row in central_graph_rows]),
                dtype=(torch_module.bool if key == "edge_mask" else torch_module.float32),
                device=policy.device,
            )
            for key in central_graph_rows[0]
        },
        "step_rows": step_rows,
        "summary": summary,
        "rollout_replay_logprob_max_abs_error": replay_error,
    }


def component_or_zeros(component: Any, fallback: Any) -> Any:
    if component is None:
        return fallback * 0.0
    return component


def active_mask_tensor(step: Any, actions: Sequence[Action], torch_module: Any) -> Any:
    if getattr(step, "active_beam_mask", None) is not None:
        return step.active_beam_mask.bool()
    return torch_module.as_tensor([action.mode != "idle" for action in actions], dtype=torch_module.bool)


def beam_selection_diagnostics(observations: Sequence[dict[str, Any]], actions: Sequence[Action]) -> dict[str, Any]:
    candidate_counts: list[float] = []
    selected_scores: list[float] = []
    score_gaps: list[float] = []
    top1_hits = 0
    top3_hits = 0
    active = 0
    rendezvous_available = 0
    rendezvous_beam_hits = 0
    rendezvous_mode_matches = 0
    rendezvous_joint_matches = 0
    for observation, action in zip(observations, actions, strict=True):
        rendezvous_score = np.asarray(
            observation.get("rendezvous_beam_score", np.zeros(0, dtype=np.float32)),
            dtype=float,
        )
        role_hint = float(np.asarray(observation.get("rendezvous_role_hint", [0.0]), dtype=float).reshape(-1)[0])
        has_rendezvous = bool(rendezvous_score.size and np.any(rendezvous_score > 0.0) and role_hint != 0.0)
        if has_rendezvous:
            rendezvous_available += 1
            expected_mode = "tx" if role_hint > 0.0 else "rx"
            mode_match = action.mode == expected_mode
            beam_hit = 0 <= int(action.beam) < rendezvous_score.size and rendezvous_score[int(action.beam)] > 0.0
            rendezvous_mode_matches += int(mode_match)
            rendezvous_beam_hits += int(beam_hit)
            rendezvous_joint_matches += int(mode_match and beam_hit)
        if action.mode == "idle":
            continue
        active += 1
        score = np.asarray(observation.get("candidate_score", np.zeros(0, dtype=np.float32)), dtype=float)
        mask = np.asarray(observation.get("candidate_mask", np.ones_like(score)), dtype=float) > 0.5
        candidates = np.flatnonzero(mask)
        candidate_counts.append(float(len(candidates)))
        if score.size == 0 or not 0 <= int(action.beam) < score.size or len(candidates) == 0:
            continue
        selected = float(score[int(action.beam)])
        visible_scores = score[candidates]
        best = float(np.max(visible_scores))
        rank = int(1 + np.sum(visible_scores > selected + 1e-12))
        selected_scores.append(selected)
        score_gaps.append(best - selected)
        top1_hits += int(rank == 1)
        top3_hits += int(rank <= 3)
    denom = max(1, active)
    rendezvous_denom = max(1, rendezvous_available)
    return {
        "active_actions": int(active),
        "beam_candidate_count_mean_active": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "beam_selected_score_mean_active": float(np.mean(selected_scores)) if selected_scores else 0.0,
        "beam_score_gap_mean_active": float(np.mean(score_gaps)) if score_gaps else 0.0,
        "beam_top1_rate_active": float(top1_hits / denom),
        "beam_top3_rate_active": float(top3_hits / denom),
        "rendezvous_observation_agent_count": int(rendezvous_available),
        "rendezvous_beam_hit_count": int(rendezvous_beam_hits),
        "rendezvous_mode_match_count": int(rendezvous_mode_matches),
        "rendezvous_joint_action_count": int(rendezvous_joint_matches),
        "rendezvous_beam_hit_rate": float(rendezvous_beam_hits / rendezvous_denom),
        "rendezvous_mode_match_rate": float(rendezvous_mode_matches / rendezvous_denom),
        "rendezvous_joint_action_rate": float(rendezvous_joint_matches / rendezvous_denom),
    }


def empty_rendezvous_action_diagnostics() -> dict[str, int]:
    return {
        "rendezvous_observation_agent_count": 0,
        "rendezvous_beam_hit_count": 0,
        "rendezvous_mode_match_count": 0,
        "rendezvous_joint_action_count": 0,
        "reciprocal_report_pair_count": 0,
        "reciprocal_scheduled_pair_count": 0,
        "reciprocal_actor_pair_count": 0,
    }


def accumulate_rendezvous_action_diagnostics(total: dict[str, int], row: dict[str, Any]) -> None:
    for key in total:
        total[key] += int(row.get(key, 0))


def summarize_rendezvous_action_diagnostics(total: dict[str, int]) -> dict[str, Any]:
    available = int(total["rendezvous_observation_agent_count"])
    denom = max(1, available)
    return {
        **total,
        "rendezvous_beam_hit_rate": float(total["rendezvous_beam_hit_count"] / denom),
        "rendezvous_mode_match_rate": float(total["rendezvous_mode_match_count"] / denom),
        "rendezvous_joint_action_rate": float(total["rendezvous_joint_action_count"] / denom),
        "reciprocal_schedule_rate": float(
            total["reciprocal_scheduled_pair_count"] / max(1, total["reciprocal_report_pair_count"])
        ),
        "reciprocal_actor_conversion_rate": float(
            total["reciprocal_actor_pair_count"] / max(1, total["reciprocal_scheduled_pair_count"])
        ),
    }


def rendezvous_pair_diagnostics(
    env: MarlNeighborDiscoveryEnv,
    observations: Sequence[dict[str, Any]],
    actions: Sequence[Action],
) -> dict[str, int]:
    """Truth-assisted offline audit; outputs never enter actor observations or rewards."""

    simulator = env._sim
    reciprocal_reports = 0
    reciprocal_scheduled = 0
    reciprocal_actor = 0
    for first in range(env.n_agents):
        for second in range(first + 1, env.n_agents):
            if not _has_anonymous_report_for(simulator, first, second, env._slot):
                continue
            if not _has_anonymous_report_for(simulator, second, first, env._slot):
                continue
            reciprocal_reports += 1
            first_beam = simulator.beam_from_to(first, second)
            second_beam = simulator.beam_from_to(second, first)
            first_active = _rendezvous_score_matches(observations[first], first_beam, env.cfg)
            second_active = _rendezvous_score_matches(observations[second], second_beam, env.cfg)
            if not (first_active and second_active):
                continue
            reciprocal_scheduled += 1
            first_action = actions[first]
            second_action = actions[second]
            aligned = beam_matches(
                int(first_action.beam),
                first_beam,
                env.cfg.azimuth_cells,
                env.cfg.alignment_tolerance_cells,
            ) and beam_matches(
                int(second_action.beam),
                second_beam,
                env.cfg.azimuth_cells,
                env.cfg.alignment_tolerance_cells,
            )
            complementary = {first_action.mode, second_action.mode} == {MODE_TX, MODE_RX}
            reciprocal_actor += int(aligned and complementary)
    return {
        "reciprocal_report_pair_count": int(reciprocal_reports),
        "reciprocal_scheduled_pair_count": int(reciprocal_scheduled),
        "reciprocal_actor_pair_count": int(reciprocal_actor),
    }


def _has_anonymous_report_for(simulator: Any, source: int, target: int, slot: int) -> bool:
    age = int(slot) - simulator.sensing_report_slot[source]
    valid = (
        (age >= 0)
        & (age <= max(1, int(simulator.cfg.sensing_report_ttl_slots)))
        & (simulator.sensing_report_confidence[source] > 0.0)
        & np.all(np.isfinite(simulator.sensing_report_position[source]), axis=1)
    )
    if not np.any(valid):
        return False
    tolerance_m = max(100.0, 4.0 * float(simulator.cfg.sensing_position_error_std_m))
    errors = np.linalg.norm(
        simulator.sensing_report_position[source, valid] - simulator.states[target].position,
        axis=1,
    )
    return bool(np.any(errors <= tolerance_m))


def _rendezvous_score_matches(observation: dict[str, Any], target_beam: int, cfg: SimulationConfig) -> bool:
    score = np.asarray(observation.get("rendezvous_beam_score", np.zeros(0)), dtype=float)
    return any(
        score[int(beam)] > 0.0
        and beam_matches(int(beam), int(target_beam), cfg.azimuth_cells, cfg.alignment_tolerance_cells)
        for beam in np.flatnonzero(score > 0.0)
    )


def rendezvous_adapter_diagnostics(policy: Any) -> dict[str, float]:
    model = policy.model
    if not hasattr(model, "rendezvous_beam_adapter") or not hasattr(model, "rendezvous_mode_adapter"):
        return {
            "rendezvous_adapter_beam_score_weight": 0.0,
            "rendezvous_adapter_beam_presence_weight": 0.0,
            "rendezvous_adapter_tx_role_weight": 0.0,
            "rendezvous_adapter_rx_role_weight": 0.0,
        }
    beam_weight = model.rendezvous_beam_adapter.weight.detach().cpu().reshape(-1)
    mode_weight = model.rendezvous_mode_adapter.weight.detach().cpu()
    return {
        "rendezvous_adapter_beam_score_weight": float(beam_weight[0].item()),
        "rendezvous_adapter_beam_presence_weight": float(beam_weight[1].item()),
        "rendezvous_adapter_tx_role_weight": float(mode_weight[MODE_TO_INDEX[MODE_TX], 0].item()),
        "rendezvous_adapter_rx_role_weight": float(mode_weight[MODE_TO_INDEX[MODE_RX], 0].item()),
    }


def optimizer_parameters(optimizer: Any) -> list[Any]:
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def update_policy(
    trajectory: dict[str, Any],
    policy: SharedBeamActorCritic,
    critic: Any,
    optimizer: Any,
    torch_module: Any,
    functional: Any,
    args: argparse.Namespace,
    centralized: bool,
) -> dict[str, float]:
    old_log_probs = trajectory["old_log_probs"]
    rewards = trajectory["rewards"]
    policy_losses = []
    mode_policy_losses = []
    beam_policy_losses = []
    gate_policy_losses = []
    value_losses = []
    entropy_values = []
    approx_kls = []
    clip_fracs = []
    bc_losses = []
    beam_rank_aux_losses = []
    rendezvous_beam_aux_losses = []
    rendezvous_role_aux_losses = []
    beam_active_fracs = []
    actor_grad_norms = []
    critic_grad_norms = []
    explained_variances = []
    normalized_candidate_entropies = []
    sample_log_ratio_means = []
    role_balance_losses = []
    mean_policy_tx_probabilities = []
    separate_action_loss = bool(getattr(args, "separate_action_loss", False))
    beam_rank_aux_coef = float(getattr(args, "beam_rank_aux_coef", 0.0))
    rendezvous_beam_aux_coef = float(getattr(args, "rendezvous_beam_aux_coef", 0.0))
    rendezvous_role_aux_coef = float(getattr(args, "rendezvous_role_aux_coef", 0.0))

    if centralized:
        if critic is None:
            raise RuntimeError("Centralized critic is required for MAPPO-style training.")
        per_agent_critic = bool(getattr(critic, "output_per_agent", False))
        return_scope = str(getattr(args, "return_scope", "team"))
        if return_scope == "per_agent":
            if not per_agent_critic:
                raise RuntimeError("Per-agent returns require a per-agent centralized critic.")
            learning_rewards = rewards
            critic_inputs = trajectory["central_graph"]
        else:
            global_rewards = rewards.mean(dim=1)
            if per_agent_critic:
                learning_rewards = global_rewards.unsqueeze(-1).expand(-1, rewards.shape[1])
                critic_inputs = trajectory["central_graph"]
            else:
                learning_rewards = global_rewards
                critic_inputs = trajectory["central_features"]
        with torch_module.no_grad():
            rollout_values = critic(critic_inputs)
            returns, fixed_advantages = fixed_ppo_targets(
                learning_rewards,
                rollout_values,
                args,
                torch_module,
            )
        for _ in range(int(args.ppo_epochs)):
            action_eval = evaluate_action_components(policy, trajectory["observations"], trajectory["actions"])
            log_probs = action_eval["log_probs"]
            entropies = action_eval["entropies"]
            values = critic(critic_inputs)
            role_balance_loss, mean_policy_tx_probability = role_balance_regularizer(
                action_eval["mode_tx_probabilities"],
                torch_module,
            )
            if separate_action_loss:
                policy_loss, component = separated_action_policy_loss(
                    action_eval,
                    trajectory,
                    fixed_advantages,
                    float(args.clip_epsilon),
                    torch_module,
                    args,
                )
                mode_policy_losses.append(component["mode_policy_loss"])
                beam_policy_losses.append(component["beam_policy_loss"])
                gate_policy_losses.append(component["gate_policy_loss"])
                clip_fracs.append(component["clip_fraction"])
            else:
                policy_loss, clip_fraction = ppo_component_loss(
                    log_probs,
                    old_log_probs,
                    fixed_advantages,
                    float(args.clip_epsilon),
                    torch_module,
                )
                mode_policy_losses.append(0.0)
                beam_policy_losses.append(0.0)
                gate_policy_losses.append(0.0)
                clip_fracs.append(clip_fraction)
            value_loss = functional.mse_loss(values, returns)
            entropy = entropies.mean()
            bc_loss = behavior_cloning_loss(policy, trajectory["observations"], trajectory["expert_actions"], torch_module)
            beam_rank_aux_loss = optional_beam_ranking_aux_loss(
                policy,
                trajectory["observations"],
                torch_module,
                log_probs,
                beam_rank_aux_coef,
                temperature=float(getattr(args, "beam_rank_temperature", 4.0)),
            )
            rendezvous_beam_aux_loss, rendezvous_role_aux_loss = optional_rendezvous_auxiliary_losses(
                policy,
                trajectory["observations"],
                torch_module,
                log_probs,
                rendezvous_beam_aux_coef,
                rendezvous_role_aux_coef,
            )
            loss = (
                policy_loss
                + float(args.value_coef) * value_loss
                - float(args.entropy_coef) * entropy
                + float(getattr(args, "expert_bc_weight", 0.0)) * bc_loss
                + beam_rank_aux_coef * beam_rank_aux_loss
                + rendezvous_beam_aux_coef * rendezvous_beam_aux_loss
                + rendezvous_role_aux_coef * rendezvous_role_aux_loss
                + float(getattr(args, "role_balance_coef", 0.0)) * role_balance_loss
            )
            optimizer.zero_grad()
            loss.backward()
            actor_grad_norm = torch_module.nn.utils.clip_grad_norm_(
                list(policy.parameters()), float(args.max_grad_norm)
            )
            critic_grad_norm = torch_module.nn.utils.clip_grad_norm_(
                list(critic.parameters()), float(args.max_grad_norm)
            )
            optimizer.step()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropy_values.append(float(entropy.item()))
            bc_losses.append(float(bc_loss.item()))
            beam_rank_aux_losses.append(float(beam_rank_aux_loss.item()))
            rendezvous_beam_aux_losses.append(float(rendezvous_beam_aux_loss.item()))
            rendezvous_role_aux_losses.append(float(rendezvous_role_aux_loss.item()))
            beam_active_fracs.append(float(trajectory["active_beam_mask"].float().mean().detach().item()))
            log_ratio = log_probs - old_log_probs
            approx_kls.append(float(((torch_module.exp(log_ratio) - 1.0) - log_ratio).mean().detach().item()))
            sample_log_ratio_means.append(float(log_ratio.mean().detach().item()))
            actor_grad_norms.append(float(actor_grad_norm.detach().item()))
            critic_grad_norms.append(float(critic_grad_norm.detach().item()))
            explained_variances.append(explained_variance(returns, values, torch_module))
            normalized_candidate_entropies.append(
                normalized_candidate_entropy(
                    action_eval["beam_entropies"], trajectory["observations"], torch_module
                )
            )
            role_balance_losses.append(float(role_balance_loss.detach().item()))
            mean_policy_tx_probabilities.append(mean_policy_tx_probability)
        return loss_summary(
            policy_losses,
            value_losses,
            entropy_values,
            approx_kls,
            clip_fracs,
            bc_losses,
            mode_policy_losses,
            beam_policy_losses,
            gate_policy_losses,
            beam_rank_aux_losses,
            rendezvous_beam_aux_losses,
            rendezvous_role_aux_losses,
            beam_active_fracs,
            actor_grad_norms,
            critic_grad_norms,
            explained_variances,
            normalized_candidate_entropies,
            sample_log_ratio_means,
            role_balance_losses,
            mean_policy_tx_probabilities,
        )

    with torch_module.no_grad():
        rollout_action_eval = evaluate_action_components(
            policy,
            trajectory["observations"],
            trajectory["actions"],
        )
        returns, fixed_advantages = fixed_ppo_targets(
            rewards,
            rollout_action_eval["values"],
            args,
            torch_module,
        )
    for _ in range(int(args.ppo_epochs)):
        action_eval = evaluate_action_components(policy, trajectory["observations"], trajectory["actions"])
        log_probs = action_eval["log_probs"]
        local_values = action_eval["values"]
        entropies = action_eval["entropies"]
        role_balance_loss, mean_policy_tx_probability = role_balance_regularizer(
            action_eval["mode_tx_probabilities"],
            torch_module,
        )
        if separate_action_loss:
            policy_loss, component = separated_action_policy_loss(
                action_eval,
                trajectory,
                fixed_advantages,
                float(args.clip_epsilon),
                torch_module,
                args,
            )
            mode_policy_losses.append(component["mode_policy_loss"])
            beam_policy_losses.append(component["beam_policy_loss"])
            gate_policy_losses.append(component["gate_policy_loss"])
            clip_fracs.append(component["clip_fraction"])
        else:
            policy_loss, clip_fraction = ppo_component_loss(
                log_probs,
                old_log_probs,
                fixed_advantages,
                float(args.clip_epsilon),
                torch_module,
            )
            mode_policy_losses.append(0.0)
            beam_policy_losses.append(0.0)
            gate_policy_losses.append(0.0)
            clip_fracs.append(clip_fraction)
        value_loss = functional.mse_loss(local_values, returns)
        entropy = entropies.mean()
        bc_loss = behavior_cloning_loss(policy, trajectory["observations"], trajectory["expert_actions"], torch_module)
        beam_rank_aux_loss = optional_beam_ranking_aux_loss(
            policy,
            trajectory["observations"],
            torch_module,
            log_probs,
            beam_rank_aux_coef,
            temperature=float(getattr(args, "beam_rank_temperature", 4.0)),
        )
        rendezvous_beam_aux_loss, rendezvous_role_aux_loss = optional_rendezvous_auxiliary_losses(
            policy,
            trajectory["observations"],
            torch_module,
            log_probs,
            rendezvous_beam_aux_coef,
            rendezvous_role_aux_coef,
        )
        loss = (
            policy_loss
            + float(args.value_coef) * value_loss
            - float(args.entropy_coef) * entropy
            + float(getattr(args, "expert_bc_weight", 0.0)) * bc_loss
            + beam_rank_aux_coef * beam_rank_aux_loss
            + rendezvous_beam_aux_coef * rendezvous_beam_aux_loss
            + rendezvous_role_aux_coef * rendezvous_role_aux_loss
            + float(getattr(args, "role_balance_coef", 0.0)) * role_balance_loss
        )
        optimizer.zero_grad()
        loss.backward()
        actor_grad_norm = torch_module.nn.utils.clip_grad_norm_(
            list(policy.parameters()), float(args.max_grad_norm)
        )
        optimizer.step()
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        entropy_values.append(float(entropy.item()))
        bc_losses.append(float(bc_loss.item()))
        beam_rank_aux_losses.append(float(beam_rank_aux_loss.item()))
        rendezvous_beam_aux_losses.append(float(rendezvous_beam_aux_loss.item()))
        rendezvous_role_aux_losses.append(float(rendezvous_role_aux_loss.item()))
        beam_active_fracs.append(float(trajectory["active_beam_mask"].float().mean().detach().item()))
        log_ratio = log_probs - old_log_probs
        approx_kls.append(float(((torch_module.exp(log_ratio) - 1.0) - log_ratio).mean().detach().item()))
        sample_log_ratio_means.append(float(log_ratio.mean().detach().item()))
        actor_grad_norms.append(float(actor_grad_norm.detach().item()))
        critic_grad_norms.append(0.0)
        explained_variances.append(explained_variance(returns, local_values, torch_module))
        normalized_candidate_entropies.append(
            normalized_candidate_entropy(
                action_eval["beam_entropies"], trajectory["observations"], torch_module
            )
        )
        role_balance_losses.append(float(role_balance_loss.detach().item()))
        mean_policy_tx_probabilities.append(mean_policy_tx_probability)
    return loss_summary(
        policy_losses,
        value_losses,
        entropy_values,
        approx_kls,
        clip_fracs,
        bc_losses,
        mode_policy_losses,
        beam_policy_losses,
        gate_policy_losses,
        beam_rank_aux_losses,
        rendezvous_beam_aux_losses,
        rendezvous_role_aux_losses,
        beam_active_fracs,
        actor_grad_norms,
        critic_grad_norms,
        explained_variances,
        normalized_candidate_entropies,
        sample_log_ratio_means,
        role_balance_losses,
        mean_policy_tx_probabilities,
    )


def expert_actions_for_env(env: MarlNeighborDiscoveryEnv, expert_protocol: str) -> list[Action]:
    """Return decentralized rule-expert actions from local simulator memory only."""

    old_protocol = env._sim.protocol
    env._sim.protocol = str(expert_protocol)
    try:
        actions: list[Action] = []
        for node in range(env.n_agents):
            mode = env._sim.select_mode(node, env._slot)
            beam = env._sim.select_beam(node, env._slot, mode, set(), set())
            gate = expert_access_gate_for_env(env, node, mode, int(beam), str(expert_protocol))
            actions.append(Action(mode, int(beam), gate))
        return actions
    finally:
        env._sim.protocol = old_protocol


def expert_access_gate_for_env(
    env: MarlNeighborDiscoveryEnv,
    node: int,
    mode: str,
    beam: int,
    expert_protocol: str,
) -> str:
    """Expose the rule expert's local collision-access decision as a gate label."""

    if expert_protocol not in {"collision_aware_isac", "budgeted_collision_aware_isac"}:
        return ACCESS_NORMAL
    sim = env._sim
    beam_collision = float(sim.collision_fail_count[node, beam])
    beam_success = float(sim.success_count[node, beam])
    beam_fail = float(sim.fail_count[node, beam])
    candidate_pool = sim.isac_candidate_pool(node, env._slot)
    degree = sum(1 for edge in sim.discovered_edges if node in edge)
    degree_need = max(0.0, float(sim.cfg.target_degree - degree)) / max(1.0, float(sim.cfg.target_degree))

    collision_pressure = 0.0
    failure_pressure = 0.0
    candidate_evidence = 0.0
    if len(candidate_pool) > 0:
        success = sim.success_count[node, candidate_pool]
        fail = sim.fail_count[node, candidate_pool]
        collision = sim.collision_fail_count[node, candidate_pool]
        belief = sim.belief[node, candidate_pool]
        collision_pressure = float(np.mean(collision / np.maximum(1.0, success + collision)))
        failure_pressure = float(np.mean(fail / np.maximum(1.0, success + fail)))
        candidate_evidence = float(np.clip(np.mean(belief + 0.15 * np.log1p(success)), 0.0, 1.0))

    if mode == MODE_TX and (beam_collision > 0.0 or collision_pressure >= 0.28 or beam_fail > beam_success + 1.0):
        return ACCESS_BACKOFF
    if (
        mode == MODE_RX
        and degree_need >= 0.35
        and candidate_evidence >= 0.20
        and beam_collision <= beam_success
        and collision_pressure <= 0.25
        and failure_pressure <= 0.65
    ):
        return ACCESS_AGGRESSIVE
    return ACCESS_NORMAL


def evaluate_actions(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    actions_by_step: Sequence[Sequence[Action]],
) -> tuple[Any, Any, Any]:
    action_eval = evaluate_action_components(policy, observations_by_step, actions_by_step)
    return action_eval["log_probs"], action_eval["values"], action_eval["entropies"]


def evaluate_action_components(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    actions_by_step: Sequence[Sequence[Action]],
) -> dict[str, Any]:
    if hasattr(policy, "evaluate_action_sequence"):
        return policy.evaluate_action_sequence(observations_by_step, actions_by_step)
    torch = policy.torch
    from torch.distributions import Categorical

    log_prob_rows = []
    mode_log_prob_rows = []
    beam_log_prob_rows = []
    gate_log_prob_rows = []
    value_rows = []
    entropy_rows = []
    mode_entropy_rows = []
    beam_entropy_rows = []
    gate_entropy_rows = []
    active_mask_rows = []
    mode_tx_probability_rows = []
    for observations, actions in zip(observations_by_step, actions_by_step, strict=True):
        step_log_probs = []
        step_mode_log_probs = []
        step_beam_log_probs = []
        step_gate_log_probs = []
        step_values = []
        step_entropies = []
        step_mode_entropies = []
        step_beam_entropies = []
        step_gate_entropies = []
        step_active_masks = []
        step_mode_tx_probabilities = []
        for observation, action in zip(observations, actions, strict=True):
            if getattr(policy, "supports_access_gate_action", False) and hasattr(policy, "action_logits_value"):
                mode_logits, beam_logits, gate_logits, value = policy.action_logits_value(observation, hard_mask=True)
            else:
                mode_logits, beam_logits, value = policy.logits_value(observation, hard_mask=True)
                gate_logits = None
            beam_dist = Categorical(logits=beam_logits)
            mode_idx = MODE_TO_INDEX[action.mode]
            beam_log_prob = torch.zeros((), dtype=beam_logits.dtype, device=policy.device)
            if action.mode != "idle":
                beam_tensor = torch.as_tensor(int(action.beam), dtype=torch.long, device=policy.device)
                beam_log_prob = beam_dist.log_prob(beam_tensor)
            beam_entropy = beam_dist.entropy()
            if is_beam_only_action_contract(
                getattr(policy, "action_contract", "joint_role_beam")
            ):
                mode_log_prob = torch.zeros((), dtype=beam_logits.dtype, device=policy.device)
                mode_entropy = torch.zeros((), dtype=beam_logits.dtype, device=policy.device)
                log_prob = beam_log_prob
            else:
                mode_dist = Categorical(logits=mode_logits)
                mode_tensor = torch.as_tensor(mode_idx, dtype=torch.long, device=policy.device)
                mode_log_prob = mode_dist.log_prob(mode_tensor)
                mode_entropy = mode_dist.entropy()
                log_prob = mode_log_prob + beam_log_prob
            mode_tx_probability = torch.softmax(mode_logits, dim=-1)[MODE_TO_INDEX[MODE_TX]]
            active_beam = action.mode != "idle"
            gate_log_prob = torch.zeros((), dtype=mode_log_prob.dtype, device=policy.device)
            gate_entropy = torch.zeros((), dtype=mode_entropy.dtype, device=policy.device)
            entropy = mode_entropy + beam_entropy
            if gate_logits is not None:
                gate_dist = Categorical(logits=gate_logits)
                gate_idx = ACCESS_GATE_TO_INDEX.get(getattr(action, "access_gate", "normal"), ACCESS_GATE_TO_INDEX["normal"])
                gate_tensor = torch.as_tensor(gate_idx, dtype=torch.long, device=policy.device)
                gate_log_prob = gate_dist.log_prob(gate_tensor)
                gate_entropy = gate_dist.entropy()
                log_prob = log_prob + gate_log_prob
                entropy = entropy + gate_entropy
            step_log_probs.append(log_prob)
            step_mode_log_probs.append(mode_log_prob)
            step_beam_log_probs.append(beam_log_prob)
            step_gate_log_probs.append(gate_log_prob)
            step_values.append(value.squeeze(-1))
            step_entropies.append(entropy)
            step_mode_entropies.append(mode_entropy)
            step_beam_entropies.append(beam_entropy)
            step_gate_entropies.append(gate_entropy)
            step_active_masks.append(torch.as_tensor(active_beam, dtype=torch.bool, device=policy.device))
            step_mode_tx_probabilities.append(mode_tx_probability)
        log_prob_rows.append(torch.stack(step_log_probs))
        mode_log_prob_rows.append(torch.stack(step_mode_log_probs))
        beam_log_prob_rows.append(torch.stack(step_beam_log_probs))
        gate_log_prob_rows.append(torch.stack(step_gate_log_probs))
        value_rows.append(torch.stack(step_values))
        entropy_rows.append(torch.stack(step_entropies))
        mode_entropy_rows.append(torch.stack(step_mode_entropies))
        beam_entropy_rows.append(torch.stack(step_beam_entropies))
        gate_entropy_rows.append(torch.stack(step_gate_entropies))
        active_mask_rows.append(torch.stack(step_active_masks))
        mode_tx_probability_rows.append(torch.stack(step_mode_tx_probabilities))
    return {
        "log_probs": torch.stack(log_prob_rows),
        "mode_log_probs": torch.stack(mode_log_prob_rows),
        "beam_log_probs": torch.stack(beam_log_prob_rows),
        "gate_log_probs": torch.stack(gate_log_prob_rows),
        "values": torch.stack(value_rows),
        "entropies": torch.stack(entropy_rows),
        "mode_entropies": torch.stack(mode_entropy_rows),
        "beam_entropies": torch.stack(beam_entropy_rows),
        "gate_entropies": torch.stack(gate_entropy_rows),
        "active_beam_mask": torch.stack(active_mask_rows),
        "mode_tx_probabilities": torch.stack(mode_tx_probability_rows),
    }


def ppo_component_loss(
    log_probs: Any,
    old_log_probs: Any,
    advantages: Any,
    clip_epsilon: float,
    torch_module: Any,
    mask: Any | None = None,
) -> tuple[Any, float]:
    ratio = torch_module.exp(log_probs - old_log_probs)
    component_advantages = advantages
    while component_advantages.dim() < log_probs.dim():
        component_advantages = component_advantages.unsqueeze(-1)
    clipped_ratio = torch_module.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss_values = -torch_module.minimum(
        ratio * component_advantages,
        clipped_ratio * component_advantages,
    )
    clipped = (torch_module.abs(ratio - 1.0) > clip_epsilon).float()
    if mask is not None:
        mask = mask.bool()
        if not bool(mask.any().item()):
            return log_probs.sum() * 0.0, 0.0
        return loss_values[mask].mean(), float(clipped[mask].mean().detach().item())
    return loss_values.mean(), float(clipped.mean().detach().item())


def role_balance_regularizer(
    mode_tx_probabilities: Any,
    torch_module: Any,
) -> tuple[Any, float]:
    """Permit local heterogeneity while centering the per-slot population mean at 0.5."""

    if mode_tx_probabilities.ndim != 2:
        raise ValueError("mode_tx_probabilities must have shape [time, agents].")
    per_slot_mean = mode_tx_probabilities.mean(dim=1)
    loss = torch_module.mean((per_slot_mean - 0.5) ** 2)
    return loss, float(per_slot_mean.mean().detach().item())


def separated_action_policy_loss(
    action_eval: dict[str, Any],
    trajectory: dict[str, Any],
    advantages: Any,
    clip_epsilon: float,
    torch_module: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    mode_loss, mode_clip = ppo_component_loss(
        action_eval["mode_log_probs"],
        trajectory["old_mode_log_probs"],
        advantages,
        clip_epsilon,
        torch_module,
    )
    active_mask = trajectory["active_beam_mask"].bool()
    beam_advantages = advantages
    feedback_coefficient = float(getattr(args, "beam_isac_feedback_coef", 0.0))
    if feedback_coefficient > 0.0:
        feedback = trajectory["beam_isac_feedback"]
        while beam_advantages.dim() < feedback.dim():
            beam_advantages = beam_advantages.unsqueeze(-1)
        beam_advantages = beam_advantages + feedback_coefficient * feedback
    beam_loss, beam_clip = ppo_component_loss(
        action_eval["beam_log_probs"],
        trajectory["old_beam_log_probs"],
        beam_advantages,
        clip_epsilon,
        torch_module,
        mask=active_mask,
    )
    gate_loss, gate_clip = ppo_component_loss(
        action_eval["gate_log_probs"],
        trajectory["old_gate_log_probs"],
        advantages,
        clip_epsilon,
        torch_module,
    )
    total = mode_loss + float(getattr(args, "beam_loss_coef", 1.0)) * beam_loss
    total = total + float(getattr(args, "gate_loss_coef", 0.25)) * gate_loss
    return total, {
        "mode_policy_loss": float(mode_loss.detach().item()),
        "beam_policy_loss": float(beam_loss.detach().item()),
        "gate_policy_loss": float(gate_loss.detach().item()),
        "clip_fraction": float(np.mean([mode_clip, beam_clip, gate_clip])),
    }


def optional_beam_ranking_aux_loss(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    torch_module: Any,
    zero_like: Any,
    coefficient: float,
    temperature: float,
) -> Any:
    if float(coefficient) <= 0.0:
        return zero_like.sum() * 0.0
    return beam_ranking_aux_loss(policy, observations_by_step, torch_module, temperature)


def beam_ranking_aux_loss(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    torch_module: Any,
    temperature: float,
) -> Any:
    losses = []
    for observations in observations_by_step:
        if not observations or not all("candidate_score" in observation for observation in observations):
            continue
        if getattr(policy, "supports_access_gate_action", False) and hasattr(policy, "batched_action_logits_value"):
            _mode_logits, beam_logits, _gate_logits, _value = policy.batched_action_logits_value(observations, hard_mask=True)
        else:
            _mode_logits, beam_logits, _value = policy.batched_logits_value(observations, hard_mask=True)
        candidate_score = torch_module.stack(
            [
                torch_module.as_tensor(observation["candidate_score"], dtype=torch_module.float32, device=policy.device)
                for observation in observations
            ],
            dim=0,
        )
        candidate_mask = torch_module.stack(
            [
                torch_module.as_tensor(
                    observation.get("candidate_mask", np.ones_like(observation["candidate_score"])),
                    dtype=torch_module.float32,
                    device=policy.device,
                )
                for observation in observations
            ],
            dim=0,
        ) > 0.5
        valid_count = candidate_mask.sum(dim=-1)
        masked_scores = candidate_score.masked_fill(~candidate_mask, -1.0e9)
        score_max = masked_scores.max(dim=-1).values
        score_min = candidate_score.masked_fill(~candidate_mask, 1.0e9).min(dim=-1).values
        informative = (valid_count >= 2) & ((score_max - score_min) > 1.0e-6)
        if not bool(informative.any().item()):
            continue
        masked_beam_logits = beam_logits.masked_fill(~candidate_mask, -1.0e9)
        log_policy = torch_module.nn.functional.log_softmax(masked_beam_logits, dim=-1)
        target_logits = float(temperature) * (candidate_score - score_max.unsqueeze(-1))
        target_logits = target_logits.masked_fill(~candidate_mask, -1.0e9)
        target = torch_module.nn.functional.softmax(target_logits, dim=-1)
        per_agent = -(target * log_policy).sum(dim=-1)
        losses.append(per_agent[informative])
    if not losses:
        parameter = next(iter(policy.parameters()))
        return parameter.sum() * 0.0
    return torch_module.cat(losses).mean()


def optional_rendezvous_auxiliary_losses(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    torch_module: Any,
    zero_like: Any,
    beam_coefficient: float,
    role_coefficient: float,
) -> tuple[Any, Any]:
    zero = zero_like.sum() * 0.0
    if float(beam_coefficient) <= 0.0 and float(role_coefficient) <= 0.0:
        return zero, zero
    return rendezvous_auxiliary_losses(policy, observations_by_step, torch_module)


def rendezvous_auxiliary_losses(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    torch_module: Any,
) -> tuple[Any, Any]:
    """Fit action heads to local sensing geometry without overriding actions."""

    informative_observations = []
    for observations in observations_by_step:
        for observation in observations:
            rendezvous_score = np.asarray(
                observation.get("rendezvous_beam_score", np.zeros(0, dtype=np.float32)),
                dtype=np.float32,
            )
            role_hint = float(np.asarray(observation.get("rendezvous_role_hint", [0.0])).reshape(-1)[0])
            if rendezvous_score.size and np.any(rendezvous_score > 0.0) and role_hint != 0.0:
                informative_observations.append(observation)

    beam_losses = []
    role_losses = []
    chunk_size = 64
    for start in range(0, len(informative_observations), chunk_size):
        observations = informative_observations[start : start + chunk_size]
        if getattr(policy, "supports_access_gate_action", False) and hasattr(policy, "batched_action_logits_value"):
            mode_logits, beam_logits, _gate_logits, _value = policy.batched_action_logits_value(
                observations,
                hard_mask=True,
            )
        else:
            mode_logits, beam_logits, _value = policy.batched_logits_value(observations, hard_mask=True)
        rendezvous_score = torch_module.stack(
            [
                torch_module.as_tensor(
                    observation.get("rendezvous_beam_score", np.zeros(policy.n_beams, dtype=np.float32)),
                    dtype=torch_module.float32,
                    device=policy.device,
                )
                for observation in observations
            ],
            dim=0,
        )
        role_hint = torch_module.as_tensor(
            [float(np.asarray(observation.get("rendezvous_role_hint", [0.0])).reshape(-1)[0]) for observation in observations],
            dtype=torch_module.float32,
            device=policy.device,
        )
        beam_targets = rendezvous_score.argmax(dim=-1)
        beam_losses.append(
            torch_module.nn.functional.cross_entropy(
                beam_logits,
                beam_targets,
                reduction="none",
            )
        )
        role_targets = torch_module.where(
            role_hint > 0.0,
            torch_module.full_like(role_hint, MODE_TO_INDEX[MODE_TX], dtype=torch_module.long),
            torch_module.full_like(role_hint, MODE_TO_INDEX[MODE_RX], dtype=torch_module.long),
        )
        role_losses.append(
            torch_module.nn.functional.cross_entropy(
                mode_logits,
                role_targets,
                reduction="none",
            )
        )
    parameter_zero = next(iter(policy.parameters())).sum() * 0.0
    beam_loss = torch_module.cat(beam_losses).mean() if beam_losses else parameter_zero
    role_loss = torch_module.cat(role_losses).mean() if role_losses else parameter_zero
    return beam_loss, role_loss


def behavior_cloning_loss(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    expert_actions_by_step: Sequence[Sequence[Action]],
    torch_module: Any,
) -> Any:
    weight = float(getattr(policy, "_expert_bc_weight_cache", 1.0))
    if weight <= 0.0:
        return torch_module.zeros((), dtype=torch_module.float32, device=policy.device)
    losses = []
    from torch.distributions import Categorical

    for observations, actions in zip(observations_by_step, expert_actions_by_step, strict=True):
        for observation, action in zip(observations, actions, strict=True):
            if getattr(policy, "supports_access_gate_action", False) and hasattr(policy, "action_logits_value"):
                mode_logits, beam_logits, gate_logits, _value = policy.action_logits_value(observation, hard_mask=False)
            else:
                mode_logits, beam_logits, _value = policy.logits_value(observation, hard_mask=False)
                gate_logits = None
            mode_dist = Categorical(logits=mode_logits)
            mode_tensor = torch_module.as_tensor(MODE_TO_INDEX[action.mode], dtype=torch_module.long, device=policy.device)
            loss = -mode_dist.log_prob(mode_tensor)
            if action.mode != "idle":
                beam_dist = Categorical(logits=beam_logits)
                beam_tensor = torch_module.as_tensor(int(action.beam), dtype=torch_module.long, device=policy.device)
                loss = loss - beam_dist.log_prob(beam_tensor)
            if gate_logits is not None:
                gate_dist = Categorical(logits=gate_logits)
                gate_idx = ACCESS_GATE_TO_INDEX.get(getattr(action, "access_gate", ACCESS_NORMAL), ACCESS_GATE_TO_INDEX[ACCESS_NORMAL])
                gate_tensor = torch_module.as_tensor(gate_idx, dtype=torch_module.long, device=policy.device)
                loss = loss - 0.10 * gate_dist.log_prob(gate_tensor)
            losses.append(loss)
    if not losses:
        return torch_module.zeros((), dtype=torch_module.float32, device=policy.device)
    return torch_module.stack(losses).mean()


def discounted_returns_2d(rewards: Any, gamma: float, torch_module: Any) -> Any:
    running = torch_module.zeros_like(rewards[-1])
    returns = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch_module.stack(returns)


def discounted_returns_1d(rewards: Any, gamma: float, torch_module: Any) -> Any:
    running = torch_module.zeros((), dtype=rewards.dtype, device=rewards.device)
    returns = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch_module.stack(returns)


def normalize_advantages(advantages: Any) -> Any:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def snapshot_normalized_advantages(returns: Any, rollout_values: Any) -> Any:
    """Freeze the behavior-rollout advantage target for all PPO reuse epochs."""

    if returns.shape != rollout_values.shape:
        raise ValueError("returns and rollout_values must have identical shapes.")
    return normalize_advantages((returns - rollout_values).detach()).clone()


def fixed_ppo_targets(
    rewards: Any,
    rollout_values: Any,
    args: argparse.Namespace,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Build fixed finite-horizon critic targets and normalized actor advantages."""

    if rewards.shape != rollout_values.shape:
        raise ValueError("rewards and rollout_values must have identical shapes.")
    estimator = str(getattr(args, "advantage_estimator", "mc"))
    if estimator == "gae":
        returns, raw_advantages = generalized_advantage_estimate(
            rewards,
            rollout_values,
            gamma=float(args.gamma),
            gae_lambda=float(getattr(args, "gae_lambda", 0.95)),
            torch_module=torch_module,
        )
        return returns.detach().clone(), normalize_advantages(raw_advantages.detach()).clone()
    if estimator != "mc":
        raise ValueError("advantage_estimator must be 'mc' or 'gae'.")
    returns = (
        discounted_returns_1d(rewards, float(args.gamma), torch_module)
        if rewards.ndim == 1
        else discounted_returns_2d(rewards, float(args.gamma), torch_module)
    )
    return returns.detach().clone(), snapshot_normalized_advantages(returns, rollout_values)


def generalized_advantage_estimate(
    rewards: Any,
    rollout_values: Any,
    *,
    gamma: float,
    gae_lambda: float,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Finite-horizon GAE with a zero value beyond the declared episode boundary."""

    if rewards.shape != rollout_values.shape:
        raise ValueError("rewards and rollout_values must have identical shapes.")
    next_value = torch_module.zeros_like(rollout_values[-1])
    running_advantage = torch_module.zeros_like(rollout_values[-1])
    advantages = []
    for reward, value in zip(reversed(rewards), reversed(rollout_values), strict=True):
        delta = reward + float(gamma) * next_value - value
        running_advantage = delta + float(gamma) * float(gae_lambda) * running_advantage
        advantages.append(running_advantage)
        next_value = value
    advantages.reverse()
    advantage_tensor = torch_module.stack(advantages)
    return advantage_tensor + rollout_values, advantage_tensor


def explained_variance(targets: Any, predictions: Any, torch_module: Any) -> float:
    target_variance = torch_module.var(targets, unbiased=False)
    if float(target_variance.detach().item()) <= 1.0e-12:
        return 0.0
    residual_variance = torch_module.var(targets - predictions.detach(), unbiased=False)
    return float((1.0 - residual_variance / target_variance).detach().item())


def normalized_candidate_entropy(
    beam_entropies: Any,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    torch_module: Any,
) -> float:
    counts = []
    for observations in observations_by_step:
        row = []
        for observation in observations:
            n_beams = len(observation["beam_belief"])
            mask = np.asarray(observation.get("candidate_mask", np.ones(n_beams)), dtype=float) > 0.5
            count = int(mask.sum())
            row.append(n_beams if count == 0 else count)
        counts.append(row)
    count_tensor = torch_module.as_tensor(
        counts, dtype=beam_entropies.dtype, device=beam_entropies.device
    )
    valid = count_tensor > 1.0
    if not bool(valid.any().item()):
        return 0.0
    normalized = beam_entropies[valid] / torch_module.log(count_tensor[valid])
    return float(normalized.mean().detach().item())


def central_feature_dim() -> int:
    return 23


def central_graph_feature_dims() -> tuple[int, int, int]:
    return 12, 6, 7


def central_graph_features(state: dict[str, Any], cfg: SimulationConfig) -> dict[str, np.ndarray]:
    """Build critic-only graph tensors; none of these fields enter the actor."""

    positions = np.asarray(state["positions"], dtype=np.float32)
    velocities = np.asarray(state["velocities"], dtype=np.float32)
    discovered = np.asarray(state["discovered_adjacency"], dtype=np.float32)
    true_adj = np.asarray(state["true_adjacency"], dtype=np.float32)
    belief = np.asarray(state["belief"], dtype=np.float32)
    n_agents = int(positions.shape[0])
    degree_scale = float(max(1, n_agents - 1))
    area = np.maximum(np.asarray(cfg.area_size_m, dtype=np.float32), 1.0e-6)
    position_norm = positions / area
    velocity_norm = velocities / max(1.0, _speed_scale(cfg))
    slot_fraction = float(state["slot"]) / max(1.0, float(cfg.slots_per_episode))
    node_features = np.concatenate(
        [
            position_norm,
            velocity_norm,
            belief.mean(axis=1, keepdims=True),
            belief.std(axis=1, keepdims=True),
            belief.max(axis=1, keepdims=True),
            discovered.sum(axis=1, keepdims=True) / degree_scale,
            true_adj.sum(axis=1, keepdims=True) / degree_scale,
            np.full((n_agents, 1), slot_fraction, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    relative = positions[None, :, :] - positions[:, None, :]
    relative_norm = relative / area.reshape(1, 1, -1)
    distance_norm = np.linalg.norm(relative, axis=-1, keepdims=True) / max(
        1.0, float(np.linalg.norm(area))
    )
    edge_features = np.concatenate(
        [
            relative_norm,
            distance_norm.astype(np.float32),
            discovered[..., None],
            true_adj[..., None],
        ],
        axis=-1,
    ).astype(np.float32)
    edge_mask = (true_adj > 0.5) | (discovered > 0.5)
    np.fill_diagonal(edge_mask, False)

    possible_edges = max(1.0, n_agents * (n_agents - 1) / 2.0)
    upper = np.triu_indices(n_agents, k=1)
    discovered_edges = float(discovered[upper].sum())
    true_edges = float(true_adj[upper].sum())
    global_features = np.asarray(
        [
            slot_fraction,
            n_agents / 100.0,
            cfg.n_beams / 2000.0,
            discovered_edges / max(1.0, true_edges),
            true_edges / possible_edges,
            float(belief.mean()),
            float(belief.std()),
        ],
        dtype=np.float32,
    )
    expected = central_graph_feature_dims()
    if (
        node_features.shape[-1] != expected[0]
        or edge_features.shape[-1] != expected[1]
        or global_features.shape[-1] != expected[2]
    ):
        raise RuntimeError("Central graph feature dimensions do not match the declared contract.")
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "global_features": global_features,
        "edge_mask": edge_mask,
    }


def central_state_features(state: dict[str, Any], cfg: SimulationConfig) -> np.ndarray:
    positions = np.asarray(state["positions"], dtype=np.float32)
    velocities = np.asarray(state["velocities"], dtype=np.float32)
    discovered = np.asarray(state["discovered_adjacency"], dtype=np.float32)
    true_adj = np.asarray(state["true_adjacency"], dtype=np.float32)
    belief = np.asarray(state["belief"], dtype=np.float32)
    n = max(1, int(cfg.n_nodes))
    possible_edges = max(1.0, n * (n - 1) / 2.0)
    area = np.asarray(cfg.area_size_m, dtype=np.float32)
    pos_norm = positions / np.maximum(area, 1e-6)
    speed = np.linalg.norm(velocities, axis=1)
    discovered_degree = discovered.sum(axis=1) / max(1, n - 1)
    true_degree = true_adj.sum(axis=1) / max(1, n - 1)
    upper = np.triu_indices(n, k=1)
    discovered_edges = float(discovered[upper].sum())
    true_edges = float(true_adj[upper].sum())
    return np.asarray(
        [
            float(state["slot"]) / max(1.0, float(cfg.slots_per_episode)),
            n / 100.0,
            cfg.n_beams / 2000.0,
            discovered_edges / possible_edges,
            true_edges / possible_edges,
            discovered_edges / max(1.0, true_edges),
            float(np.mean(belief)),
            float(np.std(belief)),
            float(np.max(belief)),
            *np.mean(pos_norm, axis=0).tolist(),
            *np.std(pos_norm, axis=0).tolist(),
            float(np.mean(speed) / max(1.0, _speed_scale(cfg))),
            float(np.std(speed) / max(1.0, _speed_scale(cfg))),
            float(np.mean(discovered_degree)),
            float(np.std(discovered_degree)),
            float(np.mean(true_degree)),
            float(np.std(true_degree)),
            float(np.mean(np.triu(discovered @ discovered, k=1)) / max(1, n)),
            float(np.mean(np.triu(true_adj @ true_adj, k=1)) / max(1, n)),
        ],
        dtype=np.float32,
    )


def _speed_scale(cfg: SimulationConfig) -> float:
    max_speed = float(cfg.mobility.get("max_speed_mps", 0.0) or 0.0)
    if max_speed > 0.0:
        return max_speed
    return float(cfg.mobility.get("speed_mean_mps", 15.0)) + 3.0 * float(cfg.mobility.get("speed_std_mps", 3.0))


def copy_observations(observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = []
    for observation in observations:
        row: dict[str, Any] = {}
        for key, value in observation.items():
            row[key] = value.copy() if hasattr(value, "copy") else value
        copied.append(row)
    return copied


def local_candidate_information_potential(
    observations: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Bounded local belief potential; higher means a smaller, sharper candidate set."""

    potentials = np.zeros(len(observations), dtype=np.float32)
    for node, observation in enumerate(observations):
        mask = np.asarray(observation["candidate_mask"], dtype=float) > 0.5
        n_beams = max(1, int(mask.size))
        count = max(1, int(mask.sum()))
        count_term = np.log1p(float(count)) / np.log1p(float(n_beams))
        scores = np.maximum(
            0.0,
            np.asarray(observation.get("candidate_score", np.zeros(n_beams)), dtype=float)[mask],
        )
        if count <= 1 or float(scores.sum()) <= 0.0:
            entropy_term = 0.0 if count <= 1 else 1.0
        else:
            probabilities = scores / scores.sum()
            entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12))))
            entropy_term = entropy / np.log(float(count))
        potentials[node] = -0.5 * float(count_term + entropy_term)
    return potentials


def local_potential_shaping_reward(
    observations: Sequence[dict[str, Any]],
    next_observations: Sequence[dict[str, Any]],
    *,
    gamma: float,
    terminal: bool,
    coefficient: float,
) -> np.ndarray:
    if float(coefficient) <= 0.0:
        return np.zeros(len(observations), dtype=np.float32)
    current = local_candidate_information_potential(observations)
    following = (
        np.zeros_like(current)
        if terminal
        else local_candidate_information_potential(next_observations)
    )
    return (float(coefficient) * (float(gamma) * following - current)).astype(np.float32)


def local_beam_isac_feedback(
    actions: Sequence[Action],
    next_observations: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Signed occupancy feedback available only after a local TX piggyback measurement."""

    if len(actions) != len(next_observations):
        raise ValueError("actions and next_observations must have identical lengths.")
    feedback = np.zeros(len(actions), dtype=np.float32)
    for node, (action, observation) in enumerate(zip(actions, next_observations, strict=True)):
        if action.mode != MODE_TX:
            continue
        beam = int(action.beam)
        target_count = float(np.asarray(observation["beam_target_count"], dtype=float)[beam])
        confidence = float(
            np.clip(
                np.asarray(observation["beam_measurement_confidence"], dtype=float)[beam],
                0.0,
                1.0,
            )
        )
        feedback[node] = confidence if target_count > 0.0 else -confidence
    return feedback


def build_episode_row(
    trajectory: dict[str, Any],
    losses: dict[str, float],
    episode: int,
    global_step: int,
    args: argparse.Namespace,
    cfg: SimulationConfig,
) -> dict[str, Any]:
    rewards = trajectory["rewards"]
    shaping_rewards = trajectory.get("potential_shaping_rewards", rewards * 0.0)
    beam_isac_feedback = trajectory.get("beam_isac_feedback", rewards * 0.0)
    summary = trajectory["summary"]
    row = {
        "episode": episode,
        "training_step": global_step,
        "seed": trajectory["seed"],
        "algorithm": str(args.algorithm),
        "slots": cfg.slots_per_episode,
        "n_nodes": cfg.n_nodes,
        "n_beams": cfg.n_beams,
        "episode_return_sum": float(rewards.sum().item()),
        "episode_return_mean_per_agent": float(rewards.sum(dim=0).mean().item()),
        "step_reward_mean": float(rewards.mean().item()),
        "step_reward_sum_mean": float(rewards.sum(dim=1).mean().item()),
        "potential_shaping_return_mean_per_agent": float(
            shaping_rewards.sum(dim=0).mean().item()
        ),
        "raw_episode_return_mean_per_agent": float(
            (rewards - shaping_rewards).sum(dim=0).mean().item()
        ),
        "beam_isac_feedback_mean": float(beam_isac_feedback.mean().item()),
        "beam_isac_feedback_nonzero_fraction": float(
            (beam_isac_feedback != 0.0).float().mean().item()
        ),
        "rollout_replay_logprob_max_abs_error": float(
            trajectory.get("rollout_replay_logprob_max_abs_error", 0.0)
        ),
    }
    row.update(losses)
    row.update({key: summary[key] for key in summary})
    return row


def evaluate_policy(
    cfg: SimulationConfig,
    policy: SharedBeamActorCritic,
    torch_module: Any,
    args: argparse.Namespace,
    env_protocol: str,
    start_episode: int,
    seed_start: int,
    stochastic_eval: bool,
) -> list[dict[str, Any]]:
    rows = []
    torch_rng_state = torch_module.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    was_training = bool(policy.model.training)
    recurrent_state = (
        policy.clone_recurrent_state() if hasattr(policy, "clone_recurrent_state") else None
    )
    policy.eval()
    try:
        with torch_module.no_grad():
            eval_modes = (False, True) if bool(args.eval_both) else (bool(stochastic_eval),)
            eval_training_step = int(start_episode) * int(getattr(args, "slots", 1) or 1)
            for mode_index, use_stochastic in enumerate(eval_modes):
                for offset in range(int(args.eval_episodes)):
                    seed = (
                        int(args.seed)
                        if str(getattr(args, "evaluation_scenario_mode", "held_out")) == "fixed"
                        else seed_start + 10_000 * mode_index + offset
                    )
                    torch_module.manual_seed(seed)
                    np.random.seed(seed)
                    env = MarlNeighborDiscoveryEnv(
                        cfg,
                        seed=seed,
                        protocol=env_protocol,
                        reward_version=str(getattr(args, "reward_version", "legacy")),
                        candidate_source=str(getattr(args, "candidate_source", "default")),
                    )
                    observations, _ = env.reset(seed=seed)
                    if hasattr(policy, "reset_recurrent_state"):
                        policy.reset_recurrent_state(env.n_agents)
                    role_rng = np.random.default_rng(seed + 777)
                    rewards = []
                    rendezvous_totals = empty_rendezvous_action_diagnostics()
                    truncated = False
                    while not truncated:
                        step = policy.act(
                            observations,
                            deterministic=not use_stochastic,
                            role_rng=(
                                role_rng
                                if str(getattr(args, "action_contract", "joint_role_beam"))
                                == "beam_only_fixed_role"
                                else None
                            ),
                        )
                        action_diagnostics = beam_selection_diagnostics(observations, step.actions)
                        action_diagnostics.update(rendezvous_pair_diagnostics(env, observations, step.actions))
                        accumulate_rendezvous_action_diagnostics(rendezvous_totals, action_diagnostics)
                        observations, reward, _terminated, truncated, _info = env.step(step.actions)
                        if (
                            bool(getattr(args, "terminate_on_full_discovery", False))
                            and len(env._sim.first_true_slot) > 0
                            and len(env._sim.discovered_edges) >= len(env._sim.first_true_slot)
                        ):
                            truncated = True
                        rewards.append(torch_module.as_tensor(reward, dtype=torch_module.float32))
                    rewards_tensor = torch_module.stack(rewards)
                    summary = env._sim.summarize(start_episode + offset).as_dict()
                    summary.update(env.access_gate_summary())
                    summary.update(summarize_rendezvous_action_diagnostics(rendezvous_totals))
                    row = {
                        "phase": "eval_stochastic" if use_stochastic else "eval_deterministic",
                        "eval_after_episode": start_episode,
                        "training_step": eval_training_step,
                        "eval_episode": offset,
                        "seed": seed,
                        "algorithm": str(args.algorithm),
                        "env_protocol": env_protocol,
                        "episode_return_sum": float(rewards_tensor.sum().item()),
                        "episode_return_mean_per_agent": float(rewards_tensor.sum(dim=0).mean().item()),
                        "step_reward_mean": float(rewards_tensor.mean().item()),
                    }
                    row.update(summary)
                    rows.append(row)
    finally:
        torch_module.random.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        if hasattr(policy, "restore_recurrent_state"):
            policy.restore_recurrent_state(recurrent_state)
        if was_training:
            policy.train()
        else:
            policy.eval()
    return rows


def loss_summary(
    policy_losses: list[float],
    value_losses: list[float],
    entropy_values: list[float],
    approx_kls: list[float],
    clip_fracs: list[float],
    bc_losses: list[float] | None = None,
    mode_policy_losses: list[float] | None = None,
    beam_policy_losses: list[float] | None = None,
    gate_policy_losses: list[float] | None = None,
    beam_rank_aux_losses: list[float] | None = None,
    rendezvous_beam_aux_losses: list[float] | None = None,
    rendezvous_role_aux_losses: list[float] | None = None,
    beam_active_fracs: list[float] | None = None,
    actor_grad_norms: list[float] | None = None,
    critic_grad_norms: list[float] | None = None,
    explained_variances: list[float] | None = None,
    normalized_candidate_entropies: list[float] | None = None,
    sample_log_ratio_means: list[float] | None = None,
    role_balance_losses: list[float] | None = None,
    mean_policy_tx_probabilities: list[float] | None = None,
) -> dict[str, float]:
    return {
        "policy_loss": float(np.mean(policy_losses)),
        "mode_policy_loss": float(np.mean(mode_policy_losses)) if mode_policy_losses else 0.0,
        "beam_policy_loss": float(np.mean(beam_policy_losses)) if beam_policy_losses else 0.0,
        "gate_policy_loss": float(np.mean(gate_policy_losses)) if gate_policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropy_values)),
        "approx_kl": float(np.mean(approx_kls)),
        "clip_fraction": float(np.mean(clip_fracs)),
        "expert_bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
        "beam_rank_aux_loss": float(np.mean(beam_rank_aux_losses)) if beam_rank_aux_losses else 0.0,
        "rendezvous_beam_aux_loss": float(np.mean(rendezvous_beam_aux_losses))
        if rendezvous_beam_aux_losses
        else 0.0,
        "rendezvous_role_aux_loss": float(np.mean(rendezvous_role_aux_losses))
        if rendezvous_role_aux_losses
        else 0.0,
        "beam_active_fraction": float(np.mean(beam_active_fracs)) if beam_active_fracs else 0.0,
        "actor_grad_norm": float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0,
        "critic_grad_norm": float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0,
        "explained_variance": float(np.mean(explained_variances)) if explained_variances else 0.0,
        "normalized_candidate_entropy": (
            float(np.mean(normalized_candidate_entropies)) if normalized_candidate_entropies else 0.0
        ),
        "sample_log_ratio_mean": float(np.mean(sample_log_ratio_means)) if sample_log_ratio_means else 0.0,
        "role_balance_loss": float(np.mean(role_balance_losses)) if role_balance_losses else 0.0,
        "mean_policy_tx_probability": (
            float(np.mean(mean_policy_tx_probabilities))
            if mean_policy_tx_probabilities
            else 0.5
        ),
    }


def should_checkpoint(index: int, interval: int) -> bool:
    return interval > 0 and index % interval == 0


def training_contract_version(args: argparse.Namespace) -> str:
    if (
        str(getattr(args, "action_contract", "joint_role_beam")) == "joint_role_beam"
        and str(getattr(args, "network", "")) == "recurrent_contention_shared"
    ):
        return "joint_role_beam_recurrent_local_v2"
    if is_beam_only_action_contract(
        str(getattr(args, "action_contract", "joint_role_beam"))
    ):
        if str(getattr(args, "network", "")) == "recurrent_contention_shared":
            return "beam_only_recurrent_local_v1"
        return "beam_only_fixed_role_v1"
    if not bool(getattr(args, "clean_ctde", False)):
        return "twc_trainable_v1"
    if bool(getattr(args, "residual_measurement_features", False)):
        return CLEAN_CTDE_RESIDUAL_CONTRACT_VERSION
    return CLEAN_CTDE_CONTRACT_VERSION


def architecture_version(args: argparse.Namespace) -> str:
    actor = str(getattr(args, "network", "shared"))
    critic = str(getattr(args, "critic_network", "pooled"))
    contract = str(getattr(args, "action_contract", "joint_role_beam"))
    if (
        actor == "recurrent_contention_shared"
        and str(getattr(args, "role_factorization", "independent")) == "beam_conditioned"
    ):
        return f"joint_beam_conditioned_role_recurrent_{critic}_direction_v1"
    if (
        actor == "recurrent_contention_shared"
        and critic == "mpnn"
        and bool(getattr(args, "candidate_score_prior", False))
    ):
        prefix = "joint" if contract == "joint_role_beam" else "beam_only"
        if contract == "joint_role_beam" and bool(
            getattr(args, "decoupled_role_tower", False)
        ):
            return "joint_decoupled_role_recurrent_beam_mpnn_score_residual_v4"
        if bool(getattr(args, "bounded_score_residual", False)):
            return f"{prefix}_recurrent_mpnn_bounded_score_residual_v3"
        return f"{prefix}_recurrent_mpnn_score_residual_v2"
    if actor == "recurrent_contention_shared" and critic == "mpnn":
        prefix = "joint" if contract == "joint_role_beam" else "beam_only"
        return f"{prefix}_recurrent_mpnn_v1"
    if actor == "recurrent_contention_shared":
        prefix = "joint" if contract == "joint_role_beam" else "beam_only"
        return f"{prefix}_recurrent_{critic}_direction_v2"
    return f"{actor}_{critic}_v1"


def save_checkpoint(
    path: Path,
    policy: SharedBeamActorCritic,
    critic: Any,
    optimizer: Any,
    args: argparse.Namespace,
    cfg: SimulationConfig,
    episode: int,
    torch_module: Any,
) -> None:
    checkpoint = {
        "episode": int(episode),
        "algorithm": str(args.algorithm),
        "action_contract": str(getattr(args, "action_contract", "joint_role_beam")),
        "sanity_only": str(getattr(args, "action_contract", "joint_role_beam"))
        == "beam_only_complementary_role",
        "training_scenario_mode": str(getattr(args, "training_scenario_mode", "varying")),
        "evaluation_scenario_mode": str(getattr(args, "evaluation_scenario_mode", "held_out")),
        "terminate_on_full_discovery": bool(
            getattr(args, "terminate_on_full_discovery", False)
        ),
        "training_contract_version": training_contract_version(args),
        "architecture_version": architecture_version(args),
        "actor_network": str(getattr(args, "network", "shared")),
        "critic_network": str(getattr(args, "critic_network", "pooled")),
        "feature_flags": resolved_feature_flags(args),
        "env_protocol": resolved_env_protocol(args),
        "policy_state_dict": policy.model.state_dict(),
        "critic_state_dict": critic.state_dict() if critic is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "config": cfg.__dict__,
    }
    torch_module.save(checkpoint, path)


def flush_outputs(
    output: Path,
    step_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
) -> None:
    write_rows(output / "step_rewards.csv", step_rows)
    write_rows(output / "episode_metrics.csv", episode_rows)
    write_rows(output / "eval_episode_metrics.csv", eval_rows)
    write_rows(output / "resource_log.csv", resource_rows)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def build_manifest(
    args: argparse.Namespace,
    cfg: SimulationConfig,
    feature_flags: dict[str, bool],
    env_protocol: str,
    centralized: bool,
    episode_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import torch

        torch_version = str(torch.__version__)
    except ImportError:  # pragma: no cover - training already requires torch
        torch_version = "unavailable"
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": str(args.algorithm),
        "action_contract": str(getattr(args, "action_contract", "joint_role_beam")),
        "network": str(getattr(args, "network", "shared")),
        "architecture_version": architecture_version(args),
        "actor_network": str(getattr(args, "network", "shared")),
        "actor_recurrent_dim": (
            int(args.hidden_dim)
            if str(getattr(args, "network", "")) == "recurrent_contention_shared"
            else None
        ),
        "actor_state_reset": (
            "zero_each_episode"
            if str(getattr(args, "network", "")) == "recurrent_contention_shared"
            else "stateless"
        ),
        "actor_global_state_access": False,
        "candidate_score_prior": bool(getattr(args, "candidate_score_prior", False)),
        "candidate_score_prior_power": float(getattr(args, "candidate_score_prior_power", 1.0)),
        "bounded_score_residual": bool(getattr(args, "bounded_score_residual", False)),
        "score_residual_max_logit": float(getattr(args, "score_residual_max_logit", 2.0)),
        "decoupled_role_tower": bool(getattr(args, "decoupled_role_tower", False)),
        "role_factorization": str(getattr(args, "role_factorization", "independent")),
        "role_conditioning": (
            "selected_beam"
            if str(getattr(args, "role_factorization", "independent")) == "beam_conditioned"
            else "none"
        ),
        "actor_gradient_isolation": (
            "role_and_beam_towers_disjoint"
            if bool(getattr(args, "decoupled_role_tower", False))
            else "shared_actor_trunk"
        ),
        "critic_network": str(getattr(args, "critic_network", "pooled")),
        "critic_hidden_dim": int(getattr(args, "critic_hidden_dim", None) or args.hidden_dim),
        "critic_message_passes": (
            2 if str(getattr(args, "critic_network", "pooled")) == "mpnn" else 0
        ),
        "critic_input_contract": (
            "central_graph_truth_training_only_v1"
            if str(getattr(args, "critic_network", "pooled")) == "mpnn"
            else "central_pooled_truth_training_only_v1"
        ),
        "reward_version": str(getattr(args, "reward_version", "legacy")),
        "reward_event_source": "dedicated_per_node_handshake_counters_v2",
        "local_potential_shaping": {
            "coefficient": float(getattr(args, "local_potential_shaping_coef", 0.0)),
            "potential": "negative_candidate_count_and_score_entropy_v1",
            "information_source": "decentralized_actor_observation_only",
            "terminal_potential": 0.0,
            "global_truth_used": False,
        },
        "scope": "real_marl_training",
        "config": str(args.config),
        "output": str(args.output),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "slots_per_episode": int(cfg.slots_per_episode),
        "slot_duration_ms": float(cfg.slot_duration_s) * 1000.0,
        "node_count": int(cfg.n_nodes),
        "beam_count": int(cfg.n_beams),
        "azimuth_cells": int(cfg.azimuth_cells),
        "elevation_cells": int(cfg.elevation_cells),
        "communication_range_m": float(cfg.communication_range_m),
        "sensing_range_m": float(cfg.sensing_range_m),
        "env_protocol": env_protocol,
        "candidate_source": str(getattr(args, "candidate_source", "default")),
        "allow_standalone_sense": bool(getattr(args, "allow_standalone_sense", False)),
        "allow_idle": bool(getattr(args, "allow_idle", True)),
        "disabled_modes": list(disabled_modes_from_args(args)),
        "expert_bc_weight": float(getattr(args, "expert_bc_weight", 0.0)),
        "expert_protocol": str(getattr(args, "expert_protocol", "collision_aware_isac")),
        "expert_gate_imitation": bool(float(getattr(args, "expert_bc_weight", 0.0)) > 0.0),
        "separate_action_loss": bool(getattr(args, "separate_action_loss", False)),
        "beam_loss_coef": float(getattr(args, "beam_loss_coef", 1.0)),
        "beam_isac_feedback_coef": float(getattr(args, "beam_isac_feedback_coef", 0.0)),
        "beam_isac_feedback_contract": "local_post_tx_anonymous_occupancy_v1",
        "role_balance_regularizer": {
            "coefficient": float(getattr(args, "role_balance_coef", 0.0)),
            "target_mean_tx_probability": 0.5,
            "scope": "training_batch_per_slot_only",
            "execution_global_information": False,
        },
        "gate_loss_coef": float(getattr(args, "gate_loss_coef", 0.25)),
        "beam_rank_aux_coef": float(getattr(args, "beam_rank_aux_coef", 0.0)),
        "beam_rank_temperature": float(getattr(args, "beam_rank_temperature", 4.0)),
        "rendezvous_beam_aux_coef": float(getattr(args, "rendezvous_beam_aux_coef", 0.0)),
        "rendezvous_role_aux_coef": float(getattr(args, "rendezvous_role_aux_coef", 0.0)),
        "rendezvous_adapter": bool(getattr(args, "rendezvous_adapter", False)),
        "rendezvous_adapter_learning_rate": float(getattr(args, "rendezvous_adapter_learning_rate", 0.03)),
        "clean_ctde": bool(getattr(args, "clean_ctde", False)),
        "actor_observation_contract": (
            training_contract_version(args) if bool(getattr(args, "clean_ctde", False)) else "legacy_local_features"
        ),
        "residual_measurement_features": bool(getattr(args, "residual_measurement_features", False)),
        "stochastic_support": {
            "role_probability_floor": float(getattr(args, "role_probability_floor", 0.0)),
            "beam_uniform_mixture": float(getattr(args, "beam_uniform_mixture", 0.0)),
        },
        "action_teacher_free": bool(getattr(args, "clean_ctde", False)),
        "role_policy": (
            "fixed_iid_bernoulli_0.5_not_learned"
            if str(getattr(args, "action_contract", "joint_role_beam"))
            == "beam_only_fixed_role"
            else "fixed_alternating_tx_rx_diagnostic_not_learned"
            if str(getattr(args, "action_contract", "joint_role_beam"))
            == "beam_only_complementary_role"
            else "learned_mode"
        ),
        "fixed_tx_probability": (
            0.5
            if str(getattr(args, "action_contract", "joint_role_beam"))
            == "beam_only_fixed_role"
            else None
        ),
        "role_rng_seed_policy": (
            "single_training_stream_seed_plus_19"
            if str(getattr(args, "action_contract", "joint_role_beam"))
            == "beam_only_fixed_role"
            else None
        ),
        "learned_mode_head_present": not is_beam_only_action_contract(
            str(getattr(args, "action_contract", "joint_role_beam"))
        ),
        "local_candidate_processing_allowed": bool(getattr(args, "clean_ctde", False)),
        "pair_derived_action_guidance_enabled": bool(cfg.rendezvous_observation_enabled),
        "post_handshake_table_exchange_protocol": env_protocol,
        "feature_flags": feature_flags,
        "rendezvous_observation_enabled": bool(cfg.rendezvous_observation_enabled),
        "contention_mode_prior": contention_mode_prior_enabled(args),
        "rule_residual_scale": float(getattr(args, "rule_residual_scale", 1.0)),
        "single_rf_chain": int(cfg.rf_chains) == 1,
        "isac_trigger": "tx_piggyback_only",
        "handshake_collision_model": (
            "unique_tx_and_unique_rx"
            if cfg.communication_phy_model == "ideal"
            else "two_phase_hello_ack_sinr_capture"
        ),
        "table_exchange_information": "confirmed_neighbor_positions_and_noisy_anonymous_sensing_reports",
        "sensing_measurement": {
            "mode": cfg.sensing_measurement_mode,
            "identity_exposed_before_handshake": False,
            "common_protocol_interface": True,
        },
        "communication_phy": communication_phy_manifest(cfg),
        "centralized_training_decentralized_execution": bool(centralized),
        "decentralized_actor_observation": True,
        "centralized_critic_uses_training_state_only": bool(centralized),
        "advantage_snapshot_contract": (
            "fixed_behavior_rollout_gae_v1"
            if str(getattr(args, "advantage_estimator", "mc")) == "gae"
            else "fixed_behavior_rollout_mc_v1"
        ),
        "advantage_estimator": str(getattr(args, "advantage_estimator", "mc")),
        "gae_lambda": (
            float(getattr(args, "gae_lambda", 0.95))
            if str(getattr(args, "advantage_estimator", "mc")) == "gae"
            else None
        ),
        "return_boundary_contract": "finite_horizon_terminal_zero_bootstrap",
        "logs_per_step_reward": True,
        "logs_episode_return": True,
        "torch_threads": int(args.torch_threads),
        "command": [sys.executable, *sys.argv],
        "git_commit": git_revision(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "torch": torch_version,
            "platform": platform.platform(),
        },
        "resource_limits": {
            "max_rss_mb": float(args.max_rss_mb),
            "max_system_memory_percent": float(args.max_system_memory_percent),
        },
        "stochastic_eval": bool(args.stochastic_eval),
        "eval_both": bool(args.eval_both),
        "final_train": episode_rows[-1] if episode_rows else {},
        "final_eval": eval_rows[-1] if eval_rows else {},
        "files": [
            "step_rewards.csv",
            "episode_metrics.csv",
            "eval_episode_metrics.csv",
            "resource_log.csv",
            "final_model.pt",
            "manifest.json",
        ],
    }


def git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def git_worktree_dirty() -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(completed.stdout.strip())


def communication_phy_manifest(cfg: SimulationConfig) -> dict[str, Any]:
    return {
        "model": cfg.communication_phy_model,
        "carrier_frequency_hz": cfg.communication_carrier_frequency_hz,
        "bandwidth_hz": cfg.communication_bandwidth_hz,
        "tx_power_w": cfg.communication_tx_power_w,
        "noise_figure_db": cfg.communication_noise_figure_db,
        "path_loss_exponent": cfg.communication_path_loss_exponent,
        "reference_distance_m": cfg.communication_reference_distance_m,
        "system_loss_db": cfg.communication_system_loss_db,
        "shadowing_std_db": cfg.communication_shadowing_std_db,
        "rician_k_db": cfg.communication_rician_k_db,
        "sinr_threshold_db": cfg.communication_sinr_threshold_db,
        "antenna_efficiency": cfg.communication_antenna_efficiency,
            "sidelobe_gain_db": cfg.communication_sidelobe_gain_db,
            "antenna_gain_mode": cfg.communication_antenna_gain_mode,
            "fixed_main_lobe_gain_db": cfg.communication_fixed_main_lobe_gain_db,
            "shared_waveform_power_enabled": cfg.shared_waveform_power_enabled,
        "fading_enabled": cfg.communication_fading_enabled,
        "shadowing_enabled": cfg.communication_shadowing_enabled,
        "handshake": "two_phase_hello_ack_sinr_capture",
        "channel_seed_policy": "scenario_seed_only",
    }


def main() -> None:
    print(json.dumps(run_training(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
