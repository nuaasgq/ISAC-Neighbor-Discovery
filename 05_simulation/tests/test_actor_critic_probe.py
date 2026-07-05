from __future__ import annotations

import importlib.util
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest

from isac_nd_sim.config import load_config
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv
from isac_nd_sim.neural_contention_actor_critic import ContentionGraphActorCritic
from isac_nd_sim.neural_scalegraph_beam_actor_critic import ScaleGraphBeamActorCritic
from isac_nd_sim.neural_shared_actor_critic import SharedBeamActorCritic


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "run_actor_critic_probe.py"
IMITATION_SCRIPT = ROOT / "05_simulation" / "run_actor_critic_imitation_probe.py"
MARL_TRAINING_SCRIPT = ROOT / "05_simulation" / "run_marl_training.py"
MARL_EVALUATE_SCRIPT = ROOT / "05_simulation" / "run_marl_evaluate.py"


def test_shared_actor_critic_samples_valid_actions() -> None:
    pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4, azimuth_cells=4, elevation_cells=2)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=123)
    policy = SharedBeamActorCritic(cfg.n_beams, hidden_dim=16)

    step = policy.act(observations)

    assert len(step.actions) == cfg.n_nodes
    assert step.log_probs.shape == (cfg.n_nodes,)
    assert step.values.shape == (cfg.n_nodes,)
    for action in step.actions:
        assert action.mode in env.modes
        assert 0 <= action.beam < cfg.n_beams


def test_shared_actor_critic_candidate_mask_samples_valid_actions() -> None:
    pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4, azimuth_cells=4, elevation_cells=2)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=123)
    policy = SharedBeamActorCritic(cfg.n_beams, hidden_dim=16, use_candidate_mask=True)

    step = policy.act(observations)

    assert len(step.actions) == cfg.n_nodes
    for observation, action in zip(observations, step.actions, strict=True):
        assert action.mode in env.modes
        assert 0 <= action.beam < cfg.n_beams
        if action.mode != "idle":
            assert observation["candidate_mask"][action.beam] > 0.5


def test_shared_actor_critic_rule_residual_can_force_candidate_action() -> None:
    pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=1, azimuth_cells=4, elevation_cells=2)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=123)
    observation = dict(observations[0])
    target_beam = 3
    observation["candidate_mask"] = observation["candidate_mask"].copy()
    observation["candidate_mask"][:] = 0.0
    observation["candidate_mask"][target_beam] = 1.0
    observation["candidate_score"] = observation["candidate_score"].copy()
    observation["candidate_score"][:] = 0.0
    observation["candidate_score"][target_beam] = 1.0
    observation["rule_mode_logits"] = observation["rule_mode_logits"].copy()
    observation["rule_mode_logits"][:] = -2.0
    observation["rule_mode_logits"][1] = 2.0
    policy = SharedBeamActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_mask=True,
        use_candidate_score=True,
        use_rule_residual=True,
        rule_residual_scale=10.0,
    )

    step = policy.act([observation], deterministic=True)

    assert step.actions[0].mode == "tx"
    assert step.actions[0].beam == target_beam


def test_scalegraph_beam_actor_critic_candidate_mask_samples_valid_actions() -> None:
    pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4, azimuth_cells=4, elevation_cells=2)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=123)
    policy = ScaleGraphBeamActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_mask=True,
        use_candidate_score=True,
        use_topology_deficit=True,
    )

    step = policy.act(observations)

    assert len(step.actions) == cfg.n_nodes
    assert step.log_probs.shape == (cfg.n_nodes,)
    assert step.values.shape == (cfg.n_nodes,)
    for observation, action in zip(observations, step.actions, strict=True):
        assert action.mode in env.modes
        assert 0 <= action.beam < cfg.n_beams
        if action.mode != "idle":
            assert observation["candidate_mask"][action.beam] > 0.5


def test_contention_graph_actor_critic_candidate_mask_samples_valid_actions() -> None:
    pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4, azimuth_cells=4, elevation_cells=2)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=123)
    policy = ContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_mask=True,
        use_candidate_score=True,
        use_topology_deficit=True,
        use_rule_residual=True,
    )

    step = policy.act(observations)

    assert len(step.actions) == cfg.n_nodes
    assert step.log_probs.shape == (cfg.n_nodes,)
    assert step.values.shape == (cfg.n_nodes,)
    for observation, action in zip(observations, step.actions, strict=True):
        assert action.mode in env.modes
        assert 0 <= action.beam < cfg.n_beams
        if action.mode != "idle":
            assert observation["candidate_mask"][action.beam] > 0.5


def test_actor_critic_probe_writes_history(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    module = load_probe_module(SCRIPT, "run_actor_critic_probe")
    output = tmp_path / "probe"
    manifest = module.run_probe(
        Namespace(
            config=str(ROOT / "05_simulation" / "configs" / "mvp.yaml"),
            output=str(output),
            episodes=1,
            slots=2,
            node_count=4,
            azimuth_cells=4,
            elevation_cells=2,
            hidden_dim=16,
            learning_rate=1e-3,
            gamma=0.95,
            value_coef=0.5,
            entropy_coef=0.01,
            seed=77,
        )
    )

    assert manifest["algorithm"] == "shared_actor_critic_probe"
    assert (output / "training_history.csv").exists()
    assert (output / "manifest.json").exists()


def test_actor_critic_imitation_probe_writes_history(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    module = load_probe_module(IMITATION_SCRIPT, "run_actor_critic_imitation_probe")
    output = tmp_path / "imitation_probe"
    manifest = module.run_probe(
        Namespace(
            config=str(ROOT / "05_simulation" / "configs" / "mvp.yaml"),
            output=str(output),
            bc_episodes=1,
            rl_episodes=0,
            eval_episodes=0,
            stochastic_eval=False,
            eval_both=False,
            slots=2,
            node_count=4,
            azimuth_cells=4,
            elevation_cells=2,
            communication_range=800.0,
            sensing_range=900.0,
            false_alarm_rate=0.0,
            miss_detection_rate=0.0,
            angular_cell_offset_std=0.0,
            sensing_period_slots=1,
            env_protocol="isac_structured_marl",
            expert_protocol="improved_rl_isac",
            hidden_dim=16,
            learning_rate=1e-3,
            gamma=0.95,
            bc_coef=1.0,
            beam_bc_coef=1.0,
            value_coef=0.25,
            entropy_coef=0.001,
            candidate_mask=False,
            candidate_score=False,
            topology_deficit=False,
            rule_residual=False,
            rule_residual_scale=1.0,
            seed=78,
            expert_seed_offset=7919,
        )
    )

    assert manifest["algorithm"] == "shared_actor_critic_imitation_probe"
    assert manifest["scope"] == "method_probe_not_paper_result"
    assert (output / "training_history.csv").exists()
    assert (output / "manifest.json").exists()


def test_marl_training_writes_step_episode_eval_and_resource_logs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    module = load_probe_module(MARL_TRAINING_SCRIPT, "run_marl_training")
    output = tmp_path / "marl_training"
    manifest = module.run_training(
        Namespace(
            config=str(ROOT / "05_simulation" / "configs" / "mvp.yaml"),
            output=str(output),
            algorithm="isac_mappo",
            network="shared",
            reward_version="legacy",
            episodes=1,
            slots=2,
            eval_episodes=1,
            eval_interval=1,
            stochastic_eval=False,
            eval_both=True,
            checkpoint_interval=0,
            node_count=4,
            azimuth_cells=4,
            elevation_cells=2,
            communication_range=800.0,
            sensing_range=900.0,
            false_alarm_rate=0.0,
            miss_detection_rate=0.0,
            angular_cell_offset_std=0.0,
            sensing_period_slots=1,
            mobility_model=None,
            env_protocol=None,
            hidden_dim=16,
            learning_rate=1e-3,
            gamma=0.95,
            ppo_epochs=1,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=1.0,
            candidate_mask=False,
            candidate_score=False,
            topology_deficit=False,
            rule_residual=False,
            rule_residual_scale=1.0,
            disable_isac_features=False,
            seed=79,
            torch_threads=1,
            step_log_period=1,
            resource_log_period=1,
            max_rss_mb=12000.0,
            max_system_memory_percent=99.0,
        )
    )

    assert manifest["scope"] == "real_marl_training"
    assert manifest["logs_per_step_reward"] is True
    assert manifest["logs_episode_return"] is True
    assert manifest["centralized_training_decentralized_execution"] is True
    assert (output / "step_rewards.csv").exists()
    assert (output / "episode_metrics.csv").exists()
    assert (output / "eval_episode_metrics.csv").exists()
    assert (output / "resource_log.csv").exists()
    assert (output / "final_model.pt").exists()

    eval_module = load_probe_module(MARL_EVALUATE_SCRIPT, "run_marl_evaluate")
    eval_output = tmp_path / "marl_transfer_eval"
    eval_manifest = eval_module.run_evaluation(
        Namespace(
            checkpoint=str(output / "final_model.pt"),
            config=str(ROOT / "05_simulation" / "configs" / "mvp.yaml"),
            output=str(eval_output),
            eval_episodes=1,
            slots=2,
            node_count=5,
            azimuth_cells=5,
            elevation_cells=2,
            communication_range=900.0,
            sensing_range=900.0,
            false_alarm_rate=0.0,
            miss_detection_rate=0.0,
            angular_cell_offset_std=0.0,
            sensing_period_slots=1,
            mobility_model=None,
            env_protocol=None,
            deterministic=False,
            stochastic=True,
            eval_both=False,
            reward_version=None,
            seed=179,
            torch_threads=1,
        )
    )

    assert eval_manifest["scope"] == "marl_transfer_evaluation"
    assert eval_manifest["node_count"] == 5
    assert eval_manifest["beam_count"] == 10
    assert (eval_output / "eval_episode_metrics.csv").exists()


def load_probe_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
