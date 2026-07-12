from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "scripts" / "run_n2_b8_isac_beam_learning_matrix.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("n2_b8_isac_beam_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_has_four_predeclared_causal_arms() -> None:
    module = load_launcher()
    assert tuple(module.METHOD_SPECS) == (
        "random_beam_antisymmetric_role",
        "learned_beam_no_isac",
        "learned_beam_raw_isac",
        "learned_beam_raw_isac_local_credit",
    )
    assert module.METHOD_SPECS["random_beam_antisymmetric_role"].beam_uniform_mixture == 1.0
    assert module.METHOD_SPECS["learned_beam_no_isac"].beam_uniform_mixture == 0.10
    assert module.METHOD_SPECS["learned_beam_raw_isac"].beam_isac_feedback_coef == 0.0
    assert module.METHOD_SPECS["learned_beam_raw_isac_local_credit"].beam_isac_feedback_coef == 0.50


def test_matrix_commands_keep_actor_local_and_forbid_rule_beam_guidance(tmp_path: Path) -> None:
    module = load_launcher()
    commands = [
        module.training_command("smoke", tmp_path, method, seed, 1)
        for seed in module.DEFAULT_SEEDS
        for method in module.METHOD_SPECS
    ]
    module.validate_matrix_contract(commands)
    assert len(commands) == 12
    for command in commands:
        assert "--clean-ctde" in command
        assert "--no-candidate-score" in command
        assert "--no-candidate-score-prior" in command
        assert "--no-bounded-score-residual" in command
        assert "--candidate-mask" not in command
        assert "--candidate-score-prior" not in command
        assert "--expert-bc-weight" in command
        assert module.command_value(command, "--expert-bc-weight") == "0.0"
        assert module.command_value(command, "--role-factorization") == (
            "beam_conditioned_antisymmetric"
        )


def test_only_isac_arms_receive_raw_local_measurements(tmp_path: Path) -> None:
    module = load_launcher()
    for method, spec in module.METHOD_SPECS.items():
        command = module.training_command("smoke", tmp_path, method, module.DEFAULT_SEEDS[0], 1)
        assert ("--residual-measurement-features" in command) == spec.residual_measurement_features
        assert ("--disable-isac-features" in command) == (not spec.residual_measurement_features)
        assert module.command_value(command, "--env-protocol") == spec.env_protocol
        assert module.command_value(command, "--beam-isac-feedback-coef") == str(
            spec.beam_isac_feedback_coef
        )


def test_formal_profile_is_exactly_100k_environment_steps_per_run() -> None:
    module = load_launcher()
    assert module.PROFILES["formal"]["episodes"] * 16 == 100_000
