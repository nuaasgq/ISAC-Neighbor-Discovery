from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from .config import SimulationConfig, load_config
from .runner import run_detailed


TRAINED_PROTOCOL = "improved_rl_isac"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float

    def encode(self, value: float) -> float:
        if self.upper <= self.lower:
            raise ValueError(f"Invalid parameter bounds for {self.name}.")
        return float(np.clip((value - self.lower) / (self.upper - self.lower), 0.0, 1.0))

    def decode(self, value: float) -> float:
        clipped = float(np.clip(value, 0.0, 1.0))
        return self.lower + clipped * (self.upper - self.lower)


DEFAULT_PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("alpha_occupancy", 0.0, 3.0),
    ParameterSpec("softmax_beta", 0.25, 8.0),
    ParameterSpec("exploration_floor", 0.01, 0.35),
    ParameterSpec("confidence_decay", 0.80, 0.995),
    ParameterSpec("piggyback_sensing_period_multiplier", 0.50, 4.00),
)


@dataclass(frozen=True)
class TrainingSettings:
    generations: int
    population: int
    episodes: int
    slots_per_episode: int
    seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    test_episodes: int
    elite_fraction: float = 0.25
    learning_rate: float = 0.70
    min_std: float = 0.03
    training_seed: int = 0


@dataclass(frozen=True)
class CandidateEvaluation:
    generation: int
    candidate: int
    vector: np.ndarray
    parameters: dict[str, float]
    score: float
    summary: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared improved_rl_isac policy parameters with CEM.")
    parser.add_argument("--config", default="05_simulation/configs/mobile_smoke.yaml", help="Base YAML config.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to 05_simulation/results_raw/training/<name>_<timestamp>.",
    )
    parser.add_argument("--generations", "--generation", dest="generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per seed for candidate training eval.")
    parser.add_argument(
        "--slots",
        "--horizon",
        dest="slots_per_episode",
        type=int,
        default=400,
        help="Episode horizon in slots. Use 300-500 for long tuning runs.",
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated training scenario seeds.")
    parser.add_argument("--test-seeds", default=None, help="Comma-separated held-out test seeds.")
    parser.add_argument("--test-episodes", type=int, default=None, help="Episodes per held-out test seed.")
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.70)
    parser.add_argument("--min-std", type=float, default=0.03)
    parser.add_argument("--training-seed", type=int, default=0, help="CEM sampler seed.")
    return parser.parse_args()


def train_from_config(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    generations: int = 8,
    population: int = 16,
    episodes: int = 2,
    slots_per_episode: int = 400,
    seeds: Sequence[int] | None = None,
    test_seeds: Sequence[int] | None = None,
    test_episodes: int | None = None,
    elite_fraction: float = 0.25,
    learning_rate: float = 0.70,
    min_std: float = 0.03,
    training_seed: int = 0,
) -> dict[str, Any]:
    config_path = Path(config_path)
    base_config = load_config(config_path)
    settings = TrainingSettings(
        generations=generations,
        population=population,
        episodes=episodes,
        slots_per_episode=slots_per_episode,
        seeds=tuple(int(seed) for seed in (seeds or (base_config.seed,))),
        test_seeds=tuple(int(seed) for seed in (test_seeds or default_test_seeds(seeds, base_config.seed))),
        test_episodes=int(test_episodes if test_episodes is not None else episodes),
        elite_fraction=elite_fraction,
        learning_rate=learning_rate,
        min_std=min_std,
        training_seed=training_seed,
    )
    validate_settings(settings)

    run_dir = resolve_output_dir(base_config, output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(settings.training_seed)
    mean = initial_vector(base_config)
    std = np.full(len(DEFAULT_PARAMETER_SPECS), 0.25, dtype=float)
    training_rows: list[dict[str, Any]] = []
    elite_rows: list[dict[str, Any]] = []
    best: CandidateEvaluation | None = None

    elite_count = max(1, min(settings.population, math.ceil(settings.population * settings.elite_fraction)))
    for generation in range(settings.generations):
        candidates: list[CandidateEvaluation] = []
        for candidate_idx in range(settings.population):
            if generation == 0 and candidate_idx == 0:
                vector = mean.copy()
            else:
                vector = np.clip(rng.normal(mean, std), 0.0, 1.0)
            parameters = decode_vector(vector)
            evaluation = evaluate_candidate(
                base_config=base_config,
                parameters=parameters,
                seeds=settings.seeds,
                episodes=settings.episodes,
                slots_per_episode=settings.slots_per_episode,
                generation=generation,
                candidate=candidate_idx,
            )
            candidates.append(evaluation)
            if best is None or evaluation.score > best.score:
                best = evaluation

        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        elites = ranked[:elite_count]
        rank_by_candidate = {item.candidate: rank for rank, item in enumerate(ranked, start=1)}
        elite_candidate_ids = {item.candidate for item in elites}
        for evaluation in candidates:
            training_rows.append(
                candidate_row(
                    evaluation,
                    settings,
                    rank=rank_by_candidate[evaluation.candidate],
                    is_elite=evaluation.candidate in elite_candidate_ids,
                )
            )
        for rank, evaluation in enumerate(elites, start=1):
            elite_rows.append(candidate_row(evaluation, settings, rank=rank, is_elite=True))

        elite_vectors = np.asarray([item.vector for item in elites], dtype=float)
        elite_mean = elite_vectors.mean(axis=0)
        elite_std = elite_vectors.std(axis=0)
        mean = (1.0 - settings.learning_rate) * mean + settings.learning_rate * elite_mean
        std = (1.0 - settings.learning_rate) * std + settings.learning_rate * elite_std
        std = np.maximum(settings.min_std, std)

    if best is None:
        raise RuntimeError("CEM training produced no candidate evaluations.")

    test_summary_rows = evaluate_test_summary(
        base_config=base_config,
        parameters=best.parameters,
        seeds=settings.test_seeds,
        episodes=settings.test_episodes,
        slots_per_episode=settings.slots_per_episode,
    )

    write_rows_csv(run_dir / "training_history.csv", training_rows)
    write_rows_csv(run_dir / "elite_history.csv", elite_rows)
    write_rows_csv(run_dir / "test_summary.csv", test_summary_rows)
    write_best_config(run_dir / "best_config.yaml", config_path, base_config, settings, best)
    write_manifest(run_dir / "manifest.json", config_path, run_dir, settings, best, test_summary_rows)

    return {
        "run_dir": str(run_dir),
        "best_score": best.score,
        "best_parameters": best.parameters,
        "generations": settings.generations,
        "population": settings.population,
        "training_rows": len(training_rows),
        "elite_rows": len(elite_rows),
        "test_rows": len(test_summary_rows),
    }


def default_test_seeds(seeds: Sequence[int] | None, base_seed: int) -> tuple[int, ...]:
    source = tuple(int(seed) for seed in seeds) if seeds else (int(base_seed),)
    return tuple(seed + 7919 for seed in source)


def validate_settings(settings: TrainingSettings) -> None:
    if settings.generations < 1:
        raise ValueError("generations must be >= 1.")
    if settings.population < 1:
        raise ValueError("population must be >= 1.")
    if settings.episodes < 1:
        raise ValueError("episodes must be >= 1.")
    if settings.test_episodes < 1:
        raise ValueError("test_episodes must be >= 1.")
    if settings.slots_per_episode < 1:
        raise ValueError("slots_per_episode must be >= 1.")
    if not settings.seeds:
        raise ValueError("At least one training seed is required.")
    if not settings.test_seeds:
        raise ValueError("At least one test seed is required.")
    if not 0.0 < settings.elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1].")
    if not 0.0 < settings.learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1].")
    if settings.min_std < 0.0:
        raise ValueError("min_std must be non-negative.")


def resolve_output_dir(base_config: SimulationConfig, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("05_simulation") / "results_raw" / "training" / f"{base_config.name}_{stamp}"


def initial_vector(config: SimulationConfig) -> np.ndarray:
    values = [spec.encode(float(getattr(config, spec.name))) for spec in DEFAULT_PARAMETER_SPECS]
    return np.asarray(values, dtype=float)


def decode_vector(vector: np.ndarray) -> dict[str, float]:
    return {spec.name: spec.decode(float(vector[idx])) for idx, spec in enumerate(DEFAULT_PARAMETER_SPECS)}


def apply_parameters(config: SimulationConfig, parameters: dict[str, float]) -> SimulationConfig:
    return replace(config, **parameters)


def evaluate_candidate(
    base_config: SimulationConfig,
    parameters: dict[str, float],
    seeds: Sequence[int],
    episodes: int,
    slots_per_episode: int,
    generation: int,
    candidate: int,
) -> CandidateEvaluation:
    rows = evaluate_episode_rows(base_config, parameters, seeds, episodes, slots_per_episode)
    summary = summarize_rows(rows, slots_per_episode)
    score = objective_score(summary)
    return CandidateEvaluation(
        generation=generation,
        candidate=candidate,
        vector=vector_from_parameters(parameters),
        parameters=parameters,
        score=score,
        summary=summary,
    )


def evaluate_episode_rows(
    base_config: SimulationConfig,
    parameters: dict[str, float],
    seeds: Sequence[int],
    episodes: int,
    slots_per_episode: int,
) -> list[dict[str, Any]]:
    configured = apply_parameters(
        replace(base_config, episodes=episodes, slots_per_episode=slots_per_episode),
        parameters,
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_config = replace(configured, seed=int(seed))
        seed_rows, _slot_rows, _edge_rows = run_detailed(seed_config, [TRAINED_PROTOCOL])
        rows.extend(seed_rows)
    return rows


def vector_from_parameters(parameters: dict[str, float]) -> np.ndarray:
    return np.asarray([spec.encode(float(parameters[spec.name])) for spec in DEFAULT_PARAMETER_SPECS], dtype=float)


def summarize_rows(rows: list[dict[str, Any]], slots_per_episode: int) -> dict[str, float]:
    if not rows:
        raise ValueError("No episode rows to summarize.")
    metric_keys = (
        "discovery_rate",
        "mean_delay_censored",
        "p90_delay_censored",
        "p95_delay_censored",
        "p99_delay_censored",
        "empty_scan_ratio",
        "collision_count",
        "discovered_edges",
        "true_edges_seen",
        "lcc_ratio",
        "isolated_node_ratio",
        "lambda2",
    )
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
    summary["episode_count"] = float(len(rows))
    summary["slots_per_episode"] = float(slots_per_episode)
    summary["collision_per_slot_mean"] = summary.get("collision_count_mean", 0.0) / max(1.0, float(slots_per_episode))
    summary["delay_fraction_mean"] = summary.get("mean_delay_censored_mean", float(slots_per_episode)) / max(
        1.0,
        float(slots_per_episode),
    )
    return summary


def objective_score(summary: dict[str, float]) -> float:
    discovery = summary.get("discovery_rate_mean", 0.0)
    lcc = summary.get("lcc_ratio_mean", 0.0)
    lambda2 = summary.get("lambda2_mean", 0.0)
    empty = summary.get("empty_scan_ratio_mean", 0.0)
    collision_per_slot = summary.get("collision_per_slot_mean", 0.0)
    delay_fraction = summary.get("delay_fraction_mean", 1.0)
    isolated = summary.get("isolated_node_ratio_mean", 1.0)
    return float(
        100.0 * discovery
        + 20.0 * lcc
        + 5.0 * lambda2
        - 20.0 * empty
        - 20.0 * collision_per_slot
        - 30.0 * delay_fraction
        - 10.0 * isolated
    )


def candidate_row(
    evaluation: CandidateEvaluation,
    settings: TrainingSettings,
    rank: int,
    is_elite: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "generation": evaluation.generation,
        "candidate": evaluation.candidate,
        "rank": rank,
        "is_elite": int(is_elite),
        "score": evaluation.score,
        "protocol": TRAINED_PROTOCOL,
        "seed_count": len(settings.seeds),
        "episodes_per_seed": settings.episodes,
        "slots_per_episode": settings.slots_per_episode,
    }
    row.update(evaluation.parameters)
    row.update(evaluation.summary)
    return row


def evaluate_test_summary(
    base_config: SimulationConfig,
    parameters: dict[str, float],
    seeds: Sequence[int],
    episodes: int,
    slots_per_episode: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    for seed in seeds:
        episode_rows = evaluate_episode_rows(base_config, parameters, [seed], episodes, slots_per_episode)
        all_episode_rows.extend(episode_rows)
        summary = summarize_rows(episode_rows, slots_per_episode)
        rows.append(test_summary_row(str(seed), episodes, slots_per_episode, parameters, summary))
    combined = summarize_rows(all_episode_rows, slots_per_episode)
    rows.append(test_summary_row("all", episodes * len(seeds), slots_per_episode, parameters, combined))
    return rows


def test_summary_row(
    seed: str,
    episodes: int,
    slots_per_episode: int,
    parameters: dict[str, float],
    summary: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "protocol": TRAINED_PROTOCOL,
        "episodes": episodes,
        "slots_per_episode": slots_per_episode,
        "score": objective_score(summary),
    }
    row.update(parameters)
    row.update(summary)
    return row


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_config(
    path: Path,
    config_path: Path,
    base_config: SimulationConfig,
    settings: TrainingSettings,
    best: CandidateEvaluation,
) -> None:
    trained_config = apply_parameters(
        replace(base_config, episodes=settings.test_episodes, slots_per_episode=settings.slots_per_episode),
        best.parameters,
    )
    doc = {
        "base_config_path": str(config_path),
        "trained_protocol": TRAINED_PROTOCOL,
        "best_score": best.score,
        "best_generation": best.generation,
        "best_candidate": best.candidate,
        "training": settings_to_dict(settings),
        "shared_policy_parameters": best.parameters,
        "resolved_simulation_config": simulation_config_to_dict(trained_config),
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def write_manifest(
    path: Path,
    config_path: Path,
    run_dir: Path,
    settings: TrainingSettings,
    best: CandidateEvaluation,
    test_summary_rows: list[dict[str, Any]],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config_path": str(config_path),
        "output_dir": str(run_dir),
        "trained_protocol": TRAINED_PROTOCOL,
        "settings": settings_to_dict(settings),
        "parameter_specs": [asdict(spec) for spec in DEFAULT_PARAMETER_SPECS],
        "best_score": best.score,
        "best_generation": best.generation,
        "best_candidate": best.candidate,
        "best_parameters": best.parameters,
        "test_score": test_summary_rows[-1]["score"] if test_summary_rows else None,
        "files": [
            "training_history.csv",
            "elite_history.csv",
            "best_config.yaml",
            "test_summary.csv",
            "manifest.json",
        ],
    }
    path.write_text(json.dumps(to_plain_data(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


def settings_to_dict(settings: TrainingSettings) -> dict[str, Any]:
    return to_plain_data(asdict(settings))


def simulation_config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    return {field.name: to_plain_data(getattr(config, field.name)) for field in fields(config)}


def to_plain_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_int_csv(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def main() -> None:
    args = parse_args()
    result = train_from_config(
        config_path=args.config,
        output_dir=args.output,
        generations=args.generations,
        population=args.population,
        episodes=args.episodes,
        slots_per_episode=args.slots_per_episode,
        seeds=parse_int_csv(args.seeds),
        test_seeds=parse_int_csv(args.test_seeds),
        test_episodes=args.test_episodes,
        elite_fraction=args.elite_fraction,
        learning_rate=args.learning_rate,
        min_std=args.min_std,
        training_seed=args.training_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
