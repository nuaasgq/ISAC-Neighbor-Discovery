from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NodeState:
    position: np.ndarray
    velocity: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    waypoint: np.ndarray | None = None
    pause_remaining: int = 0


def spatial_dimensions(mobility_cfg: dict[str, Any]) -> int:
    dimensions = int(mobility_cfg.get("spatial_dimensions", 3))
    if dimensions not in {2, 3}:
        raise ValueError("mobility spatial_dimensions must be either 2 or 3")
    return dimensions


def planar_altitude(area: np.ndarray, mobility_cfg: dict[str, Any]) -> float:
    altitude = float(mobility_cfg.get("fixed_altitude_m", area[2] / 2.0))
    return float(np.clip(altitude, 0.0, area[2]))


def random_unit_vectors(rng: np.random.Generator, n: int, dimensions: int = 3) -> np.ndarray:
    vec = rng.normal(size=(n, 3))
    if dimensions == 2:
        vec[:, 2] = 0.0
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return vec / norm


def update_attitude(state: NodeState) -> None:
    speed_xy = float(np.linalg.norm(state.velocity[:2]))
    speed = float(np.linalg.norm(state.velocity))
    if speed <= 1e-9:
        return
    state.yaw = float(np.arctan2(state.velocity[1], state.velocity[0]))
    state.pitch = float(np.arctan2(state.velocity[2], speed_xy))
    state.roll = 0.0


def initialize_states(
    n_nodes: int,
    area_size_m: tuple[float, float, float],
    mobility_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> list[NodeState]:
    area = np.asarray(area_size_m, dtype=float)
    dimensions = spatial_dimensions(mobility_cfg)
    positions = rng.uniform(low=np.zeros(3), high=area, size=(n_nodes, 3))
    if dimensions == 2:
        positions[:, 2] = planar_altitude(area, mobility_cfg)
    speed_mean = float(mobility_cfg.get("speed_mean_mps", 15.0))
    speed_std = float(mobility_cfg.get("speed_std_mps", 3.0))
    speeds = np.maximum(0.0, rng.normal(speed_mean, speed_std, size=n_nodes))
    velocities = random_unit_vectors(rng, n_nodes, dimensions) * speeds[:, None]
    states = [NodeState(positions[i], velocities[i]) for i in range(n_nodes)]
    for state in states:
        update_attitude(state)
    if mobility_cfg.get("model") == "random_waypoint":
        for state in states:
            state.waypoint = rng.uniform(low=np.zeros(3), high=area)
            if dimensions == 2:
                state.waypoint[2] = planar_altitude(area, mobility_cfg)
    return states


def apply_boundary(state: NodeState, area: np.ndarray, boundary: str) -> None:
    if boundary == "clip":
        state.position[:] = np.clip(state.position, 0.0, area)
        return
    if boundary == "wrap":
        state.position[:] = np.mod(state.position, area)
        return
    for dim in range(3):
        if state.position[dim] < 0.0:
            state.position[dim] = -state.position[dim]
            state.velocity[dim] *= -1.0
        if state.position[dim] > area[dim]:
            state.position[dim] = 2.0 * area[dim] - state.position[dim]
            state.velocity[dim] *= -1.0
        state.position[dim] = float(np.clip(state.position[dim], 0.0, area[dim]))


def cap_speed(velocity: np.ndarray, min_speed: float | None, max_speed: float | None) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    if min_speed and min_speed > 0.0 and 1e-9 < speed < min_speed:
        velocity = velocity / speed * min_speed
        speed = min_speed
    if not max_speed or max_speed <= 0.0:
        return velocity
    if speed > max_speed:
        return velocity / speed * max_speed
    return velocity


def step_states(
    states: list[NodeState],
    area_size_m: tuple[float, float, float],
    mobility_cfg: dict[str, Any],
    dt_s: float,
    slot: int,
    rng: np.random.Generator,
) -> None:
    model = str(mobility_cfg.get("model", "gauss_markov")).lower()
    area = np.asarray(area_size_m, dtype=float)
    dimensions = spatial_dimensions(mobility_cfg)
    altitude = planar_altitude(area, mobility_cfg) if dimensions == 2 else None
    boundary = str(mobility_cfg.get("boundary", "reflect")).lower()
    min_speed = float(mobility_cfg.get("min_speed_mps", 0.0)) or None
    max_speed = float(mobility_cfg.get("max_speed_mps", 0.0)) or None

    if model == "static":
        return
    if model == "gauss_markov":
        step_gauss_markov(states, mobility_cfg, dt_s, rng, min_speed, max_speed)
    elif model == "random_walk":
        step_random_walk(states, mobility_cfg, dt_s, rng, min_speed, max_speed)
    elif model == "random_direction":
        step_random_direction(states, mobility_cfg, dt_s, slot, rng, min_speed, max_speed)
    elif model == "random_waypoint":
        step_random_waypoint(states, area, mobility_cfg, dt_s, rng, min_speed, max_speed)
    else:
        raise ValueError(f"Unsupported mobility model: {model}")

    for state in states:
        if dimensions == 2:
            state.velocity[2] = 0.0
            state.velocity = cap_speed(state.velocity, min_speed, max_speed)
            if state.waypoint is not None:
                state.waypoint[2] = altitude
        state.position += state.velocity * dt_s
        apply_boundary(state, area, boundary)
        if dimensions == 2:
            state.position[2] = altitude
        update_attitude(state)


def step_gauss_markov(
    states: list[NodeState],
    mobility_cfg: dict[str, Any],
    dt_s: float,
    rng: np.random.Generator,
    min_speed: float | None,
    max_speed: float | None,
) -> None:
    alpha = float(mobility_cfg.get("alpha", 0.85))
    speed_mean = float(mobility_cfg.get("speed_mean_mps", 15.0))
    speed_std = float(mobility_cfg.get("speed_std_mps", 3.0))
    noise_scale = max(speed_std, 1e-6) * np.sqrt(max(0.0, 1.0 - alpha * alpha))
    for state in states:
        mean_dir = state.velocity / max(float(np.linalg.norm(state.velocity)), 1e-9)
        mean_velocity = mean_dir * speed_mean
        noise = rng.normal(0.0, noise_scale, size=3)
        state.velocity = cap_speed(alpha * state.velocity + (1.0 - alpha) * mean_velocity + noise, min_speed, max_speed)


def step_random_walk(
    states: list[NodeState],
    mobility_cfg: dict[str, Any],
    dt_s: float,
    rng: np.random.Generator,
    min_speed: float | None,
    max_speed: float | None,
) -> None:
    step_std = float(mobility_cfg.get("random_walk_step_std_m", 1.5))
    for state in states:
        displacement = rng.normal(0.0, step_std, size=3)
        state.velocity = cap_speed(displacement / max(dt_s, 1e-9), min_speed, max_speed)


def step_random_direction(
    states: list[NodeState],
    mobility_cfg: dict[str, Any],
    dt_s: float,
    slot: int,
    rng: np.random.Generator,
    min_speed: float | None,
    max_speed: float | None,
) -> None:
    period = max(1, int(mobility_cfg.get("direction_update_period_slots", 25)))
    speed_mean = float(mobility_cfg.get("speed_mean_mps", 15.0))
    speed_std = float(mobility_cfg.get("speed_std_mps", 3.0))
    if slot % period != 0:
        return
    directions = random_unit_vectors(rng, len(states), spatial_dimensions(mobility_cfg))
    speeds = np.maximum(0.0, rng.normal(speed_mean, speed_std, size=len(states)))
    for idx, state in enumerate(states):
        state.velocity = cap_speed(directions[idx] * speeds[idx], min_speed, max_speed)


def step_random_waypoint(
    states: list[NodeState],
    area: np.ndarray,
    mobility_cfg: dict[str, Any],
    dt_s: float,
    rng: np.random.Generator,
    min_speed: float | None,
    max_speed: float | None,
) -> None:
    speed_mean = float(mobility_cfg.get("speed_mean_mps", 15.0))
    speed_std = float(mobility_cfg.get("speed_std_mps", 3.0))
    pause_slots = int(mobility_cfg.get("waypoint_pause_slots", 0))
    for state in states:
        if state.pause_remaining > 0:
            state.pause_remaining -= 1
            state.velocity[:] = 0.0
            continue
        if state.waypoint is None:
            state.waypoint = rng.uniform(low=np.zeros(3), high=area)
        delta = state.waypoint - state.position
        distance = float(np.linalg.norm(delta))
        if distance < max(speed_mean * dt_s, 1e-6):
            state.position[:] = state.waypoint
            state.waypoint = rng.uniform(low=np.zeros(3), high=area)
            state.pause_remaining = pause_slots
            state.velocity[:] = 0.0
            continue
        speed = max(0.0, rng.normal(speed_mean, speed_std))
        state.velocity = cap_speed(delta / distance * speed, min_speed, max_speed)
