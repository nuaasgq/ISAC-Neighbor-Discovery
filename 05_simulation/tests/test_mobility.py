from __future__ import annotations

import numpy as np

from isac_nd_sim.mobility import initialize_states, step_states


def run_model(model: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    cfg = {
        "model": model,
        "speed_mean_mps": 12.0,
        "speed_std_mps": 2.0,
        "min_speed_mps": 3.0,
        "max_speed_mps": 25.0,
        "alpha": 0.8,
        "random_walk_step_std_m": 2.0,
        "direction_update_period_slots": 2,
        "boundary": "reflect",
    }
    area = (100.0, 100.0, 50.0)
    states = initialize_states(8, area, cfg, rng)
    before = np.asarray([s.position.copy() for s in states])
    for slot in range(5):
        step_states(states, area, cfg, 0.2, slot, rng)
    after = np.asarray([s.position.copy() for s in states])
    return before, after


def test_supported_mobility_models_move_and_stay_in_bounds() -> None:
    for model in ["gauss_markov", "random_walk", "random_direction", "random_waypoint"]:
        before, after = run_model(model)
        assert float(np.linalg.norm(after - before)) > 0.0
        assert np.all(after >= 0.0)
        assert np.all(after[:, 0] <= 100.0)
        assert np.all(after[:, 1] <= 100.0)
        assert np.all(after[:, 2] <= 50.0)


def test_planar_mobility_keeps_altitude_velocity_and_pitch_fixed() -> None:
    area = (100.0, 80.0, 0.0)
    for model in ["gauss_markov", "random_walk", "random_direction", "random_waypoint"]:
        rng = np.random.default_rng(1200)
        cfg = {
            "model": model,
            "spatial_dimensions": 2,
            "fixed_altitude_m": 0.0,
            "speed_mean_mps": 8.0,
            "speed_std_mps": 1.0,
            "min_speed_mps": 1.0,
            "max_speed_mps": 12.0,
            "direction_update_period_slots": 2,
        }
        states = initialize_states(8, area, cfg, rng)
        initial_xy = np.asarray([state.position[:2].copy() for state in states])
        for slot in range(20):
            step_states(states, area, cfg, 0.2, slot, rng)
            assert all(state.position[2] == 0.0 for state in states)
            assert all(state.velocity[2] == 0.0 for state in states)
            assert all(state.pitch == 0.0 for state in states)
        final_xy = np.asarray([state.position[:2] for state in states])
        assert np.any(np.linalg.norm(final_xy - initial_xy, axis=1) > 0.0)
