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
