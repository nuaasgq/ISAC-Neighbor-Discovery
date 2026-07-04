from __future__ import annotations

import numpy as np

from .mobility import NodeState


def rotation_body_to_global(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def relative_direction_body(state: NodeState, target_position: np.ndarray) -> np.ndarray:
    direction_global = target_position - state.position
    norm = float(np.linalg.norm(direction_global))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0])
    direction_global = direction_global / norm
    return rotation_body_to_global(state.yaw, state.pitch, state.roll).T @ direction_global


def direction_to_beam(direction_body: np.ndarray, azimuth_cells: int, elevation_cells: int) -> tuple[int, int, int]:
    norm = float(np.linalg.norm(direction_body))
    if norm <= 1e-12:
        direction_body = np.array([1.0, 0.0, 0.0])
    else:
        direction_body = direction_body / norm
    azimuth = float(np.arctan2(direction_body[1], direction_body[0]))
    elevation = float(np.arcsin(np.clip(direction_body[2], -1.0, 1.0)))
    az_idx = int(np.floor((azimuth + np.pi) / (2.0 * np.pi) * azimuth_cells)) % azimuth_cells
    el_idx = int(np.floor((elevation + np.pi / 2.0) / np.pi * elevation_cells))
    el_idx = int(np.clip(el_idx, 0, elevation_cells - 1))
    return el_idx * azimuth_cells + az_idx, az_idx, el_idx


def beam_for_target(
    source: NodeState,
    target_position: np.ndarray,
    azimuth_cells: int,
    elevation_cells: int,
) -> int:
    beam_idx, _, _ = direction_to_beam(
        relative_direction_body(source, target_position),
        azimuth_cells,
        elevation_cells,
    )
    return beam_idx


def beam_matches(
    selected_beam: int,
    expected_beam: int,
    azimuth_cells: int,
    tolerance_cells: int,
) -> bool:
    selected_az = selected_beam % azimuth_cells
    selected_el = selected_beam // azimuth_cells
    expected_az = expected_beam % azimuth_cells
    expected_el = expected_beam // azimuth_cells
    az_delta = abs(selected_az - expected_az)
    az_delta = min(az_delta, azimuth_cells - az_delta)
    el_delta = abs(selected_el - expected_el)
    return az_delta <= tolerance_cells and el_delta <= tolerance_cells


def shift_beam(
    beam_idx: int,
    azimuth_cells: int,
    elevation_cells: int,
    az_shift: int,
    el_shift: int,
) -> int:
    az_idx = (beam_idx % azimuth_cells + az_shift) % azimuth_cells
    el_idx = int(np.clip(beam_idx // azimuth_cells + el_shift, 0, elevation_cells - 1))
    return el_idx * azimuth_cells + az_idx
