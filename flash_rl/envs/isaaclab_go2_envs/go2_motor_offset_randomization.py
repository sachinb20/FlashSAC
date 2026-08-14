from __future__ import annotations

import torch


def motor_offset_range_is_default(offset_range: tuple[float, float], tol: float = 1.0e-7) -> bool:
    low, high = (float(offset_range[0]), float(offset_range[1]))
    return abs(low) <= tol and abs(high) <= tol


def sample_motor_offsets(
    motor_offsets: torch.Tensor,
    env_ids: torch.Tensor,
    offset_range: tuple[float, float],
) -> None:
    """Draw a per-(env, joint) joint-position bias, in radians, in place.

    Mirrors genesis_envs/go2_base.py's `_randomize_motor_offset`. genesis folds the offset into
    the PD position error inside `_compute_torques`:

        torques = kp * (actions_scaled + default_dof_pos - dof_pos + motor_offsets) - kd * dof_vel

    which is algebraically identical to shifting the position *target* by `motor_offsets`, so
    the port applies it to the joint target instead of reaching into the actuator. It models
    encoder/calibration error: the commanded joint angle and the achieved one differ by a fixed
    per-joint amount for the whole episode.
    """
    low, high = (float(offset_range[0]), float(offset_range[1]))
    if len(env_ids) == 0:
        return
    motor_offsets[env_ids] = torch.empty(
        (len(env_ids), motor_offsets.shape[1]),
        dtype=motor_offsets.dtype,
        device=motor_offsets.device,
    ).uniform_(low, high)


def clear_motor_offsets(motor_offsets: torch.Tensor, env_ids: torch.Tensor) -> None:
    if len(env_ids) > 0:
        motor_offsets[env_ids] = 0.0
