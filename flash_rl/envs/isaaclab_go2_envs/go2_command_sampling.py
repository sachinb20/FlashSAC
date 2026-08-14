from __future__ import annotations

import torch


def _uniform(shape: tuple[int, ...], value_range: tuple[float, float], *, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(float(value_range[0]), float(value_range[1]))


def apply_go2_command_deadband(
    commands: torch.Tensor,
    lin_vel_deadband: float,
    ang_vel_deadband: float,
) -> torch.Tensor:
    """Zero commands whose magnitude falls under a threshold, in place.

    Mirrors genesis_envs/go2_base.py's `_resample_commands`, which drops the planar command
    when ||(vx, vy)|| is below the threshold and the yaw command when |wz| is below it, the two
    decided independently:

        commands[:, :2] *= (norm(commands[:, :2]) > 0.2)
        commands[:, 2]  *= (abs(commands[:, 2]) > 0.2)

    With uniform commands over [-1, 1] and a 0.2 threshold this zeroes the yaw command on ~20%
    of resamples and the planar command on ~3%, so a meaningful share of genesis's training
    distribution is "hold still on this axis" -- distinct from a whole-env standing mode, which
    zeroes all three axes together.
    """
    if float(lin_vel_deadband) > 0.0:
        keep_lin = torch.norm(commands[:, :2], dim=1) > float(lin_vel_deadband)
        commands[:, :2] *= keep_lin.unsqueeze(1)
    if float(ang_vel_deadband) > 0.0:
        keep_ang = torch.abs(commands[:, 2]) > float(ang_vel_deadband)
        commands[:, 2] *= keep_ang
    return commands


def sample_go2_velocity_commands(
    count: int,
    *,
    device: torch.device,
    lin_vel_x_range: tuple[float, float],
    lin_vel_y_range: tuple[float, float],
    ang_vel_z_range: tuple[float, float],
    yaw_in_place_ang_vel_z_range: tuple[float, float],
    heading_range: tuple[float, float],
    command_resampling_time_range: tuple[float, float],
    heading_command: bool,
    rel_standing_envs: float,
    rel_yaw_in_place_envs: float,
    rel_heading_envs: float,
) -> dict[str, torch.Tensor]:
    """Sample mutually-exclusive Go2 direct velocity command modes."""
    count = int(count)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}.")
    commands = torch.zeros(count, 3, dtype=torch.float32, device=device)
    command_time_left = _uniform((count,), command_resampling_time_range, device=device)
    is_standing = torch.zeros(count, dtype=torch.bool, device=device)
    is_yaw_in_place = torch.zeros(count, dtype=torch.bool, device=device)
    is_heading = torch.zeros(count, dtype=torch.bool, device=device)
    heading_targets = torch.zeros(count, dtype=torch.float32, device=device)
    if count == 0:
        return {
            "commands": commands,
            "command_time_left": command_time_left,
            "heading_targets": heading_targets,
            "is_standing": is_standing,
            "is_yaw_in_place": is_yaw_in_place,
            "is_heading": is_heading,
        }

    commands[:, 0] = _uniform((count,), lin_vel_x_range, device=device)
    commands[:, 1] = _uniform((count,), lin_vel_y_range, device=device)
    commands[:, 2] = _uniform((count,), ang_vel_z_range, device=device)

    is_standing = _uniform((count,), (0.0, 1.0), device=device) <= float(rel_standing_envs)
    available = ~is_standing
    if float(rel_yaw_in_place_envs) > 0.0 and available.any():
        yaw_draw = _uniform((int(available.sum().item()),), (0.0, 1.0), device=device) <= float(rel_yaw_in_place_envs)
        yaw_indices = available.nonzero(as_tuple=False).flatten()[yaw_draw]
        if len(yaw_indices) > 0:
            is_yaw_in_place[yaw_indices] = True
            commands[yaw_indices, 0] = 0.0
            commands[yaw_indices, 1] = 0.0
            commands[yaw_indices, 2] = _uniform((len(yaw_indices),), yaw_in_place_ang_vel_z_range, device=device)
    available = ~(is_standing | is_yaw_in_place)
    if bool(heading_command) and available.any():
        heading_draw = _uniform((int(available.sum().item()),), (0.0, 1.0), device=device) <= float(rel_heading_envs)
        heading_indices = available.nonzero(as_tuple=False).flatten()[heading_draw]
        if len(heading_indices) > 0:
            is_heading[heading_indices] = True
            heading_targets[heading_indices] = _uniform((len(heading_indices),), heading_range, device=device)

    commands[is_standing] = 0.0
    return {
        "commands": commands,
        "command_time_left": command_time_left,
        "heading_targets": heading_targets,
        "is_standing": is_standing,
        "is_yaw_in_place": is_yaw_in_place,
        "is_heading": is_heading,
    }
