from __future__ import annotations

import torch


def _as_tensor_like(value, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def dc_motor_effort_bounds(
    joint_vel,
    effort_limit=23.5,
    saturation_effort=23.5,
    velocity_limit=30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Isaac Lab DCMotor lower and upper effort bounds for Go2-style tensors."""
    joint_vel = torch.as_tensor(joint_vel)
    if not torch.is_floating_point(joint_vel):
        joint_vel = joint_vel.to(dtype=torch.float32)
    effort_limit = _as_tensor_like(effort_limit, joint_vel)
    saturation_effort = _as_tensor_like(saturation_effort, joint_vel)
    velocity_limit = _as_tensor_like(velocity_limit, joint_vel)

    vel_at_effort_limit = velocity_limit * (1.0 + effort_limit / saturation_effort)
    clipped_vel = torch.maximum(torch.minimum(joint_vel, vel_at_effort_limit), -vel_at_effort_limit)
    torque_speed_top = saturation_effort * (1.0 - clipped_vel / velocity_limit)
    torque_speed_bottom = saturation_effort * (-1.0 - clipped_vel / velocity_limit)
    max_effort = torch.minimum(torque_speed_top, effort_limit)
    min_effort = torch.maximum(torque_speed_bottom, -effort_limit)
    return min_effort, max_effort


def unitree_go2hv_effort_limit(
    requested_effort,
    joint_vel,
    x1=13.5,
    x2=30.0,
    y1=20.2,
    y2=23.4,
) -> torch.Tensor:
    """Return the Unitree Go2HV absolute effort limit for requested effort and joint velocity."""
    requested_effort = torch.as_tensor(requested_effort)
    if not torch.is_floating_point(requested_effort):
        requested_effort = requested_effort.to(dtype=torch.float32)
    joint_vel = torch.as_tensor(joint_vel, dtype=requested_effort.dtype, device=requested_effort.device)
    x1 = _as_tensor_like(x1, requested_effort)
    x2 = _as_tensor_like(x2, requested_effort)
    y1 = _as_tensor_like(y1, requested_effort)
    y2 = _as_tensor_like(y2, requested_effort)

    same_direction = requested_effort * joint_vel > 0.0
    full_speed_limit = torch.where(same_direction, y1, y2)
    abs_vel = joint_vel.abs()
    decay_fraction = (abs_vel - x1) / (x2 - x1)
    decayed_limit = full_speed_limit * (1.0 - decay_fraction)
    limit = torch.where(abs_vel < x1, full_speed_limit, decayed_limit)
    return limit.clamp(min=0.0)


def unitree_go2hv_clip_effort(
    requested_effort,
    joint_vel,
    x1=13.5,
    x2=30.0,
    y1=20.2,
    y2=23.4,
) -> torch.Tensor:
    """Clip requested effort with the Unitree Go2HV torque-speed envelope."""
    requested_effort = torch.as_tensor(requested_effort)
    if not torch.is_floating_point(requested_effort):
        requested_effort = requested_effort.to(dtype=torch.float32)
    joint_vel = torch.as_tensor(joint_vel, dtype=requested_effort.dtype, device=requested_effort.device)
    limit = unitree_go2hv_effort_limit(requested_effort, joint_vel, x1=x1, x2=x2, y1=y1, y2=y2)
    return torch.clamp(requested_effort, min=-limit, max=limit)
