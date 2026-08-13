"""FlashSAC-original Go2 reward terms.

Ported from flash_rl/envs/genesis_envs/go2_walk.py (FlashSAC's own hand-written Go2
reward, tuned for the Genesis sim) and re-expressed against IsaacLab's
Articulation / ContactSensor state instead of Genesis's. This deliberately does NOT
reuse TDMPC2's sim-to-real reward shaping (go2_reward_terms.py in TDMPC2_isaaclab) --
only the sim, robot, actuator, and terrain are ported from TDMPC2; the reward is
FlashSAC's own.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def go2_flashsac_reward_scales(
    *,
    tracking_lin_vel: float = 1.0,
    tracking_ang_vel: float = 0.5,
    lin_vel_z: float = -2.0,
    ang_vel_xy: float = -0.05,
    orientation: float = -10.0,
    base_height: float = -50.0,
    torques: float = -0.0002,
    dof_vel: float = 0.0,
    dof_acc: float = -2.5e-7,
    action_rate: float = -0.01,
    feet_air_time: float = 1.0,
    collision: float = -1.0,
    dof_pos_limits: float = 0.0,
    termination: float = 0.0,
) -> dict[str, float]:
    """Return the FlashSAC Go2 reward-term scales (mirrors go2_walk.py's reward_cfg["reward_scales"])."""
    return {
        "tracking_lin_vel": float(tracking_lin_vel),
        "tracking_ang_vel": float(tracking_ang_vel),
        "lin_vel_z": float(lin_vel_z),
        "ang_vel_xy": float(ang_vel_xy),
        "orientation": float(orientation),
        "base_height": float(base_height),
        "torques": float(torques),
        "dof_vel": float(dof_vel),
        "dof_acc": float(dof_acc),
        "action_rate": float(action_rate),
        "feet_air_time": float(feet_air_time),
        "collision": float(collision),
        "dof_pos_limits": float(dof_pos_limits),
        "termination": float(termination),
    }


def go2_tracking_lin_vel_reward(
    commands: torch.Tensor,
    base_lin_vel_b: torch.Tensor,
    tracking_sigma: float = 0.25,
) -> torch.Tensor:
    """Tracking of linear velocity commands (xy axes)."""
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / float(tracking_sigma))


def go2_tracking_ang_vel_reward(
    commands: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    tracking_sigma: float = 0.25,
) -> torch.Tensor:
    """Tracking of angular velocity commands (yaw)."""
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / float(tracking_sigma))


def go2_lin_vel_z_penalty(base_lin_vel_b: torch.Tensor) -> torch.Tensor:
    """Penalize z axis base linear velocity."""
    return torch.square(base_lin_vel_b[:, 2])


def go2_ang_vel_xy_penalty(base_ang_vel_b: torch.Tensor) -> torch.Tensor:
    """Penalize xy axes base angular velocity."""
    return torch.sum(torch.square(base_ang_vel_b[:, :2]), dim=1)


def go2_orientation_penalty(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)


def go2_torques_penalty(applied_torque: torch.Tensor) -> torch.Tensor:
    """Penalize torques."""
    return torch.sum(torch.square(applied_torque), dim=1)


def go2_dof_vel_penalty(joint_vel: torch.Tensor) -> torch.Tensor:
    """Penalize dof velocities."""
    return torch.sum(torch.square(joint_vel), dim=1)


def go2_dof_acc_penalty(joint_acc: torch.Tensor) -> torch.Tensor:
    """Penalize dof accelerations (IsaacLab tracks this natively, unlike Genesis's finite difference)."""
    return torch.sum(torch.square(joint_acc), dim=1)


def go2_action_rate_penalty(actions: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
    """Penalize changes in raw policy actions."""
    return torch.sum(torch.square(previous_actions - actions), dim=1)


def go2_base_height_penalty(base_pos_z: torch.Tensor, base_height_target: float) -> torch.Tensor:
    """Penalize base height away from target."""
    return torch.square(base_pos_z - float(base_height_target))


def go2_collision_penalty(penalized_contact_force_norm: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Penalize contact on penalized bodies (base/thigh/calf).

    penalized_contact_force_norm: (num_envs, num_penalized_bodies), max-over-history force norm.
    """
    if penalized_contact_force_norm.shape[-1] == 0:
        return torch.zeros(penalized_contact_force_norm.shape[0], device=penalized_contact_force_norm.device)
    return torch.sum((penalized_contact_force_norm > float(threshold)).to(dtype=torch.float32), dim=1)


def go2_dof_pos_limits_penalty(joint_pos: torch.Tensor, soft_joint_pos_limits: torch.Tensor) -> torch.Tensor:
    """Penalize dof positions too close to the soft joint limits."""
    out_of_limits = -(joint_pos - soft_joint_pos_limits[..., 0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_joint_pos_limits[..., 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def go2_feet_air_time_reward(
    last_air_time: torch.Tensor,
    first_contact: torch.Tensor,
    commands: torch.Tensor,
    *,
    air_time_offset: float = 0.5,
    command_lin_vel_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward long steps; gated off for near-zero linear velocity commands."""
    reward = torch.sum((last_air_time - float(air_time_offset)) * first_contact, dim=1)
    reward = reward * (torch.linalg.norm(commands[:, :2], dim=1) > float(command_lin_vel_threshold))
    return reward


def go2_termination_penalty(terminated: torch.Tensor) -> torch.Tensor:
    """Terminal reward/penalty: 1.0 on envs that terminated (excludes time-outs)."""
    return terminated.to(dtype=torch.float32)


def go2_flashsac_reward_terms(
    *,
    commands: torch.Tensor,
    base_lin_vel_b: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    applied_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_acc: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    base_pos_z: torch.Tensor,
    penalized_contact_force_norm: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    last_air_time: torch.Tensor,
    first_contact: torch.Tensor,
    terminated: torch.Tensor,
    tracking_sigma: float = 0.25,
    base_height_target: float = 0.3,
    collision_force_threshold: float = 0.1,
    feet_air_time_offset: float = 0.5,
    feet_air_time_command_threshold: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Return all unweighted FlashSAC Go2 reward terms, keyed like go2_flashsac_reward_scales()."""
    return {
        "tracking_lin_vel": go2_tracking_lin_vel_reward(commands, base_lin_vel_b, tracking_sigma),
        "tracking_ang_vel": go2_tracking_ang_vel_reward(commands, base_ang_vel_b, tracking_sigma),
        "lin_vel_z": go2_lin_vel_z_penalty(base_lin_vel_b),
        "ang_vel_xy": go2_ang_vel_xy_penalty(base_ang_vel_b),
        "orientation": go2_orientation_penalty(projected_gravity_b),
        "base_height": go2_base_height_penalty(base_pos_z, base_height_target),
        "torques": go2_torques_penalty(applied_torque),
        "dof_vel": go2_dof_vel_penalty(joint_vel),
        "dof_acc": go2_dof_acc_penalty(joint_acc),
        "action_rate": go2_action_rate_penalty(actions, previous_actions),
        "feet_air_time": go2_feet_air_time_reward(
            last_air_time,
            first_contact,
            commands,
            air_time_offset=feet_air_time_offset,
            command_lin_vel_threshold=feet_air_time_command_threshold,
        ),
        "collision": go2_collision_penalty(penalized_contact_force_norm, collision_force_threshold),
        "dof_pos_limits": go2_dof_pos_limits_penalty(joint_pos, soft_joint_pos_limits),
        "termination": go2_termination_penalty(terminated),
    }


def go2_flashsac_scalar_reward(
    terms: Mapping[str, torch.Tensor],
    scales: Mapping[str, float],
    step_dt: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply per-term scale * dt (dropping zero-scaled terms) and sum to a scalar reward.

    Mirrors go2_base.py's `_prepare_reward_function`/`compute_reward` convention: every
    term except "termination" is scaled by dt; "termination" is applied at full scale as
    a one-shot penalty on the terminating step. Returns (scalar_reward, weighted_terms),
    where weighted_terms is suitable for per-term episode-sum logging.
    """
    weighted: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for name, value in terms.items():
        scale = float(scales.get(name, 0.0))
        if scale == 0.0:
            continue
        weight = scale if name == "termination" else scale * float(step_dt)
        weighted_term = value * weight
        weighted[name] = weighted_term
        total = weighted_term if total is None else total + weighted_term
    if total is None:
        raise ValueError("No non-zero-scaled Go2 reward terms selected.")
    return total, weighted
