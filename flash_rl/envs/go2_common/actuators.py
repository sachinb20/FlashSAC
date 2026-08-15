"""Torque-speed envelopes for the Go2 motors.

Ported from TDMPC2's ``tdmpc2/envs/go2_actuator_math.py``. That file is already pure
``torch`` with no IsaacLab dependency, so it drops into the shared core unchanged and the
Genesis arm can run the same law.

Why this can live in the shared PD law rather than behind an IsaacLab actuator:
TDMPC2 drives the robot with ``set_joint_position_target`` and lets
``MotorStrengthUnitreeActuator`` (a ``DelayedPDActuator``) compute the torque. With the
config it actually uses, that actuator reduces to

    tau = clip_envelope(Kp * (q_target - q) + Kd * (0 - qd), qd)

because ``_make_unitree_go2hv_actuator_cfg`` copies fields off a ``DCMotorCfg``, which
defines no ``min_delay``/``max_delay``, so the delay buffers default to zero length; and
``Fs``/``Fd`` (the explicit friction terms) default to 0. The actuator recomputes every
physics substep, exactly as ``Go2BaseEnv.step`` recomputes ``_compute_torques`` every
decimation substep at the same 200 Hz. The two are therefore the same computation, and
implementing the envelope here keeps one control path instead of two.
"""

from __future__ import annotations

import torch

ACTUATOR_EXPLICIT_PD = "explicit_pd_unclipped"
ACTUATOR_DC_MOTOR = "dc_motor"
ACTUATOR_UNITREE_GO2HV = "unitree_go2hv"
ACTUATOR_MODELS = (ACTUATOR_EXPLICIT_PD, ACTUATOR_DC_MOTOR, ACTUATOR_UNITREE_GO2HV)

# Unitree Go2HV envelope, from TDMPC2's UnitreeActuatorCfg_Go2HV.
GO2HV_X1 = 13.5  # rad/s: full torque available up to here
GO2HV_X2 = 30.0  # rad/s: no-load speed, limit has decayed to zero
GO2HV_Y1 = 20.2  # N.m: peak torque when torque and velocity share a sign (driving)
GO2HV_Y2 = 23.4  # N.m: peak torque when they oppose (braking)

# IsaacLab DCMotorCfg values carried by UNITREE_GO2_CFG.
DC_EFFORT_LIMIT = 23.5
DC_SATURATION_EFFORT = 23.5
DC_VELOCITY_LIMIT = 30.0


def unitree_go2hv_effort_limit(
    requested_effort: torch.Tensor,
    joint_vel: torch.Tensor,
    x1: float = GO2HV_X1,
    x2: float = GO2HV_X2,
    y1: float = GO2HV_Y1,
    y2: float = GO2HV_Y2,
) -> torch.Tensor:
    """Absolute effort ceiling for the requested torque at this joint velocity.

    Flat at ``y1``/``y2`` up to ``x1``, then linear to zero at ``x2``. The limit depends on
    whether the motor is driving or braking, which is why ``requested_effort`` is an input
    and not just its magnitude.
    """
    same_direction = requested_effort * joint_vel > 0.0
    full_speed_limit = torch.where(
        same_direction,
        torch.full_like(requested_effort, y1),
        torch.full_like(requested_effort, y2),
    )
    abs_vel = joint_vel.abs()
    decay_fraction = (abs_vel - x1) / (x2 - x1)
    decayed_limit = full_speed_limit * (1.0 - decay_fraction)
    limit = torch.where(abs_vel < x1, full_speed_limit, decayed_limit)
    return limit.clamp(min=0.0)


def unitree_go2hv_clip_effort(
    requested_effort: torch.Tensor,
    joint_vel: torch.Tensor,
    x1: float = GO2HV_X1,
    x2: float = GO2HV_X2,
    y1: float = GO2HV_Y1,
    y2: float = GO2HV_Y2,
) -> torch.Tensor:
    limit = unitree_go2hv_effort_limit(requested_effort, joint_vel, x1=x1, x2=x2, y1=y1, y2=y2)
    return torch.clamp(requested_effort, min=-limit, max=limit)


def dc_motor_clip_effort(
    requested_effort: torch.Tensor,
    joint_vel: torch.Tensor,
    effort_limit: float = DC_EFFORT_LIMIT,
    saturation_effort: float = DC_SATURATION_EFFORT,
    velocity_limit: float = DC_VELOCITY_LIMIT,
) -> torch.Tensor:
    """IsaacLab's ``DCMotor`` envelope -- a linear torque-speed line clamped to +/-effort.

    This is the actuator ``UNITREE_GO2_CFG`` ships and TDMPC2's cfg-class default. Kept
    alongside the go2hv model so the two can be bisected against each other.
    """
    vel_at_effort_limit = velocity_limit * (1.0 + effort_limit / saturation_effort)
    clipped_vel = joint_vel.clamp(min=-vel_at_effort_limit, max=vel_at_effort_limit)
    torque_speed_top = saturation_effort * (1.0 - clipped_vel / velocity_limit)
    torque_speed_bottom = saturation_effort * (-1.0 - clipped_vel / velocity_limit)
    max_effort = torch.minimum(torque_speed_top, torch.full_like(clipped_vel, effort_limit))
    min_effort = torch.maximum(torque_speed_bottom, torch.full_like(clipped_vel, -effort_limit))
    return torch.clamp(requested_effort, min=min_effort, max=max_effort)


def clip_effort(model: str, requested_effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    """Apply the configured torque ceiling. ``explicit_pd_unclipped`` applies none.

    No ceiling is the Genesis behaviour: ``go2_base.py`` reads ``torque_limits`` once and
    never uses it, and ``control_dofs_force`` does not enforce the URDF's limits either.
    """
    if model == ACTUATOR_EXPLICIT_PD:
        return requested_effort
    if model == ACTUATOR_UNITREE_GO2HV:
        return unitree_go2hv_clip_effort(requested_effort, joint_vel)
    if model == ACTUATOR_DC_MOTOR:
        return dc_motor_clip_effort(requested_effort, joint_vel)
    raise ValueError(f"Unknown actuator_model {model!r}, expected one of {ACTUATOR_MODELS}.")
