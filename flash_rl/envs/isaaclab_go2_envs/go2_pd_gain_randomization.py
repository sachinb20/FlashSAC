from __future__ import annotations

from collections.abc import Mapping

import torch

PD_GAIN_FIELDS = ("stiffness", "damping")


def pd_scale_range_is_default(scale_range: tuple[float, float], tol: float = 1.0e-7) -> bool:
    low, high = (float(scale_range[0]), float(scale_range[1]))
    return abs(low - 1.0) <= tol and abs(high - 1.0) <= tol


def restore_pd_gain_defaults(
    actuators: Mapping[str, object],
    default_actuator_state: Mapping[str, dict[str, torch.Tensor]],
    env_ids: torch.Tensor,
) -> None:
    for name, actuator in actuators.items():
        state = default_actuator_state[name]
        for field in PD_GAIN_FIELDS:
            if field not in state:
                continue
            gains = getattr(actuator, field)
            gains[env_ids] = state[field][env_ids].to(gains.device)


def randomize_pd_gains(
    actuators: Mapping[str, object],
    default_actuator_state: Mapping[str, dict[str, torch.Tensor]],
    env_ids: torch.Tensor,
    kp_scale_range: tuple[float, float],
    kd_scale_range: tuple[float, float],
) -> None:
    """Scale each resetting env's per-joint Kp/Kd off the cached default gains.

    Mirrors flash_rl/envs/genesis_envs/go2_base.py's _randomize_kp/_randomize_kd: an
    independent uniform scale per (env, joint), always applied to the *default* gains rather
    than compounding on the previous episode's values.

    Only meaningful for explicit actuators (DCMotor / DelayedPDActuator and the subclasses in
    this package), which evaluate ``stiffness * error_pos + damping * error_vel`` in Python
    each step -- so mutating the tensors here changes the torque the very next step, with no
    write back into PhysX required.
    """
    for name, actuator in actuators.items():
        state = default_actuator_state[name]
        for field, scale_range in zip(PD_GAIN_FIELDS, (kp_scale_range, kd_scale_range)):
            if field not in state or pd_scale_range_is_default(tuple(scale_range)):
                continue
            low, high = (float(scale_range[0]), float(scale_range[1]))
            gains = getattr(actuator, field)
            scales = torch.empty(
                (len(env_ids), gains.shape[1]),
                dtype=gains.dtype,
                device=gains.device,
            ).uniform_(low, high)
            gains[env_ids] = state[field][env_ids].to(gains.device) * scales
