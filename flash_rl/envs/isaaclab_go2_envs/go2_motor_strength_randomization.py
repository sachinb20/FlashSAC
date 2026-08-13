from __future__ import annotations

from collections.abc import Callable, Mapping

import torch


def motor_strength_range_is_default(strength_range: tuple[float, float], tol: float = 1.0e-7) -> bool:
    low, high = (float(strength_range[0]), float(strength_range[1]))
    return abs(low - 1.0) <= tol and abs(high - 1.0) <= tol


def as_actuator_tensor(value, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device=like.device, dtype=like.dtype)
    return torch.full_like(like, float(value))


def restore_motor_strength_defaults(
    actuators: Mapping[str, object],
    default_actuator_state: Mapping[str, dict[str, torch.Tensor]],
    env_ids: torch.Tensor,
    refresh_velocity_limit: Callable[[object], None] | None = None,
) -> None:
    for name, actuator in actuators.items():
        state = default_actuator_state[name]
        actuator.effort_limit[env_ids] = state["effort_limit"][env_ids].to(actuator.effort_limit.device)
        if "saturation_effort" in state:
            saturation = as_actuator_tensor(actuator._saturation_effort, actuator.effort_limit)
            saturation[env_ids] = state["saturation_effort"][env_ids].to(saturation.device)
            actuator._saturation_effort = saturation
        if refresh_velocity_limit is not None:
            refresh_velocity_limit(actuator)
        if "motor_strength" in state:
            actuator.motor_strength[env_ids] = state["motor_strength"][env_ids].to(actuator.motor_strength.device)
        if "scaled_motor_strength" in state:
            actuator.scaled_motor_strength[env_ids] = state["scaled_motor_strength"][env_ids].to(
                actuator.scaled_motor_strength.device
            )


def randomize_motor_strength(
    actuators: Mapping[str, object],
    default_actuator_state: Mapping[str, dict[str, torch.Tensor]],
    env_ids: torch.Tensor,
    strength_range: tuple[float, float],
    per_joint: bool,
    refresh_velocity_limit: Callable[[object], None] | None = None,
    require_motor_strength: bool = False,
) -> torch.Tensor | None:
    low, high = (float(strength_range[0]), float(strength_range[1]))
    env_strength_values = []
    for name, actuator in actuators.items():
        state = default_actuator_state[name]
        actuator.effort_limit[env_ids] = state["effort_limit"][env_ids].to(actuator.effort_limit.device)
        if "saturation_effort" in state:
            saturation = as_actuator_tensor(actuator._saturation_effort, actuator.effort_limit)
            saturation[env_ids] = state["saturation_effort"][env_ids].to(saturation.device)
            actuator._saturation_effort = saturation
        if refresh_velocity_limit is not None:
            refresh_velocity_limit(actuator)
        if not hasattr(actuator, "motor_strength"):
            continue
        if per_joint:
            strength = torch.empty(
                (len(env_ids), actuator.motor_strength.shape[1]),
                dtype=actuator.motor_strength.dtype,
                device=actuator.motor_strength.device,
            ).uniform_(low, high)
        else:
            strength = torch.empty(
                (len(env_ids), 1),
                dtype=actuator.motor_strength.dtype,
                device=actuator.motor_strength.device,
            ).uniform_(low, high)
            strength = strength.expand(-1, actuator.motor_strength.shape[1])
        actuator.motor_strength[env_ids] = strength
        if hasattr(actuator, "scaled_motor_strength"):
            if low == high:
                actuator.scaled_motor_strength[env_ids] = 0.0
            else:
                actuator.scaled_motor_strength[env_ids] = 2.0 * (strength - low) / (high - low) - 1.0
        env_strength_values.append(strength.detach())
    if not env_strength_values:
        if require_motor_strength:
            raise RuntimeError(
                "Motor-strength domain randomization is enabled with a non-default range, "
                "but no Go2 actuator exposes a motor_strength tensor."
            )
        return None
    return torch.cat(env_strength_values, dim=1)
