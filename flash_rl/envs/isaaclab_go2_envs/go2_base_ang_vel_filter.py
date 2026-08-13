from __future__ import annotations

import torch


def base_ang_vel_with_optional_ema(
    raw_base_ang_vel: torch.Tensor,
    filtered_base_ang_vel: torch.Tensor,
    alpha: float | None,
    *,
    commit: bool,
) -> torch.Tensor:
    """Return policy-input base angular velocity and optionally commit EMA state."""
    if alpha is None:
        return raw_base_ang_vel
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"base angular velocity EMA alpha must be in (0, 1], got {alpha}.")
    next_filtered = filtered_base_ang_vel * (1.0 - alpha) + raw_base_ang_vel * alpha
    if commit:
        filtered_base_ang_vel.copy_(next_filtered)
        return filtered_base_ang_vel.clone()
    return next_filtered


def reset_base_ang_vel_ema_state(
    filtered_base_ang_vel: torch.Tensor,
    last_policy_base_ang_vel: torch.Tensor,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Clear committed EMA/debug state for all envs or a subset."""
    if env_ids is None:
        filtered_base_ang_vel.zero_()
        last_policy_base_ang_vel.zero_()
        return
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=filtered_base_ang_vel.device).reshape(-1)
    if env_ids.numel() == 0:
        return
    filtered_base_ang_vel[env_ids] = 0.0
    last_policy_base_ang_vel[env_ids] = 0.0
