from __future__ import annotations

import math

import torch


def go2_body_up_alignment_from_projected_gravity(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """Return body-up alignment from Isaac projected gravity in the body frame.

    Isaac projected gravity is the gravity direction expressed in the body frame.
    For an upright base this is approximately ``[0, 0, -1]``, so ``-z`` is
    ``dot(base_z_axis_world, world_up)``.
    """
    if not isinstance(projected_gravity_b, torch.Tensor):
        projected_gravity_b = torch.as_tensor(projected_gravity_b, dtype=torch.float32)
    if projected_gravity_b.shape[-1] != 3:
        raise ValueError(
            f"projected_gravity_b must have last dimension 3, got shape {tuple(projected_gravity_b.shape)}."
        )
    return -projected_gravity_b[..., 2]


def go2_bad_orientation_mask_from_projected_gravity(
    projected_gravity_b: torch.Tensor,
    *,
    body_up_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bad-orientation mask and body-up alignment."""
    threshold = float(body_up_threshold)
    if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
        raise ValueError(f"body_up_threshold must be finite and in [-1, 1], got {body_up_threshold}.")
    alignment = go2_body_up_alignment_from_projected_gravity(projected_gravity_b)
    return alignment < threshold, alignment


def go2_update_bad_orientation_hysteresis(
    raw_bad_orientation: torch.Tensor,
    counts: torch.Tensor,
    *,
    hysteresis_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update consecutive-frame bad-orientation counters and return terminal mask."""
    steps = int(hysteresis_steps)
    if steps <= 0:
        raise ValueError(f"hysteresis_steps must be positive, got {hysteresis_steps}.")
    if counts.ndim == 0:
        raise ValueError("counts must be at least 1-D.")
    raw = torch.as_tensor(raw_bad_orientation, dtype=torch.bool, device=counts.device)
    if raw.shape != counts.shape:
        raise ValueError(
            f"raw_bad_orientation and counts must have matching shapes, got {raw.shape} and {counts.shape}."
        )
    next_counts = torch.where(raw, torch.clamp(counts + 1, max=steps), torch.zeros_like(counts))
    return next_counts >= steps, next_counts
