from __future__ import annotations

import torch

ACTION_DECODER_SCALAR = "scalar"
ACTION_DECODER_DEFAULT_CENTERED_SOFT_LIMITS = "default_centered_soft_limits"
ACTION_DECODER_SAC_AFFINE_SOFT_LIMITS = "sac_affine_soft_limits"

ACTION_DECODER_MODES = (
    ACTION_DECODER_SCALAR,
    ACTION_DECODER_DEFAULT_CENTERED_SOFT_LIMITS,
    ACTION_DECODER_SAC_AFFINE_SOFT_LIMITS,
)

ACTION_RATE_AUTO = "auto"
ACTION_RATE_RAW_RESCALED = "raw_rescaled"
ACTION_RATE_DECODED_TARGET_DELTA = "decoded_target_delta"

ACTION_RATE_MODES = (
    ACTION_RATE_AUTO,
    ACTION_RATE_RAW_RESCALED,
    ACTION_RATE_DECODED_TARGET_DELTA,
)

RANDOM_ACTION_CENTER_ZERO = "zero"
RANDOM_ACTION_CENTER_NEUTRAL = "neutral"

RANDOM_ACTION_CENTERS = (
    RANDOM_ACTION_CENTER_ZERO,
    RANDOM_ACTION_CENTER_NEUTRAL,
)

ACTION_DECODER_MODE_IDS = {
    ACTION_DECODER_SCALAR: 0,
    ACTION_DECODER_DEFAULT_CENTERED_SOFT_LIMITS: 1,
    ACTION_DECODER_SAC_AFFINE_SOFT_LIMITS: 2,
}

ACTION_RATE_MODE_IDS = {
    ACTION_RATE_AUTO: -1,
    ACTION_RATE_RAW_RESCALED: 0,
    ACTION_RATE_DECODED_TARGET_DELTA: 1,
}


def validate_action_decoder_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in ACTION_DECODER_MODES:
        raise ValueError(f"Unsupported Go2 action decoder {mode!r}; expected one of {ACTION_DECODER_MODES}.")
    return mode


def validate_action_rate_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in ACTION_RATE_MODES:
        raise ValueError(f"Unsupported Go2 action-rate mode {mode!r}; expected one of {ACTION_RATE_MODES}.")
    return mode


def validate_random_action_center(center: str) -> str:
    center = str(center)
    if center not in RANDOM_ACTION_CENTERS:
        raise ValueError(f"Unsupported Isaac random action center {center!r}; expected one of {RANDOM_ACTION_CENTERS}.")
    return center


def resolve_action_rate_mode(action_decoder: str, action_rate_mode: str) -> str:
    action_decoder = validate_action_decoder_mode(action_decoder)
    action_rate_mode = validate_action_rate_mode(action_rate_mode)
    if action_rate_mode != ACTION_RATE_AUTO:
        return action_rate_mode
    if action_decoder == ACTION_DECODER_SCALAR:
        return ACTION_RATE_RAW_RESCALED
    return ACTION_RATE_DECODED_TARGET_DELTA


def _split_soft_limits(soft_joint_pos_limits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if soft_joint_pos_limits.shape[-1] != 2:
        raise ValueError(
            "soft_joint_pos_limits must have a final dimension of size 2 "
            f"for lower/upper limits, got shape {tuple(soft_joint_pos_limits.shape)}."
        )
    return soft_joint_pos_limits[..., 0], soft_joint_pos_limits[..., 1]


def _clamp_to_soft_limits(target: torch.Tensor, soft_joint_pos_limits: torch.Tensor) -> torch.Tensor:
    soft_lower, soft_upper = _split_soft_limits(soft_joint_pos_limits)
    return torch.max(torch.min(target, soft_upper), soft_lower)


def decode_go2_action(
    raw_action: torch.Tensor,
    default_joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    mode: str,
    action_scale: float = 0.85,
    clip_joint_targets: bool = False,
) -> torch.Tensor:
    """Decode raw TD-MPC2 actions in [-1, 1] into Go2 joint-position targets."""
    mode = validate_action_decoder_mode(mode)
    u = torch.as_tensor(raw_action, device=default_joint_pos.device, dtype=default_joint_pos.dtype).clamp(-1.0, 1.0)
    soft_lower, soft_upper = _split_soft_limits(soft_joint_pos_limits)

    if mode == ACTION_DECODER_SCALAR:
        target = default_joint_pos + float(action_scale) * u
    elif mode == ACTION_DECODER_DEFAULT_CENTERED_SOFT_LIMITS:
        lower_delta = default_joint_pos - soft_lower
        upper_delta = soft_upper - default_joint_pos
        target = default_joint_pos + torch.where(u >= 0.0, u * upper_delta, u * lower_delta)
    elif mode == ACTION_DECODER_SAC_AFFINE_SOFT_LIMITS:
        soft_mid = 0.5 * (soft_lower + soft_upper)
        soft_half = 0.5 * (soft_upper - soft_lower)
        target = soft_mid + soft_half * u
    else:
        raise AssertionError(f"validated unknown decoder mode {mode!r}")

    if clip_joint_targets:
        target = _clamp_to_soft_limits(target, soft_joint_pos_limits)
    return target


def go2_neutral_action(
    default_joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    mode: str,
    action_scale: float = 0.85,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return the raw action vector that decodes to the default joint position."""
    mode = validate_action_decoder_mode(mode)
    if mode in {ACTION_DECODER_SCALAR, ACTION_DECODER_DEFAULT_CENTERED_SOFT_LIMITS}:
        return torch.zeros_like(default_joint_pos)

    soft_lower, soft_upper = _split_soft_limits(soft_joint_pos_limits)
    soft_half = 0.5 * (soft_upper - soft_lower)
    if torch.any(torch.abs(soft_half) <= float(eps)):
        raise ValueError("Cannot compute Go2 neutral action because at least one soft joint range is degenerate.")
    soft_mid = 0.5 * (soft_lower + soft_upper)
    return (default_joint_pos - soft_mid) / soft_half


def neutral_action_pre_tanh_bias(neutral_action: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Return atanh(neutral_action) for initializing a tanh-squashed policy mean."""
    neutral_action = neutral_action.clamp(-1.0 + float(eps), 1.0 - float(eps))
    return torch.atanh(neutral_action)


def decoded_action_rate_l2(
    current_target: torch.Tensor,
    previous_target: torch.Tensor,
    reference_scale: float = 0.25,
) -> torch.Tensor:
    """Return sum(((q_t - q_{t-1}) / reference_scale) ** 2) over joints."""
    reference_scale = float(reference_scale)
    if reference_scale <= 0.0:
        raise ValueError(f"reference_scale must be positive, got {reference_scale}.")
    q_delta = current_target - previous_target
    return torch.sum(torch.square(q_delta / reference_scale), dim=-1)
