from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

GO2_TERRAIN_MODE_FLAT = "flat"
GO2_TERRAIN_MODE_ROUGH_MEDIUM = "rough_medium"
GO2_TERRAIN_MODE_ROUGH_HARD = "rough_hard"
GO2_TERRAIN_MODES = frozenset(
    {
        GO2_TERRAIN_MODE_FLAT,
        GO2_TERRAIN_MODE_ROUGH_MEDIUM,
        GO2_TERRAIN_MODE_ROUGH_HARD,
    }
)

GO2_TERRAIN_PRESET_ROUGH_MEDIUM = "go2_rough_medium"
GO2_TERRAIN_PRESET_ROUGH_HARD = "go2_rough_hard"
GO2_TERRAIN_PRESETS = frozenset(
    {
        GO2_TERRAIN_PRESET_ROUGH_MEDIUM,
        GO2_TERRAIN_PRESET_ROUGH_HARD,
    }
)
GO2_TERRAIN_MODE_TO_PRESET = {
    GO2_TERRAIN_MODE_ROUGH_MEDIUM: GO2_TERRAIN_PRESET_ROUGH_MEDIUM,
    GO2_TERRAIN_MODE_ROUGH_HARD: GO2_TERRAIN_PRESET_ROUGH_HARD,
}

GO2_TERRAIN_CURRICULUM_MODE_DISTANCE = "distance"
GO2_TERRAIN_CURRICULUM_MODE_NONE = "none"
GO2_TERRAIN_CURRICULUM_MODES = frozenset(
    {
        GO2_TERRAIN_CURRICULUM_MODE_DISTANCE,
        GO2_TERRAIN_CURRICULUM_MODE_NONE,
    }
)

GO2_ROUGH_TERRAIN_SUB_TERRAIN_NAMES = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
)

GO2_TERRAIN_DEFAULT_SIZE = (8.0, 8.0)
GO2_TERRAIN_DEFAULT_BORDER_WIDTH = 20.0
GO2_TERRAIN_DEFAULT_HORIZONTAL_SCALE = 0.1
GO2_TERRAIN_DEFAULT_VERTICAL_SCALE = 0.005
GO2_TERRAIN_DEFAULT_SLOPE_THRESHOLD = 0.75
GO2_HEIGHT_SCAN_DEFAULT_SIZE = (1.6, 1.0)
GO2_HEIGHT_SCAN_DEFAULT_RESOLUTION = 0.1
GO2_HEIGHT_SCAN_DEFAULT_VERTICAL_OFFSET = 20.0
GO2_HEIGHT_SCAN_DEFAULT_REFERENCE_OFFSET = 0.5
GO2_HEIGHT_SCAN_DEFAULT_CLIP = (-1.0, 1.0)


@dataclass(frozen=True)
class Go2SubTerrainSpec:
    class_name: str
    proportion: float
    params: Mapping[str, Any]


@dataclass(frozen=True)
class Go2TerrainGeneratorSpec:
    preset: str
    curriculum: bool
    size: tuple[float, float]
    border_width: float
    num_rows: int
    num_cols: int
    horizontal_scale: float
    vertical_scale: float
    slope_threshold: float
    use_cache: bool
    sub_terrains: Mapping[str, Go2SubTerrainSpec]


@dataclass(frozen=True)
class Go2TerrainCurriculumUpdate:
    move_up: torch.Tensor
    move_down: torch.Tensor
    stationary_guard: torch.Tensor
    distance: torch.Tensor
    command_xy_norm: torch.Tensor


def validate_go2_terrain_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in GO2_TERRAIN_MODES:
        raise ValueError(f"Invalid Go2 terrain mode {mode!r}. Expected one of {sorted(GO2_TERRAIN_MODES)}.")
    return mode


def validate_go2_terrain_preset(preset: str) -> str:
    preset = str(preset)
    if preset not in GO2_TERRAIN_PRESETS:
        raise ValueError(f"Invalid Go2 terrain preset {preset!r}. Expected one of {sorted(GO2_TERRAIN_PRESETS)}.")
    return preset


def validate_go2_terrain_curriculum_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in GO2_TERRAIN_CURRICULUM_MODES:
        raise ValueError(
            f"Invalid Go2 terrain curriculum mode {mode!r}. Expected one of {sorted(GO2_TERRAIN_CURRICULUM_MODES)}."
        )
    return mode


def resolve_go2_terrain_preset(
    mode: str,
    preset: str | None,
    *,
    default_preset: str = GO2_TERRAIN_PRESET_ROUGH_MEDIUM,
) -> str:
    mode = validate_go2_terrain_mode(mode)
    if preset is None:
        preset = default_preset
    preset = validate_go2_terrain_preset(preset)
    if mode in GO2_TERRAIN_MODE_TO_PRESET:
        expected_preset = GO2_TERRAIN_MODE_TO_PRESET[mode]
        if preset not in {default_preset, expected_preset}:
            raise ValueError(
                f"Go2 terrain mode {mode!r} is incompatible with terrain preset {preset!r}; "
                f"expected {expected_preset!r}."
            )
        return expected_preset
    return preset


def validate_go2_terrain_positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    if float(value) != float(int(value)):
        raise ValueError(f"{field} must be a positive integer, got {value}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value}.")
    return value


def validate_go2_terrain_nonnegative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
    if float(value) != float(int(value)):
        raise ValueError(f"{field} must be a non-negative integer, got {value}.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value}.")
    return value


def validate_go2_terrain_nonnegative_float(value: float, *, field: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{field} must be non-negative, got {value}.")
    return value


def validate_go2_terrain_positive_float(value: float, *, field: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{field} must be positive, got {value}.")
    return value


def validate_go2_height_scan_size(value, *, field: str = "height_scan_size") -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{field} must contain exactly two values, got {value!r}.")
    length = validate_go2_terrain_positive_float(value[0], field=f"{field}[0]")
    width = validate_go2_terrain_positive_float(value[1], field=f"{field}[1]")
    return (length, width)


def validate_go2_height_scan_clip(value, *, field: str = "height_scan_clip") -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{field} must contain exactly two values, got {value!r}.")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        raise ValueError(f"{field} lower bound must be <= upper bound, got {(low, high)}.")
    return (low, high)


def go2_height_scan_dot_count(
    size: tuple[float, float] = GO2_HEIGHT_SCAN_DEFAULT_SIZE,
    resolution: float = GO2_HEIGHT_SCAN_DEFAULT_RESOLUTION,
) -> int:
    size = validate_go2_height_scan_size(size)
    resolution = validate_go2_terrain_positive_float(resolution, field="height_scan_resolution")
    x_count = int(torch.arange(-size[0] / 2.0, size[0] / 2.0 + 1.0e-9, resolution).numel())
    y_count = int(torch.arange(-size[1] / 2.0, size[1] / 2.0 + 1.0e-9, resolution).numel())
    return x_count * y_count


def go2_direct_velocity_base_observation_dim(
    *,
    observe_base_lin_vel: bool = False,
    height_scan_enabled: bool = False,
    height_scan_observe: bool | None = None,
    height_scan_size: tuple[float, float] = GO2_HEIGHT_SCAN_DEFAULT_SIZE,
    height_scan_resolution: float = GO2_HEIGHT_SCAN_DEFAULT_RESOLUTION,
) -> int:
    base_dim = 48 if bool(observe_base_lin_vel) else 45
    observe_height_scan = bool(height_scan_enabled) if height_scan_observe is None else bool(height_scan_observe)
    if bool(height_scan_enabled) and observe_height_scan:
        base_dim += go2_height_scan_dot_count(height_scan_size, height_scan_resolution)
    return base_dim


def go2_direct_velocity_policy_observation_dim(
    *,
    observe_base_lin_vel: bool = False,
    observation_history_enabled: bool = True,
    observation_history_length: int = 4,
    height_scan_enabled: bool = False,
    height_scan_observe: bool | None = None,
    height_scan_size: tuple[float, float] = GO2_HEIGHT_SCAN_DEFAULT_SIZE,
    height_scan_resolution: float = GO2_HEIGHT_SCAN_DEFAULT_RESOLUTION,
) -> int:
    history_length = validate_go2_terrain_nonnegative_int(
        observation_history_length,
        field="observation_history_length",
    )
    frame_count = history_length + 1 if bool(observation_history_enabled) else 1
    return (
        go2_direct_velocity_base_observation_dim(
            observe_base_lin_vel=observe_base_lin_vel,
            height_scan_enabled=height_scan_enabled,
            height_scan_observe=height_scan_observe,
            height_scan_size=height_scan_size,
            height_scan_resolution=height_scan_resolution,
        )
        * frame_count
    )


def go2_terrain_mode_is_rough(mode: str) -> bool:
    return validate_go2_terrain_mode(mode) != GO2_TERRAIN_MODE_FLAT


def go2_terrain_type_for_mode(mode: str) -> str:
    return "generator" if go2_terrain_mode_is_rough(mode) else "plane"


def go2_terrain_sub_terrain_names_for_mode(mode: str, preset: str | None = None) -> tuple[str, ...]:
    mode = validate_go2_terrain_mode(mode)
    if mode == GO2_TERRAIN_MODE_FLAT:
        return tuple()
    preset = resolve_go2_terrain_preset(mode, preset)
    spec = build_go2_terrain_generator_spec(preset, num_rows=1, num_cols=len(GO2_ROUGH_TERRAIN_SUB_TERRAIN_NAMES))
    return tuple(spec.sub_terrains)


def _sub_terrain_spec(class_name: str, proportion: float, **params) -> Go2SubTerrainSpec:
    proportion = float(proportion)
    if not 0.0 < proportion <= 1.0:
        raise ValueError(f"sub-terrain proportion must be in (0, 1], got {proportion}.")
    return Go2SubTerrainSpec(
        class_name=class_name,
        proportion=proportion,
        params=MappingProxyType(dict(params)),
    )


def _rough_sub_terrains(
    *,
    box_height_range: tuple[float, float],
    random_rough_noise_range: tuple[float, float],
    random_rough_noise_step: float,
    slope_range: tuple[float, float],
    stair_height_range: tuple[float, float],
) -> Mapping[str, Go2SubTerrainSpec]:
    return MappingProxyType(
        {
            "pyramid_stairs": _sub_terrain_spec(
                "MeshPyramidStairsTerrainCfg",
                0.2,
                step_height_range=stair_height_range,
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": _sub_terrain_spec(
                "MeshInvertedPyramidStairsTerrainCfg",
                0.2,
                step_height_range=stair_height_range,
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "boxes": _sub_terrain_spec(
                "MeshRandomGridTerrainCfg",
                0.2,
                grid_width=0.45,
                grid_height_range=box_height_range,
                platform_width=2.0,
            ),
            "random_rough": _sub_terrain_spec(
                "HfRandomUniformTerrainCfg",
                0.2,
                noise_range=random_rough_noise_range,
                noise_step=float(random_rough_noise_step),
                border_width=0.25,
            ),
            "hf_pyramid_slope": _sub_terrain_spec(
                "HfPyramidSlopedTerrainCfg",
                0.1,
                slope_range=slope_range,
                platform_width=2.0,
                border_width=0.25,
            ),
            "hf_pyramid_slope_inv": _sub_terrain_spec(
                "HfInvertedPyramidSlopedTerrainCfg",
                0.1,
                slope_range=slope_range,
                platform_width=2.0,
                border_width=0.25,
            ),
        }
    )


def _build_go2_rough_medium_sub_terrains() -> Mapping[str, Go2SubTerrainSpec]:
    return _rough_sub_terrains(
        box_height_range=(0.025, 0.10),
        random_rough_noise_range=(0.01, 0.06),
        random_rough_noise_step=0.01,
        slope_range=(0.0, 0.30),
        stair_height_range=(0.03, 0.16),
    )


def _build_go2_rough_hard_sub_terrains() -> Mapping[str, Go2SubTerrainSpec]:
    return _rough_sub_terrains(
        box_height_range=(0.05, 0.15),
        random_rough_noise_range=(0.02, 0.10),
        random_rough_noise_step=0.02,
        slope_range=(0.0, 0.40),
        stair_height_range=(0.05, 0.23),
    )


def build_go2_terrain_generator_spec(
    preset: str,
    *,
    num_rows: int,
    num_cols: int,
    curriculum_enabled: bool = True,
    use_cache: bool = False,
) -> Go2TerrainGeneratorSpec:
    preset = validate_go2_terrain_preset(preset)
    num_rows = validate_go2_terrain_positive_int(num_rows, field="num_rows")
    num_cols = validate_go2_terrain_positive_int(num_cols, field="num_cols")
    sub_terrain_builders = {
        GO2_TERRAIN_PRESET_ROUGH_MEDIUM: _build_go2_rough_medium_sub_terrains,
        GO2_TERRAIN_PRESET_ROUGH_HARD: _build_go2_rough_hard_sub_terrains,
    }
    return Go2TerrainGeneratorSpec(
        preset=preset,
        curriculum=bool(curriculum_enabled),
        size=GO2_TERRAIN_DEFAULT_SIZE,
        border_width=GO2_TERRAIN_DEFAULT_BORDER_WIDTH,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=GO2_TERRAIN_DEFAULT_HORIZONTAL_SCALE,
        vertical_scale=GO2_TERRAIN_DEFAULT_VERTICAL_SCALE,
        slope_threshold=GO2_TERRAIN_DEFAULT_SLOPE_THRESHOLD,
        use_cache=bool(use_cache),
        sub_terrains=sub_terrain_builders[preset](),
    )


def compute_go2_distance_terrain_curriculum(
    root_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    commands: torch.Tensor,
    *,
    terrain_tile_length: float,
    episode_length_s: float,
    stationary_xy_command_threshold: float,
    enabled: bool = True,
) -> Go2TerrainCurriculumUpdate:
    """Compute IsaacLab-style distance curriculum moves for Go2 rough terrain.

    The returned tensors are per-environment and side-effect free. Callers decide
    whether to pass the move flags to an IsaacLab TerrainImporter.
    """
    if root_pos_w.ndim != 2 or root_pos_w.shape[-1] < 2:
        raise ValueError(f"root_pos_w must have shape (N, >=2), got {tuple(root_pos_w.shape)}.")
    if env_origins.ndim != 2 or env_origins.shape[-1] < 2:
        raise ValueError(f"env_origins must have shape (N, >=2), got {tuple(env_origins.shape)}.")
    if commands.ndim != 2 or commands.shape[-1] < 2:
        raise ValueError(f"commands must have shape (N, >=2), got {tuple(commands.shape)}.")
    num_envs = int(root_pos_w.shape[0])
    if int(env_origins.shape[0]) != num_envs or int(commands.shape[0]) != num_envs:
        raise ValueError(
            "root_pos_w, env_origins, and commands must have matching leading dimensions: "
            f"{tuple(root_pos_w.shape)}, {tuple(env_origins.shape)}, {tuple(commands.shape)}."
        )
    tile_length = validate_go2_terrain_nonnegative_float(terrain_tile_length, field="terrain_tile_length")
    episode_length_s = validate_go2_terrain_nonnegative_float(episode_length_s, field="episode_length_s")
    stationary_threshold = validate_go2_terrain_nonnegative_float(
        stationary_xy_command_threshold,
        field="stationary_xy_command_threshold",
    )

    distance = torch.linalg.norm(root_pos_w[:, :2] - env_origins[:, :2], dim=1)
    command_xy_norm = torch.linalg.norm(commands[:, :2], dim=1)
    stationary_guard = command_xy_norm < stationary_threshold
    move_up = distance > (0.5 * tile_length)
    move_down = distance < (0.5 * command_xy_norm * episode_length_s)
    move_down = move_down & ~move_up

    if not bool(enabled):
        move_up = torch.zeros(num_envs, dtype=torch.bool, device=root_pos_w.device)
        move_down = torch.zeros_like(move_up)
    else:
        move_up = move_up & ~stationary_guard
        move_down = move_down & ~stationary_guard
    return Go2TerrainCurriculumUpdate(
        move_up=move_up,
        move_down=move_down,
        stationary_guard=stationary_guard,
        distance=distance,
        command_xy_norm=command_xy_norm,
    )


def build_go2_terrain_generator_cfg(
    preset: str,
    *,
    num_rows: int,
    num_cols: int,
    curriculum_enabled: bool = True,
    use_cache: bool = False,
):
    spec = build_go2_terrain_generator_spec(
        preset,
        num_rows=num_rows,
        num_cols=num_cols,
        curriculum_enabled=curriculum_enabled,
        use_cache=use_cache,
    )
    try:
        import isaaclab.terrains as terrain_gen
        from isaaclab.terrains import TerrainGeneratorCfg
    except Exception as exc:
        raise RuntimeError(
            "IsaacLab terrain config classes are unavailable. Build Go2 terrain configs after "
            "the Isaac/Omniverse runtime has initialized, or run through isaaclab.sh."
        ) from exc

    terrain_class_by_name = {
        "MeshPyramidStairsTerrainCfg": terrain_gen.MeshPyramidStairsTerrainCfg,
        "MeshInvertedPyramidStairsTerrainCfg": terrain_gen.MeshInvertedPyramidStairsTerrainCfg,
        "MeshRandomGridTerrainCfg": terrain_gen.MeshRandomGridTerrainCfg,
        "HfRandomUniformTerrainCfg": terrain_gen.HfRandomUniformTerrainCfg,
        "HfPyramidSlopedTerrainCfg": terrain_gen.HfPyramidSlopedTerrainCfg,
        "HfInvertedPyramidSlopedTerrainCfg": terrain_gen.HfInvertedPyramidSlopedTerrainCfg,
    }
    sub_terrains = {
        name: terrain_class_by_name[sub_spec.class_name](
            proportion=sub_spec.proportion,
            **dict(sub_spec.params),
        )
        for name, sub_spec in spec.sub_terrains.items()
    }
    return TerrainGeneratorCfg(
        size=spec.size,
        border_width=spec.border_width,
        num_rows=spec.num_rows,
        num_cols=spec.num_cols,
        horizontal_scale=spec.horizontal_scale,
        vertical_scale=spec.vertical_scale,
        slope_threshold=spec.slope_threshold,
        use_cache=spec.use_cache,
        curriculum=spec.curriculum,
        sub_terrains=sub_terrains,
    )
