"""Build a UnitreeGo2VelocityDirectEnvCfg from FlashSAC's `env.isaac_*` config keys.

Mirrors TDMPC2_isaaclab/tdmpc2/envs/isaaclab.py's `_make_velocity_direct_env` field-by-field
(same isaac_* flag names, same defaults from TDMPC2's config.yaml), minus the flags that have
no home in the trimmed env cfg: TDMPC2's reward-scale flags (isaac_direct_velocity_*_reward_scale
for swing_clearance/foot_lift/trot/hip_ab_ad/support_plane/standing_default_pose/bad_contact/
low_base_height), action_rate_mode/reference_scale, bad_orientation_terminal_reward, reward_mode,
eval-command-sweep, and AMP/SMP knobs -- none of those apply to FlashSAC's own reward
(go2_flashsac_rewards.py) or agent-side (TD-MPC2 planner) concerns.
"""

from __future__ import annotations

from typing import Any

from .go2_action_decoder import (
    ACTION_DECODER_SCALAR,
    RANDOM_ACTION_CENTER_ZERO,
    validate_action_decoder_mode,
    validate_random_action_center,
)
from .go2_terrain_cfg import (
    GO2_HEIGHT_SCAN_DEFAULT_CLIP,
    GO2_HEIGHT_SCAN_DEFAULT_REFERENCE_OFFSET,
    GO2_HEIGHT_SCAN_DEFAULT_RESOLUTION,
    GO2_HEIGHT_SCAN_DEFAULT_SIZE,
    GO2_HEIGHT_SCAN_DEFAULT_VERTICAL_OFFSET,
    GO2_TERRAIN_CURRICULUM_MODE_DISTANCE,
    GO2_TERRAIN_MODE_FLAT,
)
from .go2_urdf_asset import GO2_ASSET_SOURCE_ISAACLAB_USD, GO2_UNITREE_ROS_URDF_PATH, validate_go2_asset_source
from .isaaclab_go2_velocity_direct import (
    GO2_ACTUATOR_MODE_DC_MOTOR,
    UnitreeGo2VelocityDirectEnvCfg,
    _make_go2_robot_cfg,
    _validate_go2_actuator_model,
)


def build_go2_velocity_env_cfg(
    *,
    num_envs: int,
    device: str = "cuda:0",
    seed: int = 0,
    # actuator / asset -- isaac_go2_actuator_model=unitree_go2hv, isaac_go2_asset_source=unitree_urdf
    isaac_go2_actuator_model: str = GO2_ACTUATOR_MODE_DC_MOTOR,
    isaac_go2_asset_source: str = GO2_ASSET_SOURCE_ISAACLAB_USD,
    isaac_go2_urdf_path: str | None = None,
    # PD gains. None keeps the asset's own values (UNITREE_GO2_CFG: Kp=25, Kd=0.5); genesis's
    # go2-walk uses Kp=30, Kd=1.5. Selecting isaac_go2_actuator_model=unitree_go2hv does NOT
    # change the gains on its own -- it only swaps the torque-speed envelope.
    isaac_go2_pd_stiffness: float | None = None,
    isaac_go2_pd_damping: float | None = None,
    # Use genesis's nominal stance (front thigh 0.8 / rear thigh 1.0, calf -1.5, spawn z 0.42)
    # instead of TDMPC2's fore/aft-symmetric PyMPC stance (thigh 0.9, calf -1.8, z 0.29017).
    isaac_go2_genesis_style_nominal_pose: bool = False,
    # Drop the *_rotor links genesis's go2.urdf lacks (1.068 kg). unitree_urdf source only.
    isaac_go2_strip_rotor_links: bool = False,
    # Spawn z, overriding the nominal-pose preset's own (genesis 0.42, PyMPC 0.29017). The
    # preset's joint angles are unaffected -- only the drop distance changes. None = preset.
    isaac_go2_spawn_height: float | None = None,
    # genesis's reset_idx: joints offset by U(-0.3, 0.3) rad off the default, base xy spread
    # +/-1.0 with a U(-0.1, 0.1) roll/pitch tilt and U(0, pi) yaw.
    isaac_direct_velocity_genesis_style_reset_enabled: bool = False,
    # Zero a sampled command below this magnitude (genesis uses 0.2 on both). 0.0 disables.
    isaac_velocity_command_deadband_lin_vel: float = 0.0,
    isaac_velocity_command_deadband_ang_vel: float = 0.0,
    # action pipeline
    isaac_action_scale: float = 0.85,
    isaac_direct_velocity_action_decoder: str = ACTION_DECODER_SCALAR,
    isaac_direct_velocity_clip_joint_targets: bool = True,
    isaac_direct_velocity_use_neutral_action: bool = False,
    isaac_random_action_center: str = RANDOM_ACTION_CENTER_ZERO,
    # 1 executes the previous control step's action, matching genesis's action_latency=0.02.
    isaac_direct_velocity_action_latency_steps: int = 0,
    # termination
    isaac_direct_velocity_enable_termination: bool = True,
    isaac_direct_velocity_bad_orientation_termination_enabled: bool = False,
    isaac_direct_velocity_bad_orientation_body_up_threshold: float = 0.25,
    isaac_direct_velocity_bad_orientation_hysteresis_steps: int = 3,
    # genesis-parity termination -- matches go2_base.py's check_termination() exactly
    # (|roll|/|pitch| Euler-angle thresholds + a base-height floor), independent of the
    # bad_orientation_* projected-gravity check above.
    isaac_direct_velocity_genesis_style_termination_enabled: bool = False,
    isaac_direct_velocity_termination_roll_threshold: float = 0.4,
    isaac_direct_velocity_termination_pitch_threshold: float = 0.4,
    isaac_direct_velocity_termination_min_base_height: float = 0.0,
    # observation
    isaac_direct_velocity_observe_base_lin_vel: bool = False,
    isaac_direct_velocity_observation_history_enabled: bool = True,
    isaac_direct_velocity_observation_history_length: int = 4,
    isaac_direct_velocity_base_ang_vel_filter_alpha: float | None = None,
    isaac_disable_obs_noise: bool = False,
    isaac_randomize_episode_lengths: bool = False,
    # privileged critic-only observation (off by default -- see UnitreeGo2VelocityDirectEnvCfg
    # .privileged_base_lin_vel). Only meaningful together with agent.asymmetric_observation=true.
    isaac_direct_velocity_privileged_base_lin_vel: bool = False,
    # Appends the previous raw action (12 cols) as a further critic-only tail. With
    # privileged_base_lin_vel this reproduces genesis's 60-col privileged_obs_buf exactly.
    isaac_direct_velocity_privileged_last_actions: bool = False,
    # None follows the nominal stance height; genesis's go2-walk uses 0.3.
    isaac_direct_velocity_base_height_target: float | None = None,
    # Applies genesis's obs_scales (lin_vel 2.0 / ang_vel 0.25 / dof_vel 0.05) together with
    # genesis's matching post-scale noise magnitudes. The port is otherwise raw physical units.
    isaac_direct_velocity_genesis_style_obs_scaling_enabled: bool = False,
    # genesis shares one noise draw across the whole batch each step; the port draws per-env.
    isaac_direct_velocity_obs_noise_shared_across_envs: bool = False,
    # genesis clips observations to +/-100. null disables.
    isaac_direct_velocity_obs_clip: float | None = None,
    # record_video's tracking camera: 'swarm' (default) is a wide overview of every env's
    # robot; 'single_env' chases one robot like genesis_envs/go2_base.py's render() does.
    isaac_video_camera_mode: str = "swarm",
    isaac_video_camera_env_index: int = 0,
    # velocity commands
    isaac_velocity_command_resampling_time_range: tuple[float, float] = (5.0, 7.0),
    isaac_velocity_command_lin_vel_x_range: tuple[float, float] = (-1.0, 1.0),
    isaac_velocity_command_lin_vel_y_range: tuple[float, float] = (-0.5, 0.5),
    isaac_velocity_command_ang_vel_z_range: tuple[float, float] = (-1.0, 1.0),
    isaac_velocity_command_yaw_in_place_ang_vel_z_range: tuple[float, float] = (-0.7, 0.7),
    isaac_velocity_command_heading_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793),
    isaac_velocity_command_heading_enabled: bool = True,
    isaac_velocity_command_heading_control_stiffness: float = 1.0,
    isaac_velocity_command_rel_standing_envs: float = 0.06,
    isaac_velocity_command_rel_heading_envs: float = 1.0,
    isaac_velocity_command_rel_yaw_in_place_envs: float = 0.0,
    # domain randomization
    isaac_dr_enabled: bool = False,
    isaac_dr_train_only: bool = True,
    isaac_dr_static_friction_range: tuple[float, float] = (0.2, 1.25),
    isaac_dr_dynamic_friction_range: tuple[float, float] = (0.2, 1.25),
    isaac_dr_restitution_range: tuple[float, float] = (0.0, 0.15),
    isaac_dr_base_mass_add_range: tuple[float, float] = (-1.0, 3.0),
    isaac_dr_base_com_range_x: tuple[float, float] = (-0.15, 0.15),
    isaac_dr_base_com_range_y: tuple[float, float] = (-0.10, 0.10),
    isaac_dr_base_com_range_z: tuple[float, float] = (-0.05, 0.08),
    isaac_dr_motor_strength_range: tuple[float, float] = (0.9, 1.1),
    isaac_dr_motor_strength_per_joint: bool = True,
    # Per-joint, per-episode Kp/Kd scaling, matching genesis's kp_scale_range/kd_scale_range.
    # (1.0, 1.0) disables it.
    isaac_dr_kp_scale_range: tuple[float, float] = (1.0, 1.0),
    isaac_dr_kd_scale_range: tuple[float, float] = (1.0, 1.0),
    # Per-(env, joint) joint-position bias in radians; genesis uses [-0.02, 0.02]. (0,0) off.
    isaac_dr_motor_offset_range: tuple[float, float] = (0.0, 0.0),
    # terrain (flat + rough)
    isaac_direct_velocity_terrain_mode: str = GO2_TERRAIN_MODE_FLAT,
    isaac_direct_velocity_terrain_preset: str | None = None,
    isaac_direct_velocity_terrain_curriculum_enabled: bool = True,
    isaac_direct_velocity_terrain_curriculum_mode: str = GO2_TERRAIN_CURRICULUM_MODE_DISTANCE,
    isaac_direct_velocity_terrain_stationary_xy_command_threshold: float = 0.1,
    isaac_direct_velocity_terrain_max_init_level: int = 5,
    isaac_direct_velocity_terrain_num_rows: int = 10,
    isaac_direct_velocity_terrain_num_cols: int = 20,
    isaac_direct_velocity_terrain_debug_vis: bool = False,
    # height scan (disabled by default, matches TDMPC2 config.yaml)
    isaac_direct_velocity_height_scan_enabled: bool = False,
    isaac_direct_velocity_height_scan_observe: bool = True,
    isaac_direct_velocity_height_scan_size: tuple[float, float] = GO2_HEIGHT_SCAN_DEFAULT_SIZE,
    isaac_direct_velocity_height_scan_resolution: float = GO2_HEIGHT_SCAN_DEFAULT_RESOLUTION,
    isaac_direct_velocity_height_scan_vertical_offset: float = GO2_HEIGHT_SCAN_DEFAULT_VERTICAL_OFFSET,
    isaac_direct_velocity_height_scan_reference_offset: float = GO2_HEIGHT_SCAN_DEFAULT_REFERENCE_OFFSET,
    isaac_direct_velocity_height_scan_clip: tuple[float, float] = GO2_HEIGHT_SCAN_DEFAULT_CLIP,
    # sim / physx timing overrides (None = inert, matches TDMPC2 config.yaml defaults)
    isaac_direct_velocity_sim_dt: float | None = None,
    isaac_direct_velocity_decimation: int | None = None,
    isaac_physx_enable_external_forces_every_iteration: bool | None = None,
    isaac_physx_min_velocity_iteration_count: int | None = None,
    **_ignored: Any,
) -> UnitreeGo2VelocityDirectEnvCfg:
    env_cfg = UnitreeGo2VelocityDirectEnvCfg()

    actuator_model = _validate_go2_actuator_model(isaac_go2_actuator_model)
    asset_source = validate_go2_asset_source(isaac_go2_asset_source)
    urdf_path = str(isaac_go2_urdf_path) if isaac_go2_urdf_path else str(env_cfg.urdf_path or GO2_UNITREE_ROS_URDF_PATH)
    env_cfg.actuator_model = actuator_model
    env_cfg.asset_source = asset_source
    env_cfg.urdf_path = urdf_path
    env_cfg.pd_stiffness = None if isaac_go2_pd_stiffness is None else float(isaac_go2_pd_stiffness)
    env_cfg.pd_damping = None if isaac_go2_pd_damping is None else float(isaac_go2_pd_damping)
    # The robot cfg is built once at dataclass-definition time with the default actuator/asset;
    # rebuild it now that the actual actuator_model/asset_source/urdf_path/PD gains are known.
    env_cfg.genesis_style_nominal_pose = bool(isaac_go2_genesis_style_nominal_pose)
    env_cfg.genesis_style_reset_enabled = bool(isaac_direct_velocity_genesis_style_reset_enabled)
    env_cfg.strip_rotor_links = bool(isaac_go2_strip_rotor_links)
    env_cfg.spawn_height = None if isaac_go2_spawn_height is None else float(isaac_go2_spawn_height)
    env_cfg.obs_noise_shared_across_envs = bool(isaac_direct_velocity_obs_noise_shared_across_envs)
    env_cfg.obs_clip = None if isaac_direct_velocity_obs_clip is None else float(isaac_direct_velocity_obs_clip)
    env_cfg.command_deadband_lin_vel = float(isaac_velocity_command_deadband_lin_vel)
    env_cfg.command_deadband_ang_vel = float(isaac_velocity_command_deadband_ang_vel)
    env_cfg.robot = _make_go2_robot_cfg(
        actuator_model,
        asset_source,
        urdf_path,
        env_cfg.pd_stiffness,
        env_cfg.pd_damping,
        env_cfg.genesis_style_nominal_pose,
        bool(isaac_go2_strip_rotor_links),
        env_cfg.spawn_height,
    )
    # The base-height reward target defaults to the nominal stance height, so it follows the
    # pose unless explicitly overridden. genesis's go2-walk uses a flat 0.3.
    if isaac_direct_velocity_base_height_target is not None:
        env_cfg.base_height_target = float(isaac_direct_velocity_base_height_target)
    elif env_cfg.genesis_style_nominal_pose:
        env_cfg.base_height_target = 0.3

    env_cfg.seed = int(seed)
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.sim.device = str(device)

    env_cfg.action_scale = float(isaac_action_scale)
    env_cfg.action_decoder = validate_action_decoder_mode(isaac_direct_velocity_action_decoder)
    env_cfg.clip_joint_targets = bool(isaac_direct_velocity_clip_joint_targets)
    env_cfg.use_neutral_action = bool(isaac_direct_velocity_use_neutral_action)
    env_cfg.random_action_center = validate_random_action_center(isaac_random_action_center)
    env_cfg.action_latency_steps = int(isaac_direct_velocity_action_latency_steps)

    env_cfg.enable_termination = bool(isaac_direct_velocity_enable_termination)
    env_cfg.bad_orientation_termination_enabled = bool(isaac_direct_velocity_bad_orientation_termination_enabled)
    env_cfg.bad_orientation_body_up_threshold = float(isaac_direct_velocity_bad_orientation_body_up_threshold)
    env_cfg.bad_orientation_hysteresis_steps = int(isaac_direct_velocity_bad_orientation_hysteresis_steps)
    env_cfg.genesis_style_termination_enabled = bool(isaac_direct_velocity_genesis_style_termination_enabled)
    env_cfg.termination_roll_threshold = float(isaac_direct_velocity_termination_roll_threshold)
    env_cfg.termination_pitch_threshold = float(isaac_direct_velocity_termination_pitch_threshold)
    env_cfg.termination_min_base_height = float(isaac_direct_velocity_termination_min_base_height)

    env_cfg.observe_base_lin_vel = bool(isaac_direct_velocity_observe_base_lin_vel)
    env_cfg.observation_history_enabled = bool(isaac_direct_velocity_observation_history_enabled)
    env_cfg.observation_history_length = int(isaac_direct_velocity_observation_history_length)
    env_cfg.base_ang_vel_filter_alpha = isaac_direct_velocity_base_ang_vel_filter_alpha
    env_cfg.enable_observation_noise = not bool(isaac_disable_obs_noise)
    env_cfg.randomize_episode_lengths = bool(isaac_randomize_episode_lengths)
    env_cfg.privileged_base_lin_vel = bool(isaac_direct_velocity_privileged_base_lin_vel)
    env_cfg.privileged_last_actions = bool(isaac_direct_velocity_privileged_last_actions)
    env_cfg.genesis_style_obs_scaling_enabled = bool(isaac_direct_velocity_genesis_style_obs_scaling_enabled)

    if isaac_video_camera_mode not in ("swarm", "single_env"):
        raise ValueError(f"isaac_video_camera_mode must be 'swarm' or 'single_env', got {isaac_video_camera_mode!r}")
    env_cfg.video_camera_mode = str(isaac_video_camera_mode)
    env_cfg.video_camera_env_index = int(isaac_video_camera_env_index)

    env_cfg.command_resampling_time_range = tuple(float(v) for v in isaac_velocity_command_resampling_time_range)
    env_cfg.lin_vel_x_range = tuple(float(v) for v in isaac_velocity_command_lin_vel_x_range)
    env_cfg.lin_vel_y_range = tuple(float(v) for v in isaac_velocity_command_lin_vel_y_range)
    env_cfg.ang_vel_z_range = tuple(float(v) for v in isaac_velocity_command_ang_vel_z_range)
    env_cfg.yaw_in_place_ang_vel_z_range = tuple(float(v) for v in isaac_velocity_command_yaw_in_place_ang_vel_z_range)
    env_cfg.heading_range = tuple(float(v) for v in isaac_velocity_command_heading_range)
    env_cfg.heading_command = bool(isaac_velocity_command_heading_enabled)
    env_cfg.heading_control_stiffness = float(isaac_velocity_command_heading_control_stiffness)
    env_cfg.rel_standing_envs = float(isaac_velocity_command_rel_standing_envs)
    env_cfg.rel_heading_envs = float(isaac_velocity_command_rel_heading_envs)
    env_cfg.rel_yaw_in_place_envs = float(isaac_velocity_command_rel_yaw_in_place_envs)

    env_cfg.domain_randomization_enabled = bool(isaac_dr_enabled)
    env_cfg.domain_randomization_train_only = bool(isaac_dr_train_only)
    env_cfg.dr_static_friction_range = tuple(float(v) for v in isaac_dr_static_friction_range)
    env_cfg.dr_dynamic_friction_range = tuple(float(v) for v in isaac_dr_dynamic_friction_range)
    env_cfg.dr_restitution_range = tuple(float(v) for v in isaac_dr_restitution_range)
    env_cfg.dr_base_mass_add_range = tuple(float(v) for v in isaac_dr_base_mass_add_range)
    env_cfg.dr_base_com_range_x = tuple(float(v) for v in isaac_dr_base_com_range_x)
    env_cfg.dr_base_com_range_y = tuple(float(v) for v in isaac_dr_base_com_range_y)
    env_cfg.dr_base_com_range_z = tuple(float(v) for v in isaac_dr_base_com_range_z)
    env_cfg.dr_motor_strength_range = tuple(float(v) for v in isaac_dr_motor_strength_range)
    env_cfg.dr_motor_strength_per_joint = bool(isaac_dr_motor_strength_per_joint)
    env_cfg.dr_kp_scale_range = tuple(float(v) for v in isaac_dr_kp_scale_range)
    env_cfg.dr_kd_scale_range = tuple(float(v) for v in isaac_dr_kd_scale_range)
    env_cfg.dr_motor_offset_range = tuple(float(v) for v in isaac_dr_motor_offset_range)

    env_cfg.terrain_mode = str(isaac_direct_velocity_terrain_mode)
    if isaac_direct_velocity_terrain_preset is not None:
        env_cfg.terrain_preset = str(isaac_direct_velocity_terrain_preset)
    env_cfg.terrain_curriculum_enabled = bool(isaac_direct_velocity_terrain_curriculum_enabled)
    env_cfg.terrain_curriculum_mode = str(isaac_direct_velocity_terrain_curriculum_mode)
    env_cfg.terrain_stationary_xy_command_threshold = float(
        isaac_direct_velocity_terrain_stationary_xy_command_threshold
    )
    env_cfg.terrain_max_init_level = int(isaac_direct_velocity_terrain_max_init_level)
    env_cfg.terrain_num_rows = int(isaac_direct_velocity_terrain_num_rows)
    env_cfg.terrain_num_cols = int(isaac_direct_velocity_terrain_num_cols)
    env_cfg.terrain_debug_vis = bool(isaac_direct_velocity_terrain_debug_vis)

    env_cfg.height_scan_enabled = bool(isaac_direct_velocity_height_scan_enabled)
    env_cfg.height_scan_observe = bool(isaac_direct_velocity_height_scan_observe)
    env_cfg.height_scan_size = tuple(float(v) for v in isaac_direct_velocity_height_scan_size)
    env_cfg.height_scan_resolution = float(isaac_direct_velocity_height_scan_resolution)
    env_cfg.height_scan_vertical_offset = float(isaac_direct_velocity_height_scan_vertical_offset)
    env_cfg.height_scan_reference_offset = float(isaac_direct_velocity_height_scan_reference_offset)
    env_cfg.height_scan_clip = tuple(float(v) for v in isaac_direct_velocity_height_scan_clip)

    if isaac_direct_velocity_sim_dt is not None:
        env_cfg.sim.dt = float(isaac_direct_velocity_sim_dt)
    if isaac_direct_velocity_decimation is not None:
        env_cfg.decimation = int(isaac_direct_velocity_decimation)
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.contact_sensor.update_period = env_cfg.sim.dt
    if isaac_physx_enable_external_forces_every_iteration is not None:
        env_cfg.sim.physx.enable_external_forces_every_iteration = bool(
            isaac_physx_enable_external_forces_every_iteration
        )
    if isaac_physx_min_velocity_iteration_count is not None:
        env_cfg.sim.physx.min_velocity_iteration_count = int(isaac_physx_min_velocity_iteration_count)

    return env_cfg
