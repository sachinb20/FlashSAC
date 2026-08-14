"""Direct Unitree Go2 velocity-tracking IsaacLab environment.

Ported from TDMPC2_isaaclab/tdmpc2/envs/isaaclab_go2_velocity_direct.py: sim setup,
robot/URDF asset, actuator model, terrain (flat + rough), domain randomization,
observation composition, and action decoding are kept faithful to that source.

The reward is NOT ported from TDMPC2 -- it uses FlashSAC's own Go2 reward
(go2_flashsac_rewards.py, ported from flash_rl/envs/genesis_envs/go2_walk.py).
AMP/SMP auxiliary features, gait/foot-lift/support-plane reward diagnostics (those
existed only to feed TDMPC2's reward terms), terrain-curriculum/contact debug metric
logging, and TDMPC2-wrapper-only hooks (eval command sweep, debug snapshots,
checkpoint fingerprinting) are dropped.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import DCMotor, IdealPDActuator, IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from .go2_action_decoder import (
    ACTION_DECODER_SCALAR,
    RANDOM_ACTION_CENTER_ZERO,
    decode_go2_action,
    go2_neutral_action,
    validate_action_decoder_mode,
    validate_random_action_center,
)
from .go2_base_ang_vel_filter import base_ang_vel_with_optional_ema, reset_base_ang_vel_ema_state
from .go2_command_sampling import apply_go2_command_deadband, sample_go2_velocity_commands
from .go2_flashsac_rewards import go2_flashsac_reward_scales, go2_flashsac_reward_terms, go2_flashsac_scalar_reward
from .go2_motor_offset_randomization import (
    clear_motor_offsets,
    motor_offset_range_is_default,
    sample_motor_offsets,
)
from .go2_motor_strength_randomization import (
    as_actuator_tensor,
    motor_strength_range_is_default,
    randomize_motor_strength,
    restore_motor_strength_defaults,
)
from .go2_pd_gain_randomization import (
    PD_GAIN_FIELDS,
    randomize_pd_gains,
    restore_pd_gain_defaults,
)
from .go2_termination import go2_bad_orientation_mask_from_projected_gravity, go2_update_bad_orientation_hysteresis
from .go2_terrain_cfg import (
    GO2_HEIGHT_SCAN_DEFAULT_CLIP,
    GO2_HEIGHT_SCAN_DEFAULT_REFERENCE_OFFSET,
    GO2_HEIGHT_SCAN_DEFAULT_RESOLUTION,
    GO2_HEIGHT_SCAN_DEFAULT_SIZE,
    GO2_HEIGHT_SCAN_DEFAULT_VERTICAL_OFFSET,
    GO2_TERRAIN_CURRICULUM_MODE_DISTANCE,
    GO2_TERRAIN_MODE_FLAT,
    GO2_TERRAIN_PRESET_ROUGH_MEDIUM,
    build_go2_terrain_generator_cfg,
    compute_go2_distance_terrain_curriculum,
    go2_direct_velocity_base_observation_dim,
    go2_terrain_mode_is_rough,
    go2_terrain_type_for_mode,
    resolve_go2_terrain_preset,
    validate_go2_height_scan_clip,
    validate_go2_height_scan_size,
    validate_go2_terrain_curriculum_mode,
    validate_go2_terrain_mode,
    validate_go2_terrain_nonnegative_float,
    validate_go2_terrain_nonnegative_int,
    validate_go2_terrain_positive_float,
    validate_go2_terrain_positive_int,
)
from .go2_urdf_asset import (
    GO2_ASSET_SOURCE_ISAACLAB_USD,
    GO2_ASSET_SOURCE_UNITREE_URDF,
    GO2_POLICY_JOINT_NAMES,
    GO2_UNITREE_ROS_URDF_PATH,
    go2_policy_to_sim_joint_ids,
    stage_go2_urdf_for_isaaclab,
    validate_go2_asset_source,
    validate_go2_urdf_joint_contract,
)
from .isaaclab_unitree_actuators import UnitreeActuatorCfg_Go2HV

GO2_ACTUATOR_MODE_DC_MOTOR = "dc_motor"
GO2_ACTUATOR_MODE_UNITREE_GO2HV = "unitree_go2hv"
# Unclipped explicit PD -- no torque ceiling of any kind, matching genesis. See
# MotorStrengthIdealPD.
GO2_ACTUATOR_MODE_IDEAL_PD = "ideal_pd"
GO2_ACTUATOR_MODES = frozenset(
    {GO2_ACTUATOR_MODE_DC_MOTOR, GO2_ACTUATOR_MODE_UNITREE_GO2HV, GO2_ACTUATOR_MODE_IDEAL_PD}
)

# Nominal stance used by TDMPC2's sim-to-real tuning; overrides IsaacLab's stock
# UNITREE_GO2_CFG defaults (base_z=0.4, thigh 0.8/1.0, calf -1.5).
PYMPC_GO2_NOMINAL_BASE_Z = 0.29017
PYMPC_GO2_NOMINAL_JOINT_POS = {
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
}

# genesis_envs/go2_walk.py's default_joint_angles + base_init_pos. Note the front/rear thigh
# split (0.8 vs 1.0): genesis stands rear-loaded and nose-down, whereas the PyMPC stance above
# is fore/aft symmetric at 0.9. Because the decoder is
# `target = default_joint_pos + action_scale * action`, this pose is the policy's operating
# point, so the difference biases forward vs backward locomotion, not just posture.
GENESIS_GO2_NOMINAL_BASE_Z = 0.42
GENESIS_GO2_NOMINAL_JOINT_POS = {
    ".*_hip_joint": 0.0,
    "F[L,R]_thigh_joint": 0.8,
    "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
}


class MotorStrengthDCMotor(DCMotor):
    """DC motor that scales computed PD effort before torque-speed clipping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motor_strength = torch.ones_like(self.computed_effort)
        self.scaled_motor_strength = torch.zeros_like(self.computed_effort)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        feedforward_effort = control_action.joint_efforts
        if feedforward_effort is None:
            feedforward_effort = torch.zeros_like(error_pos)

        effort = self.stiffness * error_pos + self.damping * error_vel + feedforward_effort
        self.computed_effort = effort * self.motor_strength
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


class MotorStrengthIdealPD(IdealPDActuator):
    """Explicit PD with motor-strength scaling and **no torque ceiling at all**.

    Matches genesis_envs/go2_base.py's `_compute_torques`, which computes
    `kp * error_pos - kd * joint_vel`, scales by motor_strengths, and hands the result to
    `control_dofs_force` without ever consulting a torque limit (go2_base.py reads
    `self.torque_limits` once and never uses it).

    `_clip_effort` is overridden to the identity, so the actuator's own `effort_limit` is
    inert. PhysX does not re-clamp either: IsaacLab defaults `effort_limit_sim` to 1e9 for
    explicit actuators, so the solver enforces no ceiling of its own.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motor_strength = torch.ones_like(self.computed_effort)
        self.scaled_motor_strength = torch.zeros_like(self.computed_effort)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        feedforward_effort = control_action.joint_efforts
        if feedforward_effort is None:
            feedforward_effort = torch.zeros_like(error_pos)

        effort = self.stiffness * error_pos + self.damping * error_vel + feedforward_effort
        self.computed_effort = effort * self.motor_strength
        self.applied_effort = self.computed_effort

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        return effort


def _make_ideal_pd_actuator_cfg(source_cfg):
    """Rebuild an actuator cfg as an unclipped IdealPDActuatorCfg, keeping gains/limits."""
    kwargs = {"joint_names_expr": source_cfg.joint_names_expr}
    for field_name in (
        "effort_limit",
        "effort_limit_sim",
        "velocity_limit",
        "velocity_limit_sim",
        "stiffness",
        "damping",
        "armature",
        "friction",
    ):
        if hasattr(source_cfg, field_name):
            kwargs[field_name] = getattr(source_cfg, field_name)
    cfg = IdealPDActuatorCfg(**kwargs)
    cfg.class_type = MotorStrengthIdealPD
    return cfg


def _validate_go2_actuator_model(actuator_model: str) -> str:
    actuator_model = str(actuator_model)
    if actuator_model not in GO2_ACTUATOR_MODES:
        raise ValueError(
            f"Invalid isaac_go2_actuator_model '{actuator_model}'. Expected one of {sorted(GO2_ACTUATOR_MODES)}."
        )
    return actuator_model


def _make_unitree_go2hv_actuator_cfg(source_cfg):
    kwargs = {"joint_names_expr": source_cfg.joint_names_expr}
    for field_name in (
        "effort_limit",
        "effort_limit_sim",
        "velocity_limit",
        "velocity_limit_sim",
        "stiffness",
        "damping",
        "armature",
        "friction",
        "dynamic_friction",
        "viscous_friction",
        "min_delay",
        "max_delay",
    ):
        if hasattr(source_cfg, field_name):
            kwargs[field_name] = getattr(source_cfg, field_name)
    return UnitreeActuatorCfg_Go2HV(**kwargs)


def _make_go2_urdf_spawn_cfg(urdf_path: str, strip_rotor_links: bool = False):
    staged_urdf_path = stage_go2_urdf_for_isaaclab(urdf_path, strip_rotor_links=strip_rotor_links)
    return sim_utils.UrdfFileCfg(
        asset_path=str(staged_urdf_path),
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    )


def _make_go2_robot_cfg(
    actuator_model: str = GO2_ACTUATOR_MODE_DC_MOTOR,
    asset_source: str = GO2_ASSET_SOURCE_ISAACLAB_USD,
    urdf_path: str | None = None,
    pd_stiffness: float | None = None,
    pd_damping: float | None = None,
    genesis_style_nominal_pose: bool = False,
    strip_rotor_links: bool = False,
):
    actuator_model = _validate_go2_actuator_model(actuator_model)
    asset_source = validate_go2_asset_source(asset_source)
    robot_cfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    nominal_base_z = GENESIS_GO2_NOMINAL_BASE_Z if genesis_style_nominal_pose else PYMPC_GO2_NOMINAL_BASE_Z
    nominal_joint_pos = GENESIS_GO2_NOMINAL_JOINT_POS if genesis_style_nominal_pose else PYMPC_GO2_NOMINAL_JOINT_POS
    robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, nominal_base_z),
        joint_pos=dict(nominal_joint_pos),
        joint_vel={".*": 0.0},
    )
    if asset_source == GO2_ASSET_SOURCE_UNITREE_URDF:
        urdf_path = str(GO2_UNITREE_ROS_URDF_PATH if urdf_path is None else urdf_path)
        validate_go2_urdf_joint_contract(urdf_path)
        robot_cfg.spawn = _make_go2_urdf_spawn_cfg(urdf_path, strip_rotor_links)
    for name, actuator_cfg in robot_cfg.actuators.items():
        if getattr(actuator_cfg, "class_type", None) is DCMotor:
            if actuator_model == GO2_ACTUATOR_MODE_DC_MOTOR:
                actuator_cfg.class_type = MotorStrengthDCMotor
            elif actuator_model == GO2_ACTUATOR_MODE_IDEAL_PD:
                actuator_cfg = _make_ideal_pd_actuator_cfg(actuator_cfg)
                robot_cfg.actuators[name] = actuator_cfg
            else:
                actuator_cfg = _make_unitree_go2hv_actuator_cfg(actuator_cfg)
                robot_cfg.actuators[name] = actuator_cfg
        # Applied after the actuator-model swap: _make_unitree_go2hv_actuator_cfg copies
        # stiffness/damping straight off the source cfg, so selecting unitree_go2hv changes
        # only the torque-speed envelope -- the gains stay at whatever UNITREE_GO2_CFG
        # declares (Kp=25, Kd=0.5) unless overridden here.
        if pd_stiffness is not None:
            actuator_cfg.stiffness = float(pd_stiffness)
        if pd_damping is not None:
            actuator_cfg.damping = float(pd_damping)
    return robot_cfg


@configclass
class UnitreeGo2VelocityEventCfg:
    """Startup and reset events matched to Isaac Lab's manager Go2 flat task."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 96,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class UnitreeGo2VelocityDirectEnvCfg(DirectRLEnvCfg):
    """Direct Unitree Go2 flat/rough velocity tracking, with FlashSAC's own reward."""

    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.85
    action_space = 12
    observation_space = 45
    state_space = 0
    clip_joint_targets = True
    enable_termination = True
    bad_orientation_termination_enabled = False
    bad_orientation_body_up_threshold = 0.25
    bad_orientation_hysteresis_steps = 3
    # Alternative termination check matching flash_rl/envs/genesis_envs/go2_base.py's
    # check_termination() exactly: |roll| or |pitch| past a threshold (world-frame Euler
    # angles, radians), or base height below a floor. Independent of
    # bad_orientation_termination_enabled above, which uses a different (projected-gravity
    # body-up alignment + hysteresis) criterion -- this one has no hysteresis, matching
    # genesis's single-frame check.
    genesis_style_termination_enabled = False
    termination_roll_threshold = 0.4
    termination_pitch_threshold = 0.4
    termination_min_base_height = 0.0
    randomize_episode_lengths = False
    enable_observation_noise = True
    observe_base_lin_vel = False
    observation_history_enabled = True
    observation_history_length = 4
    # Adds true (unnoised) base_lin_vel_b as a critic-only tail appended after the actor's
    # observation_space columns -- mirrors genesis_envs/go2_walk.py's privileged_obs_buf
    # (its primary addition over the actor obs). Off by default so existing symmetric runs
    # keep their validated 225-dim shape; the FlashSACAgent's own asymmetric_observation
    # flag (not this one) decides whether the actor is actually restricted to the prefix --
    # see flash_rl/agents/flashSAC/agent.py's actor_observation_dim resolution.
    privileged_base_lin_vel = False
    # Adds the previous raw action (12 cols) as a further critic-only tail. Together with
    # privileged_base_lin_vel this reproduces genesis_envs/go2_base.py's privileged_obs_buf
    # exactly: [actor obs | base_lin_vel (3) | last_actions (12)], i.e. 45 -> 60 at
    # observation_history_enabled=false. Off by default. Order matters -- base_lin_vel first,
    # then last_actions -- to match genesis's concatenation order column for column.
    privileged_last_actions = False
    action_decoder = ACTION_DECODER_SCALAR
    use_neutral_action = False
    random_action_center = RANDOM_ACTION_CENTER_ZERO
    domain_randomization_enabled = False
    domain_randomization_train_only = True
    dr_static_friction_range = (0.2, 1.25)
    dr_dynamic_friction_range = (0.2, 1.25)
    dr_restitution_range = (0.0, 0.15)
    dr_base_mass_add_range = (-1.0, 3.0)
    dr_base_com_range_x = (-0.05, 0.05)
    dr_base_com_range_y = (-0.05, 0.05)
    dr_base_com_range_z = (-0.05, 0.05)
    dr_motor_strength_range = (0.9, 1.1)
    dr_motor_strength_per_joint = True
    # Per-joint, per-episode uniform scaling of the actuator's Kp/Kd, matching
    # genesis_envs/go2_walk.py's kp_scale_range/kd_scale_range. (1.0, 1.0) disables it, which
    # is the default -- the port's own DR surface randomizes motor_strength instead.
    dr_kp_scale_range = (1.0, 1.0)
    dr_kd_scale_range = (1.0, 1.0)
    # Per-(env, joint) joint-position bias in radians, held for the episode -- genesis's
    # motor_offset_range ([-0.02, 0.02] there). (0.0, 0.0) disables it. genesis randomizes this
    # INSTEAD of motor strength (its randomize_motor_strength is False).
    dr_motor_offset_range = (0.0, 0.0)
    actuator_model = GO2_ACTUATOR_MODE_DC_MOTOR
    asset_source = GO2_ASSET_SOURCE_ISAACLAB_USD
    urdf_path = str(GO2_UNITREE_ROS_URDF_PATH)
    # None keeps whatever the asset declares (UNITREE_GO2_CFG: Kp=25, Kd=0.5). Genesis's
    # go2-walk uses Kp=30, Kd=1.5 -- see _make_go2_robot_cfg.
    pd_stiffness = None
    pd_damping = None
    # Swap the PyMPC nominal stance for genesis's (front thigh 0.8 / rear thigh 1.0, calf -1.5,
    # spawn z 0.42). See GENESIS_GO2_NOMINAL_JOINT_POS for why this is fore/aft relevant.
    genesis_style_nominal_pose = False
    # Reset-state randomization, matching genesis_envs/go2_base.py's reset_idx:
    #   dof_pos  = default + U(-0.3, 0.3)      (ADDITIVE -- IsaacLab's stock Go2 event instead
    #                                           scales the default by U(1.0, 1.0), i.e. none)
    #   base xy += U(-1.0, 1.0), roll/pitch = U(-0.1, 0.1), yaw = U(0.0, 3.14)
    # genesis_style_reset_enabled swaps all of the above in at once; leave it off to keep the
    # port's own reset (no joint noise, base xy +/-0.5, yaw +/-pi, level).
    genesis_style_reset_enabled = False
    # Drop the twelve 0.089 kg *_rotor links TDMPC2's URDF fixes to the base and genesis's
    # go2.urdf does not have (1.068 kg total: base 7.99 -> 6.92 kg, robot 16.09 -> 15.02 kg).
    # Only meaningful with asset_source='unitree_urdf'. See strip_go2_rotor_links.
    strip_rotor_links = False
    # genesis draws ONE noise vector of shape (num_obs,) per step and broadcasts it across the
    # whole batch, so every env sees the identical perturbation; the port draws independently
    # per env. Same marginal distribution, very different correlation across the batch.
    obs_noise_shared_across_envs = False
    # genesis clips obs (and privileged obs) to +/-100. None disables.
    obs_clip = None
    # Zero a freshly sampled command whose magnitude is under the threshold; genesis uses 0.2
    # on both. 0.0 disables. See apply_go2_command_deadband.
    command_deadband_lin_vel = 0.0
    command_deadband_ang_vel = 0.0

    # video_camera_mode='swarm' points the tracking camera at the centroid of every env's
    # robot (a wide overview of the whole batch). 'single_env' instead chases one robot
    # (video_camera_env_index) the way flash_rl/envs/genesis_envs/go2_base.py's render()
    # always does -- see _update_tracking_camera().
    video_camera_mode = "swarm"
    video_camera_env_index = 0

    base_lin_vel_noise = (-0.1, 0.1)
    base_ang_vel_noise = (-0.2, 0.2)
    base_ang_vel_filter_alpha = None
    projected_gravity_noise = (-0.05, 0.05)
    joint_pos_noise = (-0.01, 0.01)
    joint_vel_noise = (-1.5, 1.5)

    # Per-channel observation scaling, applied before the noise (see
    # _build_current_policy_observation). All 1.0 = the port's native raw-physical-units
    # observation. genesis_envs/go2_walk.py's obs_scales instead uses lin_vel=2.0,
    # ang_vel=0.25, dof_pos=1.0, dof_vel=0.05 -- set genesis_style_obs_scaling_enabled to
    # apply those together with genesis's matching (post-scale) noise magnitudes, since the
    # two are only meaningful as a pair.
    obs_scale_base_lin_vel = 1.0
    obs_scale_base_ang_vel = 1.0
    obs_scale_projected_gravity = 1.0
    obs_scale_commands_lin_vel = 1.0
    obs_scale_commands_ang_vel = 1.0
    obs_scale_joint_pos = 1.0
    obs_scale_joint_vel = 1.0
    genesis_style_obs_scaling_enabled = False

    # action_latency_steps=1 executes the previous control step's action, matching genesis's
    # action_latency=0.02 (one 20ms control step). genesis asserts the value is 0 or 0.02, so
    # only 0 and 1 are accepted here.
    action_latency_steps = 0

    terrain_mode = GO2_TERRAIN_MODE_FLAT
    terrain_preset = GO2_TERRAIN_PRESET_ROUGH_MEDIUM
    terrain_curriculum_enabled = True
    terrain_curriculum_mode = GO2_TERRAIN_CURRICULUM_MODE_DISTANCE
    terrain_stationary_xy_command_threshold = 0.1
    terrain_max_init_level = 5
    terrain_num_rows = 10
    terrain_num_cols = 20
    height_scan_enabled = False
    height_scan_observe = True
    height_scan_size = GO2_HEIGHT_SCAN_DEFAULT_SIZE
    height_scan_resolution = GO2_HEIGHT_SCAN_DEFAULT_RESOLUTION
    height_scan_vertical_offset = GO2_HEIGHT_SCAN_DEFAULT_VERTICAL_OFFSET
    height_scan_reference_offset = GO2_HEIGHT_SCAN_DEFAULT_REFERENCE_OFFSET
    height_scan_clip = GO2_HEIGHT_SCAN_DEFAULT_CLIP
    terrain_debug_vis = False

    lin_vel_x_range = (0.2, 0.8)
    lin_vel_y_range = (0.0, 0.0)
    ang_vel_z_range = (-0.5, 0.5)
    yaw_in_place_ang_vel_z_range = (-0.5, 0.5)
    heading_range = (-3.141592653589793, 3.141592653589793)
    heading_command = False
    heading_control_stiffness = 1.0
    rel_standing_envs = 0.0
    rel_heading_envs = 0.0
    rel_yaw_in_place_envs = 0.0
    command_resampling_time_range = (10.0, 10.0)

    # FlashSAC-original reward (see go2_flashsac_rewards.py) -- ported from
    # flash_rl/envs/genesis_envs/go2_walk.py's reward_cfg, NOT from TDMPC2.
    tracking_sigma = 0.25
    base_height_target = PYMPC_GO2_NOMINAL_BASE_Z
    tracking_lin_vel_reward_scale = 1.0
    tracking_ang_vel_reward_scale = 0.5
    lin_vel_z_reward_scale = -2.0
    ang_vel_xy_reward_scale = -0.05
    orientation_reward_scale = -10.0
    base_height_reward_scale = -50.0
    torques_reward_scale = -0.0002
    dof_vel_reward_scale = 0.0
    dof_acc_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 1.0
    feet_air_time_offset = 0.5
    feet_air_time_command_lin_vel_threshold = 0.1
    collision_reward_scale = -1.0
    collision_force_threshold = 0.1
    dof_pos_limits_reward_scale = 0.0
    termination_reward_scale = 0.0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    events: UnitreeGo2VelocityEventCfg = UnitreeGo2VelocityEventCfg()
    robot = _make_go2_robot_cfg(GO2_ACTUATOR_MODE_DC_MOTOR)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )


class UnitreeGo2VelocityDirectEnv(DirectRLEnv):
    """Direct Unitree Go2 velocity task: TDMPC2's sim/robot/actuator, FlashSAC's reward."""

    cfg: UnitreeGo2VelocityDirectEnvCfg

    def __init__(self, cfg: UnitreeGo2VelocityDirectEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.actuator_model = _validate_go2_actuator_model(cfg.actuator_model)
        cfg.asset_source = validate_go2_asset_source(cfg.asset_source)
        cfg.urdf_path = str(GO2_UNITREE_ROS_URDF_PATH if cfg.urdf_path is None else cfg.urdf_path)
        cfg.action_decoder = validate_action_decoder_mode(cfg.action_decoder)
        cfg.random_action_center = validate_random_action_center(cfg.random_action_center)
        cfg.terrain_mode = validate_go2_terrain_mode(cfg.terrain_mode)
        cfg.terrain_preset = resolve_go2_terrain_preset(cfg.terrain_mode, cfg.terrain_preset)
        cfg.terrain_curriculum_enabled = bool(cfg.terrain_curriculum_enabled)
        cfg.terrain_curriculum_mode = validate_go2_terrain_curriculum_mode(cfg.terrain_curriculum_mode)
        cfg.terrain_stationary_xy_command_threshold = validate_go2_terrain_nonnegative_float(
            cfg.terrain_stationary_xy_command_threshold,
            field="terrain_stationary_xy_command_threshold",
        )
        cfg.terrain_max_init_level = validate_go2_terrain_nonnegative_int(
            cfg.terrain_max_init_level, field="terrain_max_init_level"
        )
        cfg.terrain_num_rows = validate_go2_terrain_positive_int(cfg.terrain_num_rows, field="terrain_num_rows")
        cfg.terrain_num_cols = validate_go2_terrain_positive_int(cfg.terrain_num_cols, field="terrain_num_cols")
        cfg.height_scan_enabled = bool(cfg.height_scan_enabled)
        cfg.height_scan_observe = bool(cfg.height_scan_observe)
        cfg.height_scan_size = validate_go2_height_scan_size(cfg.height_scan_size, field="height_scan_size")
        cfg.height_scan_resolution = validate_go2_terrain_positive_float(
            cfg.height_scan_resolution, field="height_scan_resolution"
        )
        cfg.height_scan_vertical_offset = validate_go2_terrain_nonnegative_float(
            cfg.height_scan_vertical_offset, field="height_scan_vertical_offset"
        )
        cfg.height_scan_reference_offset = float(cfg.height_scan_reference_offset)
        cfg.height_scan_clip = validate_go2_height_scan_clip(cfg.height_scan_clip, field="height_scan_clip")
        cfg.terrain_debug_vis = bool(cfg.terrain_debug_vis)
        cfg.bad_orientation_termination_enabled = bool(cfg.bad_orientation_termination_enabled)
        cfg.bad_orientation_body_up_threshold = float(cfg.bad_orientation_body_up_threshold)
        if not math.isfinite(cfg.bad_orientation_body_up_threshold) or not (
            -1.0 <= cfg.bad_orientation_body_up_threshold <= 1.0
        ):
            raise ValueError(
                "bad_orientation_body_up_threshold must be finite and in [-1, 1], "
                f"got {cfg.bad_orientation_body_up_threshold}."
            )
        cfg.bad_orientation_hysteresis_steps = int(cfg.bad_orientation_hysteresis_steps)
        if cfg.bad_orientation_hysteresis_steps <= 0:
            raise ValueError(
                f"bad_orientation_hysteresis_steps must be positive, got {cfg.bad_orientation_hysteresis_steps}."
            )
        if int(cfg.observation_history_length) < 0:
            raise ValueError(f"observation_history_length must be non-negative, got {cfg.observation_history_length}.")
        if cfg.base_ang_vel_filter_alpha is not None:
            alpha = float(cfg.base_ang_vel_filter_alpha)
            if not 0.0 < alpha <= 1.0:
                raise ValueError(
                    f"base_ang_vel_filter_alpha must be in (0, 1] when provided, got {cfg.base_ang_vel_filter_alpha}."
                )
            cfg.base_ang_vel_filter_alpha = alpha
        for name in ("rel_standing_envs", "rel_heading_envs", "rel_yaw_in_place_envs"):
            value = float(getattr(cfg, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
            setattr(cfg, name, value)
        yaw_low, yaw_high = (float(v) for v in cfg.yaw_in_place_ang_vel_z_range)
        if yaw_low > yaw_high:
            raise ValueError(
                f"yaw_in_place_ang_vel_z_range lower bound must be <= upper bound, got {(yaw_low, yaw_high)}."
            )
        cfg.yaw_in_place_ang_vel_z_range = (yaw_low, yaw_high)

        if bool(cfg.genesis_style_reset_enabled):
            # genesis_envs/go2_base.py reset_idx: joints are offset ADDITIVELY off the default
            # (reset_joints_by_scale, the port's stock event, multiplies instead -- and its
            # (1.0, 1.0) range means no randomization at all), and the base gets a wider xy
            # spread plus a small roll/pitch tilt that the port's level reset never applies.
            cfg.events.reset_robot_joints = EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={"position_range": (-0.3, 0.3), "velocity_range": (0.0, 0.0)},
            )
            cfg.events.reset_base = EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-1.0, 1.0),
                        "y": (-1.0, 1.0),
                        "roll": (-0.1, 0.1),
                        "pitch": (-0.1, 0.1),
                        "yaw": (0.0, 3.14),
                    },
                    "velocity_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                },
            )

        cfg.action_latency_steps = int(cfg.action_latency_steps)
        if cfg.action_latency_steps not in (0, 1):
            raise ValueError(
                f"action_latency_steps must be 0 or 1 (genesis supports only a single-control-step "
                f"delay), got {cfg.action_latency_steps}."
            )
        if bool(cfg.genesis_style_obs_scaling_enabled):
            # genesis_envs/go2_walk.py obs_cfg: obs_scales lin_vel=2.0, ang_vel=0.25,
            # dof_pos=1.0, dof_vel=0.05 (projected gravity unscaled), and obs_noise ang_vel=0.1,
            # gravity=0.02, dof_pos=0.01, dof_vel=0.5. The noise magnitudes are genesis's
            # POST-scale values, which is why they travel with the scales rather than being
            # left at this cfg's own raw-unit defaults.
            cfg.obs_scale_base_lin_vel = 2.0
            cfg.obs_scale_base_ang_vel = 0.25
            cfg.obs_scale_projected_gravity = 1.0
            cfg.obs_scale_commands_lin_vel = 2.0
            cfg.obs_scale_commands_ang_vel = 0.25
            cfg.obs_scale_joint_pos = 1.0
            cfg.obs_scale_joint_vel = 0.05
            cfg.base_ang_vel_noise = (-0.1, 0.1)
            cfg.projected_gravity_noise = (-0.02, 0.02)
            cfg.joint_pos_noise = (-0.01, 0.01)
            cfg.joint_vel_noise = (-0.5, 0.5)

        base_observation_dim = go2_direct_velocity_base_observation_dim(
            observe_base_lin_vel=bool(cfg.observe_base_lin_vel),
            height_scan_enabled=bool(cfg.height_scan_enabled),
            height_scan_observe=bool(cfg.height_scan_observe),
            height_scan_size=cfg.height_scan_size,
            height_scan_resolution=float(cfg.height_scan_resolution),
        )
        frame_count = int(cfg.observation_history_length) + 1 if bool(cfg.observation_history_enabled) else 1
        # actor_observation_dim is the prefix width FlashSAC's agent slices to when
        # asymmetric_observation=true (see flash_rl/envs/isaaclab_go2.py). cfg.observation_space
        # is the FULL width the base DirectRLEnv builds single_observation_space["policy"] from,
        # so any critic-only privileged addition must be included here, not just the actor part.
        self.actor_observation_dim = base_observation_dim * frame_count
        critic_extra_dim = 3 if bool(cfg.privileged_base_lin_vel) else 0
        if bool(cfg.privileged_last_actions):
            critic_extra_dim += int(cfg.action_space)
        cfg.observation_space = self.actor_observation_dim + critic_extra_dim
        super().__init__(cfg, render_mode, **kwargs)

        self._configure_joint_order_mapping()
        action_dim = self.single_action_space.shape[0]
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = self._robot.data.default_joint_pos.clone()
        self._previous_processed_actions = self._processed_actions.clone()
        self.actions = self._actions
        self._set_action_buffers_to_reset(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

        # Sim-order joint-position bias added to every target in _apply_action. Allocated here
        # (not in the DR block) because _apply_action reads it even with DR disabled.
        self._motor_offsets = torch.zeros_like(self._robot.data.default_joint_pos)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Mirrors genesis's commands_scale = [lin_vel, lin_vel, ang_vel]; a (3,) row vector so
        # it broadcasts over the env dimension. Only touches the observation -- the reward's
        # tracking terms read the unscaled self._commands.
        self._command_obs_scale = torch.tensor(
            [
                float(self.cfg.obs_scale_commands_lin_vel),
                float(self.cfg.obs_scale_commands_lin_vel),
                float(self.cfg.obs_scale_commands_ang_vel),
            ],
            device=self.device,
        )
        self._filtered_base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_policy_base_ang_vel = torch.zeros_like(self._filtered_base_ang_vel)
        self._heading_targets = torch.zeros(self.num_envs, device=self.device)
        self._is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_yaw_in_place_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        self._command_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_command_update_step = -1

        # FlashSAC reward scales: drop zero-scaled terms, mirroring go2_base.py's
        # _prepare_reward_function convention in the Genesis env this was ported from.
        self._reward_scales = {
            key: value
            for key, value in go2_flashsac_reward_scales(
                tracking_lin_vel=float(self.cfg.tracking_lin_vel_reward_scale),
                tracking_ang_vel=float(self.cfg.tracking_ang_vel_reward_scale),
                lin_vel_z=float(self.cfg.lin_vel_z_reward_scale),
                ang_vel_xy=float(self.cfg.ang_vel_xy_reward_scale),
                orientation=float(self.cfg.orientation_reward_scale),
                base_height=float(self.cfg.base_height_reward_scale),
                torques=float(self.cfg.torques_reward_scale),
                dof_vel=float(self.cfg.dof_vel_reward_scale),
                dof_acc=float(self.cfg.dof_acc_reward_scale),
                action_rate=float(self.cfg.action_rate_reward_scale),
                feet_air_time=float(self.cfg.feet_air_time_reward_scale),
                collision=float(self.cfg.collision_reward_scale),
                dof_pos_limits=float(self.cfg.dof_pos_limits_reward_scale),
                termination=float(self.cfg.termination_reward_scale),
            ).items()
            if value != 0.0
        }
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device) for key in self._reward_scales
        }
        self._last_reward_terms: dict[str, torch.Tensor] = {}
        # Per-episode reward-rate summary (python floats), refreshed in _reset_idx before
        # _episode_sums is zeroed for the resetting envs. Mirrors go2_base.py's
        # extras["episode"][name] = mean(episode_sums[env_ids]) / max_episode_length_s.
        self._last_episode_info: dict[str, float] = {}

        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, contact_feet_names = self._contact_sensor.find_bodies(".*_foot")
        self._init_penalized_contact_body_ids()
        self._bad_orientation_hysteresis_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._eval_mode = False
        self._init_observation_history_storage()
        # True terminal observation, snapshotted at the top of _reset_idx (before any reset
        # mutation) for whichever envs are resetting this step. IsaacLab's DirectRLEnv.step()
        # only calls _get_observations() AFTER _reset_idx, so the obs it returns for a done env
        # is already post-reset (see github.com/isaac-sim/IsaacLab/issues/1362) -- the FlashSAC
        # wrapper reads this buffer instead, mirroring genesis_envs/go2_base.py's
        # final_obs_history_buf.
        self._final_observations = torch.zeros(
            self.num_envs,
            int(self.cfg.observation_space),
            dtype=torch.float32,
            device=self.device,
        )
        # Same staleness problem as _final_observations: the base class's _reset_idx zeroes
        # episode_length_buf, so it must be snapshotted before that runs too.
        self._final_episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._base_body_ids, _ = self._robot.find_bodies("base")
        self._feet_body_ids, articulation_feet_names = self._robot.find_bodies(".*_foot")
        if tuple(contact_feet_names) != tuple(articulation_feet_names):
            raise RuntimeError(
                "Foot-body ordering mismatch between contact sensor and articulation: "
                f"{contact_feet_names} vs {articulation_feet_names}"
            )

        self._cache_domain_randomization_defaults()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = None
        if bool(self.cfg.height_scan_enabled):
            self._height_scanner = RayCaster(self._build_height_scanner_cfg())
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain = self._build_runtime_terrain_cfg()
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # The reset event reads scene.env_origins. Register the manually created terrain
        # so direct helpers and IsaacLab reset code share the same origin tensor.
        self.scene._terrain = self._terrain
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _build_height_scanner_cfg(self) -> RayCasterCfg:
        return RayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/base",
            update_period=float(self.cfg.decimation) * float(self.cfg.sim.dt),
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, float(self.cfg.height_scan_vertical_offset))),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=float(self.cfg.height_scan_resolution),
                size=tuple(float(v) for v in self.cfg.height_scan_size),
            ),
            debug_vis=bool(self.cfg.terrain_debug_vis),
            mesh_prim_paths=[self.cfg.terrain.prim_path],
        )

    def _build_runtime_terrain_cfg(self) -> TerrainImporterCfg:
        base_cfg = self.cfg.terrain
        if not go2_terrain_mode_is_rough(self.cfg.terrain_mode):
            base_cfg.terrain_type = "plane"
            base_cfg.terrain_generator = None
            base_cfg.max_init_terrain_level = None
            base_cfg.debug_vis = bool(self.cfg.terrain_debug_vis)
            return base_cfg

        terrain_generator = build_go2_terrain_generator_cfg(
            self.cfg.terrain_preset,
            num_rows=int(self.cfg.terrain_num_rows),
            num_cols=int(self.cfg.terrain_num_cols),
            curriculum_enabled=bool(self.cfg.terrain_curriculum_enabled),
        )
        return TerrainImporterCfg(
            prim_path=base_cfg.prim_path,
            terrain_type=go2_terrain_type_for_mode(self.cfg.terrain_mode),
            terrain_generator=terrain_generator,
            max_init_terrain_level=int(self.cfg.terrain_max_init_level),
            collision_group=int(base_cfg.collision_group),
            physics_material=base_cfg.physics_material,
            visual_material=getattr(base_cfg, "visual_material", None),
            debug_vis=bool(self.cfg.terrain_debug_vis),
        )

    def _configure_joint_order_mapping(self):
        sim_joint_names = tuple(str(name) for name in self._robot.joint_names)
        policy_joint_names = tuple(GO2_POLICY_JOINT_NAMES)
        try:
            policy_to_sim = go2_policy_to_sim_joint_ids(sim_joint_names, policy_joint_names)
        except ValueError as exc:
            raise RuntimeError(
                "Go2 joint-name contract mismatch. Refusing to run because the policy/action adapter "
                "cannot be built safely. "
                f"asset_source={self.cfg.asset_source}, actual={sim_joint_names}, "
                f"policy={policy_joint_names}."
            ) from exc
        self._policy_to_sim_joint_ids = torch.as_tensor(policy_to_sim, dtype=torch.long, device=self.device)

    def _joints_sim_to_policy(self, values: torch.Tensor) -> torch.Tensor:
        return values.index_select(1, self._policy_to_sim_joint_ids.to(values.device))

    def _joints_policy_to_sim(self, values: torch.Tensor) -> torch.Tensor:
        policy_to_sim = self._policy_to_sim_joint_ids.to(values.device)
        out = torch.empty_like(values)
        out.index_copy_(1, policy_to_sim, values)
        return out

    def _default_joint_pos_policy_order(self) -> torch.Tensor:
        return self._joints_sim_to_policy(self._robot.data.default_joint_pos)

    def _soft_joint_pos_limits_policy_order(self) -> torch.Tensor:
        return self._joints_sim_to_policy(self._robot.data.soft_joint_pos_limits)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._previous_processed_actions = self._processed_actions.clone()
        self._actions = actions.clamp(-1.0, 1.0).clone()
        # action_latency_steps=1 executes the PREVIOUS control step's action while the policy
        # still observes the one it just emitted -- exactly genesis_envs/go2_base.py's
        # `exec_actions = self.last_actions if self.action_latency > 0 else self.actions`.
        # The observation and the action_rate penalty both keep using self._actions, matching
        # genesis (its obs_buf carries self.actions, not the delayed copy).
        exec_actions = self._previous_actions if int(self.cfg.action_latency_steps) > 0 else self._actions
        targets_policy_order = self._decode_actions_to_joint_targets(exec_actions)
        self._processed_actions = self._joints_policy_to_sim(targets_policy_order)
        self.actions = self._actions

    def _decode_actions_to_joint_targets(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return decode_go2_action(
            raw_actions,
            self._default_joint_pos_policy_order(),
            self._soft_joint_pos_limits_policy_order(),
            mode=self.cfg.action_decoder,
            action_scale=float(self.cfg.action_scale),
            clip_joint_targets=bool(self.cfg.clip_joint_targets),
        )

    def neutral_action(self) -> torch.Tensor:
        neutral = go2_neutral_action(
            self._default_joint_pos_policy_order(),
            self._soft_joint_pos_limits_policy_order(),
            mode=self.cfg.action_decoder,
            action_scale=float(self.cfg.action_scale),
        )
        return neutral

    def _reset_raw_action(self) -> torch.Tensor:
        if bool(self.cfg.use_neutral_action):
            return self.neutral_action().clamp(-1.0, 1.0)
        return torch.zeros_like(self._actions)

    def _set_action_buffers_to_reset(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        reset_action = self._reset_raw_action()
        reset_target_policy_order = self._decode_actions_to_joint_targets(reset_action)
        reset_target_sim_order = self._joints_policy_to_sim(reset_target_policy_order)
        self._actions[env_ids] = reset_action[env_ids]
        self._previous_actions[env_ids] = reset_action[env_ids]
        self._processed_actions[env_ids] = reset_target_sim_order[env_ids]
        self._previous_processed_actions[env_ids] = reset_target_sim_order[env_ids]
        self.actions = self._actions

    def _apply_action(self):
        # The offset is added here rather than in _pre_physics_step so it also biases the
        # targets written by _set_action_buffers_to_reset, matching genesis -- where the offset
        # sits inside _compute_torques and therefore affects every torque the episode computes.
        self._robot.set_joint_position_target(self._processed_actions + self._motor_offsets)

    def _get_observations(self) -> dict:
        return self.compute_policy_observations(update_history=True)

    def compute_policy_observations(self, update_history: bool = True) -> dict:
        current_obs = self._build_current_policy_observation(commit_filter=update_history)
        actor_obs = self._compose_policy_observation(current_obs, update_history=update_history)
        # Privileged critic-only tails, appended after the actor's own columns so the prefix
        # property (critic[:, :actor_observation_dim] == actor) holds. Neither ever enters the
        # actor's observation_history ring buffer -- they are raw current-frame values, not
        # stacked -- unlike the rest of the actor observation. Concatenation order matches
        # genesis_envs/go2_base.py's privileged_obs_buf: base_lin_vel then last_actions.
        critic_extras = []
        if self.cfg.privileged_base_lin_vel:
            # Scaled but never noised, matching genesis's privileged_obs_buf tail
            # (base_lin_vel * obs_scales["lin_vel"]).
            critic_extras.append(self._robot.data.root_lin_vel_b * float(self.cfg.obs_scale_base_lin_vel))
        if self.cfg.privileged_last_actions:
            # _pre_physics_step rotates _previous_actions before overwriting _actions, so at
            # observation time this is the action from the prior step -- the same quantity
            # genesis calls last_actions (it assigns last_actions = actions only at the very
            # end of its step(), after compute_observations()).
            critic_extras.append(self._previous_actions)
        if not critic_extras:
            return {"policy": self._maybe_clip_observation(actor_obs)}
        return {"policy": self._maybe_clip_observation(torch.cat([actor_obs, *critic_extras], dim=-1))}

    def _snapshot_before_reset(self) -> None:
        """Snapshot true current-step terminal state before any reset mutation runs.

        Called at the very top of _reset_idx, while the articulation is still in its genuine
        terminal pose and before super()._reset_idx() zeroes episode_length_buf. The
        observation snapshot uses update_history=False / commit_filter=False so this extra
        read does not double-shift the observation-history ring buffer or double-commit the
        ang-vel EMA filter for envs that are NOT resetting this step.
        """
        self._final_observations = self.compute_policy_observations(update_history=False)["policy"].detach().clone()
        self._final_episode_length = self.episode_length_buf.detach().clone()

    def _build_current_policy_observation(self, commit_filter: bool = True) -> torch.Tensor:
        self._maybe_update_commands_after_step()
        # Default slices: base ang vel 0:3, projected gravity 3:6, velocity command 6:9,
        # joint pos delta 9:21, joint vel delta 21:33, raw action 33:45.
        # observe_base_lin_vel prepends base lin vel 0:3 and shifts these slices by 3.
        # Per-channel scaling is applied BEFORE the noise, matching genesis's ordering
        # (compute_observations() builds obs_buf from already-scaled terms, then adds
        # `gs_rand_float(-1,1) * obs_noise` on top). With every scale left at its 1.0 default
        # the two orderings are identical, so this costs nothing for unscaled runs.
        obs_terms = []
        if self.cfg.observe_base_lin_vel:
            obs_terms.append(
                self._add_uniform_obs_noise(
                    self._robot.data.root_lin_vel_b * float(self.cfg.obs_scale_base_lin_vel),
                    self.cfg.base_lin_vel_noise,
                )
            )
        base_ang_vel = self._observe_base_ang_vel(commit=commit_filter)
        projected_gravity = self._add_uniform_obs_noise(
            self._robot.data.projected_gravity_b * float(self.cfg.obs_scale_projected_gravity),
            self.cfg.projected_gravity_noise,
        )
        joint_pos_rel = self._add_uniform_obs_noise(
            self._joints_sim_to_policy(self._robot.data.joint_pos - self._robot.data.default_joint_pos)
            * float(self.cfg.obs_scale_joint_pos),
            self.cfg.joint_pos_noise,
        )
        joint_vel_rel = self._add_uniform_obs_noise(
            self._joints_sim_to_policy(self._robot.data.joint_vel - self._robot.data.default_joint_vel)
            * float(self.cfg.obs_scale_joint_vel),
            self.cfg.joint_vel_noise,
        )
        # Commands carry no noise in either sim; genesis scales them by
        # commands_scale = [lin_vel, lin_vel, ang_vel].
        commands = self._commands * self._command_obs_scale
        obs_terms.extend([base_ang_vel, projected_gravity, commands, joint_pos_rel, joint_vel_rel, self._actions])
        height_scan = self._height_scan_observation()
        if height_scan is not None:
            obs_terms.append(height_scan)
        return torch.cat(obs_terms, dim=-1)

    def _observe_base_ang_vel(self, commit: bool = True) -> torch.Tensor:
        base_ang_vel = self._add_uniform_obs_noise(
            self._robot.data.root_ang_vel_b * float(self.cfg.obs_scale_base_ang_vel),
            self.cfg.base_ang_vel_noise,
        )
        base_ang_vel = base_ang_vel_with_optional_ema(
            base_ang_vel,
            self._filtered_base_ang_vel,
            self.cfg.base_ang_vel_filter_alpha,
            commit=commit,
        )
        if commit:
            self._last_policy_base_ang_vel = base_ang_vel.detach().clone()
        return base_ang_vel

    def _height_scan_values(self) -> torch.Tensor | None:
        if not bool(self.cfg.height_scan_enabled):
            return None
        if self._height_scanner is None:
            raise RuntimeError("Go2 height scan is enabled but the RayCaster sensor was not created.")
        scanner_data = self._height_scanner.data
        height = (
            scanner_data.pos_w[:, 2].unsqueeze(1)
            - scanner_data.ray_hits_w[..., 2]
            - float(self.cfg.height_scan_reference_offset)
        )
        height = torch.nan_to_num(height, nan=0.0, posinf=1.0, neginf=-1.0)
        clip_low, clip_high = self.cfg.height_scan_clip
        return torch.clamp(height, min=float(clip_low), max=float(clip_high)).to(dtype=torch.float32)

    def _height_scan_observation(self) -> torch.Tensor | None:
        if not bool(self.cfg.height_scan_observe):
            return None
        return self._height_scan_values()

    def _reset_base_ang_vel_filter(self, env_ids: torch.Tensor | None = None) -> None:
        reset_base_ang_vel_ema_state(self._filtered_base_ang_vel, self._last_policy_base_ang_vel, env_ids)

    def _policy_observation_base_dim(self) -> int:
        return go2_direct_velocity_base_observation_dim(
            observe_base_lin_vel=bool(self.cfg.observe_base_lin_vel),
            height_scan_enabled=bool(self.cfg.height_scan_enabled),
            height_scan_observe=bool(self.cfg.height_scan_observe),
            height_scan_size=self.cfg.height_scan_size,
            height_scan_resolution=float(self.cfg.height_scan_resolution),
        )

    def _policy_observation_frame_count(self) -> int:
        if not bool(self.cfg.observation_history_enabled):
            return 1
        return int(self.cfg.observation_history_length) + 1

    def _init_observation_history_storage(self) -> None:
        frame_count = self._policy_observation_frame_count()
        base_dim = self._policy_observation_base_dim()
        self._observation_history = torch.zeros(
            (self.num_envs, frame_count, base_dim), dtype=torch.float32, device=self.device
        )
        self._observation_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_observation_history(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if len(env_ids) == 0:
            return
        self._observation_history[env_ids] = 0.0
        self._observation_history_valid[env_ids] = False

    def _compose_policy_observation(self, current_obs: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        frame_count = self._policy_observation_frame_count()
        if frame_count <= 1:
            return current_obs
        current_obs = current_obs.detach().clone()
        if current_obs.shape[-1] != self._policy_observation_base_dim():
            raise RuntimeError(
                "Direct Go2 velocity base observation dim mismatch: "
                f"got {current_obs.shape[-1]}, expected {self._policy_observation_base_dim()}."
            )
        valid = self._observation_history_valid
        if update_history:
            if torch.any(valid):
                self._observation_history[valid, :-1] = self._observation_history[valid, 1:].clone()
                self._observation_history[valid, -1] = current_obs[valid]
            invalid = ~valid
            if torch.any(invalid):
                self._observation_history[invalid] = current_obs[invalid].unsqueeze(1).expand(-1, frame_count, -1)
                self._observation_history_valid[invalid] = True
            return self._observation_history.reshape(self.num_envs, -1).clone()

        # Non-mutating variant: what the composed history WOULD be if current_obs were
        # appended, without touching self._observation_history. Used by
        # _snapshot_before_reset() so envs that are not resetting this step keep an
        # untouched history ring buffer.
        transient_history = self._observation_history.clone()
        if torch.any(valid):
            transient_history[valid, :-1] = self._observation_history[valid, 1:]
            transient_history[valid, -1] = current_obs[valid]
        invalid = ~valid
        if torch.any(invalid):
            transient_history[invalid] = current_obs[invalid].unsqueeze(1).expand(-1, frame_count, -1)
        return transient_history.reshape(self.num_envs, -1)

    def _add_uniform_obs_noise(self, value: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
        if not self.cfg.enable_observation_noise:
            return value
        low, high = bounds
        if bool(self.cfg.obs_noise_shared_across_envs):
            # One draw per channel, broadcast over the env dimension -- genesis's
            # `gs_rand_float(-1, 1, (num_single_obs,))` is a (45,) tensor added to a
            # (num_envs, 45) buffer, so the whole batch shares a perturbation each step.
            shape = (1,) * (value.dim() - 1) + (value.shape[-1],)
            noise = torch.empty(shape, dtype=value.dtype, device=value.device).uniform_(float(low), float(high))
            return value + noise
        return value + torch.empty_like(value).uniform_(float(low), float(high))

    def _maybe_clip_observation(self, value: torch.Tensor) -> torch.Tensor:
        if self.cfg.obs_clip is None:
            return value
        limit = abs(float(self.cfg.obs_clip))
        return torch.clamp(value, min=-limit, max=limit)

    def _body_ids_list(self, body_ids) -> list[int]:
        if isinstance(body_ids, torch.Tensor):
            return [int(value) for value in body_ids.detach().cpu().reshape(-1).tolist()]
        if isinstance(body_ids, (list, tuple)):
            return [int(value) for value in body_ids]
        return [int(body_ids)]

    def _body_ids_tensor(self, body_ids) -> torch.Tensor:
        return torch.as_tensor(self._body_ids_list(body_ids), dtype=torch.long, device=self.device).reshape(-1)

    def _find_contact_bodies_optional(self, *patterns: str) -> tuple[torch.Tensor, tuple[str, ...]]:
        for pattern in patterns:
            try:
                body_ids, body_names = self._contact_sensor.find_bodies(pattern)
            except Exception:
                continue
            body_ids = self._body_ids_tensor(body_ids)
            if body_ids.numel() > 0:
                return body_ids, tuple(str(name) for name in body_names)
        return torch.empty(0, dtype=torch.long, device=self.device), tuple()

    def _init_penalized_contact_body_ids(self) -> None:
        """Contact-sensor body indices for the collision penalty (base + thigh + calf)."""
        thigh_ids, _ = self._find_contact_bodies_optional(".*_thigh", ".*thigh.*")
        calf_ids, _ = self._find_contact_bodies_optional(".*_calf", ".*calf.*")
        self._penalized_contact_body_ids = torch.cat(
            [self._body_ids_tensor(self._base_id), thigh_ids, calf_ids]
        ).unique()

    def _contact_force_norm_history(self, body_ids) -> torch.Tensor:
        body_ids = self._body_ids_tensor(body_ids)
        if body_ids.numel() == 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float32, device=self.device)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        force_norm = torch.linalg.norm(net_contact_forces[:, :, body_ids, :], dim=-1).max(dim=1)[0]
        if force_norm.ndim == 1:
            force_norm = force_norm.unsqueeze(1)
        return force_norm

    def _get_rewards(self) -> torch.Tensor:
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        penalized_contact_force_norm = self._contact_force_norm_history(self._penalized_contact_body_ids)

        terms = go2_flashsac_reward_terms(
            commands=self._commands,
            base_lin_vel_b=self._robot.data.root_lin_vel_b,
            base_ang_vel_b=self._robot.data.root_ang_vel_b,
            projected_gravity_b=self._robot.data.projected_gravity_b,
            applied_torque=self._robot.data.applied_torque,
            joint_vel=self._robot.data.joint_vel,
            joint_acc=self._robot.data.joint_acc,
            actions=self._actions,
            previous_actions=self._previous_actions,
            base_pos_z=self._robot.data.root_pos_w[:, 2],
            penalized_contact_force_norm=penalized_contact_force_norm,
            joint_pos=self._robot.data.joint_pos,
            soft_joint_pos_limits=self._robot.data.soft_joint_pos_limits,
            last_air_time=last_air_time,
            first_contact=first_contact,
            terminated=self.reset_terminated,
            tracking_sigma=float(self.cfg.tracking_sigma),
            base_height_target=float(self.cfg.base_height_target),
            collision_force_threshold=float(self.cfg.collision_force_threshold),
            feet_air_time_offset=float(self.cfg.feet_air_time_offset),
            feet_air_time_command_threshold=float(self.cfg.feet_air_time_command_lin_vel_threshold),
        )
        reward, weighted_terms = go2_flashsac_scalar_reward(terms, self._reward_scales, self.step_dt)
        for key, value in weighted_terms.items():
            self._episode_sums[key] += value
        self._last_reward_terms = {key: value.detach().clone() for key, value in weighted_terms.items()}
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_contact_force = self._contact_force_norm_history(self._base_id).amax(dim=1)
        base_contact = base_contact_force > 1.0
        raw_bad_orientation, _ = go2_bad_orientation_mask_from_projected_gravity(
            self._robot.data.projected_gravity_b,
            body_up_threshold=float(self.cfg.bad_orientation_body_up_threshold),
        )
        bad_orientation_hysteresis, self._bad_orientation_hysteresis_count = go2_update_bad_orientation_hysteresis(
            raw_bad_orientation,
            self._bad_orientation_hysteresis_count,
            hysteresis_steps=int(self.cfg.bad_orientation_hysteresis_steps),
        )
        terminated = torch.zeros_like(base_contact)
        if bool(self.cfg.enable_termination):
            terminated = base_contact
            if bool(self.cfg.bad_orientation_termination_enabled):
                terminated = terminated | bad_orientation_hysteresis
            if bool(self.cfg.genesis_style_termination_enabled):
                # Matches go2_base.py's check_termination() exactly: base_euler there is
                # relative to base_init_quat, which for go2-walk is the identity quaternion,
                # so it reduces to the plain world-frame roll/pitch computed here.
                roll, pitch, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
                bad_roll_pitch = (roll.abs() > float(self.cfg.termination_roll_threshold)) | (
                    pitch.abs() > float(self.cfg.termination_pitch_threshold)
                )
                too_low = self._robot.data.root_pos_w[:, 2] < float(self.cfg.termination_min_base_height)
                terminated = terminated | bad_roll_pitch | too_low
        return terminated, time_out

    def _terrain_curriculum_enabled_for_reset(self) -> bool:
        if not go2_terrain_mode_is_rough(self.cfg.terrain_mode):
            return False
        if not bool(self.cfg.terrain_curriculum_enabled):
            return False
        if str(self.cfg.terrain_curriculum_mode) != GO2_TERRAIN_CURRICULUM_MODE_DISTANCE:
            return False
        if getattr(self._terrain, "terrain_origins", None) is None:
            return False
        if not hasattr(self._terrain, "update_env_origins"):
            return False
        return True

    def _terrain_tile_length(self) -> float:
        generator = getattr(self.cfg.terrain, "terrain_generator", None)
        size = getattr(generator, "size", None)
        if size is not None and len(size) > 0:
            return float(size[0])
        return float(getattr(self.scene.cfg, "env_spacing", 0.0))

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        if not self._terrain_curriculum_enabled_for_reset():
            return
        result = compute_go2_distance_terrain_curriculum(
            self._robot.data.root_pos_w[env_ids],
            self.scene.env_origins[env_ids],
            self._commands[env_ids],
            terrain_tile_length=self._terrain_tile_length(),
            episode_length_s=float(self.max_episode_length) * float(self.step_dt),
            stationary_xy_command_threshold=float(self.cfg.terrain_stationary_xy_command_threshold),
            enabled=True,
        )
        eligible = self._command_counter[env_ids] > 0
        move_up = result.move_up & eligible
        move_down = result.move_down & eligible
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Must run before any reset mutation below -- the articulation is still in its
        # genuine terminal pose at this point. See _snapshot_before_reset().
        self._snapshot_before_reset()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        self._update_terrain_curriculum(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._apply_or_restore_domain_randomization(env_ids)
        if self.cfg.randomize_episode_lengths and len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._set_action_buffers_to_reset(env_ids)
        self._reset_base_ang_vel_filter(env_ids)
        self._resample_commands(env_ids)
        self._reset_observation_history(env_ids)
        self._bad_orientation_hysteresis_count[env_ids] = 0
        # Must run before the zeroing loop just below.
        self._update_episode_info(env_ids)
        for values in self._episode_sums.values():
            values[env_ids] = 0.0

    def _update_episode_info(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            self._last_episode_info = {}
            return
        episode_length_s = float(self.max_episode_length) * float(self.step_dt)
        self._last_episode_info = {
            f"Reward/{name}": (values[env_ids].mean() / episode_length_s).item()
            for name, values in self._episode_sums.items()
        }
        self._last_episode_info["episode_length"] = self._final_episode_length[env_ids].to(torch.float32).mean().item()

    def _resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        sample = sample_go2_velocity_commands(
            len(env_ids),
            device=self.device,
            lin_vel_x_range=self.cfg.lin_vel_x_range,
            lin_vel_y_range=self.cfg.lin_vel_y_range,
            ang_vel_z_range=self.cfg.ang_vel_z_range,
            yaw_in_place_ang_vel_z_range=self.cfg.yaw_in_place_ang_vel_z_range,
            heading_range=self.cfg.heading_range,
            command_resampling_time_range=self.cfg.command_resampling_time_range,
            heading_command=bool(self.cfg.heading_command),
            rel_standing_envs=float(self.cfg.rel_standing_envs),
            rel_yaw_in_place_envs=float(self.cfg.rel_yaw_in_place_envs),
            rel_heading_envs=float(self.cfg.rel_heading_envs),
        )
        # Applied after mode selection so a standing/yaw-in-place env keeps its zeroed axes;
        # genesis has no such modes and applies the deadband straight to the fresh sample.
        sampled_commands = apply_go2_command_deadband(
            sample["commands"],
            float(self.cfg.command_deadband_lin_vel),
            float(self.cfg.command_deadband_ang_vel),
        )
        self._command_time_left[env_ids] = sample["command_time_left"]
        self._commands[env_ids] = sampled_commands
        self._heading_targets[env_ids] = sample["heading_targets"]
        self._is_standing_env[env_ids] = sample["is_standing"]
        self._is_yaw_in_place_env[env_ids] = sample["is_yaw_in_place"]
        self._is_heading_env[env_ids] = sample["is_heading"]
        self._apply_command_modes(env_ids)
        self._command_counter[env_ids] += 1

    def set_eval_mode(self, enabled: bool):
        self._eval_mode = bool(enabled)

    def _cache_domain_randomization_defaults(self) -> None:
        self._dr_default_masses = self._robot.root_physx_view.get_masses().clone()
        self._dr_default_inertias = self._robot.root_physx_view.get_inertias().clone()
        self._dr_default_coms = self._robot.root_physx_view.get_coms().clone()
        self._dr_default_materials = self._robot.root_physx_view.get_material_properties().clone()
        self._dr_default_actuator_state = {}
        for name, actuator in self._robot.actuators.items():
            state = {"effort_limit": actuator.effort_limit.detach().clone()}
            for gain_field in PD_GAIN_FIELDS:
                gains = getattr(actuator, gain_field, None)
                if isinstance(gains, torch.Tensor):
                    state[gain_field] = gains.detach().clone()
            if hasattr(actuator, "motor_strength"):
                state["motor_strength"] = actuator.motor_strength.detach().clone()
            if hasattr(actuator, "scaled_motor_strength"):
                state["scaled_motor_strength"] = actuator.scaled_motor_strength.detach().clone()
            if hasattr(actuator, "_saturation_effort"):
                state["saturation_effort"] = as_actuator_tensor(actuator._saturation_effort, actuator.effort_limit)
            if hasattr(actuator, "_vel_at_effort_lim"):
                state["vel_at_effort_lim"] = as_actuator_tensor(actuator._vel_at_effort_lim, actuator.effort_limit)
            self._dr_default_actuator_state[name] = state

    def _apply_or_restore_domain_randomization(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0 or not self.cfg.domain_randomization_enabled:
            return
        randomize = (not self._eval_mode) or (not self.cfg.domain_randomization_train_only)
        if not randomize:
            self._restore_domain_randomization_defaults(env_ids)
            return
        env_ids_cpu = env_ids.detach().cpu()
        self._randomize_base_mass(env_ids_cpu)
        self._randomize_base_com(env_ids_cpu)
        self._randomize_friction(env_ids_cpu)
        self._randomize_motor_strength(env_ids)
        self._randomize_pd_gains(env_ids)
        self._randomize_motor_offsets(env_ids)

    def _restore_domain_randomization_defaults(self, env_ids: torch.Tensor) -> None:
        env_ids_cpu = env_ids.detach().cpu()
        masses = self._robot.root_physx_view.get_masses()
        inertias = self._robot.root_physx_view.get_inertias()
        masses[env_ids_cpu] = self._dr_default_masses[env_ids_cpu]
        inertias[env_ids_cpu] = self._dr_default_inertias[env_ids_cpu]
        self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
        self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
        coms = self._robot.root_physx_view.get_coms()
        coms[env_ids_cpu] = self._dr_default_coms[env_ids_cpu]
        self._robot.root_physx_view.set_coms(coms, env_ids_cpu)
        materials = self._robot.root_physx_view.get_material_properties()
        materials[env_ids_cpu] = self._dr_default_materials[env_ids_cpu]
        self._robot.root_physx_view.set_material_properties(materials, env_ids_cpu)
        restore_motor_strength_defaults(
            self._robot.actuators,
            self._dr_default_actuator_state,
            env_ids,
            refresh_velocity_limit=self._refresh_dc_motor_velocity_limit,
        )
        restore_pd_gain_defaults(self._robot.actuators, self._dr_default_actuator_state, env_ids)
        clear_motor_offsets(self._motor_offsets, env_ids)

    def _randomize_base_mass(self, env_ids_cpu: torch.Tensor) -> None:
        body_ids = torch.as_tensor(self._base_body_ids, dtype=torch.long)
        masses = self._robot.root_physx_view.get_masses()
        inertias = self._robot.root_physx_view.get_inertias()
        masses[env_ids_cpu[:, None], body_ids] = self._dr_default_masses[env_ids_cpu[:, None], body_ids].clone()
        low, high = self.cfg.dr_base_mass_add_range
        delta = torch.empty((len(env_ids_cpu), len(body_ids)), dtype=masses.dtype).uniform_(float(low), float(high))
        new_masses = torch.clamp(masses[env_ids_cpu[:, None], body_ids] + delta, min=1.0e-6)
        masses[env_ids_cpu[:, None], body_ids] = new_masses
        ratios = new_masses / self._dr_default_masses[env_ids_cpu[:, None], body_ids].clamp(min=1.0e-6)
        inertias[env_ids_cpu[:, None], body_ids] = (
            self._dr_default_inertias[env_ids_cpu[:, None], body_ids] * ratios[..., None]
        )
        self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
        self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

    def _randomize_base_com(self, env_ids_cpu: torch.Tensor) -> None:
        body_ids = torch.as_tensor(self._base_body_ids, dtype=torch.long)
        coms = self._dr_default_coms.clone()
        ranges = [self.cfg.dr_base_com_range_x, self.cfg.dr_base_com_range_y, self.cfg.dr_base_com_range_z]
        offsets = torch.stack(
            [torch.empty(len(env_ids_cpu), dtype=coms.dtype).uniform_(float(low), float(high)) for low, high in ranges],
            dim=1,
        )
        coms[env_ids_cpu[:, None], body_ids, :3] += offsets[:, None, :]
        self._robot.root_physx_view.set_coms(coms, env_ids_cpu)

    def _randomize_friction(self, env_ids_cpu: torch.Tensor) -> None:
        materials = self._dr_default_materials.clone()
        num_shapes = materials.shape[1]
        static_low, static_high = self.cfg.dr_static_friction_range
        dynamic_low, dynamic_high = self.cfg.dr_dynamic_friction_range
        restitution_low, restitution_high = self.cfg.dr_restitution_range
        static = torch.empty((len(env_ids_cpu), num_shapes), dtype=materials.dtype).uniform_(
            float(static_low), float(static_high)
        )
        dynamic = torch.empty((len(env_ids_cpu), num_shapes), dtype=materials.dtype).uniform_(
            float(dynamic_low), float(dynamic_high)
        )
        dynamic = torch.minimum(dynamic, static)
        restitution = torch.empty((len(env_ids_cpu), num_shapes), dtype=materials.dtype).uniform_(
            float(restitution_low), float(restitution_high)
        )
        materials[env_ids_cpu, :, 0] = static
        materials[env_ids_cpu, :, 1] = dynamic
        materials[env_ids_cpu, :, 2] = restitution
        self._robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def _randomize_motor_strength(self, env_ids: torch.Tensor) -> None:
        randomize_motor_strength(
            self._robot.actuators,
            self._dr_default_actuator_state,
            env_ids,
            tuple(self.cfg.dr_motor_strength_range),
            bool(self.cfg.dr_motor_strength_per_joint),
            refresh_velocity_limit=self._refresh_dc_motor_velocity_limit,
            require_motor_strength=not motor_strength_range_is_default(tuple(self.cfg.dr_motor_strength_range)),
        )

    def _randomize_motor_offsets(self, env_ids: torch.Tensor) -> None:
        offset_range = tuple(self.cfg.dr_motor_offset_range)
        if motor_offset_range_is_default(offset_range):
            clear_motor_offsets(self._motor_offsets, env_ids)
            return
        sample_motor_offsets(self._motor_offsets, env_ids, offset_range)

    def _randomize_pd_gains(self, env_ids: torch.Tensor) -> None:
        randomize_pd_gains(
            self._robot.actuators,
            self._dr_default_actuator_state,
            env_ids,
            tuple(self.cfg.dr_kp_scale_range),
            tuple(self.cfg.dr_kd_scale_range),
        )

    def _refresh_dc_motor_velocity_limit(self, actuator) -> None:
        if not hasattr(actuator, "_vel_at_effort_lim") or not hasattr(actuator, "_saturation_effort"):
            return
        saturation = as_actuator_tensor(actuator._saturation_effort, actuator.effort_limit).clamp(min=1.0e-6)
        actuator._saturation_effort = saturation
        actuator._vel_at_effort_lim = actuator.velocity_limit * (1.0 + actuator.effort_limit / saturation)

    def _maybe_update_commands_after_step(self):
        step = int(self.common_step_counter)
        if step <= 0 or step == self._last_command_update_step:
            return
        self._command_time_left -= self.step_dt
        resample_env_ids = (self._command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(resample_env_ids) > 0:
            self._resample_commands(resample_env_ids)
        self._apply_command_modes()
        self._last_command_update_step = step

    def _apply_command_modes(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.heading_command:
            heading_env_ids = env_ids[self._is_heading_env[env_ids]]
            if len(heading_env_ids) > 0:
                heading_error = math_utils.wrap_to_pi(
                    self._heading_targets[heading_env_ids] - self._robot.data.heading_w[heading_env_ids]
                )
                self._commands[heading_env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=float(self.cfg.ang_vel_z_range[0]),
                    max=float(self.cfg.ang_vel_z_range[1]),
                )
        standing_env_ids = env_ids[self._is_standing_env[env_ids]]
        if len(standing_env_ids) > 0:
            self._commands[standing_env_ids, :] = 0.0

    def set_velocity_command(self, command: torch.Tensor):
        command = command.to(self.device, dtype=torch.float32)
        if command.ndim == 1:
            command = command.unsqueeze(0)
        self._commands[:] = command[:, :3]
        self._command_time_left[:] = float(self.cfg.command_resampling_time_range[1])
        self._is_heading_env[:] = False
        self._is_standing_env[:] = False
        self._is_yaw_in_place_env[:] = False

    def _update_tracking_camera(self) -> None:
        """Point the viewport camera at the robot(s), tracking as they move.

        cfg.video_camera_mode='swarm' (default) targets the centroid of every env's robot --
        a wide overview of the whole batch. 'single_env' instead chases one robot
        (cfg.video_camera_env_index) with the same close chase-cam offset
        flash_rl/envs/genesis_envs/go2_base.py's render() always uses for its single-env view.
        """
        if self.cfg.video_camera_mode == "single_env":
            pos = self._robot.data.root_pos_w[self.cfg.video_camera_env_index]
            target = (float(pos[0]), float(pos[1]), float(pos[2]) - 0.1)
            eye = (float(pos[0]) - 1.0, float(pos[1]) - 1.0, float(pos[2]) + 0.5)
        else:
            centroid = self._robot.data.root_pos_w.mean(dim=0)
            target = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
            eye = (float(centroid[0]) - 3.5, float(centroid[1]) - 3.5, float(centroid[2]) + 2.2)
        self.sim.set_camera_view(eye, target)

    def render(self, recompute: bool = False):
        # Only the base DirectRLEnv's built-in rgb_array path needs a positioned camera;
        # render_mode is None during normal (non-recording) training/eval, so this is a no-op
        # unless the env was constructed with render_mode="rgb_array".
        if self.render_mode == "rgb_array":
            self._update_tracking_camera()
        return super().render(recompute=recompute)
