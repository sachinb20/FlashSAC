"""IsaacLab/PhysX implementation of ``Go2SimBackend``.

This is a *physics backend only*. It deliberately does not use ``DirectRLEnv``,
``ManagerBasedRLEnv``, or any IsaacLab reward/observation/termination machinery: all of
that already exists in ``go2_env.py`` and is shared with the Genesis arm. What lives here
is a ``SimulationContext``, an ``Articulation``, a ``ContactSensor``, and the index and
frame translation needed to make PhysX answer the same questions Genesis answers.

Two robot descriptions are selectable via ``asset_source`` (see ``go2_urdf.py``):

``genesis_merged``  the pre-merged URDF the Genesis backend loads, with
                    ``merge_fixed_joints=False`` so PhysX keeps exactly the 17 links
                    Genesis produces. 15.019 kg.
``unitree_urdf``    the upstream Unitree description TDMPC2 trains against, spawned with
                    TDMPC2's own settings. Imports to 31 bodies / 16.087 kg. Rung 1 of
                    the port; not yet paired with TDMPC2's actuator or rewards.

Torque control: the actuator is configured with zero stiffness and damping and an
effectively unbounded effort limit, so ``set_joint_effort_target`` writes the torque the
shared PD law computed, unmodified. Genesis applies no torque ceiling either -- the
URDF's 23.7/35.55 N.m limits are not enforced by ``control_dofs_force``.

Written against IsaacLab 2.3 (the ``python_version == '3.11'`` pin in ``pyproject.toml``).
IsaacLab's converter API moved between 2.1 and 2.3; ``_build_urdf_spawn_cfg`` handles both
shapes and raises a legible error rather than a stray ``TypeError`` if it meets a third.

Runs in ``.venv-isaaclab`` against isaacsim 5.1.0.0; it cannot be exercised from the
Genesis virtualenv (``pyproject.toml`` declares the extras mutually exclusive). See
``GO2_SIM_BACKEND.md``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from ..go2_urdf import get_merged_urdf, get_staged_unitree_urdf
from ..sim_backend import resolve_link_indices_by_pattern

_SIM_APP = None

GROUND_MATERIALS = ("isaaclab_default", "tdmpc2")


def _ground_material_cfg(sim_utils: Any, ground_material: str) -> Any:
    """Contact material for the ground plane.

    Two presets:

    ``isaaclab_default``  static/dynamic 0.5, combine ``"average"`` -- what a bare
                          ``GroundPlaneCfg()`` gives you.
    ``tdmpc2``            static/dynamic 1.0, combine ``"multiply"`` -- the material
                          TDMPC2 hands to its ``TerrainImporterCfg``.

    The **combine mode** is the load-bearing half. PhysX gives each collider its own
    material, so a robot-foot/ground contact has two friction values and must reduce them
    to one; the mode decides how. Under ``average`` the ground drags the result toward its
    own 0.5 and the foot only ever contributes half. Under ``multiply`` against a ground of
    exactly 1.0 the product is the foot's own coefficient, so the ground becomes
    transparent and the robot's material alone sets the contact -- which is what makes
    TDMPC2's friction randomisation land on the contact undiluted.

    That difference is not symmetric between the two arms and is not merely a scale factor:
    with our friction DR the robot's material is a random draw, and ``(x + 0.5)/2`` is a
    different distribution from ``x`` in both mean and spread.

    Setting it on the ground alone is sufficient. When two colliding materials disagree on
    the combine mode PhysX takes the higher-priority one (``PxCombineMode`` order
    average < min < multiply < max), so ``multiply`` on the ground wins over ``average``
    on the robot regardless of what the URDF import produced.
    """
    if ground_material not in GROUND_MATERIALS:
        raise ValueError(f"Unknown ground_material {ground_material!r}, expected one of {GROUND_MATERIALS}.")
    if ground_material == "tdmpc2":
        return sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    return sim_utils.RigidBodyMaterialCfg()


def _launch_app(headless: bool, enable_cameras: bool) -> Any:
    """Boot Isaac Sim once per process.

    IsaacLab modules cannot be imported before the app exists, which is why every
    IsaacLab import in this file is function-local.
    """
    global _SIM_APP
    if _SIM_APP is None:
        from isaaclab.app import AppLauncher

        _SIM_APP = AppLauncher(headless=headless, enable_cameras=enable_cameras).app
    return _SIM_APP


class IsaacLabSimBackend:
    def __init__(
        self,
        num_envs: int,
        num_build_envs: int,
        sim_dt: float,
        control_dt: float,
        urdf_path: str,
        links_to_keep: Sequence[str],
        base_init_pos: torch.Tensor,
        base_init_quat: torch.Tensor,
        show_viewer: bool,
        debug: bool,
        device: torch.device,
        default_joint_angles: Optional[dict[str, float]] = None,
        dof_armature: float = 0.1,
        enable_camera: bool = False,
        env_spacing: float = 4.0,
        asset_source: str = "genesis_merged",
        ground_material: str = "isaaclab_default",
    ) -> None:
        if asset_source not in ("genesis_merged", "unitree_urdf"):
            raise ValueError(f"Unknown asset_source {asset_source!r}, expected 'genesis_merged' or 'unitree_urdf'.")
        # enable_cameras must be decided at app launch, before any IsaacLab import.
        _launch_app(headless=not show_viewer, enable_cameras=show_viewer or enable_camera)

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils import configclass

        self.device = device
        self.num_envs = num_envs
        self.num_build_envs = num_build_envs
        self.debug = debug
        self._sim_dt = sim_dt
        self._enable_camera = enable_camera
        self._camera = None

        self.asset_source = asset_source
        if asset_source == "unitree_urdf":
            asset_urdf = get_staged_unitree_urdf(urdf_path)
        else:
            asset_urdf = get_merged_urdf(urdf_path, links_to_keep)

        self.sim = SimulationContext(SimulationCfg(dt=sim_dt, device=str(device)))

        spawn_cfg = self._build_urdf_spawn_cfg(sim_utils, asset_urdf, asset_source)

        robot_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=spawn_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(base_init_pos.cpu().numpy().tolist()),
                rot=tuple(base_init_quat.cpu().numpy().tolist()),  # wxyz, as Genesis
                # Must be a valid stance. IsaacLab validates the spawn pose against the
                # joint limits and hard-errors, and the URDF's implicit all-zeros default
                # is illegal: the calf joints are limited to [-2.723, -0.838]. Genesis only
                # warns about the same thing, then the env's reset overwrites it. Seeding
                # the genesis nominal stance here keeps the two spawns equivalent.
                joint_pos=dict(default_joint_angles or {}),
            ),
            # Zero gains + unbounded effort => PhysX applies our torque verbatim.
            actuators={
                "all_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0,
                    damping=0.0,
                    effort_limit_sim=1.0e9,
                    velocity_limit_sim=1.0e9,
                    # Matches Genesis's solver default (see env_cfg["dof_armature"]).
                    # PhysX defaults this to 0, which puts the shared explicit PD past its
                    # stability limit and makes the legs oscillate at +/-10 rad/s.
                    armature=dof_armature,
                )
            },
        )

        @configclass
        class _Go2SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(physics_material=_ground_material_cfg(sim_utils, ground_material)),
            )
            # Lights. IsaacLab's InteractiveScene ships none, and without one the RTX
            # renderer returns a near-black image. Purely visual: lights are inert prims
            # with no effect on physics, so this cannot perturb the dynamics comparison.
            #
            # These are TDMPC2's exact values (its _setup_scene ends with
            # DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))), so the two
            # recordings are directly comparable.
            #
            # The dome is deliberately left VISIBLE in primary rays. An earlier version
            # hid it, reasoning that it wanted a light source and not a backdrop -- but
            # the ground plane is solid black on both sides (IsaacLab's
            # TerrainImporterCfg.visual_material defaults to diffuse_color=(0,0,0), which
            # it forwards into GroundPlaneCfg's color, and GroundPlaneCfg's own default is
            # the same black). With the dome hidden the sky is black too, so a black
            # ground against a black sky is simply invisible: the recording showed a white
            # robot floating in a void with no ground, horizon or contact shadow. The
            # visible dome is what gives the ground something to read against.
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
            )
            robot: ArticulationCfg = robot_cfg
            contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=0,
                track_air_time=False,
            )

        self.scene = InteractiveScene(_Go2SceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True))

        # The camera must exist BEFORE sim.reset(): sensors are initialised by a callback
        # fired during reset, and that callback is what populates Camera._ALL_INDICES.
        # Creating it afterwards yields a camera that raises AttributeError on first use.
        if enable_camera:
            # No robot visual material is bound. TDMPC2 binds none either, so its Go2
            # renders in the converter's default white -- matching that is the point.
            # (The override that used to live here was also silently failing: the frame
            # showed a white robot, not the dark grey it asked for.)
            self._setup_camera(sim_utils)

        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor = self.scene["contact_forces"]

        self.body_names: list[str] = list(self.robot.body_names)
        self.n_links = len(self.body_names)
        self.env_origins = self.scene.env_origins.to(self.device)

        # The contact sensor keeps its own body ordering; map it onto the articulation's.
        sensor_names = list(self.contact_sensor.body_names)
        self._contact_perm = torch.tensor(
            [sensor_names.index(name) for name in self.body_names], device=self.device, dtype=torch.long
        )

        self._base_friction: Optional[torch.Tensor] = None

    def _setup_camera(self, sim_utils: Any) -> None:
        """Prepare the viewport render path used for recording.

        Renders through the persp viewport camera (``/OmniverseKit_Persp``) at 1280x720
        via a replicator render product, which is how ``DirectRLEnv``'s ``rgb_array`` mode
        works and what the earlier isaaclab_go2 port used for its recordings.

        A ``Camera`` sensor was tried first and looked markedly worse: at Genesis's 320x320
        it falls under the RTX pipeline's minimum input resolution (Isaac logs
        ``DLSS increasing input dimensions: Render resolution of (186, 186) is below
        minimal input resolution of 300``) and bypasses the viewport's post-processing.
        Matching Genesis's exact frame size is not worth a visibly degraded image, so the
        frame is larger here than on the Genesis arm.

        Failure is downgraded to a warning: a missing camera should cost the video, not
        the training run.
        """
        del sim_utils
        try:
            import omni.replicator.core as rep

            # 640x480 rather than the viewer default 1280x720: smaller frames keep the
            # recorded gifs light, and this still clears the RTX pipeline's minimum input
            # resolution of 300 (the renderer works at roughly 58% of output, so anything
            # below ~520 wide starts getting upscaled and looks soft). Genesis's own
            # 320x320 is well under that, which is why matching it exactly looked worse.
            self._render_resolution = (640, 480)
            self._render_product = rep.create.render_product("/OmniverseKit_Persp", self._render_resolution)
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
            self._camera = "viewport"
        except Exception as exc:  # noqa: BLE001 - never let the camera kill training
            print(f"[go2/isaaclab] camera setup failed, video disabled: {exc}")
            self._camera = None

    @staticmethod
    def _build_urdf_spawn_cfg(sim_utils: Any, merged_urdf: str, asset_source: str = "genesis_merged") -> Any:
        """Build a ``UrdfFileCfg`` across IsaacLab converter API revisions.

        2.3 nests drive settings under ``joint_drive=JointDriveCfg(gains=PDGainsCfg(...))``;
        2.1 exposes flat ``default_drive_*`` fields. We want a drive with zero gains either
        way, because the shared PD law is what actually drives the robot.

        ``asset_source`` selects between the two robot descriptions. The settings that
        differ are exactly the ones TDMPC2 sets in ``_make_go2_urdf_spawn_cfg``: it leaves
        ``merge_fixed_joints`` at IsaacLab's default (True) because its URDF is unmerged,
        turns cylinders into capsules, and runs a stiffer solver (8/4 rather than 4/0).
        """
        is_unitree = asset_source == "unitree_urdf"

        common = dict(
            asset_path=merged_urdf,
            fix_base=False,
            # genesis_merged is already merged by go2_urdf.py, so ask the importer to keep
            # every link. The upstream asset is unmerged and TDMPC2 never sets this, so it
            # gets IsaacLab's default of True.
            #
            # Note what that default actually does under Isaac Sim 5.1: a fixed-joint child
            # is merged only when it carries no mass. Measured on the upstream file, 42
            # links become 31 -- calflower/calflower1/front_camera (no <inertial>) and
            # imu/radar (mass 0.0) fold in, while the twelve 0.089 kg *_rotor links, the
            # 0.001 kg heads, and the 0.04 kg feet all survive as separate bodies. The feet
            # surviving is load-bearing and is pure luck: they are kept because they weigh
            # 40 g, not because anything asked for them.
            merge_fixed_joints=True if is_unitree else False,
            # Upstream ships cylinder collision primitives; TDMPC2 converts them to
            # capsules, which changes contact geometry, so it is part of the asset choice.
            replace_cylinders_with_capsules=is_unitree,
            # Without this PhysX attaches no contact-reporter API to the bodies and the
            # ContactSensor fails to initialise. Genesis reports contact forces for every
            # link unconditionally; on PhysX it is opt-in at spawn time. The environment
            # needs it for both termination (base contact) and the feet_air_time reward.
            activate_contact_sensors=True,
            # Leaving these unset means PhysX defaults, under which the robot chatters:
            # an unbounded depenetration velocity turns any contact penetration into a
            # violent pop, the legs get kicked, and the PD law fights back with torques
            # far above the motor limits. IsaacLab's own Unitree/ANYmal configs set
            # exactly these values. Genesis's Newton solver needs no equivalent.
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
                # Genesis runs with enable_self_collision=True, so match it here rather
                # than copying IsaacLab's quadruped default of False.
                enabled_self_collisions=True,
                solver_position_iteration_count=8 if is_unitree else 4,
                solver_velocity_iteration_count=4 if is_unitree else 0,
            ),
        )

        cfg_cls = sim_utils.UrdfFileCfg
        converter = getattr(sim_utils, "UrdfConverterCfg", None)

        if converter is not None and hasattr(converter, "JointDriveCfg"):
            drive_cls = converter.JointDriveCfg
            gains_cls = getattr(drive_cls, "PDGainsCfg", None)
            if gains_cls is not None:
                return cfg_cls(
                    **common,
                    joint_drive=drive_cls(target_type="none", gains=gains_cls(stiffness=0.0, damping=0.0)),
                )
            return cfg_cls(**common, joint_drive=drive_cls(target_type="none"))

        try:
            return cfg_cls(
                **common,
                default_drive_type="force",
                default_drive_stiffness=0.0,
                default_drive_damping=0.0,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Could not build a UrdfFileCfg for this IsaacLab version. Update "
                "IsaacLabSimBackend._build_urdf_spawn_cfg for the installed converter API."
            ) from exc

    # ---------------------------------------------------------------- setup

    def resolve_dof_indices(self, dof_names: Sequence[str]) -> list[int]:
        ids, names = self.robot.find_joints(list(dof_names), preserve_order=True)
        # Kept for diagnosis: if these ever come back in a different order than requested,
        # every joint-indexed tensor in the shared core is silently permuted.
        self.requested_dof_names = list(dof_names)
        self.resolved_dof_names = list(names)
        return list(ids)

    def resolve_link_indices(self, name_patterns: Sequence[str]) -> list[int]:
        return resolve_link_indices_by_pattern(self.body_names, name_patterns)

    def get_dof_pos_limits(self, dof_indices: Sequence[int]) -> torch.Tensor:
        limits = self.robot.data.joint_pos_limits[0, list(dof_indices), :]
        return limits.clone().to(self.device)

    def get_dof_force_range(self, dof_indices: Sequence[int]) -> torch.Tensor:
        return self.robot.data.joint_effort_limits[0, list(dof_indices)].clone().to(self.device)

    def describe_actuation(self) -> dict[str, Any]:
        """Report the gains PhysX is *actually* using, for bring-up diagnosis.

        The shared PD law is the only intended controller, so sim-side stiffness and
        damping must both read zero. Anything else means the engine is applying a second
        controller on top of ours.
        """
        d = self.robot.data
        idx = 0

        def first(name: str) -> Any:
            t = getattr(d, name, None)
            return None if t is None else round(float(t[0, idx]), 6)

        return {
            "sim_joint_stiffness": first("joint_stiffness"),
            "sim_joint_damping": first("joint_damping"),
            "sim_joint_armature": first("joint_armature"),
            "sim_joint_friction": first("joint_friction_coeff") or first("joint_friction"),
            "sim_effort_limit": first("joint_effort_limits"),
        }

    def set_pd_gains(self, p_gains: torch.Tensor, d_gains: torch.Tensor, dof_indices: Sequence[int]) -> None:
        # Intentionally a no-op: the engine-side drive stays at zero gain so that the
        # shared PD law in Go2BaseEnv._compute_torques is the only controller, exactly as
        # on the Genesis arm (use_implicit_controller=False).
        return

    # ---------------------------------------------------------------- state

    def get_base_pos(self) -> torch.Tensor:
        return self.robot.data.root_pos_w - self.env_origins

    def get_base_quat(self) -> torch.Tensor:
        return self.robot.data.root_quat_w  # wxyz

    def get_base_lin_vel_world(self) -> torch.Tensor:
        # Link-frame, to match Genesis's robot.get_vel() (base link, not COM).
        data = self.robot.data
        return getattr(data, "root_link_lin_vel_w", data.root_lin_vel_w)

    def get_base_ang_vel_world(self) -> torch.Tensor:
        data = self.robot.data
        return getattr(data, "root_link_ang_vel_w", data.root_ang_vel_w)

    def get_dofs_position(self, dof_indices: Sequence[int]) -> torch.Tensor:
        return self.robot.data.joint_pos[:, list(dof_indices)]

    def get_dofs_velocity(self, dof_indices: Sequence[int]) -> torch.Tensor:
        return self.robot.data.joint_vel[:, list(dof_indices)]

    def get_links_net_contact_force(self) -> torch.Tensor:
        forces = self.contact_sensor.data.net_forces_w
        if forces.dim() == 4:  # (envs, history, bodies, 3)
            forces = forces[:, 0]
        return forces[:, self._contact_perm, :]

    def get_links_pos(self, link_indices: Sequence[int]) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, list(link_indices), :] - self.env_origins.unsqueeze(1)

    def get_links_quat(self, link_indices: Sequence[int]) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, list(link_indices), :]

    def get_links_vel(self, link_indices: Sequence[int]) -> torch.Tensor:
        data = self.robot.data
        vel = getattr(data, "body_link_lin_vel_w", data.body_lin_vel_w)
        return vel[:, list(link_indices), :]

    def get_links_com(self, link_indices: Sequence[int]) -> torch.Tensor:
        data = self.robot.data
        com = getattr(data, "body_com_pos_w", data.body_pos_w)
        return com[:, list(link_indices), :] - self.env_origins.unsqueeze(1)

    # -------------------------------------------------------------- control

    def apply_dof_force(self, torques: torch.Tensor, dof_indices: Sequence[int]) -> None:
        self.robot.set_joint_effort_target(torques, joint_ids=list(dof_indices))
        self.robot.write_data_to_sim()

    def step(self) -> None:
        self.sim.step(render=False)
        self.scene.update(self._sim_dt)

    # ---------------------------------------------------------------- reset

    def set_dofs_position(
        self,
        position: torch.Tensor,
        dof_indices: Sequence[int],
        zero_velocity: bool,
        envs_idx: torch.Tensor,
    ) -> None:
        vel = torch.zeros_like(position) if zero_velocity else self.robot.data.joint_vel[envs_idx][:, list(dof_indices)]
        self.robot.write_joint_state_to_sim(position, vel, joint_ids=list(dof_indices), env_ids=envs_idx)

    def set_base_pos(self, pos: torch.Tensor, envs_idx: torch.Tensor) -> None:
        pose = self.robot.data.root_state_w[envs_idx, :7].clone()
        pose[:, :3] = pos + self.env_origins[envs_idx]
        self.robot.write_root_pose_to_sim(pose, env_ids=envs_idx)

    def set_base_quat(self, quat: torch.Tensor, envs_idx: torch.Tensor) -> None:
        pose = self.robot.data.root_state_w[envs_idx, :7].clone()
        pose[:, 3:7] = quat
        self.robot.write_root_pose_to_sim(pose, env_ids=envs_idx)

    def set_base_velocity(self, lin_vel: torch.Tensor, ang_vel: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self.robot.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=1), env_ids=envs_idx)

    def zero_all_dofs_velocity(self, envs_idx: torch.Tensor) -> None:
        joint_vel = torch.zeros_like(self.robot.data.joint_vel[envs_idx])
        joint_pos = self.robot.data.joint_pos[envs_idx]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=envs_idx)
        self.robot.write_root_velocity_to_sim(torch.zeros((len(envs_idx), 6), device=self.device), env_ids=envs_idx)

    def get_all_dofs_velocity(self) -> torch.Tensor:
        # Genesis exposes the floating base as DoFs 0..5 ahead of the joints; PhysX keeps
        # root velocity separate, so present the same layout the core expects.
        root = self.robot.data.root_vel_w
        return torch.cat([root, self.robot.data.joint_vel], dim=1)

    def set_all_dofs_velocity(self, vel: torch.Tensor) -> None:
        self.robot.write_root_velocity_to_sim(vel[:, :6])
        self.robot.write_joint_state_to_sim(self.robot.data.joint_pos, vel[:, 6:])

    # ------------------------------------------------- domain randomisation

    def set_geoms_friction_ratio(self, ratios: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Scale friction by a per-env ratio, matching Genesis's multiplicative semantics.

        PhysX takes absolute material properties, so the asset's own friction is cached on
        first call and every later write is ``base * ratio``.
        """
        view = self.robot.root_physx_view
        materials = view.get_material_properties()  # (envs, shapes, 3) on CPU
        if self._base_friction is None:
            self._base_friction = materials.clone()

        idx_cpu = envs_idx.detach().cpu()
        ratio_cpu = ratios.detach().cpu().reshape(-1, 1)
        updated = materials.clone()
        # columns 0/1 are static/dynamic friction; 2 is restitution, left untouched
        updated[idx_cpu, :, 0] = self._base_friction[idx_cpu, :, 0] * ratio_cpu
        updated[idx_cpu, :, 1] = self._base_friction[idx_cpu, :, 1] * ratio_cpu
        view.set_material_properties(updated, idx_cpu)

    def set_links_mass_shift(
        self, added_mass: torch.Tensor, link_indices: Sequence[int], envs_idx: torch.Tensor
    ) -> None:
        view = self.robot.root_physx_view
        if not hasattr(self, "_default_masses"):
            self._default_masses = view.get_masses().clone()
        masses = view.get_masses().clone()
        idx_cpu = envs_idx.detach().cpu()
        delta = added_mass.detach().cpu().reshape(-1, 1)
        for link in link_indices:
            masses[idx_cpu, link] = self._default_masses[idx_cpu, link] + delta.squeeze(1)
        view.set_masses(masses, idx_cpu)

    def set_links_com_shift(self, com_shift: torch.Tensor, link_indices: Sequence[int], envs_idx: torch.Tensor) -> None:
        view = self.robot.root_physx_view
        if not hasattr(self, "_default_coms"):
            self._default_coms = view.get_coms().clone()
        coms = view.get_coms().clone()
        idx_cpu = envs_idx.detach().cpu()
        shift = com_shift.detach().cpu().reshape(len(idx_cpu), 1, 3)
        for link in link_indices:
            coms[idx_cpu, link, :3] = self._default_coms[idx_cpu, link, :3] + shift[:, 0, :]
        view.set_coms(coms, idx_cpu)

    # --------------------------------------------------------------- render

    def render(self, track_pos: Optional[torch.Tensor] = None) -> Any:
        """Return one 320x320 RGB frame of env 0, or ``None`` if no camera is attached.

        Mirrors Genesis's chase framing. ``track_pos`` arrives in env-local coordinates,
        so env 0's origin is added back to place the camera in world space.
        """
        if self._camera is None:
            return None

        import numpy as np

        base = track_pos if track_pos is not None else torch.zeros(3, device=self.device)
        base = (base.to(self.device) + self.env_origins[0]).tolist()
        # Genesis's chase framing: eye behind/left/above, looking slightly below the base.
        self.sim.set_camera_view(
            (base[0] - 1.0, base[1] - 1.0, base[2] + 0.5),
            (base[0], base[1], base[2] - 0.1),
        )
        self.sim.render()

        rgb = self._rgb_annotator.get_data()
        rgb = np.frombuffer(rgb, dtype=np.uint8).reshape(*rgb.shape)
        if rgb.size == 0:  # renderer still warming up
            w, h = self._render_resolution
            return np.zeros((h, w, 3), dtype=np.uint8)
        return rgb[:, :, :3]

    def draw_debug(self, foot_positions: torch.Tensor, com: torch.Tensor, terrain_heights: torch.Tensor) -> None:
        return

    def close(self) -> None:
        return
