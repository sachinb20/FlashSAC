"""Genesis implementation of ``Go2SimBackend``.

Behaviour-preserving with respect to ``flash_rl/envs/genesis_envs/go2_base.py`` -- the
same scene options, the same solver, the same calls in the same order. The one change is
the asset: this backend loads the *pre-merged* URDF with ``merge_fixed_links=False``
instead of asking Genesis to merge at load time. Both routes were verified to produce an
identical articulation (17 links, same names and order, inertia and COM differing by
exactly 0.0, 27 collision geoms), so this costs nothing and lets the IsaacLab arm consume
the same file -- which it must, since Genesis cannot be installed alongside it.

Index translation lives here: Genesis's rigid solver numbers links globally across
entities, so the robot's base sits at solver index ``robot.link_start`` (1, behind the
ground plane) while the core addresses it as robot-local 0.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

from ..go2_urdf import get_merged_urdf
from ..sim_backend import resolve_link_indices_by_pattern


class GenesisSimBackend:
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
        enable_camera: bool = True,
        asset_source: str = "genesis_merged",
        ground_material: str = "isaaclab_default",
    ) -> None:
        # The upstream Unitree description is a PhysX-side port step; it has never been
        # run on Genesis and its topology (rotor links, unmerged calflower chain) would
        # silently change which bodies the substring contact matching selects. Refuse
        # rather than run a robot nobody asked for.
        # Genesis's ground is its own plane.urdf with Genesis's solver defaults; it has
        # no PhysX-style per-material combine mode, so the preset cannot be honoured here.
        if ground_material != "isaaclab_default":
            raise ValueError(
                f"The Genesis backend cannot set ground_material={ground_material!r}; its ground is "
                "plane.urdf under the Genesis solver. Use sim_backend='isaaclab'."
            )
        if asset_source != "genesis_merged":
            raise ValueError(
                f"The Genesis backend supports asset_source='genesis_merged' only, got {asset_source!r}. "
                "Use sim_backend='isaaclab' for the upstream Unitree description."
            )
        # Genesis already defaults every DoF to 0.1, so writing it is a no-op here -- but
        # it is written anyway so the value is explicit and identical on both backends.
        self._dof_armature = dof_armature
        # Genesis's offscreen camera is cheap and the baseline always attaches it, so the
        # flag is accepted for interface parity but the camera is always created.
        del enable_camera
        # Genesis spawns at the URDF's zero pose and only warns that it violates the joint
        # limits; the env's reset_idx immediately writes default_dof_pos anyway. Accepted
        # for interface parity with the IsaacLab backend, which must spawn in a valid pose.
        del default_joint_angles
        try:
            gs.init(logging_level="warning")
        except Exception as e:  # already initialised in this process
            print(e)

        self.device = device
        self.num_envs = num_envs
        self.num_build_envs = num_build_envs
        self.debug = debug
        self.headless = not show_viewer

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=sim_dt, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / control_dt * (control_dt / sim_dt)),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.rigid_solver: RigidSolver = next(s for s in self.scene.sim.solvers if isinstance(s, RigidSolver))

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        merged_urdf = get_merged_urdf(urdf_path, links_to_keep)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=merged_urdf,
                merge_fixed_links=False,
                links_to_keep=[],
                pos=base_init_pos.cpu().numpy(),
                quat=base_init_quat.cpu().numpy(),
            ),
            visualize_contact=debug,
        )

        if gs.platform != "macOS":
            self._set_camera()

        self.scene.build(n_envs=num_build_envs)

        self.n_links = self.robot.n_links
        self._link_offset = self.robot.link_start
        self._motor_dofs: list[int] = []

    # ---------------------------------------------------------------- setup

    def _solver_links(self, link_indices: Sequence[int]) -> list[int]:
        return [int(i) + self._link_offset for i in link_indices]

    def resolve_dof_indices(self, dof_names: Sequence[str]) -> list[int]:
        # Genesis resolves each joint by exact name, so the order is the requested one by
        # construction. Recorded to match the IsaacLab backend's diagnostic.
        self.requested_dof_names = list(dof_names)
        self.resolved_dof_names = list(dof_names)
        self._motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in dof_names]
        return self._motor_dofs

    def resolve_link_indices(self, name_patterns: Sequence[str]) -> list[int]:
        names = [link.name for link in self.robot.links]
        offsets = [link.idx - self._link_offset for link in self.robot.links]
        return [offsets[i] for i in resolve_link_indices_by_pattern(names, name_patterns)]

    def get_dof_pos_limits(self, dof_indices: Sequence[int]) -> torch.Tensor:
        return torch.stack(self.robot.get_dofs_limit(dof_indices), dim=1)

    def get_dof_force_range(self, dof_indices: Sequence[int]) -> torch.Tensor:
        return self.robot.get_dofs_force_range(dof_indices)[1]

    def set_pd_gains(self, p_gains: torch.Tensor, d_gains: torch.Tensor, dof_indices: Sequence[int]) -> None:
        self.robot.set_dofs_kp(p_gains, dof_indices)
        self.robot.set_dofs_kv(d_gains, dof_indices)
        armature = torch.full((len(dof_indices),), self._dof_armature, device=self.device)
        self.robot.set_dofs_armature(armature, dof_indices)

    def describe_actuation(self) -> dict[str, Any]:
        """Counterpart to the IsaacLab diagnostic. Genesis drives by explicit force only,
        so the registered kp/kv are inert here -- reported for comparison, not because the
        engine uses them. Armature, by contrast, is very much live."""
        arm = self.robot.get_dofs_armature(self._motor_dofs) if self._motor_dofs else None
        return {
            "engine": "genesis (explicit torque via control_dofs_force)",
            "sim_joint_armature": None if arm is None else round(float(arm[0]), 6),
        }

    # ---------------------------------------------------------------- state

    def get_base_pos(self) -> torch.Tensor:
        return self.robot.get_pos()

    def get_base_quat(self) -> torch.Tensor:
        return self.robot.get_quat()

    def get_base_lin_vel_world(self) -> torch.Tensor:
        return self.robot.get_vel()

    def get_base_ang_vel_world(self) -> torch.Tensor:
        return self.robot.get_ang()

    def get_dofs_position(self, dof_indices: Sequence[int]) -> torch.Tensor:
        return self.robot.get_dofs_position(dof_indices)

    def get_dofs_velocity(self, dof_indices: Sequence[int]) -> torch.Tensor:
        return self.robot.get_dofs_velocity(dof_indices)

    def get_links_net_contact_force(self) -> torch.Tensor:
        return torch.as_tensor(self.robot.get_links_net_contact_force(), device=self.device, dtype=torch.float32)

    def get_links_pos(self, link_indices: Sequence[int]) -> torch.Tensor:
        return self.rigid_solver.get_links_pos(self._solver_links(link_indices))

    def get_links_quat(self, link_indices: Sequence[int]) -> torch.Tensor:
        return self.rigid_solver.get_links_quat(self._solver_links(link_indices))

    def get_links_vel(self, link_indices: Sequence[int]) -> torch.Tensor:
        return self.rigid_solver.get_links_vel(self._solver_links(link_indices))

    def get_links_com(self, link_indices: Sequence[int]) -> torch.Tensor:
        return self.rigid_solver.get_links_root_COM(self._solver_links(link_indices))

    # -------------------------------------------------------------- control

    def apply_dof_force(self, torques: torch.Tensor, dof_indices: Sequence[int]) -> None:
        if self.num_build_envs == 0:
            torques = torques.squeeze()
        self.robot.control_dofs_force(torques, dof_indices)

    def step(self) -> None:
        self.scene.step()

    # ---------------------------------------------------------------- reset

    def set_dofs_position(
        self,
        position: torch.Tensor,
        dof_indices: Sequence[int],
        zero_velocity: bool,
        envs_idx: torch.Tensor,
    ) -> None:
        self.robot.set_dofs_position(
            position=position,
            dofs_idx_local=dof_indices,
            zero_velocity=zero_velocity,
            envs_idx=envs_idx,
        )

    def set_base_pos(self, pos: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self.robot.set_pos(pos, zero_velocity=False, envs_idx=envs_idx)

    def set_base_quat(self, quat: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self.robot.set_quat(quat, zero_velocity=False, envs_idx=envs_idx)

    def set_base_velocity(self, lin_vel: torch.Tensor, ang_vel: torch.Tensor, envs_idx: torch.Tensor) -> None:
        base_vel = torch.concat([lin_vel, ang_vel], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=envs_idx)

    def zero_all_dofs_velocity(self, envs_idx: torch.Tensor) -> None:
        self.robot.zero_all_dofs_velocity(envs_idx)

    def get_all_dofs_velocity(self) -> torch.Tensor:
        return self.robot.get_dofs_velocity()

    def set_all_dofs_velocity(self, vel: torch.Tensor) -> None:
        self.robot.set_dofs_velocity(vel)

    # ------------------------------------------------- domain randomisation

    def set_geoms_friction_ratio(self, ratios: torch.Tensor, envs_idx: torch.Tensor) -> None:
        solver = self.rigid_solver
        # Genesis expects one ratio per geom; the core samples one per env.
        per_geom = ratios.repeat(1, solver.n_geoms)
        solver.set_geoms_friction_ratio(per_geom, torch.arange(0, solver.n_geoms), envs_idx)

    def set_links_mass_shift(
        self, added_mass: torch.Tensor, link_indices: Sequence[int], envs_idx: torch.Tensor
    ) -> None:
        self.rigid_solver.set_links_mass_shift(added_mass, self._solver_links(link_indices), envs_idx)

    def set_links_com_shift(self, com_shift: torch.Tensor, link_indices: Sequence[int], envs_idx: torch.Tensor) -> None:
        self.rigid_solver.set_links_COM_shift(com_shift, self._solver_links(link_indices), envs_idx)

    # --------------------------------------------------------------- render

    def _set_camera(self) -> None:
        self._floating_camera = self.scene.add_camera(
            pos=np.array([0, -1, 1]),
            lookat=np.array([0, 0, 0]),
            fov=40,
            GUI=False,
        )

    def update_viewer(self, track_pos: Optional[torch.Tensor] = None) -> None:
        """No-op: Genesis repaints its own viewer inside ``scene.step()``."""
        return

    def render(self, track_pos: Optional[torch.Tensor] = None) -> Any:
        robot_pos = np.array(track_pos.cpu()) if track_pos is not None else np.zeros(3)
        self._floating_camera.set_pose(
            pos=robot_pos + np.array([-1, -1, 0.5]), lookat=robot_pos + np.array([0, 0, -0.1])
        )
        frame, _, _, _ = self._floating_camera.render()
        return frame

    def draw_debug(self, foot_positions: torch.Tensor, com: torch.Tensor, terrain_heights: torch.Tensor) -> None:
        self.scene.clear_debug_objects()
        foot_poss = foot_positions[0].reshape(-1, 3).cpu()
        self.scene.draw_debug_line(foot_poss[0], foot_poss[3], radius=0.002, color=(1, 0, 0, 0.7))
        self.scene.draw_debug_line(foot_poss[1], foot_poss[2], radius=0.002, color=(1, 0, 0, 0.7))
        c = com[0].clone()
        c[2] = 0.02 + terrain_heights[0]
        self.scene.draw_debug_sphere(pos=c, radius=0.02, color=(0, 0, 1, 0.7))

    def close(self) -> None:
        return
