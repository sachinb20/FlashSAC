"""Simulator-agnostic Go2 velocity-tracking environment.

This is ``flash_rl/envs/genesis_envs/go2_base.py`` + ``go2_walk.py`` with every physics
call routed through ``Go2SimBackend``. The RL-visible behaviour is unchanged and is
*byte-identical between backends* by construction: observations, reward terms, command
sampling, the action pipeline, the PD torque law, termination, episode bookkeeping and
all domain-randomisation sampling live here, in plain ``torch``, and run the same code
whichever engine is underneath.

What genuinely differs between backends is only what an engine swap *must* change:
contact resolution, the constraint solver, integration, and collision geometry.

Scope: flat ground only. ``use_terrain=True`` is rejected rather than silently ignored
-- the ``go2-walk`` baseline this port targets runs flat, and Genesis's height-field
terrain has no faithful one-to-one IsaacLab counterpart.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .math_utils import (
    TC_FLOAT,
    TC_INT,
    euler2quat,
    inv_quat,
    quat2euler,
    quat_from_angle_axis,
    quat_mul,
    rand_float,
    transform_by_quat,
    wrap_to_pi,
)
from .sim_backend import Go2SimBackend, make_backend

__all__ = ["Go2WalkEnv", "get_cfgs", "get_env"]


class Go2BaseEnv:
    def __init__(
        self,
        num_envs: int,
        env_cfg: dict[str, Any],
        obs_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
        command_cfg: dict[str, Any],
        show_viewer: bool,
        eval: bool,
        debug: bool,
        sim_backend: str = "genesis",
        enable_camera: bool = False,
        device: str = "cuda",
    ) -> None:
        self.num_envs = 1 if num_envs == 0 else num_envs
        self.num_build_envs = num_envs
        self.num_single_obs = obs_cfg["num_obs"]
        assert obs_cfg["num_history_obs"] == 1, "num_history_obs is now fixed to 1."
        self.num_obs = self.num_single_obs * obs_cfg["num_history_obs"]
        self.num_privileged_obs = obs_cfg["num_priv_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.headless = not show_viewer
        self.eval = eval
        self.debug = debug
        self.sim_backend_name = sim_backend

        self.dt = 1 / env_cfg["control_freq"]
        if env_cfg["use_implicit_controller"]:
            raise NotImplementedError(
                "The shared Go2 core drives the robot by explicit torque only "
                "(use_implicit_controller=False), so the PD law is identical across backends."
            )
        sim_dt = self.dt / env_cfg["decimation"]
        self.max_episode_length_s = env_cfg["episode_length_s"]
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.obs_cfg = obs_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_cfg = reward_cfg
        self.reward_scales = reward_cfg["reward_scales"]
        self.env_cfg = env_cfg
        self.command_cfg = command_cfg

        self.command_type = env_cfg["command_type"]
        assert self.command_type in ["heading", "ang_vel_yaw"]

        self.action_latency = env_cfg["action_latency"]
        assert self.action_latency in [0, 0.02]

        if env_cfg["use_terrain"]:
            raise NotImplementedError("The shared Go2 core supports flat ground only (use_terrain=False).")

        self.num_dof = env_cfg["num_dofs"]
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            assert device in ["cpu", "cuda"]
            self.device = torch.device(device)

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)

        self.sim: Go2SimBackend = make_backend(
            sim_backend,
            num_envs=self.num_envs,
            num_build_envs=self.num_build_envs,
            sim_dt=sim_dt,
            control_dt=self.dt,
            urdf_path=self.env_cfg["urdf_path"],
            links_to_keep=self.env_cfg["links_to_keep"],
            base_init_pos=self.base_init_pos,
            base_init_quat=self.base_init_quat,
            # The spawn pose must be a valid stance, not the all-zeros default: the calf
            # joints are limited to [-2.723, -0.838] and cannot reach 0.
            default_joint_angles=self.env_cfg["default_joint_angles"],
            dof_armature=float(self.env_cfg["dof_armature"]),
            show_viewer=show_viewer,
            debug=debug,
            enable_camera=enable_camera,
            device=self.device,
        )

        self._init_buffers()
        self._prepare_reward_function()

        # domain randomization
        self._randomize_controls()
        self._randomize_rigids()

    def _prepare_reward_function(self) -> None:
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros((self.num_envs,), device=self.device, dtype=TC_FLOAT)
            for name in self.reward_scales.keys()
        }

    def _init_buffers(self) -> None:
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=TC_FLOAT)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=TC_FLOAT)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=TC_FLOAT)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=TC_FLOAT)
        self.global_gravity = torch.tensor(np.array([0.0, 0.0, -1.0]), device=self.device, dtype=TC_FLOAT)
        self.forward_vec = torch.zeros((self.num_envs, 3), device=self.device, dtype=TC_FLOAT)
        self.forward_vec[:, 0] = 1.0

        self.obs_buf = torch.zeros((self.num_envs, self.num_single_obs), device=self.device, dtype=TC_FLOAT)
        self.final_obs_history_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=TC_FLOAT)
        self.obs_history_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=TC_FLOAT)
        self.obs_noise = torch.zeros((self.num_envs, self.num_single_obs), device=self.device, dtype=TC_FLOAT)
        self._prepare_obs_noise()
        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=TC_FLOAT)
        )
        self.final_privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=TC_FLOAT)
        )
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=TC_FLOAT)
        self.rew_buf_pos = torch.zeros((self.num_envs,), device=self.device, dtype=TC_FLOAT)
        self.rew_buf_neg = torch.zeros((self.num_envs,), device=self.device, dtype=TC_FLOAT)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=TC_INT)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=TC_INT)
        self.time_out_buf = torch.zeros((self.num_envs,), device=self.device, dtype=TC_INT)

        # commands
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=TC_FLOAT)
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=self.device,
            dtype=TC_FLOAT,
        )
        self.stand_still = torch.zeros((self.num_envs,), device=self.device, dtype=TC_INT)

        # names to indices
        self.motor_dofs = self.sim.resolve_dof_indices(self.env_cfg["dof_names"])

        # Robot-local link indices (base link is 0); the backend maps these onto whatever
        # numbering its engine uses.
        self.termination_contact_link_indices = self.sim.resolve_link_indices(
            self.env_cfg["termination_contact_link_names"]
        )
        self.penalized_contact_link_indices = self.sim.resolve_link_indices(
            self.env_cfg["penalized_contact_link_names"]
        )
        self.feet_link_indices = self.sim.resolve_link_indices(self.env_cfg["feet_link_names"])
        assert len(self.termination_contact_link_indices) > 0
        assert len(self.penalized_contact_link_indices) > 0
        assert len(self.feet_link_indices) > 0
        self.base_link_index = 0

        # actions
        self.actions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=TC_FLOAT)
        self.last_actions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=TC_FLOAT)
        self.last_last_actions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=TC_FLOAT)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=TC_FLOAT)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=TC_FLOAT)
        self.last_dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=TC_FLOAT)
        self.root_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=TC_FLOAT)
        self.last_root_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=TC_FLOAT)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=TC_FLOAT)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=TC_FLOAT)
        self.link_contact_forces = torch.zeros((self.num_envs, self.sim.n_links, 3), device=self.device, dtype=TC_FLOAT)

        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=TC_FLOAT
        )
        self.feet_max_height = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=TC_FLOAT
        )
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=TC_INT)

        # extras
        self.continuous_push = torch.zeros((self.num_envs, 3), device=self.device, dtype=TC_FLOAT)
        self.env_identities = torch.arange(self.num_envs, device=self.device, dtype=TC_INT)
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {}

        self.terrain_heights = torch.zeros((self.num_envs,), device=self.device, dtype=TC_FLOAT)

        # PD control
        stiffness = self.env_cfg["PD_stiffness"]
        damping = self.env_cfg["PD_damping"]

        p_gains, d_gains = [], []
        for dof_name in self.env_cfg["dof_names"]:
            for key in stiffness.keys():
                if key in dof_name:
                    p_gains.append(stiffness[key])
                    d_gains.append(damping[key])
        self.p_gains = torch.tensor(p_gains, device=self.device)
        self.d_gains = torch.tensor(d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)

        self.sim.set_pd_gains(self.p_gains, self.d_gains, self.motor_dofs)

        default_joint_angles = self.env_cfg["default_joint_angles"]
        self.default_dof_pos = torch.tensor(
            [default_joint_angles[name] for name in self.env_cfg["dof_names"]],
            device=self.device,
        )

        self.dof_pos_limits = self.sim.get_dof_pos_limits(self.motor_dofs)
        self.torque_limits = self.sim.get_dof_force_range(self.motor_dofs)
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.reward_cfg["soft_dof_pos_limit"]
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.reward_cfg["soft_dof_pos_limit"]

        self.motor_strengths = torch.ones((self.num_envs, self.num_dof), device=self.device, dtype=TC_FLOAT)
        self.motor_offsets = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=TC_FLOAT)

        # gait control
        n_feet = len(self.feet_link_indices)
        self.foot_positions = torch.ones(self.num_envs, n_feet, 3, device=self.device, dtype=TC_FLOAT)
        self.foot_quaternions = torch.ones(self.num_envs, n_feet, 4, device=self.device, dtype=TC_FLOAT)
        self.foot_velocities = torch.ones(self.num_envs, n_feet, 3, device=self.device, dtype=TC_FLOAT)

        self.com = torch.zeros(self.num_envs, 3, device=self.device, dtype=TC_FLOAT)

    def _update_buffers(self) -> None:
        self.base_pos[:] = self.sim.get_base_pos()
        self.base_quat[:] = self.sim.get_base_quat()
        base_quat_rel = quat_mul(self.base_quat, inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = quat2euler(base_quat_rel)

        inv_quat_yaw = quat_from_angle_axis(
            -self.base_euler[:, 2], torch.tensor([0, 0, 1], device=self.device, dtype=torch.float)
        )

        inv_base_quat = inv_quat(self.base_quat)
        # NOTE: linear velocity uses the yaw-only frame, angular velocity the full body
        # frame. Inherited from the Genesis env; preserved deliberately.
        self.base_lin_vel[:] = transform_by_quat(self.sim.get_base_lin_vel_world(), inv_quat_yaw)
        self.base_ang_vel[:] = transform_by_quat(self.sim.get_base_ang_vel_world(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        self.dof_pos[:] = self.sim.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.sim.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = self.sim.get_links_net_contact_force()
        self.com[:] = self.sim.get_links_com([self.base_link_index]).squeeze(dim=1)

        self.foot_positions[:] = self.sim.get_links_pos(self.feet_link_indices)
        self.foot_quaternions[:] = self.sim.get_links_quat(self.feet_link_indices)
        self.foot_velocities[:] = self.sim.get_links_vel(self.feet_link_indices)

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        # control_type = 'P'
        actions_scaled = actions * self.env_cfg["action_scale"]
        torques = (
            self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
            - self.batched_d_gains * self.dof_vel
        )
        return torques * self.motor_strengths

    def check_termination(self) -> None:
        self.reset_buf = torch.any(
            torch.norm(self.link_contact_forces[:, self.termination_contact_link_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= torch.logical_or(
            torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"],
            torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"],
        )
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_height_lower_than"]
        self.reset_buf |= self.time_out_buf

    def compute_reward(self) -> None:
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.obs_history_buf, {}

    def get_privileged_observations(self) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
        return self.privileged_obs_buf, {}

    def post_physics_step(self) -> None:
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_buffers()

        resampling_time_s = self.env_cfg["resampling_time_s"]
        envs_idx = (self.episode_length_buf % int(resampling_time_s / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)
        if self.command_type == "heading":
            forward = transform_by_quat(self.forward_vec, self.base_quat)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        # random push
        push_interval_s = self.env_cfg["push_interval_s"]
        if push_interval_s > 0 and not (self.debug or self.eval):
            max_push_vel_xy = self.env_cfg["max_push_vel_xy"]
            dofs_vel = self.sim.get_all_dofs_velocity()
            push_vel = rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.sim.set_all_dofs_velocity(dofs_vel)

        self.check_termination()
        self.compute_reward()

        if torch.any(self.reset_buf):
            self.extras["episode_length"] = (
                (self.episode_length_buf * self.reset_buf).sum() / self.reset_buf.sum()
            ).item()
        envs_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.num_build_envs > 0:
            self.compute_observations()
            self.final_obs_history_buf = self.obs_history_buf.detach().clone()
            self.final_privileged_obs_buf = self.privileged_obs_buf.detach().clone()
            self.reset_idx(envs_idx)
        self.compute_observations()

        if not self.headless and self.debug:
            self.sim.draw_debug(self.foot_positions, self.com, self.terrain_heights)

        self.last_actions[:] = self.actions[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.sim.get_base_lin_vel_world()

    def compute_observations(self) -> None:
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        # add noise
        if not self.eval:
            self.obs_buf += rand_float(-1.0, 1.0, (self.num_single_obs,), self.device) * self.obs_noise

        clip_obs = 100.0
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        self.obs_history_buf = torch.cat([self.obs_history_buf[:, self.num_single_obs :], self.obs_buf.detach()], dim=1)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.obs_buf,  # 45 (noised)
                    self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                    self.last_actions,  # 12
                ],
                axis=-1,
            )
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

    def _prepare_obs_noise(self) -> None:
        self.obs_noise[:, :3] = self.obs_cfg["obs_noise"]["ang_vel"]
        self.obs_noise[:, 3:6] = self.obs_cfg["obs_noise"]["gravity"]
        self.obs_noise[:, 21:33] = self.obs_cfg["obs_noise"]["dof_pos"]
        self.obs_noise[:, 33:45] = self.obs_cfg["obs_noise"]["dof_vel"]

    def _resample_commands(self, envs_idx: torch.Tensor) -> None:
        # lin_vel
        self.commands[envs_idx, 0] = rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, :2] *= (torch.norm(self.commands[envs_idx, :2], dim=1) > 0.2).unsqueeze(1)

        # ang_vel
        if self.command_type == "heading":
            self.commands[envs_idx, 3] = rand_float(-3.14, 3.14, (len(envs_idx),), self.device)
        elif self.command_type == "ang_vel_yaw":
            self.commands[envs_idx, 2] = rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
            self.commands[envs_idx, 2] *= torch.abs(self.commands[envs_idx, 2]) > 0.2

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = (self.default_dof_pos) + rand_float(
            -0.3, 0.3, (len(envs_idx), self.num_dof), self.device
        )
        self.dof_vel[envs_idx] = 0.0
        self.sim.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dof_indices=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset root states - position
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx, :2] += rand_float(-1.0, 1.0, (len(envs_idx), 2), self.device)
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        base_euler = rand_float(-0.1, 0.1, (len(envs_idx), 3), self.device)
        base_euler[:, 2] = rand_float(0.0, 3.14, (len(envs_idx),), self.device)
        self.base_quat[envs_idx] = quat_mul(euler2quat(base_euler), self.base_quat[envs_idx])
        self.sim.set_base_pos(self.base_pos[envs_idx], envs_idx=envs_idx)
        self.sim.set_base_quat(self.base_quat[envs_idx], envs_idx=envs_idx)
        self.sim.zero_all_dofs_velocity(envs_idx)

        # update projected gravity
        inv_base_quat = inv_quat(self.base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        # reset root states - velocity
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0.0
        self.sim.set_base_velocity(
            lin_vel=self.base_lin_vel[envs_idx], ang_vel=self.base_ang_vel[envs_idx], envs_idx=envs_idx
        )

        self._resample_commands(envs_idx)

        # reset buffers
        self.obs_history_buf[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.max_episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0
        # send timeout info to the algorithm
        if self.env_cfg["send_timeouts"]:
            self.extras["time_outs"] = self.time_out_buf
        self.time_out_buf[envs_idx] = 0

    def reset(self) -> tuple[None, None]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return None, None

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_range = self.env_cfg["action_range"]
        self.actions = torch.clip(actions, -action_range, action_range)
        exec_actions = self.last_actions if self.action_latency > 0 else self.actions

        for _ in range(self.env_cfg["decimation"]):
            self.torques = self._compute_torques(exec_actions)
            self.sim.apply_dof_force(self.torques, self.motor_dofs)
            self.sim.step()
            self.dof_pos[:] = self.sim.get_dofs_position(self.motor_dofs)
            self.dof_vel[:] = self.sim.get_dofs_velocity(self.motor_dofs)

        self.post_physics_step()

        self.extras["privileged_observations"] = self.privileged_obs_buf
        self.extras["final_observations"] = self.final_obs_history_buf
        self.extras["final_privileged_observations"] = self.final_privileged_obs_buf

        return self.obs_history_buf, self.rew_buf, self.reset_buf, self.extras

    # ------------ domain randomization----------------

    def _randomize_rigids(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if self.eval:
            return

        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs, device=self.device)
        elif len(env_ids) == 0:
            return

        if self.env_cfg["randomize_friction"]:
            self._randomize_link_friction(env_ids)
        if self.env_cfg["randomize_base_mass"]:
            self._randomize_base_mass(env_ids)
        if self.env_cfg["randomize_com_displacement"]:
            self._randomize_com_displacement(env_ids)

    def _randomize_controls(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if self.eval:
            return

        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs, device=self.device)
        elif len(env_ids) == 0:
            return

        if self.env_cfg["randomize_motor_strength"]:
            self._randomize_motor_strength(env_ids)
        if self.env_cfg["randomize_motor_offset"]:
            self._randomize_motor_offset(env_ids)
        if self.env_cfg["randomize_kp_scale"]:
            self._randomize_kp(env_ids)
        if self.env_cfg["randomize_kd_scale"]:
            self._randomize_kd(env_ids)

    def _rand(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, device=self.device, dtype=TC_FLOAT)

    def _randomize_link_friction(self, env_ids: torch.Tensor) -> None:
        min_friction, max_friction = self.env_cfg["friction_range"]
        ratios = self._rand((len(env_ids), 1)) * (max_friction - min_friction) + min_friction
        self.sim.set_geoms_friction_ratio(ratios, env_ids)

    def _randomize_base_mass(self, env_ids: torch.Tensor) -> None:
        min_mass, max_mass = self.env_cfg["added_mass_range"]
        added_mass = self._rand((len(env_ids), 1)) * (max_mass - min_mass) + min_mass
        self.sim.set_links_mass_shift(added_mass, [self.base_link_index], env_ids)

    def _randomize_com_displacement(self, env_ids: torch.Tensor) -> None:
        min_displacement, max_displacement = self.env_cfg["com_displacement_range"]
        com_displacement = self._rand((len(env_ids), 1, 3)) * (max_displacement - min_displacement) + min_displacement
        self.sim.set_links_com_shift(com_displacement, [self.base_link_index], env_ids)

    def _randomize_motor_strength(self, env_ids: torch.Tensor) -> None:
        min_strength, max_strength = self.env_cfg["motor_strength_range"]
        self.motor_strengths[env_ids, :] = self._rand((len(env_ids), 1)) * (max_strength - min_strength) + min_strength

    def _randomize_motor_offset(self, env_ids: torch.Tensor) -> None:
        min_offset, max_offset = self.env_cfg["motor_offset_range"]
        self.motor_offsets[env_ids, :] = (
            self._rand((len(env_ids), self.num_dof)) * (max_offset - min_offset) + min_offset
        )

    def _randomize_kp(self, env_ids: torch.Tensor) -> None:
        min_scale, max_scale = self.env_cfg["kp_scale_range"]
        kp_scales = self._rand((len(env_ids), self.num_dof)) * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids: torch.Tensor) -> None:
        min_scale, max_scale = self.env_cfg["kd_scale_range"]
        kd_scales = self._rand((len(env_ids), self.num_dof)) * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]

    def render(self) -> Any:
        return self.sim.render(track_pos=self.base_pos[0])

    def close(self) -> None:
        self.sim.close()


class Go2WalkEnv(Go2BaseEnv):
    """Reward terms for ``go2-walk``. Pure buffer math -- no simulator contact at all."""

    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self) -> torch.Tensor:
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self) -> torch.Tensor:
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self) -> torch.Tensor:
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self) -> torch.Tensor:
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self) -> torch.Tensor:
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self) -> torch.Tensor:
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self) -> torch.Tensor:
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_base_height(self) -> torch.Tensor:
        # Penalize base height away from target
        base_height = self.base_pos[:, 2]
        base_height_target = self.reward_cfg["base_height_target"]
        return torch.square(base_height - base_height_target)

    def _reward_collision(self) -> torch.Tensor:
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0 * (torch.norm(self.link_contact_forces[:, self.penalized_contact_link_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_termination(self) -> torch.Tensor:
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self) -> torch.Tensor:
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_air_time(self) -> torch.Tensor:
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        # reward only on first contact with the ground
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime


def get_cfgs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The ``go2-walk`` configuration, unchanged from ``genesis_envs/go2_walk.py``.

    ``urdf_path`` is resolved by each backend: Genesis reads it relative to its own
    bundled asset root, the IsaacLab backend resolves the same file to an absolute path
    inside the installed ``genesis`` package so both engines import identical geometry
    and inertia. See ``go2_urdf.py``.
    """
    env_cfg = {
        "urdf_path": "urdf/go2/urdf/go2.urdf",
        "links_to_keep": [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ],
        "num_actions": 12,
        "num_dofs": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "termination_contact_link_names": ["base"],
        "penalized_contact_link_names": ["base", "thigh", "calf"],
        "feet_link_names": ["foot"],
        "base_link_name": ["base"],
        # PD
        "PD_stiffness": {"joint": 30.0},
        "PD_damping": {"joint": 1.5},
        "use_implicit_controller": False,
        # Rotor inertia added to every DoF. NOT in the URDF (it declares no <dynamics> at
        # all) -- this is Genesis's *solver default*, which the original env never had to
        # name. It is load-bearing: the calf link's own inertia about the knee is only
        # ~0.003 kg.m^2, so 0.1 raises the effective inertia ~34x. Without it the explicit
        # PD is past its stability limit at this timestep (Kd*dt/I = 2.5 > 2) and the legs
        # ring at +/-10 rad/s. PhysX defaults armature to 0, so the IsaacLab backend must
        # set it. Stated here so the two engines cannot silently disagree again.
        "dof_armature": 0.1,
        # termination
        "termination_if_roll_greater_than": 0.4,
        "termination_if_pitch_greater_than": 0.4,
        "termination_if_height_lower_than": 0.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # random push
        "push_interval_s": -1,
        "max_push_vel_xy": 1.0,
        # time (second)
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "command_type": "ang_vel_yaw",  # 'ang_vel_yaw' or 'heading'
        "action_scale": 0.25,
        "action_latency": 0.02,
        "action_range": 3.0,  # originally clip_actions
        "send_timeouts": True,
        "control_freq": 50,
        "decimation": 4,
        "feet_geom_offset": 1,
        "use_terrain": False,
        # domain randomization
        "randomize_friction": True,
        "friction_range": [0.2, 1.5],
        "randomize_base_mass": True,
        "added_mass_range": [-1.0, 3.0],
        "randomize_com_displacement": True,
        "com_displacement_range": [-0.01, 0.01],
        "randomize_motor_strength": False,
        "motor_strength_range": [0.9, 1.1],
        "randomize_motor_offset": True,
        "motor_offset_range": [-0.02, 0.02],
        "randomize_kp_scale": True,
        "kp_scale_range": [0.8, 1.2],
        "randomize_kd_scale": True,
        "kd_scale_range": [0.8, 1.2],
        # coupling
        "coupling": False,
    }
    obs_cfg = {
        "num_obs": 9 + 3 * env_cfg["num_dofs"],  # 45
        "num_history_obs": 1,
        "obs_noise": {
            "ang_vel": 0.1,
            "gravity": 0.02,
            "dof_pos": 0.01,
            "dof_vel": 0.5,
        },
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
        "num_priv_obs": 12 + 4 * env_cfg["num_dofs"],  # 60
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "soft_dof_pos_limit": 0.9,
        "base_height_target": 0.3,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -10.0,
            "base_height": -50.0,
            "torques": -0.0002,
            "collision": -1.0,
            "dof_vel": -0.0,
            "dof_acc": -2.5e-7,
            "feet_air_time": 1.0,
            "action_rate": -0.01,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-1.0, 1.0],
        "ang_vel_range": [-1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_env(
    num_envs: int,
    eval_mode: bool,
    sim_backend: str = "genesis",
    enable_camera: bool = False,
) -> Go2WalkEnv:
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    return Go2WalkEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        eval=eval_mode,
        debug=False,
        sim_backend=sim_backend,
        enable_camera=enable_camera,
    )
