"""The simulator seam for the Go2 environment.

``Go2BaseEnv`` talks to the physics engine only through ``Go2SimBackend``. Everything
else in the environment -- observations, rewards, commands, action pipeline, PD torque
law, termination, episode bookkeeping, domain-randomisation *sampling* -- is plain
``torch`` on buffers and is shared verbatim between backends.

Two conventions every backend must honour, because the shared core assumes them:

**Link indices are robot-local.** Index 0 is the robot's own base link. Genesis's
rigid solver numbers links globally across entities (the ground plane occupies slot 0,
so the robot's base is solver index 1); ``GenesisSimBackend`` adds that offset
internally. IsaacLab resolves its own body ordering. The core never sees either.

**Positions are per-environment local.** Genesis simulates ``n_envs`` overlapping
parallel worlds that all share an origin, so ``base_pos`` is already relative to the
env. IsaacLab lays environments out on a grid, so ``IsaacLabSimBackend`` subtracts
``env_origins`` on read and adds it on write. This matters: the core's reset spreads
the base by +/-1.0 in xy, which must not walk a robot into its neighbour's tile.

**Velocity frames.** ``get_base_lin_vel_world``/``get_base_ang_vel_world`` return the
base link's *world-frame* linear/angular velocity (Genesis ``get_vel()``/``get_ang()``).
The core does the rotation into the observation frames itself -- and note it uses a
yaw-only frame for linear velocity but the full body frame for angular velocity. That
asymmetry is inherited from the Genesis env and is deliberately preserved.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

import torch


class Go2SimBackend(Protocol):
    """Physics operations the Go2 environment needs. See module docstring for conventions."""

    device: torch.device
    num_envs: int
    n_links: int

    # ---------------------------------------------------------------- setup

    def resolve_dof_indices(self, dof_names: Sequence[str]) -> list[int]:
        """Joint indices for ``dof_names``, in exactly the order given."""
        ...

    def resolve_link_indices(self, name_substrings: Sequence[str]) -> list[int]:
        """Robot-local link indices whose name *contains* any of ``name_substrings``.

        Substring matching, not regex, and ordered by link index -- matching Genesis's
        ``find_link_indices``.
        """
        ...

    def get_dof_pos_limits(self, dof_indices: Sequence[int]) -> torch.Tensor:
        """``(n_dof, 2)`` lower/upper joint position limits from the asset."""
        ...

    def get_dof_force_range(self, dof_indices: Sequence[int]) -> torch.Tensor:
        """``(n_dof,)`` upper effort limit from the asset."""
        ...

    def set_pd_gains(self, p_gains: torch.Tensor, d_gains: torch.Tensor, dof_indices: Sequence[int]) -> None:
        """Register nominal gains with the engine.

        Only meaningful for an engine-side controller. Both backends drive the robot by
        explicit torque, so this is bookkeeping the core mirrors from Genesis; the PD law
        itself lives in ``Go2BaseEnv._compute_torques``.
        """
        ...

    # ---------------------------------------------------------------- state

    def get_base_pos(self) -> torch.Tensor:
        """``(num_envs, 3)`` base position, env-local."""
        ...

    def get_base_quat(self) -> torch.Tensor:
        """``(num_envs, 4)`` base orientation, ``wxyz``."""
        ...

    def get_base_lin_vel_world(self) -> torch.Tensor: ...

    def get_base_ang_vel_world(self) -> torch.Tensor: ...

    def get_dofs_position(self, dof_indices: Sequence[int]) -> torch.Tensor: ...

    def get_dofs_velocity(self, dof_indices: Sequence[int]) -> torch.Tensor: ...

    def get_links_net_contact_force(self) -> torch.Tensor:
        """``(num_envs, n_links, 3)`` net contact force per robot-local link."""
        ...

    def get_links_pos(self, link_indices: Sequence[int]) -> torch.Tensor: ...

    def get_links_quat(self, link_indices: Sequence[int]) -> torch.Tensor: ...

    def get_links_vel(self, link_indices: Sequence[int]) -> torch.Tensor: ...

    def get_links_com(self, link_indices: Sequence[int]) -> torch.Tensor:
        """``(num_envs, len(link_indices), 3)`` centre of mass in world/root frame."""
        ...

    # -------------------------------------------------------------- control

    def apply_dof_force(self, torques: torch.Tensor, dof_indices: Sequence[int]) -> None:
        """Apply joint torques directly. No engine-side clamping (Genesis applies none)."""
        ...

    def step(self) -> None:
        """Advance one physics substep of ``sim_dt``."""
        ...

    # ---------------------------------------------------------------- reset

    def set_dofs_position(
        self,
        position: torch.Tensor,
        dof_indices: Sequence[int],
        zero_velocity: bool,
        envs_idx: torch.Tensor,
    ) -> None: ...

    def set_base_pos(self, pos: torch.Tensor, envs_idx: torch.Tensor) -> None: ...

    def set_base_quat(self, quat: torch.Tensor, envs_idx: torch.Tensor) -> None: ...

    def set_base_velocity(self, lin_vel: torch.Tensor, ang_vel: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Write the 6-DoF root velocity (Genesis: free-joint DoFs 0..5)."""
        ...

    def zero_all_dofs_velocity(self, envs_idx: torch.Tensor) -> None: ...

    def get_all_dofs_velocity(self) -> torch.Tensor:
        """Every DoF's velocity, root first. Entries 0..2 are base linear velocity.

        Used only by the random-push path, which writes back a modified copy. Genesis
        exposes the floating base as DoFs 0..5 of the articulation; the shape is therefore
        engine-dependent and the core only ever touches ``[:, :2]``.
        """
        ...

    def set_all_dofs_velocity(self, vel: torch.Tensor) -> None: ...

    # ------------------------------------------------- domain randomisation

    def set_geoms_friction_ratio(self, ratios: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Scale every collision geom's friction by a per-env ratio.

        Genesis semantics: a *multiplier* on the asset's own friction, not an absolute
        value.
        """
        ...

    def set_links_mass_shift(
        self, added_mass: torch.Tensor, link_indices: Sequence[int], envs_idx: torch.Tensor
    ) -> None:
        """Add ``added_mass`` kg to the given links (a delta, not an assignment)."""
        ...

    def set_links_com_shift(self, com_shift: torch.Tensor, link_indices: Sequence[int], envs_idx: torch.Tensor) -> None:
        """Displace the given links' centre of mass by ``com_shift`` metres."""
        ...

    # --------------------------------------------------------------- render

    def render(self, track_pos: Optional[torch.Tensor] = None) -> Any:
        """RGB frame of env 0, or ``None`` if the backend has no camera."""
        ...

    def close(self) -> None: ...


"""Backend constructor keyword arguments (all backends must accept them):

``num_envs``, ``num_build_envs``, ``sim_dt``, ``control_dt``, ``urdf_path``,
``links_to_keep``, ``base_init_pos``, ``base_init_quat``, ``default_joint_angles``,
``show_viewer``, ``debug``, ``enable_camera``, ``device``.

``default_joint_angles`` is the nominal stance. Genesis spawns at the URDF zero pose and
merely warns that it violates the joint limits, but IsaacLab validates the spawn pose and
raises, and the all-zeros default is illegal for this robot (the calf joints are limited
to ``[-2.723, -0.838]``). Backends that need a valid spawn pose must use it.
"""


def make_backend(sim_backend: str, **kwargs: Any) -> Go2SimBackend:
    """Construct a backend by name.

    Imports are deferred because Genesis and IsaacLab **cannot be installed in the same
    virtualenv** -- ``pyproject.toml``'s ``[tool.uv] conflicts`` declares the two extras
    mutually exclusive (they pin different torch versions). Importing this module must
    therefore never pull in either engine.
    """
    if sim_backend == "genesis":
        from .backends.genesis_backend import GenesisSimBackend

        return GenesisSimBackend(**kwargs)
    if sim_backend == "isaaclab":
        from .backends.isaaclab_backend import IsaacLabSimBackend

        return IsaacLabSimBackend(**kwargs)
    raise ValueError(f"Unknown sim_backend {sim_backend!r}, expected 'genesis' or 'isaaclab'.")
