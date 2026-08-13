from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import NDArray


def recursive_to_numpy(
    data: Union[torch.Tensor, dict[str, Any], list[Any], tuple[Any, ...], NDArray],
) -> Union[NDArray, dict[str, Any], list[Any], tuple[Any, ...]]:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)
    else:
        return data


class IsaacLabGo2VectorEnv(VectorEnv[NDArray, NDArray, NDArray]):
    """Gymnasium "SyncVectorEnv" wrapper for the ported direct Unitree Go2 IsaacLab env.

    Modeled on flash_rl/envs/genesis.py's GenesisVectorEnv (episode-info translation, true
    final_obs, render contract) rather than flash_rl/envs/isaaclab.py's IsaacLabVectorEnv,
    since the latter's generic gym.make() path doesn't apply here (this env is not
    gym-registered) and has known gaps this env avoids: final_obs is the post-reset obs
    there, and render() is unimplemented.
    """

    def __init__(
        self,
        env: Any,
        rescale_action: bool = False,
        to_numpy: bool = True,
        **kwargs: Any,
    ):
        self.envs = env
        self.num_envs = self.envs.num_envs
        self.device = self.envs.device
        self.rescale_action = rescale_action
        self.to_numpy = to_numpy

        # obs_size is the actor's own width (used for actor_observation_size in info, read by
        # FlashSACAgent when asymmetric_observation=true). cfg.observation_space is the FULL
        # width -- actor columns plus any privileged critic-only tail (see
        # UnitreeGo2VelocityDirectEnvCfg.privileged_base_lin_vel) -- and is what the Box space
        # must reflect, since the buffer/critic always sees the full-width observation.
        self.obs_size = int(self.envs.actor_observation_dim)
        self.critic_obs_size = int(self.envs.cfg.observation_space)
        self.single_observation_space = gym.spaces.Box(
            low=0.0, high=0.0, shape=(self.critic_obs_size,), dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        action_size = int(self.envs.single_action_space.shape[0])
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}

    @property
    def base_env(self) -> Any:
        return self.envs

    @property
    def unwrapped(self) -> VectorEnv[NDArray, NDArray, NDArray]:
        return self.base_env  # type: ignore

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[NDArray, dict[str, Any]]:
        obs_dict, _ = self.envs.reset()
        obs = obs_dict["policy"]
        info: dict[str, Any] = {"actor_observation_size": [self.obs_size]}
        if self.to_numpy:
            obs = recursive_to_numpy(obs)
            info = recursive_to_numpy(info)  # type: ignore
        return obs, info

    def step(
        self, actions: Union[NDArray, dict[str, NDArray]]
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, dict[str, Any]]:
        if isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.from_numpy(np.asarray(actions)).to(self.device)
        if self.rescale_action:
            torch_actions = torch_actions.clamp(-1.0, 1.0)

        obs_dict, rewards, terminations, truncations, _ = self.envs.step(torch_actions)
        obs = obs_dict["policy"]

        infos: dict[str, Any] = {}
        # The ported env snapshots the true pre-reset terminal observation in _reset_idx
        # (before any reset mutation runs) -- see _snapshot_before_reset() in
        # isaaclab_go2_envs/isaaclab_go2_velocity_direct.py. This avoids the post-reset
        # final_obs gap that flash_rl/envs/isaaclab.py has (isaac-sim/IsaacLab#1362).
        infos["final_obs"] = self.envs._final_observations.detach().clone()

        done = terminations | truncations
        if done.any():
            # Already-reduced python floats (mean reward rate over the just-finished episode,
            # mirroring genesis_envs/go2_base.py's extras["episode"]); NOT raw per-env tensors,
            # and NOT the single terminal-step reward -- see _update_episode_info().
            infos["episode_info"] = dict(self.envs._last_episode_info)
            infos["final_info"] = {}
            infos["_final_info"] = done
            infos["_final_obs"] = done
            infos["_elapsed_steps"] = done

        if self.to_numpy:
            obs = recursive_to_numpy(obs)
            rewards = recursive_to_numpy(rewards)
            terminations = recursive_to_numpy(terminations)  # type: ignore
            truncations = recursive_to_numpy(truncations)
            infos = recursive_to_numpy(infos)

        return obs, rewards, terminations, truncations, infos  # type: ignore

    def close(self, **kwargs: Any) -> None:
        return

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        function = getattr(self.envs, name)
        return function(*args, **kwargs)

    def render(self) -> Optional[NDArray]:
        # Requires the env to have been constructed with render_mode="rgb_array" (via
        # make_isaaclab_go2_env(..., record_video=True)) and enable_cameras=True at AppLauncher
        # boot -- otherwise the base DirectRLEnv.render() returns None (render_mode is None).
        image = self.envs.render()
        if image is None:
            raise RuntimeError(
                "render() returned None -- this env was constructed without render_mode='rgb_array' "
                "(pass record_video=True to make_isaaclab_go2_env) and/or enable_cameras=True."
            )
        if self.to_numpy:
            image = recursive_to_numpy(image)
        # Match genesis.py's convention: (H, W, C) -> (1, H, W, C), one camera for all envs.
        return image[np.newaxis, ...]


def make_isaaclab_go2_env(
    env_name: str,
    num_envs: int,
    seed: int,
    eval_mode: bool,
    rescale_action: bool = False,
    headless: bool = True,
    enable_cameras: bool = False,
    device: str = "cuda:0",
    record_video: bool = False,
    **isaac_overrides: Any,
) -> VectorEnv[NDArray, NDArray, NDArray]:
    """Build the IsaacLab app, construct the direct Go2 velocity env, and wrap it.

    `isaaclab.*` cannot be imported before the sim app boots (only `isaaclab.app.AppLauncher`
    is importable cold), so the env-construction imports are deferred to inside this function,
    after AppLauncher(...) returns -- mirroring flash_rl/envs/isaaclab.py's ordering.

    record_video=True sets render_mode="rgb_array" so IsaacLabGo2VectorEnv.render() works; it
    requires enable_cameras=True too (headless offscreen rendering needs both).
    """
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless, device=device, enable_cameras=enable_cameras)
    simulation_app = app_launcher.app  # noqa: F841  -- keep the sim app alive for this process

    from .isaaclab_go2_envs import get_isaaclab_go2_env

    env = get_isaaclab_go2_env(
        env_name=env_name,
        num_envs=num_envs,
        seed=seed,
        device=device,
        render_mode="rgb_array" if record_video else None,
        **isaac_overrides,
    )
    env.set_eval_mode(bool(eval_mode))
    return IsaacLabGo2VectorEnv(env, rescale_action=rescale_action, to_numpy=True)
