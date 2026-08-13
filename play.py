import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import random
from typing import Any, MutableMapping

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.types import Tensor


def make_live_env(cfg: Any, num_envs: int) -> Any:
    """Construct this run's env with live rendering enabled, dispatched by cfg.env.env_type.

    Unlike train.py (which goes through flash_rl.envs.create_envs), this calls each backend's
    factory directly so it can pass headless=False / show_viewer=True -- knobs create_envs
    doesn't expose since normal training always runs headless.
    """
    env_type = cfg.env.env_type
    if env_type == "genesis":
        from flash_rl.envs.genesis import make_genesis_env

        return make_genesis_env(
            env_name=cfg.env.env_name,
            num_envs=num_envs,
            rescale_action=cfg.env.rescale_action,
            eval_mode=True,
            show_viewer=True,
        )

    elif env_type == "isaaclab_go2":
        from flash_rl.envs.isaaclab_go2 import make_isaaclab_go2_env

        # Forward every isaac_* key from the training config so the live env matches what the
        # checkpoint was trained on (actuator model, asset source, action scale, DR, ...).
        isaac_overrides = {str(k): v for k, v in cfg.env.items() if str(k).startswith("isaac_")}
        return make_isaaclab_go2_env(
            env_name=cfg.env.env_name,
            num_envs=num_envs,
            seed=int(cfg.seed),
            eval_mode=True,
            headless=False,
            enable_cameras=True,
            device=str(cfg.env.get("device", "cuda:0")),
            **isaac_overrides,
        )

    elif env_type == "isaaclab":
        from flash_rl.envs.isaaclab import make_isaaclab_env

        return make_isaaclab_env(env_name=cfg.env.env_name, num_envs=num_envs, seed=int(cfg.seed), headless=False)

    else:
        raise NotImplementedError(
            f"Live play is not implemented for env_type={env_type!r}. Supported: 'genesis', 'isaaclab_go2', 'isaaclab'."
        )


def play(args: argparse.Namespace) -> None:
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    checkpoint_path = args.checkpoint_path
    num_envs = args.num_envs
    num_episodes = args.num_episodes

    # Load config (same as train.py) -- pass the same --overrides you trained with, so the
    # network architecture (obs/action dims, asymmetric_observation) matches the checkpoint.
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Create environment with rendering on, in whichever sim this config trained on
    env = make_live_env(cfg, num_envs)

    # isaaclab's generic wrapper supports the RSL-RL-style decorrelated reset; genesis and
    # isaaclab_go2 don't take that kwarg (see evaluation.py's own env_type branch for the
    # same distinction).
    reset_kwargs = {"random_start_init": False} if cfg.env.env_type == "isaaclab" else {}
    observations, env_info = env.reset(**reset_kwargs)

    # Create agent using config (same as train.py)
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    # Load checkpoint
    agent.load(checkpoint_path)

    # Play loop
    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
    completed_episodes = 0
    episode_returns = np.zeros(num_envs)

    while completed_episodes < num_episodes:
        actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, infos = env.step(actions)

        episode_returns += rewards
        episode_dones = np.logical_or(terminateds, truncateds)

        for idx in range(num_envs):
            if episode_dones[idx]:
                completed_episodes += 1
                print(f"Episode {completed_episodes}: return = {episode_returns[idx]:.2f}")
                episode_returns[idx] = 0.0
                if completed_episodes >= num_episodes:
                    break

        observations = next_observations
        prev_transition = {"next_observation": observations}

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained FlashSAC agent live, in whichever sim it trained on")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to agent checkpoint directory")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments to visualize")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to play")
    args = parser.parse_args()
    play(args)
