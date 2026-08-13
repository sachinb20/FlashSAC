import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import random
from pathlib import Path
from typing import MutableMapping

import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.types import Tensor

# (lin_vel_x, lin_vel_y, ang_vel_z), body frame: +x forward, +y left, +z yaw ccw.
DIRECTIONS = {
    "forward": (0.5, 0.0, 0.0),
    "backward": (-0.5, 0.0, 0.0),
    "left": (0.0, 0.5, 0.0),
    "right": (0.0, -0.5, 0.0),
    "yaw": (0.0, 0.0, 0.7),
}


def record(args: argparse.Namespace) -> None:
    if args.env_type != "isaaclab_go2":
        raise NotImplementedError(
            f"eval_directions.py only supports env_type='isaaclab_go2' right now (got {args.env_type!r}) -- "
            "it uses set_velocity_command() and the tracking-camera render(), both specific to that env."
        )

    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    from flash_rl.envs.isaaclab_go2 import make_isaaclab_go2_env

    isaac_overrides = {str(k): v for k, v in cfg.env.items() if str(k).startswith("isaac_")}
    env = make_isaaclab_go2_env(
        env_name=cfg.env.env_name,
        num_envs=args.num_envs,
        seed=int(cfg.seed),
        eval_mode=True,
        headless=True,
        enable_cameras=True,
        record_video=True,
        device=str(cfg.env.get("device", "cuda:0")),
        **isaac_overrides,
    )
    # Smaller than IsaacLab's 1280x720 default -- keeps GIF file size and encode time sane
    # for 5 x 7s clips. Must be set before the first render() call (lazily creates the
    # render product on first use).
    env.envs.cfg.viewer.resolution = (args.width, args.height)

    observations, env_info = env.reset()
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args.checkpoint_path)

    control_hz = 1.0 / (float(env.envs.cfg.decimation) * float(env.envs.cfg.sim.dt))
    steps_per_direction = int(round(args.duration_s * control_hz))
    print(f"control rate: {control_hz:.1f} Hz -> {steps_per_direction} steps per direction", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, command in DIRECTIONS.items():
        print(f"=== {name}: command=(lin_x={command[0]}, lin_y={command[1]}, ang_z={command[2]}) ===", flush=True)
        # Fresh spawn for this direction -- also re-rolls domain randomization (mass/friction/
        # motor strength) since eval_mode's DR-restore-defaults only applies when
        # isaac_dr_train_only=true, matching what "new robots" means physically, not just visually.
        observations, _ = env.reset()
        command_tensor = torch.tensor([command] * args.num_envs, dtype=torch.float32)

        prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
        out_path = out_dir / f"{name}.mp4"
        # fps=control_hz (50) plays back at the same rate frames were captured -- real-time.
        # GIF's per-frame delay is in whole centiseconds and most browsers/viewers additionally
        # clamp very short delays to a much larger minimum (~100ms) regardless of what's
        # encoded, which is why a "correctly timed" 50fps GIF often visibly plays back ~5x slow;
        # mp4 doesn't have either problem.
        writer = imageio.get_writer(out_path, fps=control_hz, codec="libx264", quality=8)
        num_frames = 0
        for step in range(steps_per_direction):
            # Re-issued every step: set_velocity_command's internal countdown (
            # command_resampling_time_range[1], e.g. 7s) can otherwise expire near the end of
            # this exact-length clip and get silently resampled to something random.
            env.call("set_velocity_command", command_tensor)

            actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
            actions = np.array(actions)
            observations, rewards, terminateds, truncateds, infos = env.step(actions)

            frame = env.render()[0]  # (1, H, W, C) -> (H, W, C); one shared tracking camera
            writer.append_data(frame)
            num_frames += 1
            prev_transition = {"next_observation": observations}

            if (terminateds | truncateds).any() and step < steps_per_direction - 1:
                # A robot fell and auto-reset mid-clip -- expected, not an error. The command
                # re-issue above corrects the freshly-resampled command back to `name` next step.
                print(f"  step {step}: {int((terminateds | truncateds).sum())} env(s) reset mid-clip", flush=True)

        writer.close()
        print(f"  saved {out_path} ({num_frames} frames, {out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a trained Go2 policy stepping through forward/backward/left/right/yaw commands."
    )
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--env_type", type=str, default="isaaclab_go2")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to agent checkpoint directory")
    parser.add_argument("--num_envs", type=int, default=4, help="Robots recorded simultaneously per direction")
    parser.add_argument("--duration_s", type=float, default=7.0, help="Seconds per direction")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--out_dir", type=str, default="videos/eval_directions")
    args = parser.parse_args()
    record(args)
