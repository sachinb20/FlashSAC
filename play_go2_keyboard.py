"""Drive a trained `env=go2` policy live from the terminal keyboard.

Unlike an ordinary play script -- where the environment resamples its own velocity command
on a timer -- this pins the command to whatever you type, so you steer the robot yourself.
The network, observation layout, actuator model and asset all come from the same hydra
config the checkpoint was trained with, so **pass the same `--overrides` you trained with**.

    Up / Down     forward / back      (vx)
    Left / Right  strafe left / right (vy)
    Q / W         yaw left / right    (wz)
    Space or L    stop
    Ctrl-C        quit

Keys are read straight from this terminal, so the shell that launched the run keeps focus
-- you do not have to click into the Isaac Sim window. The terminal reports key *presses*
only, never releases, so commands are **sticky**: each press steps the axis by
``--key_step`` and the value holds until you change it or press space.

The keyboard-reading half is taken from `play_joystick.py` on the `feat/isaaclab-go2-port`
branch. Everything touching the environment is new, because that script drives a
`DirectRLEnv` whose command buffer and hooks do not exist here.

Example, for the run that completed 50M (unitree URDF + go2hv actuator):

    OMNI_KIT_ACCEPT_EULA=YES .venv-isaaclab/bin/python play_go2_keyboard.py \\
        --checkpoint_path models/go2-port-ladder/rung1-unitree-urdf/go2-walk/seed0-0814-174351/step48829 \\
        --overrides env=go2 --overrides env.sim_backend=isaaclab \\
        --overrides env.asset_source=unitree_urdf \\
        --overrides env.actuator_model=unitree_go2hv \\
        --overrides env.pd_stiffness=25.0 --overrides env.pd_damping=0.5 \\
        --overrides agent=flashSAC --overrides agent.asymmetric_observation=true \\
        --overrides num_train_envs=1
"""

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import random
import select
import sys
import termios
import time
import tty
from typing import Any, MutableMapping

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.types import Tensor


class TerminalKeys:
    """Read driving keys straight from the terminal, bypassing the Omniverse window.

    IsaacLab's own ``Se2Keyboard`` subscribes to the *Isaac Sim app window's* keyboard, so
    it only sees keys while that window has focus -- stay in the shell and every keystroke
    goes to the terminal instead, and the command never moves. This reads stdin directly.

    Commands are sticky because a terminal reports presses, not releases: each press steps
    the corresponding axis by ``step`` and the value holds until changed.
    """

    ESCAPES = {"[A": (0, +1.0), "[B": (0, -1.0), "[D": (1, +1.0), "[C": (1, -1.0)}

    def __init__(self, step: float, yaw_left_key: str, yaw_right_key: str, low, high):
        self._step = float(step)
        self._yaw_left = yaw_left_key.lower()
        self._yaw_right = yaw_right_key.lower()
        self._low = low
        self._high = high
        self._command = np.zeros(3, dtype=np.float32)
        self._pending = ""
        # No tty when stdin is a pipe or file (nohup, CI, `< /dev/null`). Degrade to a
        # fixed zero command rather than dying in termios, so the run is still usable for
        # checking that the policy stands.
        self._enabled = sys.stdin.isatty()
        if not self._enabled:
            print("[play] stdin is not a terminal -- keyboard disabled, command held at zero.")
            return
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def restore(self) -> None:
        if self._enabled:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def _apply(self, axis: int, direction: float) -> None:
        self._command[axis] = float(
            np.clip(self._command[axis] + direction * self._step, self._low[axis], self._high[axis])
        )

    def poll(self) -> np.ndarray:
        """Drain everything buffered since the last call, then return the current command.

        Reads the raw fd via ``os.read`` rather than ``sys.stdin.read``: ``sys.stdin`` is a
        buffered ``TextIOWrapper``, so one read can pull a whole chunk into Python's buffer,
        after which ``select()`` reports the fd empty and the drain loop exits with
        keystrokes still unread. ``os.read`` and ``select`` agree on what "has data" means.
        """
        if not self._enabled:
            return self._command.copy()
        while select.select([self._fd], [], [], 0.0)[0]:
            chunk = os.read(self._fd, 1024)
            if not chunk:
                break
            self._pending += chunk.decode("utf-8", errors="ignore")
        self._consume()
        return self._command.copy()

    def _consume(self) -> None:
        while self._pending:
            if self._pending[0] == "\x1b":
                # Arrow keys arrive as ESC [ A/B/C/D. Hold a partial tail for the next poll.
                if len(self._pending) < 3:
                    return
                tail = self._pending[1:3]
                self._pending = self._pending[3:]
                if tail in self.ESCAPES:
                    axis, direction = self.ESCAPES[tail]
                    self._apply(axis, direction)
                continue
            ch = self._pending[0].lower()
            self._pending = self._pending[1:]
            if ch == self._yaw_left:
                self._apply(2, +1.0)
            elif ch == self._yaw_right:
                self._apply(2, -1.0)
            elif ch in (" ", "l"):
                self._command[:] = 0.0
            elif ch == "\x03":  # Ctrl-C never reaches the signal handler in cbreak mode
                raise KeyboardInterrupt


def make_live_env(cfg: Any, num_envs: int, show_viewer: bool) -> Any:
    """Build the same env `env=go2` trains on, with the viewer turned on.

    Reads the rung flags off the hydra config so the plant matches the checkpoint: pass the
    same overrides you trained with or the policy will be driving a different robot.
    """
    from flash_rl.envs.go2_common.go2_env import get_env
    from flash_rl.envs.go2_sim import Go2VectorEnv

    env_cfg = cfg.env
    if env_cfg.env_type != "go2":
        raise NotImplementedError(
            f"This script drives env_type='go2', got {env_cfg.env_type!r}. The command-pinning "
            "hook below is specific to that env's command buffer."
        )

    env = get_env(
        num_envs=num_envs,
        eval_mode=True,  # also disables domain randomisation
        sim_backend=env_cfg.sim_backend,
        enable_camera=False,
        asset_source=env_cfg.get("asset_source"),
        actuator_model=env_cfg.get("actuator_model"),
        ground_material=env_cfg.get("ground_material"),
        pd_stiffness=env_cfg.get("pd_stiffness"),
        pd_damping=env_cfg.get("pd_damping"),
        dof_armature=env_cfg.get("dof_armature"),
        show_viewer=show_viewer,
    )
    return Go2VectorEnv(env, rescale_action=bool(env_cfg.rescale_action), to_numpy=True)


def play(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    # CUDA graphs cannot coexist with Isaac Sim in one process: they demand strict
    # accounting of every allocation in their memory pool, and Isaac's own CUDA
    # allocations land there unaccounted, so the first policy forward dies with
    #   "These live storage data ptrs are in the cudagraph pool but not accounted for"
    # The config default 'auto' resolves to 'max-autotune' on torch >= 2.9 and to
    # 'reduce-overhead' below it -- *both* force graphs on, so neither is safe here.
    # Overriding after compose rather than asking the caller to remember a flag.
    cfg.agent.compile_mode = args.compile_mode
    print(f"[play] agent.compile_mode = {args.compile_mode}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    env = make_live_env(cfg, args.num_envs, show_viewer=not args.headless)
    base_env = env.base_env

    # Clamp to the ranges the policy was trained on, so we stay in distribution.
    cmd_cfg = base_env.command_cfg
    low = np.array([cmd_cfg["lin_vel_x_range"][0], cmd_cfg["lin_vel_y_range"][0], cmd_cfg["ang_vel_range"][0]])
    high = np.array([cmd_cfg["lin_vel_x_range"][1], cmd_cfg["lin_vel_y_range"][1], cmd_cfg["ang_vel_range"][1]])
    print(
        f"[play] command clamp: vx[{low[0]:+.2f},{high[0]:+.2f}] "
        f"vy[{low[1]:+.2f},{high[1]:+.2f}] wz[{low[2]:+.2f},{high[2]:+.2f}]"
    )

    keys = TerminalKeys(args.key_step, args.yaw_left_key, args.yaw_right_key, low, high)
    held = {"command": np.zeros(3, dtype=np.float32)}

    # The env resamples its own velocity command -- on the resampling timer in
    # post_physics_step, and again inside reset_idx. Replacing the sampler with a writer
    # pins the command through both, so a reset cannot steer the robot out from under you.
    def _pin_command(envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        value = torch.as_tensor(held["command"], dtype=base_env.commands.dtype, device=base_env.commands.device)
        base_env.commands[envs_idx, :3] = value

    base_env._resample_commands = _pin_command

    observations, env_info = env.reset()
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args.checkpoint_path)

    yl, yr = args.yaw_left_key.upper(), args.yaw_right_key.upper()
    print(
        f"[play] Keep THIS terminal focused (not the sim window).\n"
        f"[play]   Up/Down = forward/back, Left/Right = strafe, {yl}/{yr} = yaw left/right,\n"
        f"[play]   Space or L = stop, Ctrl-C = quit.\n"
        f"[play]   Sticky: each press steps by {args.key_step} and holds."
    )

    dt = base_env.dt
    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
    step = 0
    next_frame = time.perf_counter()
    try:
        while True:
            held["command"] = keys.poll()
            # Write every step, not only on the resample tick: otherwise a new keypress
            # would not take effect until the env happened to resample.
            _pin_command(torch.arange(base_env.num_envs, device=base_env.device))

            actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
            observations, _, _, _, _ = env.step(np.array(actions))
            prev_transition = {"next_observation": observations}

            # Repaint and pump the Omniverse UI. Physics steps with render=False, so
            # without this the window never redraws and the WM reports it unresponsive.
            base_env.sim.update_viewer(base_env.base_pos[0])

            step += 1
            if args.print_every > 0 and step % args.print_every == 0:
                c = held["command"]
                lin = base_env.base_lin_vel[0]
                ang = base_env.base_ang_vel[0]
                print(
                    f"[play] cmd vx={c[0]:+.2f} vy={c[1]:+.2f} wz={c[2]:+.2f} | "
                    f"measured vx={float(lin[0]):+.2f} vy={float(lin[1]):+.2f} wz={float(ang[2]):+.2f} | "
                    f"h={float(base_env.base_pos[0, 2]):.3f}"
                )

            # Pace to wall clock. With one environment the sim runs far faster than real
            # time, which makes the robot unsteerable.
            if not args.free_run:
                next_frame += dt
                remaining = next_frame - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_frame = time.perf_counter()  # fell behind; do not accumulate debt
    except KeyboardInterrupt:
        print("\n[play] stopping.")
    finally:
        # Restore cooked mode first, so a crash cannot leave the shell in cbreak.
        keys.restore()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drive a trained env=go2 policy from the terminal keyboard")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the agent checkpoint directory")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--yaw_left_key", type=str, default="q", help="Key that yaws left (+wz)")
    parser.add_argument("--yaw_right_key", type=str, default="w", help="Key that yaws right (-wz)")
    parser.add_argument("--key_step", type=float, default=0.25, help="Command increment per keypress")
    parser.add_argument("--print_every", type=int, default=50, help="Print commanded vs measured every N steps (0=off)")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune-no-cudagraphs",
        help="Overrides agent.compile_mode. Must not enable CUDA graphs -- Isaac Sim shares the "
        "process and its allocations break cudagraph accounting. 'default' disables autotuning too.",
    )
    parser.add_argument("--headless", action="store_true", help="No viewer window (for checking it runs)")
    parser.add_argument("--free_run", action="store_true", help="Do not pace to wall clock")
    args = parser.parse_args()
    play(args)
