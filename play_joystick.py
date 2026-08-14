"""Drive a trained isaaclab_go2 policy live from a gamepad or keyboard.

Unlike play.py -- which lets the env sample its own velocity commands on a timer -- this
script pins the velocity command to whatever the input device reports, so you steer the robot
yourself. Everything else (network, observation layout, actuator model) comes from the same
hydra config the checkpoint was trained with, so pass the SAME --overrides you trained with.

Key/stick bindings come from IsaacLab's own Se2 devices:

    gamepad   left stick = forward/strafe, right stick X = turn
    keyboard  Up/Down = forward/back, Left/Right = strafe, Q/W = turn left/right,
              L = zero the command

The keyboard yaw keys default to Q/W (IsaacLab's own Se2Keyboard uses Z/X); override with
--yaw_left_key / --yaw_right_key.

Both devices are created by default and their commands are summed, so the keyboard works
whether or not a gamepad is plugged in.
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

    `Se2Keyboard` subscribes to the *Isaac Sim app window's* keyboard, so it only sees keys
    while that window has focus -- if you stay in the shell, every keystroke goes to the
    terminal instead and the command never moves. This reads stdin directly, so it works from
    the terminal that launched the run.

    The terminal reports key *presses* only, never releases, so commands are sticky: each
    press steps the corresponding axis by `step` and the value holds until changed. Space or
    'l' zeroes everything.
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
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def restore(self) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def _apply(self, axis: int, direction: float) -> None:
        self._command[axis] = float(
            np.clip(self._command[axis] + direction * self._step, self._low[axis], self._high[axis])
        )

    def poll(self) -> np.ndarray:
        """Drain everything buffered since the last call, then return the current command.

        Reads the raw fd via os.read rather than sys.stdin.read: sys.stdin is a buffered
        TextIOWrapper, so a single read can pull a whole chunk into Python's buffer, after
        which select() reports the fd as empty and the drain loop exits with keystrokes still
        unread. os.read and select agree on what "has data" means.
        """
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


def _rebind_keyboard(keyboard, yaw_left_key: str, yaw_right_key: str) -> None:
    """Arrows (or numpad) translate; two chosen keys yaw. Replaces IsaacLab's Z/X yaw keys.

    `_INPUT_KEY_MAPPING` is a plain name->delta dict that `_on_keyboard_event` looks up on both
    press and release, so swapping the whole dict before the loop starts is sufficient -- there
    is no cached per-key state that could fall out of sync.
    """
    vx = keyboard.v_x_sensitivity
    vy = keyboard.v_y_sensitivity
    wz = keyboard.omega_z_sensitivity
    forward = np.asarray([1.0, 0.0, 0.0]) * vx
    strafe = np.asarray([0.0, 1.0, 0.0]) * vy
    yaw = np.asarray([0.0, 0.0, 1.0]) * wz
    keyboard._INPUT_KEY_MAPPING = {
        "UP": forward,
        "NUMPAD_8": forward,
        "DOWN": -forward,
        "NUMPAD_2": -forward,
        # +y is the robot's left in body frame, so LEFT strafes left.
        "LEFT": strafe,
        "NUMPAD_4": strafe,
        "RIGHT": -strafe,
        "NUMPAD_6": -strafe,
        yaw_left_key.upper(): yaw,
        yaw_right_key.upper(): -yaw,
    }


def _build_devices(args: argparse.Namespace, sim_device: str) -> list:
    """Create the requested Se2 input devices (imported post-app-launch: they need carb)."""
    from isaaclab.devices.gamepad import Se2Gamepad, Se2GamepadCfg
    from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg

    devices = []
    if args.device in ("both", "gamepad"):
        devices.append(
            Se2Gamepad(
                Se2GamepadCfg(
                    v_x_sensitivity=args.v_x_sensitivity,
                    v_y_sensitivity=args.v_y_sensitivity,
                    omega_z_sensitivity=args.omega_z_sensitivity,
                    dead_zone=args.dead_zone,
                    sim_device=sim_device,
                )
            )
        )
    if args.device in ("both", "keyboard"):
        keyboard = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=args.v_x_sensitivity,
                v_y_sensitivity=args.v_y_sensitivity,
                omega_z_sensitivity=args.omega_z_sensitivity,
                sim_device=sim_device,
            )
        )
        _rebind_keyboard(keyboard, args.yaw_left_key, args.yaw_right_key)
        devices.append(keyboard)
    return devices


def _command_limits(cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    """Clamp bounds taken from the training command ranges, so we stay in-distribution."""
    env_cfg = cfg.env

    def rng(key, fallback):
        value = env_cfg.get(key, None)
        return (float(value[0]), float(value[1])) if value is not None else fallback

    x_lo, x_hi = rng("isaac_velocity_command_lin_vel_x_range", (-1.0, 1.0))
    y_lo, y_hi = rng("isaac_velocity_command_lin_vel_y_range", (-1.0, 1.0))
    w_lo, w_hi = rng("isaac_velocity_command_ang_vel_z_range", (-1.0, 1.0))
    return (np.array([x_lo, y_lo, w_lo]), np.array([x_hi, y_hi, w_hi]))


def make_live_env(cfg: Any, num_envs: int) -> Any:
    """Same construction path as play.py's isaaclab_go2 branch, with the viewer on."""
    if cfg.env.env_type != "isaaclab_go2":
        raise NotImplementedError(
            f"Joystick play is only implemented for env_type='isaaclab_go2', got {cfg.env.env_type!r}. "
            "The command-pinning hook below is specific to that env's command buffer."
        )
    from flash_rl.envs.isaaclab_go2 import make_isaaclab_go2_env

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


def play(args: argparse.Namespace) -> None:
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

    env = make_live_env(cfg, args.num_envs)
    base_env = env.base_env
    sim_device = str(base_env.device)

    low, high = _command_limits(cfg)
    print(f"[joystick] command clamp: vx{low[0], high[0]}  vy{low[1], high[1]}  wz{low[2], high[2]}")

    terminal_keys = None
    if args.terminal_keys:
        terminal_keys = TerminalKeys(args.key_step, args.yaw_left_key, args.yaw_right_key, low, high)
        devices = []
    else:
        devices = _build_devices(args, sim_device)
        for device in devices:
            device.reset()

    # The env normally advances/resamples its own velocity command inside every observation
    # build. Replacing that hook makes the observation carry the device command instead --
    # and because it is the hook the observation itself calls, this holds through resets,
    # which would otherwise resample the command out from under us.
    held = {"command": np.zeros(3, dtype=np.float32)}

    def _pin_command_to_device():
        base_env._commands[:] = torch.as_tensor(
            held["command"], dtype=base_env._commands.dtype, device=base_env._commands.device
        )

    base_env._maybe_update_commands_after_step = _pin_command_to_device

    observations, env_info = env.reset()
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args.checkpoint_path)

    if terminal_keys is not None:
        print(
            f"[joystick] TERMINAL keys (keep this shell focused, NOT the sim window).\n"
            f"[joystick]   Up/Down = fwd/back, Left/Right = strafe, "
            f"{args.yaw_left_key.upper()}/{args.yaw_right_key.upper()} = turn, "
            f"Space or L = stop, Ctrl-C = quit.\n"
            f"[joystick]   Commands are sticky: each press steps by {args.key_step} and holds."
        )
    else:
        print("[joystick] driving. gamepad: left stick = move, right stick X = turn.")
        print(
            f"[joystick] keyboard: Up/Down = fwd/back, Left/Right = strafe, "
            f"{args.yaw_left_key.upper()}/{args.yaw_right_key.upper()} = turn left/right, "
            f"L = stop. Ctrl-C to quit."
        )
        print("[joystick] NOTE: click the Isaac Sim window first -- these keys only register there.")

    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
    step = 0
    try:
        while True:
            if terminal_keys is not None:
                held["command"] = terminal_keys.poll()
            else:
                raw = np.zeros(3, dtype=np.float32)
                for device in devices:
                    raw += np.asarray(device.advance().cpu(), dtype=np.float32)
                held["command"] = np.clip(raw, low, high).astype(np.float32)

            actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
            observations, _, _, _, _ = env.step(np.array(actions))
            prev_transition = {"next_observation": observations}

            if args.follow_camera:
                base_env._update_tracking_camera()

            step += 1
            if args.print_every > 0 and step % args.print_every == 0:
                c = held["command"]
                measured = base_env._robot.data.root_lin_vel_b[0]
                print(
                    f"[joystick] cmd vx={c[0]:+.2f} vy={c[1]:+.2f} wz={c[2]:+.2f} | "
                    f"measured vx={float(measured[0]):+.2f} vy={float(measured[1]):+.2f}"
                )
    except KeyboardInterrupt:
        print("\n[joystick] stopping.")
    finally:
        # Restore cooked mode before anything else, so a crash cannot leave the shell in cbreak.
        if terminal_keys is not None:
            terminal_keys.restore()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drive a trained isaaclab_go2 policy from a gamepad/keyboard")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to agent checkpoint directory")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", choices=("both", "gamepad", "keyboard"), default="both")
    parser.add_argument("--v_x_sensitivity", type=float, default=1.0)
    parser.add_argument("--v_y_sensitivity", type=float, default=1.0)
    parser.add_argument("--omega_z_sensitivity", type=float, default=1.0)
    parser.add_argument("--dead_zone", type=float, default=0.05, help="Gamepad stick dead zone")
    parser.add_argument("--yaw_left_key", type=str, default="q", help="Key that yaws left (+omega_z)")
    parser.add_argument("--yaw_right_key", type=str, default="w", help="Key that yaws right (-omega_z)")
    parser.add_argument(
        "--terminal_keys",
        action="store_true",
        help="Read keys from this terminal instead of the Isaac Sim window (no window focus needed)",
    )
    parser.add_argument("--key_step", type=float, default=0.25, help="Command increment per keypress in --terminal_keys")
    parser.add_argument("--follow_camera", action="store_true", help="Chase the robot with the viewport camera")
    parser.add_argument("--print_every", type=int, default=0, help="Print commanded vs measured velocity every N steps")
    args = parser.parse_args()
    play(args)
