# Genesis-parity work log

Everything done in one session to make `env=isaaclab_go2` a fair comparison arm against the
vanilla FlashSAC baseline (`env=genesis`, `env.env_name=go2-walk`). The goal is that the only
remaining differences are the ones the port exists to test.

Every flag added here **defaults to the port's previous behaviour**, so existing runs and
checkpoints are unaffected unless a flag is passed explicitly.

Branch: `feat/isaaclab-go2-port`. Baseline before this work: `d91eeb4`.

---

## Commits

### `d9c4041` — Drop `rew_` prefix from genesis episode reward keys

`genesis.py`'s `step()` turns `extras["episode"]` keys into `Reward/<key>` metrics. The genesis
envs prefixed each term with `rew_` while the IsaacLab env did not, so the two simulators logged
non-overlapping wandb series (`Reward/rew_tracking_lin_vel` vs `Reward/tracking_lin_vel`) and
their panels could not be overlaid.

Touches all four genesis envs (`go2_base`, `go2_walk_easy`, `go2_backflip`, `panda_grasp`).

### `93aca74` — PD gains, Kp/Kd randomization, privileged observation width

| Flag | Default | Genesis value |
|---|---|---|
| `env.isaac_go2_pd_stiffness` | `null` (keeps asset's 25.0) | `30.0` |
| `env.isaac_go2_pd_damping` | `null` (keeps asset's 0.5) | `1.5` |
| `env.isaac_dr_kp_scale_range` | `[1.0, 1.0]` (off) | `[0.8, 1.2]` |
| `env.isaac_dr_kd_scale_range` | `[1.0, 1.0]` (off) | `[0.8, 1.2]` |
| `env.isaac_direct_velocity_privileged_last_actions` | `false` | `true` |

**Key correction:** selecting `isaac_go2_actuator_model=unitree_go2hv` does **not** change the PD
gains. `_make_unitree_go2hv_actuator_cfg` copies `stiffness`/`damping` straight off the source
config, so both actuator models sit at `UNITREE_GO2_CFG`'s Kp=25 / Kd=0.5. The swap only replaces
the torque-speed envelope.

**Privileged obs:** genesis's `privileged_obs_buf` is 60 columns —
`[obs 45 | base_lin_vel 3 | last_actions 12]`. The port's `privileged_base_lin_vel` alone reached
only 48, so the critic was missing the 12 `last_actions` columns even in the "matched" config.

New file: `go2_pd_gain_randomization.py`.

### `7b86e4b` — Observation scaling, action latency, unclipped PD actuator

| Flag | Default | Genesis value |
|---|---|---|
| `env.isaac_direct_velocity_genesis_style_obs_scaling_enabled` | `false` | `true` |
| `env.isaac_direct_velocity_action_latency_steps` | `0` | `1` |
| `env.isaac_go2_actuator_model` | `dc_motor` | `ideal_pd` |

**Observation scaling** applies genesis's `obs_scales` (lin_vel 2.0, ang_vel 0.25, dof_pos 1.0,
dof_vel 0.05) *together with* genesis's post-scale noise magnitudes, since genesis's noise values
are defined against the already-scaled signal. Scaling is applied before the noise, matching
genesis's ordering; with all scales at 1.0 the two orderings are identical, so unscaled runs are
untouched.

**Action latency** executes the previous control step's action while the policy still observes the
one it just emitted — genesis's
`exec_actions = self.last_actions if self.action_latency > 0 else self.actions`.

**`ideal_pd`** is a new actuator model: explicit PD with `_clip_effort` overridden to the identity,
i.e. no torque ceiling at all. This matches genesis, which hands raw kp/kd torques to
`control_dofs_force` and never consults a limit (`go2_base.py` reads `torque_limits` once and never
uses it). PhysX does not re-clamp — IsaacLab defaults `effort_limit_sim` to 1e9 for explicit
actuators. Verified: 33.09 N·m applied against a 23.5 N·m `effort_limit`, `applied == computed`.

### `9a80e98` — Nominal stance, reset randomization, command deadband, motor-offset DR

| Flag | Default | Genesis value |
|---|---|---|
| `env.isaac_go2_genesis_style_nominal_pose` | `false` | `true` |
| `env.isaac_direct_velocity_base_height_target` | `null` (follows stance) | `0.3` |
| `env.isaac_direct_velocity_genesis_style_reset_enabled` | `false` | `true` |
| `env.isaac_velocity_command_deadband_lin_vel` | `0.0` (off) | `0.2` |
| `env.isaac_velocity_command_deadband_ang_vel` | `0.0` (off) | `0.2` |
| `env.isaac_dr_motor_offset_range` | `[0.0, 0.0]` (off) | `[-0.02, 0.02]` |

**Nominal stance** — the most consequential of the four. Genesis stands with the front thighs at
0.8 and the **rear** thighs at 1.0, calves at −1.5, spawning at z=0.42. The port used TDMPC2's
PyMPC stance, which is fore/aft *symmetric* (thigh 0.9 front and rear), calves 0.3 rad more folded,
spawning 13 cm lower. Because the decoder is `target = default_joint_pos + action_scale × action`,
the nominal pose is the policy's operating point — a symmetric stance removes genesis's rear-loaded
forward bias.

**Reset randomization** — genesis offsets joints *additively* by U(−0.3, 0.3) rad off the default;
the port's stock event multiplies by U(1.0, 1.0), i.e. **no joint randomization at all**. Also
widens base xy spread to ±1.0 and adds the U(−0.1, 0.1) roll/pitch tilt genesis applies.

**Command deadband** — genesis zeroes a fresh command when ‖(vx,vy)‖ or |ωz| is under 0.2,
independently per axis. At genesis's ranges that is ~20% of yaw commands and ~3% of planar
commands, so a real share of its training distribution is "hold this axis still" — distinct from
`rel_standing_envs`, which zeroes all three axes together.

**Motor-offset DR** — genesis randomizes a per-joint position bias and leaves motor strength fixed;
the port did the opposite. Applied to the joint target in `_apply_action`, algebraically identical
to genesis folding it into the PD position error inside `_compute_torques`.

New file: `go2_motor_offset_randomization.py`. Also adds `play_joystick.py` (below).

### `HEAD` — Rotor-link mass parity, batch-shared observation noise, observation clipping

| Flag | Default | Genesis value |
|---|---|---|
| `env.isaac_go2_strip_rotor_links` | `false` | `true` |
| `env.isaac_direct_velocity_obs_noise_shared_across_envs` | `false` | `true` |
| `env.isaac_direct_velocity_obs_clip` | `null` | `100.0` |

**Rotor links — a 7% robot mass difference that had gone unnoticed.** A link-by-link diff of the
two URDFs showed every shared link's mass, COM and inertia tensor is *identical*. But TDMPC2's
`go2_description.urdf` has 33 links to genesis's 21, including twelve 0.089 kg `*_rotor` links —
one per actuated joint — that genesis's file does not model at all. Stripping them reproduces
genesis's 15.0190 kg exactly.

They are **not** all on the base: each rotor is fixed to the link that *carries* its motor, so
after PhysX merges fixed-joint children into their parent the extra mass lands as follows.

| Rigid body | Genesis | Port | Delta | Rotors merged in |
|---|---|---|---|---|
| `base` | 6.921 kg | 7.277 kg | **+5.1%** | 4 × hip rotor |
| `*_hip` (×4) | 0.678 kg | 0.767 kg | **+13.1%** | 1 × thigh rotor |
| `*_thigh` (×4) | 1.152 kg | 1.241 kg | **+7.7%** | 1 × calf rotor |
| `*_calf` (×4) | 0.154 kg | 0.154 kg | — | none |
| **Whole robot** | **15.019 kg** | **16.087 kg** | **+7.1%** | 12 total |

Per leg that is 2.024 kg → 2.202 kg (+8.8%), and the added mass sits high on the limb (hip and
thigh, never the calf), so it raises swing-leg inertia as well as total weight.

**Batch-shared observation noise.** Genesis draws
`gs_rand_float(-1.0, 1.0, (num_single_obs,))` — a `(45,)` tensor added to a `(num_envs, 45)`
buffer, so it broadcasts and **every env sees the identical perturbation each step**. The port drew
independently per env. Same marginal distribution, completely different correlation structure:
genesis's noise is a single shared perturbation the batch cannot average out, while i.i.d. noise
largely cancels across 1024 envs. This looks like a genesis bug, but it is what the baseline
trained under.

**Observation clipping** to ±100 on both the actor and privileged observation, as genesis does.

---

## Tooling added

**`play_joystick.py`** — drive a trained policy live from a gamepad, the Isaac Sim window's
keyboard, or the terminal. Pins the env's velocity command to the input device by replacing
`_maybe_update_commands_after_step` (the hook the observation build itself calls), so the pin
survives resets that would otherwise resample the command. Commands are clamped to the training
ranges to stay in-distribution.

- Gamepad: left stick = move, right stick X = turn.
- Sim-window keyboard: arrows = move, Q/W = turn, L = stop.
- `--terminal_keys`: same keys read from the shell instead, for when the sim window does not have
  focus. Presses are sticky (the terminal reports no key releases), stepping by `--key_step`.

---

## Verified without a full training run

- **Deadband** is bit-identical to genesis's formula over 400k samples and reproduces its analytic
  zeroing rates (3.13% planar vs π·0.04/4 = 3.14%; 19.92% yaw vs 20%).
- **Rotor strip** yields exactly 15.0190 kg, matching genesis.
- **`ideal_pd`** applies 33.09 N·m against a 23.5 N·m limit with `applied == computed`.
- **Obs scaling** — with noise off, every channel equals `raw × scale` exactly, including the
  privileged `base_lin_vel` tail.
- **Action latency** — `step(+0.4)` leaves the target at the default; the next `step(-0.4)` drives
  it to +0.3 = 0.4 × 0.75, i.e. the previous action.
- **Kp/Kd DR** — per-joint gains land inside 30·[0.8,1.2] and 1.5·[0.8,1.2].
- **Defaults regression** — with no flags passed the env still reports 225/225 obs, Kp=25, Kd=0.5,
  no gain randomization, no action delay, all scales 1.0.
- **Shared noise / clipping** — batch-shared draw is identical across envs and varies per channel;
  clipping saturates at ±100 and is inert when `null`.

Confirmed already matching, so no change was needed: control timing (both 50 Hz control over
200 Hz physics, decimation 4), termination (base contact >1.0 N, |roll|/|pitch| > 0.4, height floor
0.0), all 13 reward scales and the `tracking_lin_vel` / `feet_air_time` implementations, and random
pushes (genesis sets `push_interval_s: -1`, i.e. disabled).

**Not verified:** no flag has been exercised in a full-length training run, and the stance, reset,
rotor-strip, noise and clip flags have only been checked through hydra composition and standalone
unit tests — not against a live sim.

---

## What is still different

### Fundamental — the point of the experiment

- **Physics engine.** Genesis vs Isaac Sim / PhysX: different contact models, solvers, integrators.
- **Collision geometry.** The port's URDF converts with `replace_cylinders_with_capsules=True` and
  carries visual/collision meshes that differ from genesis's; masses now match but contact shapes
  are not guaranteed to.

### Real, no flag yet

- **`dof_acc`** — genesis finite-differences `(last_dof_vel − dof_vel)/dt`; the port uses IsaacLab's
  native `joint_acc`. Feeds a −2.5e-7 penalty, so numerically small.
- **Contact force sampling** — the port takes `amax` over a 3-step contact history; genesis reads
  instantaneous forces. Makes the port's contact termination and collision penalty marginally more
  trigger-happy.
- **Joint ordering** — per-leg (genesis) vs per-joint-type (port). Harmless for training since it is
  a consistent relabeling, but checkpoints are not cross-loadable between the two arms.
