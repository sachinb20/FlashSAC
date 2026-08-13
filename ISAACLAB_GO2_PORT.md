# Go2 IsaacLab Port

Ports the Unitree Go2 sim, robot/URDF, and actuator setup from `TDMPC2_isaaclab` (a separate,
standalone repo used only as a porting reference -- not a FlashSAC dependency) into FlashSAC as
a new environment, trained with FlashSAC's own SAC implementation. `TDMPC2_isaaclab`'s planner,
world model, AMP/SMP auxiliary features, and reward shaping are **not** part of this port.

New env: `env=isaaclab_go2`, `env.env_name=go2-vel-direct`.

## What was ported from `TDMPC2_isaaclab`

All under `flash_rl/envs/isaaclab_go2_envs/`, kept close to the source structure/logic:

| File | What it is |
|---|---|
| `go2_urdf_asset.py` | Asset source selection (stock USD vs. Unitree URDF), URDF staging, joint-name/order contract |
| `go2_actuator_math.py`, `isaaclab_unitree_actuators.py` | Go2HV nonlinear torque-speed envelope actuator, plus the plain DC-motor option |
| `go2_terrain_cfg.py` | Flat + rough terrain presets, height-scan geometry, distance curriculum |
| `go2_motor_strength_randomization.py` | Per-joint motor-strength domain randomization |
| `go2_base_ang_vel_filter.py` | Base angular-velocity EMA filter |
| `go2_command_sampling.py` | Velocity command sampling (forward/lateral/yaw, standing, yaw-in-place, heading modes) |
| `go2_termination.py` | Bad-orientation (projected-gravity) termination check with hysteresis |
| `go2_action_decoder.py` | Raw `[-1,1]` action -> joint-position-target decoding (scalar / default-centered / SAC-affine modes) |
| `isaaclab_go2_velocity_direct.py` | The `DirectRLEnv` subclass itself -- scene setup, action pipeline, observation composition, domain randomization, reset flow |
| `go2_env_builder.py` | Translates `isaac_*` hydra flags into the env cfg (new file, but a direct port of `TDMPC2_isaaclab`'s `_make_velocity_direct_env` injection logic) |
| `assets/go2_description/` | The URDF + mesh files themselves |

Dropped from the source env: AMP/SMP auxiliary observations, gait/foot-lift/support-plane reward
diagnostics (existed only to feed TDMPC2's reward), TDMPC2-wrapper-only hooks (checkpoint
fingerprinting, debug snapshots).

## What's FlashSAC-original, not from TDMPC2

**The reward.** `go2_flashsac_rewards.py` is ported from FlashSAC's own
`flash_rl/envs/genesis_envs/go2_walk.py` (14 terms: tracking lin/ang vel, lin_vel_z, ang_vel_xy,
orientation, torques, dof_vel/acc, action_rate, base_height, collision, dof_pos_limits,
termination), re-expressed against IsaacLab's `Articulation`/`ContactSensor` state instead of
Genesis's. TDMPC2's own 23-term sim-to-real reward (`go2_reward_terms.py` in `TDMPC2_isaaclab`)
was **not** ported at all.

## Built new this round (not from either source)

- `flash_rl/envs/isaaclab_go2.py` -- the FlashSAC-facing `VectorEnv` wrapper. Modeled on
  `flash_rl/envs/genesis.py`'s pattern (not the generic `flash_rl/envs/isaaclab.py`, which can't
  be reused here: it depends on `gym.make()` task registration this env doesn't have, and has two
  bugs this port avoids -- `final_obs` is the *post-reset* observation there, and `render()` is
  unimplemented).
- **True `final_obs`/episode-length/episode-reward snapshotting.** IsaacLab's `DirectRLEnv.step()`
  only computes observations *after* `_reset_idx()` runs, so the raw obs/episode_length_buf a done
  env returns are already post-reset. `_snapshot_before_reset()` in the env class captures the
  true terminal values first, without corrupting the observation-history ring buffer or the ang-vel
  EMA filter for envs that aren't resetting that step.
- **Privileged/asymmetric observation support** (`privileged_base_lin_vel` cfg flag) -- appends true
  (unnoised) `base_lin_vel_b` as a critic-only tail after the actor's own columns. Off by default.
- **Genesis-parity termination** (`genesis_style_termination_enabled` cfg flag) -- an additional
  termination check replicating `go2_base.py`'s `check_termination()` exactly (`|roll|`/`|pitch|`
  Euler-angle thresholds + a base-height floor), independent of the ported bad-orientation check.
- **Tracking-camera rendering.** `render_mode="rgb_array"` + a camera that follows the centroid of
  all envs' robots each frame (`_update_tracking_camera`), reusing IsaacLab's built-in RGB
  annotator (`DirectRLEnv.render()`) rather than anything custom.
- `play.py` -- live-viewer playback for a trained checkpoint, dispatched by `env_type`
  (`genesis` / `isaaclab_go2` / `isaaclab`).
- `eval_directions.py` -- records a checkpoint stepping through fixed forward/backward/left/right/yaw
  velocity commands, fresh env reset per direction, MP4 output at the sim's real control rate.
- `show_viewer` threaded through `genesis.py` -> `genesis_envs/__init__.py` -> `go2_walk.py` (was
  hardcoded off) so genesis's own live viewer can be requested; only the `go2-walk` path touched.

## Matching vanilla FlashSAC (Genesis `go2-walk`) for a fair comparison

The first working version used each simulator's own natural defaults. Getting a comparison that
isolates the sim/robot swap (rather than also comparing incidental settings) took the following,
verified via a real hydra-compose + live env test at each step, not just read from config:

| Setting | Before | Now (matches vanilla) |
|---|---|---|
| `num_train_envs` | 256 | 1024 |
| `num_env_steps` | 5,000,000 | 50,000,896 |
| `agent.buffer_max_length` / `buffer_min_length` | 2M / 50K | 10M / 100K |
| `gamma` | 0.99 | 0.95 |
| `n_step` | 3 | 1 |
| `agent.asymmetric_observation` | false | true (needed `privileged_base_lin_vel` built first) |
| Observation width | 225 (4-frame history stack) | 45 (`observation_history_enabled=false`) |
| Termination | disabled | enabled, `genesis_style_termination_enabled=true` |

**Action scale turned out to already match** (0.75 effective joint-delta-per-unit-action in both --
genesis's `action_range=3.0 x action_scale=0.25` and this port's direct `action_scale=0.75` reduce
to the identical formula `target = default_joint_pos + 0.75 x policy_output`); an earlier claim in
this doc's own history that these differed was a miscalculation, corrected once the full genesis
pipeline was traced.

## What's still different (the actual thing under test, or not fixable via a flag)

| | Vanilla (genesis) | This port |
|---|---|---|
| Physics engine | Genesis | Isaac Sim / PhysX |
| Robot asset | genesis-world's bundled `go2.urdf` | TDMPC2's `go2_description.urdf` |
| Actuator / PD gains | Linear PD, `Kp=30, Kd=1.5`, **randomized +/-20% per episode** | Go2HV nonlinear torque-speed envelope, `Kp=25, Kd=0.5`, **fixed** |
| Action-execution delay | 1 control step (20ms) | none |
| Observation scaling | fixed per-channel constants (`ang_vel x0.25`, `dof_vel x0.05`, ...) | raw physical units, unscaled |
| Domain-randomization surface | friction + mass + com + motor_offset + **kp_scale + kd_scale** | friction + mass + com + **motor_strength** |
| Joint action ordering | grouped per-leg | grouped per-joint-type (hip/thigh/calf) |

None of these have a config flag that makes them equivalent -- the actuator/PD row is the actual
subject of the port, and the rest would need new code, not a different override.

## Known checkpoints (this machine)

| Run | Path | Config |
|---|---|---|
| Vanilla baseline | `models/test/test/go2-walk/seed0-0812-022306/step48829` | Genesis `go2-walk`, full 1024-env/50M-step run |
| v1 | `models/isaaclab_go2_test/v1/go2-vel-direct/seed0-0812-041000/step19531` | isaaclab_go2, unmatched (256 envs/5M steps, symmetric obs) |
| v2 | `models/isaaclab_go2_test/v2/go2-vel-direct/seed0-0812-042709/step195312` | isaaclab_go2, unmatched, 10x longer than v1 |
| v3 | `models/isaaclab_go2_test/v3/go2-vel-direct/seed0-0812-163347/step48829` | isaaclab_go2, **fully matched** to vanilla (see table above) |

Checkpoints/tensorboard runs/videos are gitignored (`models/**`, `runs/**`, `videos/**`) -- this
table is the pointer to what exists on disk, not something git tracks.

## Commands

Train (v3-equivalent, matched comparison):
```bash
/home/sachin/miniconda3/envs/isaaclab/bin/python train.py \
    --overrides env=isaaclab_go2 \
    --overrides env.isaac_go2_actuator_model=unitree_go2hv \
    --overrides env.isaac_go2_asset_source=unitree_urdf \
    --overrides env.isaac_action_scale=0.75 \
    --overrides env.isaac_direct_velocity_clip_joint_targets=false \
    --overrides env.isaac_dr_enabled=true \
    --overrides env.isaac_dr_train_only=true \
    --overrides env.isaac_dr_motor_strength_per_joint=true \
    --overrides env.isaac_direct_velocity_observe_base_lin_vel=false \
    --overrides env.isaac_direct_velocity_enable_termination=true \
    --overrides env.isaac_direct_velocity_genesis_style_termination_enabled=true \
    --overrides env.isaac_direct_velocity_base_ang_vel_filter_alpha=0.2 \
    --overrides env.isaac_direct_velocity_privileged_base_lin_vel=true \
    --overrides env.isaac_direct_velocity_observation_history_enabled=false \
    --overrides agent=flashSAC --overrides agent.asymmetric_observation=true \
    --overrides gamma=0.95 --overrides n_step=1 \
    --overrides num_train_envs=1024 --overrides num_env_steps=50_000_896 \
    --overrides num_eval_envs=null --overrides num_record_envs=null \
    --overrides num_eval_episodes=1024 --overrides num_record_episodes=0 \
    --overrides agent.buffer_max_length=10_000_000 --overrides agent.buffer_min_length=100_000 \
    --overrides agent.sample_batch_size=2048 --overrides agent.buffer_device_type=cuda \
    --overrides updates_per_interaction_step=2 \
    --overrides seed=0 --overrides group_name=<your_group> --overrides exp_name=<your_exp> \
    --overrides logger_type=tensorboard
```

Play (live viewer) / record (fixed-direction MP4s) -- pass the **same `env.*`/`agent.*` overrides
used for training** so the network shape matches the checkpoint:
```bash
/home/sachin/miniconda3/envs/isaaclab/bin/python play.py \
    --overrides env=isaaclab_go2 --overrides <same env.*/agent.* overrides as training> \
    --checkpoint_path <path>/step<N> --num_envs 4 --num_episodes 5

/home/sachin/miniconda3/envs/isaaclab/bin/python eval_directions.py \
    --overrides env=isaaclab_go2 --overrides <same env.*/agent.* overrides as training> \
    --checkpoint_path <path>/step<N> --num_envs 4 --duration_s 7.0 --out_dir videos/<name>
```

Genesis (`env=genesis`) commands run under `uv run` (FlashSAC's own `.venv` -- genesis-world lives
there). Everything `isaaclab_go2` runs under the conda `isaaclab` environment
(`/home/sachin/miniconda3/envs/isaaclab`), which has isaacsim/isaaclab installed; FlashSAC's own
`.venv` does not, by design (`isaaclab` and `genesis` extras conflict in the same environment).
