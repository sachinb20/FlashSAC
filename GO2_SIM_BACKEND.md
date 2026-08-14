# Go2: one environment, swappable simulator

Branch: `feat/go2-sim-backend`, cut fresh from `main`. None of the earlier
`feat/isaaclab-go2-port` work is carried over.

## The idea

The previous attempt rebuilt the Go2 task natively in IsaacLab and then tried to recover
parity with the Genesis baseline flag by flag. This branch inverts that: **keep the whole
environment and swap only the physics engine.**

Everything the RL algorithm can observe or be rewarded by now lives in one
simulator-agnostic file and runs identically on both backends. Only the engine differs.

```
env=go2 env.sim_backend=genesis     # reference arm
env=go2 env.sim_backend=isaaclab    # same environment, PhysX underneath
```

The old `env=genesis env.env_name=go2-walk` path is **untouched** and stays the reference.

## What is shared vs. what differs

| | Where it lives | Same across backends? |
|---|---|---|
| Rewards (all 14 terms) | `go2_common/go2_env.py` | Yes — one code path |
| Observations, scaling, noise, clipping | `go2_common/go2_env.py` | Yes |
| Command sampling, deadband, resampling | `go2_common/go2_env.py` | Yes |
| Action pipeline (clip → rescale → scale → latency) | `go2_common/go2_env.py` + `go2_sim.py` | Yes |
| PD torque law | `go2_common/go2_env.py` | Yes |
| Termination | `go2_common/go2_env.py` | Yes |
| Domain-randomisation *sampling* | `go2_common/go2_env.py` | Yes |
| URDF, mass, inertia, collision geometry | `assets/go2_genesis/` | Yes — same file |
| DR *application*, state I/O, stepping | `go2_common/backends/*` | Engine-specific by necessity |
| Contact resolution, constraint solver, integration | the engine | **No — this is the residual** |

That last row is the point. Holding everything else byte-identical is what makes the
remaining difference attributable to contact physics.

## Files

**New**

- `flash_rl/envs/go2_common/go2_env.py` — `Go2BaseEnv` + `Go2WalkEnv` + `get_cfgs()`, the
  shared core. Ported from `genesis_envs/go2_base.py` + `go2_walk.py` with every physics
  call routed through the backend seam. Config values are unchanged.
- `flash_rl/envs/go2_common/sim_backend.py` — the `Go2SimBackend` protocol (~25 methods)
  and `make_backend()`. Documents the three conventions backends must honour: robot-local
  link indices, per-env local positions, and world-frame velocity returns.
- `flash_rl/envs/go2_common/math_utils.py` — the quaternion/Euler helpers, moved verbatim.
  They were always pure `torch`; the `gs_` prefix was naming, not a dependency.
- `flash_rl/envs/go2_common/go2_urdf.py` — URDF resolution, vendoring, and the fixed-link
  pre-merge (see below).
- `flash_rl/envs/go2_common/backends/genesis_backend.py` — Genesis implementation.
- `flash_rl/envs/go2_common/backends/isaaclab_backend.py` — IsaacLab/PhysX implementation.
- `flash_rl/envs/go2_sim.py` — `Go2VectorEnv`, the gymnasium wrapper. Same behaviour as
  `GenesisVectorEnv` but holds no simulator reference.
- `configs/env/go2.yaml` — the new env config.
- `scripts/vendor_go2_asset.py` — one-time asset vendoring + merge, with verification.

**Modified**

- `flash_rl/envs/__init__.py` — added the `env_type == "go2"` branch.
- `.gitignore` — ignore the generated `assets/go2_genesis/`.

**Untouched** — `flash_rl/envs/genesis_envs/**` and `flash_rl/envs/genesis.py`. The
existing baseline still runs exactly as before.

## Three things that needed real work

### 1. The URDF merge (the one that would have silently broken parity)

Genesis loads `go2.urdf` with `merge_fixed_links=True, links_to_keep=[the 4 feet]`,
collapsing 29 links to 17: `base` absorbs `Head_upper`/`Head_lower`/`imu`/`radar`, and
each `*_calf` absorbs `*_calflower`/`*_calflower1`.

This matters because the environment selects contact bodies by **substring**, and
`"calf"` also matches `"calflower"`. On an unmerged asset the `collision` penalty would
watch 12 bodies instead of 4. IsaacLab's importer only offers all-or-nothing
`merge_fixed_joints`, so *neither* of its settings reproduces Genesis's behaviour — one
loses the feet, the other changes the contact sets.

So `go2_urdf.py` performs the merge itself (composing mass, inertia via parallel-axis,
and geometry origins), and both engines load that pre-merged file with their own merging
disabled. Topology is then identical by construction.

Verified against a live Genesis build:

```
ref n_links 17 | merged n_links 17
names identical: True
masses  maxdiff: 8.88e-16
inertia maxdiff: 0.0
com     maxdiff: 0.0
n_geoms: 27 vs 27
```

Helpful discovery along the way: **collision geometry is entirely primitives** (5 boxes,
17 cylinders, 5 spheres) — no collision meshes. The 25 MB of `.dae` files are visual only,
so contact geometry transfers to PhysX exactly rather than through mesh approximation.

### 2. Index and frame translation

Genesis numbers links globally across entities (ground plane at 0, robot base at 1) and
simulates `n_envs` overlapping worlds sharing one origin. IsaacLab uses per-articulation
body indices and lays environments out on a grid. The core sees only robot-local indices
(base = 0) and per-env local positions; each backend translates. Without the origin
handling, the reset's ±1.0 m xy spread would walk robots into neighbouring tiles.

### 3. The two simulators cannot share a virtualenv

`pyproject.toml`'s `[tool.uv] conflicts` declares the `isaaclab` and `genesis` extras
mutually exclusive (different torch pins). Every engine import is therefore deferred —
`make_backend()` imports a backend module only after one has been selected, and nothing in
`go2_common/` imports an engine at module scope. It also means the vendored asset must
live in the repo: the IsaacLab venv cannot import `genesis` to find it.

## Bring-up log: every problem found running the IsaacLab arm

Five real engine differences, found by running it. None were typos — each is something
Genesis does implicitly that PhysX does not, which is exactly the class of bug this design
was meant to surface rather than hide.

### 1. Spawn pose is validated on IsaacLab, only warned about on Genesis

```
ValueError: The following joints have default positions out of the limits:
        - 'FL_calf_joint': 0.000 not in [-2.723, -0.838]
```

`ArticulationCfg.InitialStateCfg` had a base pose but no `joint_pos`, so IsaacLab used the
URDF's implicit all-zeros — illegal for the calf joints. Genesis spawns at the same zero
pose, prints `Reference robot position exceeds joint limits`, and carries on because
`reset_idx` overwrites it a moment later.

**Fix:** `default_joint_angles` now plumbs from `get_cfgs()` into the backend and seeds
`InitialStateCfg.joint_pos`. Verified all 12 values are in range.

### 2. Contact reporting is opt-in on PhysX

```
RuntimeError: Sensor at path '/World/envs/env_.*/Robot/.*' could not find any bodies
with contact reporter API.
```

**Fix:** `activate_contact_sensors=True` on the spawn cfg. Genesis reports net contact
force for every link unconditionally, so nothing in the shared env hinted this was needed
-- and it is load-bearing twice over: termination reads base contact, and `feet_air_time`
reads foot contact.

### 3. Sensors must exist before the reset that initialises them

```
AttributeError: 'Camera' object has no attribute '_ALL_INDICES'
```

`_ALL_INDICES` is populated by the sensor-initialisation callback fired during
`sim.reset()`; the camera was being created after it. **Fix:** create it before.

### 4. PhysX solver properties were left at their defaults

Unset `rigid_props`/`articulation_props` meant an effectively unbounded depenetration
velocity, so any contact penetration became a violent pop. **Fix**, matching IsaacLab's own
Unitree/ANYmal configs:

```python
max_depenetration_velocity=1.0
solver_position_iteration_count=4, solver_velocity_iteration_count=0
enabled_self_collisions=True   # matches Genesis; IsaacLab's quadruped default is False
```

### 5. ⭐ Armature: Genesis defaults to 0.1, PhysX to 0

The one that actually mattered, and the hardest to see. With the first four fixed the robot
still would not stand -- it rang at +/-10 rad/s with torques of 100-146 N.m, far above the
motor limits, **even in free fall with no contact at all**:

```
                    Genesis      IsaacLab (before)
mean|q-q*|           0.004          0.396
mean|tau|            0.04         104
height after 0.5s    0.764  (correct free fall)   1.860  (barely fell)
```

The free-fall test is what cracked it: with zero contacts the problem persisted, so it was
never contacts. Querying both engines directly:

```
genesis  get_dofs_armature: [0.1, 0.1, 0.1]
isaac    sim_joint_armature: 0.0
```

The URDF declares **no `<dynamics>` tags at all** -- this is purely Genesis's solver
default, a number the original environment never had to name. It is load-bearing: the calf
link's inertia about the knee is only ~0.003 kg.m^2, so armature 0.1 raises the effective
inertia ~34x, which decides whether the shared explicit PD is numerically stable at this
timestep:

| | effective inertia | `Kd*dt/I` | stable? |
|---|---|---|---|
| PhysX, armature 0 | 0.003 | **2.5** | no (limit is 2) |
| Genesis, armature 0.1 | 0.103 | 0.07 | yes |

**Fix:** promoted to an explicit `env_cfg["dof_armature"] = 0.1`, written by *both*
backends -- a no-op on Genesis, decisive on IsaacLab -- so the two can never silently
disagree on it again.

### Result

| after 120 zero-action steps | Genesis | IsaacLab |
|---|---|---|
| `mean\|q-q*\|` | 0.0779 | 0.0712 |
| `max\|q-q*\|` | 0.2021 | 0.1918 |
| `mean\|tau\|` | 2.346 | 2.147 |
| base height | 0.3006 | 0.3080 |
| contact force | 150.31 | 148.47 |

Step 0 is bit-identical on both (`0.1581 / 3.928 / 0.4177`), so the reset and the first
torque agree exactly; the small residual accumulates purely from contact and solver
behaviour, which is the quantity we set out to isolate.

### Also fixed along the way

`flash_rl/types.py` imported `jax.numpy` at module scope purely to widen a type alias.
Nothing on the torch path uses JAX, and installing it into an IsaacSim environment drags
numpy 2.x in with it. The import is now optional, which is what lets the port run in an
existing IsaacSim conda env with no package installs.

## Verification

Run on the Genesis backend (the arm that can be executed in this venv):

- **Structural parity** vs. the original env — `motor_dofs`, all three contact-link index
  sets, `n_links`, `dof_pos_limits`, `torque_limits`, `default_dof_pos`, `p_gains`,
  `d_gains`: **all MATCH**.
- **Trajectory equivalence** — 40 steps, 8 envs, eval mode (DR off so both consume RNG
  identically), same seeds and actions: observations and rewards **max|diff| = 0.000e+00**.
  The refactor is bit-for-bit behaviour-preserving.
- **Training smoke test** — `train.py` with `env=go2 env.sim_backend=genesis` ran 128
  interaction steps and checkpointed cleanly (exit 0).

The IsaacLab arm has now been run end to end (see the bring-up log above) in
`.venv-isaaclab` against isaacsim 5.1.0.0 on an RTX 4090. It reproduces Genesis's link
topology, kinematics and standing behaviour. What has **not** been run on IsaacLab yet is
a full training job -- only the smoke test.

## Bring-up checklist for the IsaacLab arm

1. From the **Genesis** venv, vendor the asset (needs `genesis` importable):
   `uv run --extra genesis python scripts/vendor_go2_asset.py`
   Should print 17 links / 15.019 kg and `OK: topology matches the Genesis reference.`
2. Switch to the IsaacLab venv and confirm `assets/go2_genesis/urdf/go2_merged.urdf` exists.
3. Run a tiny job first (`num_train_envs=16`, `num_env_steps=8192`) and check:
   - `resolve_link_indices` gives 4 feet and 9 penalised bodies (base + 4 thigh + 4 calf),
     matching the Genesis values `[13,14,15,16]` and `[0,5,6,7,8,9,10,11,12]`. Body
     *order* may differ; the *sets* must not.
   - the robot settles at roughly the 0.3 m base-height target rather than sinking or
     launching (a sign the torque path or effort limit is wrong).
4. Known soft spots, in the order I'd suspect them:
   - `UrdfFileCfg` converter API — `_build_urdf_spawn_cfg` covers IsaacLab 2.1 and 2.3
     shapes and raises a legible error otherwise.
   - `root_link_lin_vel_w` vs `root_com_lin_vel_w` — Genesis's `get_vel()` is the base
     *link* velocity; the backend prefers the link variant where available.
   - Ground-plane friction. Genesis's `plane.urdf` and IsaacLab's `GroundPlaneCfg` do not
     necessarily start from the same coefficients, and the friction DR is *multiplicative*
     on both sides, so a different base value shifts the whole distribution.
   - The chase camera (`env.enable_camera=true`). It is wrapped in a try/except that
     downgrades failure to a warning and disables video, so a camera problem costs you the
     recording rather than the run — but then `num_record_episodes` must be `0`, since
     `record_video()` cannot stack `None` frames.

## Training commands

Matched pair for the `go2-walk-compare` wandb group. Everything outside the environment
block is character-for-character identical between the two.

**Genesis (reference):**

```bash
uv run python train.py \
    --config_name flashSAC_base --overrides seed=0 \
    --overrides group_name=go2-walk-compare --overrides exp_name=genesis \
    --overrides logger_type=wandb --overrides entity_name=null \
    --overrides evaluation_per_interaction_step=4882 \
    --overrides metrics_per_interaction_step=4882 \
    --overrides recording_per_interaction_step=4882 \
    --overrides logging_per_interaction_step=488 \
    --overrides env=go2 \
    --overrides env.sim_backend=genesis \
    --overrides env.env_name=go2-walk \
    --overrides num_env_steps=50_000_896 --overrides num_train_envs=1024 \
    --overrides num_eval_envs=null --overrides num_record_envs=null \
    --overrides num_eval_episodes=1024 --overrides num_record_episodes=1 \
    --overrides agent=flashSAC \
    --overrides agent.buffer_max_length=10_000_000 \
    --overrides agent.buffer_min_length=100_000 \
    --overrides agent.buffer_device_type=cuda \
    --overrides agent.sample_batch_size=2048 --overrides agent.use_amp=true \
    --overrides updates_per_interaction_step=2 \
    --overrides agent.asymmetric_observation=true \
    --overrides gamma=0.95 --overrides n_step=1
```

**IsaacLab:**

```bash
uv run --extra isaaclab python train.py \
    --config_name flashSAC_base --overrides seed=0 \
    --overrides group_name=go2-walk-compare --overrides exp_name=isaaclab \
    --overrides logger_type=wandb --overrides entity_name=null \
    --overrides evaluation_per_interaction_step=4882 \
    --overrides metrics_per_interaction_step=4882 \
    --overrides recording_per_interaction_step=4882 \
    --overrides logging_per_interaction_step=488 \
    --overrides env=go2 \
    --overrides env.sim_backend=isaaclab \
    --overrides env.enable_camera=true \
    --overrides env.env_name=go2-walk \
    --overrides num_env_steps=50_000_896 --overrides num_train_envs=1024 \
    --overrides num_eval_envs=null --overrides num_record_envs=null \
    --overrides num_eval_episodes=1024 --overrides num_record_episodes=1 \
    --overrides agent=flashSAC \
    --overrides agent.buffer_max_length=10_000_000 \
    --overrides agent.buffer_min_length=100_000 \
    --overrides agent.buffer_device_type=cuda \
    --overrides agent.sample_batch_size=2048 --overrides agent.use_amp=true \
    --overrides updates_per_interaction_step=2 \
    --overrides agent.asymmetric_observation=true \
    --overrides gamma=0.95 --overrides n_step=1
```

### The complete list of differences

| | Genesis | IsaacLab | Why |
|---|---|---|---|
| `env.sim_backend` | `genesis` | `isaaclab` | the point of the exercise |
| `exp_name` | `genesis` | `isaaclab` | wandb run label |
| `env.enable_camera` | *(unset)* | `true` | Genesis always attaches its camera; on IsaacLab the renderer costs throughput so it is opt-in. **Required** whenever `num_record_episodes > 0` — `record_video()` `np.stack`s the frames and will crash on `None`. |
| virtualenv | default | `--extra isaaclab` | the two engines cannot share one |

Nothing else changes. Every environment-side flag the previous branch needed —
`isaac_action_scale`, `isaac_go2_pd_stiffness/damping`, `clip_joint_targets`,
`action_latency_steps`, `genesis_style_*`, the entire DR block — is **gone**, because
those values are no longer duplicated per backend. There is one `action_scale = 0.25`,
one `PD_stiffness = 30.0`, one `PD_damping = 1.5` in `get_cfgs()`, read by both arms.

### If your reference run used `env=genesis`

The original `env=genesis env.env_name=go2-walk` path still works and is still the
baseline. It is a valid comparison arm: it was verified bit-identical to
`env=go2 env.sim_backend=genesis` (obs and reward `max|diff| = 0.000e+00` over 40 steps).
Routing both arms through `env=go2` is nonetheless preferable, because then the two runs
share one env implementation rather than two implementations proven equal at one point in
time.

### Video comparability

Both backends record env 0 at **320x320**, 40 degree FOV, from `robot_pos + [-1, -1, 0.5]`
looking at `robot_pos + [0, 0, -0.1]` — the Genesis framing, reproduced on IsaacLab by
deriving the focal length from the FOV (`f = 20.955 / (2 tan 20 deg) = 28.79 mm`). The two
videos are meant to be watchable side by side.
