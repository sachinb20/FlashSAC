# Go2 day 2: porting the TDMPC2 environment, one rung at a time

Day 1 (`GO2_SIM_BACKEND.md`) swapped the *physics engine* under a fixed environment:
Genesis's `go2-walk` task running on PhysX, verified to train as well as the Genesis
reference. Day 2 starts moving the *environment itself* toward TDMPC2's, so that
FlashSAC-vs-TDMPC2 becomes an algorithm comparison rather than an environment comparison.

**Source of truth is the `TDMPC2_isaaclab/` checkout, never the old
`feat/isaaclab-go2-port` branch.** That branch was itself derived from TDMPC2 and carried
its own drift — ~30 `genesis_style_*` reconciliation flags, and a parity effort where each
added flag closed a verified gap while training got *worse* (12.84 with the most parity vs
25.04 with the least). It is useful only as a list of pitfalls.

**Method.** One axis per rung, behind a config flag defaulting to the existing behaviour,
with a training run at each. The target is the user's real training command
(`exp_name=he_h1_s075_gait_r5m_b256_t01425_u16_s2`), *not*
`UnitreeGo2VelocityDirectEnvCfg`'s class defaults — they disagree on ~20 fields.

---

## Results

| run | config | steps | avg_return | avg_length | tracking_lin_vel |
|---|---|---|---|---|---|
| `fq0mkatm` (day 1) | genesis_merged asset | 50M | 23.49 | 1001 | 0.892 |
| **A** `eep2pu0n` | + unitree URDF | 19.5M (stopped) | 18.52 | 914 | 0.824 |
| **B** `9k1k4x9o` | + actuator, **armature 0** | 13.5M (stopped) | **−0.21** | **4.0** | 0.001 |
| **C** `ngu37ifo` | + actuator, armature 0.1 | **50M complete** | **23.44** | **1000** | **0.901** |
| **D** `ftzh0srg` | + tdmpc2 ground material | 27.5M (running) | 20.25 | 955 | 0.558 |

**C is the headline: 23.44 vs the day-1 baseline's 23.49.** Swapping in TDMPC2's robot
description and its actuator model is performance-neutral on this task.

B is the failure that taught us the most — see the armature section.

---

## Rung 1 — TDMPC2's robot description

`env.asset_source = genesis_merged | unitree_urdf`

### They are not the same robot

Both descend from `unitreerobotics/go2_description`. Diffed link by link they differ in
exactly three ways; the other 25 links match to 1e-12 on mass, COM, inertia and collision
geometry, and the seven `.dae` visual meshes are **byte-identical** (md5-checked), so only
the 27 KB URDF is vendored and it borrows the existing mesh directory.

| | `genesis_merged` | `unitree_urdf` |
|---|---|---|
| total mass | 15.019 kg | **16.087 kg** (+7.1%) |
| extra links | — | 12 `*_rotor` (0.089 kg each) + `front_camera` |
| thigh collision box | 0.11 m (Unitree's "amended" model) | 0.213 m (stock) |
| calf joint limits | 35.55 N·m / 20.07 rad/s (23.7 × **1.5**) | 45.43 / 15.7 (23.7 × **1.9169**) |

The rotors are pure inertia lumps — no visual, no collision, COM at the joint origin — and
they hang off the *proximal* link by a **fixed** joint, so they contribute no reflected
inertia at all. That job belongs to `armature`. Unitree spec the knee at 45.43 N·m, so the
Genesis asset understates knee torque by 22%; inert while no torque ceiling is enforced,
live as soon as a real actuator model lands.

### ⭐ How Isaac Sim 5.1 actually merges fixed joints

Measured, not assumed. `merge_fixed_joints` defaults to `True` and TDMPC2 never overrides
it, yet TDMPC2 resolves feet with `find_bodies(".*_foot")` and the foot joints are fixed.
Converting the asset and dumping the topology explains it:

**A fixed-joint child is merged only if it carries no mass.** 42 links import to **31**:

- merged: `*_calflower`, `*_calflower1`, `front_camera` (no `<inertial>`) and `imu`,
  `radar` (`<inertial>` with mass exactly 0.0)
- kept: the twelve 0.089 kg rotors, `Head_upper`/`Head_lower` at 0.001 kg, and **the four
  0.04 kg feet**

TDMPC2's feet survive because they weigh 40 grams, not because anything asked for them.

### ⭐ The silent bug this would have caused

The shared env selected contact bodies by **substring**, safe only on the merged 17-link
asset where no body name is a prefix of another. On the 31-body asset `"calf"` also matches
`FL_calf_rotor` and `"thigh"` matches `FL_thigh_rotor`: the penalised-contact set grows
from 9 bodies to **17**. Invisible, too — those bodies have no collider, so they always
report zero force and the reward looks correct while the config is wrong.

`resolve_link_indices` is now full-match **regex** (`sim_backend.py`), with the same
patterns TDMPC2 uses: `["base", ".*_thigh", ".*_calf"]`, `[".*_foot"]`. Verified equivalent
on the merged asset — still `[0]`, `[0,5..12]`, `[13,14,15,16]`, settling trajectory
unchanged to every printed digit.

### Bring-up

```
                         genesis_merged      unitree_urdf
n_links                        17                 31
termination / penalized / feet  1 / 9 / 4          1 / 9 / 4
step 0  mean|q-q*| / |tau|     0.1581 / 3.928     0.1581 / 3.930
after 30 steps  base height    0.3080             0.3084
```

Step 0 still matches the Genesis arm to three digits, so the reset and first torque are
unaffected by the asset swap.

**Command (run A):**

```bash
cd /home/sachin/FlashSAC && OMNI_KIT_ACCEPT_EULA=YES .venv-isaaclab/bin/python train.py \
    --config_name flashSAC_base --overrides seed=0 \
    --overrides group_name=go2-port-ladder --overrides exp_name=rung1-unitree-urdf \
    --overrides logger_type=wandb --overrides entity_name=null \
    --overrides evaluation_per_interaction_step=4882 \
    --overrides metrics_per_interaction_step=4882 \
    --overrides recording_per_interaction_step=4882 \
    --overrides logging_per_interaction_step=488 \
    --overrides env=go2 \
    --overrides env.sim_backend=isaaclab \
    --overrides env.asset_source=unitree_urdf \
    --overrides env.enable_camera=true \
    --overrides env.env_name=go2-walk \
    --overrides num_env_steps=50_000_896 --overrides num_train_envs=1024 \
    --overrides num_eval_envs=null --overrides num_record_envs=null \
    --overrides num_eval_episodes=1024 --overrides num_record_episodes=1 \
    --overrides agent=flashSAC \
    --overrides agent.compile_mode=max-autotune-no-cudagraphs \
    --overrides agent.buffer_max_length=10_000_000 \
    --overrides agent.buffer_min_length=100_000 \
    --overrides agent.buffer_device_type=cuda \
    --overrides agent.sample_batch_size=2048 --overrides agent.use_amp=true \
    --overrides updates_per_interaction_step=2 \
    --overrides agent.asymmetric_observation=true \
    --overrides gamma=0.95 --overrides n_step=1
```

---

## Rung 3 — the actuator

```
env.actuator_model  explicit_pd_unclipped | dc_motor | unitree_go2hv
env.pd_stiffness    30.0 (Genesis)  ->  25.0 (TDMPC2)
env.pd_damping       1.5 (Genesis)  ->   0.5 (TDMPC2)
env.dof_armature     0.1 (Genesis)  ->   0.0 (TDMPC2)
```

### Why the envelope lives in the shared PD law

TDMPC2 drives with `set_joint_position_target` and lets `MotorStrengthUnitreeActuator` (a
`DelayedPDActuator`) compute torque. With the config it actually uses, that reduces to

```
tau = clip_envelope(Kp * (q_target - q) + Kd * (0 - qd), qd)
```

— `_make_unitree_go2hv_actuator_cfg` copies fields off a `DCMotorCfg`, which defines no
`min_delay`/`max_delay`, so the delay buffers are zero-length; `Fs`/`Fd` default to 0. It
recomputes every physics substep, exactly as `Go2BaseEnv.step` recomputes
`_compute_torques` every decimation substep, both at 200 Hz. Same computation, so
implementing the envelope in `actuators.py` keeps one control path instead of two — and
lets the rung be exercised on the fast Genesis backend.

`unitree_go2hv_clip_effort` and `dc_motor_clip_effort` are **bit-identical** to TDMPC2's
`go2_actuator_math.py` over 200k random (torque, velocity) pairs.

### The envelope is not a no-op

Under zero actions peak torque is ~8.5 N·m, nowhere near the 20.2/23.4 ceiling — a
zero-action smoke test cannot tell this rung from a no-op. Under random actions across the
full action range it binds on **14.6%** of joint-steps at Genesis gains and **20.2%** at
TDMPC2's.

### ⭐ Armature is not an independent axis

Run B set `dof_armature=0.0` and collapsed to 4-step episodes, never recovering through
13.5M steps. Ablated on Genesis, 300 steps × 256 envs, random actions:

| config | max \|qd\| | % qd>30 | tilt terminations |
|---|---|---|---|
| baseline 30/1.5 arm 0.1 unclipped | 11.4 | 0.00% | 669 |
| + go2hv clip only | 11.5 | 0.00% | 722 |
| + gains only (25/0.5, arm 0.1) | 16.9 | 0.00% | 1565 |
| + gains + clip (arm 0.1) | 15.3 | 0.00% | 1654 |
| + gains + **armature 0** | 48.3 | 0.37% | 2867 |
| **full rung** | **115.2** | **3.70%** | 3256 |

**The armature drop is the cause; the envelope amplifies it.** Removing the ~34×
effective-inertia buffer lets joint velocities run past the motor's 30 rad/s no-load speed
— exactly where the go2hv envelope caps torque at *zero*. The leg goes limp, cannot
recover, and the robot tips past the 0.4 rad termination. The clip alone is nearly free;
the gains alone are survivable.

Two guards now exist in `get_env()`:

- **raise** when `Kd*dt/I > 2` (the calf's ~0.003 kg·m² about the knee plus armature).
  Genesis 1.5/0.1 gives 0.07; TDMPC2 0.5/0.0 gives 0.83; **1.5/0.0 gives 2.50**.
- **warn** when armature < 0.05 while tilt termination is still on. The ratio test is
  necessary but not sufficient — armature 0 scores 0.83 and still destabilises.

TDMPC2 gets away with armature 0 only in combination with config not yet ported:
`enable_termination=false`, a crouched stance 13 cm lower, and gentler command ranges. It
must move together with those rungs.

**Command (run C — the one that completed 50M at 23.44):** run A's command plus

```
    --overrides env.actuator_model=unitree_go2hv \
    --overrides env.pd_stiffness=25.0 \
    --overrides env.pd_damping=0.5 \
```

Run B was the same plus `--overrides env.dof_armature=0.0` — **do not repeat**.

---

## Scene: lighting and the ground contact material

### The recordings had no ground

Frame mean brightness was 12.5/255, bottom half RGB [5.3, 5.4, 5.5] — a white robot in a
black void, no ground, horizon or contact shadow.

It was **not** the ground colour. `TerrainImporterCfg.visual_material` defaults to
`diffuse_color=(0,0,0)` and forwards it into `GroundPlaneCfg(color=...)`, and
`GroundPlaneCfg`'s own default is the same black — *TDMPC2's ground is black too*.

The cause was one flag. An earlier version set `visible_in_primary_ray=False` on the dome
light, reasoning that a light source should not double as a backdrop. With the dome hidden
the sky renders black, and a black ground against a black sky is invisible.

Now matching TDMPC2's `_setup_scene` exactly: `DomeLightCfg(intensity=2000.0,
color=(0.75, 0.75, 0.75))`, visible. Mean brightness **12.5 → 107.8**. The robot
visual-material override was removed too — TDMPC2 binds none, and ours was silently
failing anyway (asked for dark grey, rendered white). All inert prims; no dynamics effect.

### `env.ground_material`

```
isaaclab_default   static/dynamic 0.5, combine "average"   (GroundPlaneCfg defaults)
tdmpc2             static/dynamic 1.0, combine "multiply"  (TDMPC2's TerrainImporterCfg)
```

The **combine mode** is the load-bearing half, not the coefficients. PhysX gives every
collider its own material, so a foot/ground contact holds two friction values and must
reduce them to one:

| mode | contact μ |
|---|---|
| `average` | (μ_foot + μ_ground) / 2 |
| `multiply` | μ_foot × μ_ground |

With `multiply` against a ground of exactly **1.0**, the contact *is* the robot's own
coefficient — 1.0 is the identity element, so the ground becomes transparent and TDMPC2's
friction randomisation lands undiluted. With `average` against 0.5 the ground permanently
drags every contact toward 0.5 and halves the robot's contribution: even a randomised
μ_foot of 0 still yields 0.25. The range is compressed *and* re-centred, so it is not a
scale factor absorbable elsewhere.

Setting it on the ground alone suffices: when two materials disagree on the mode PhysX
takes the higher-priority one (`PxCombineMode` order average < min < multiply < max).

Genesis rejects anything but `isaaclab_default` — its ground is `plane.urdf` under the
Genesis solver, with no per-material combine mode.

**Command (run D):** run C's command plus `--overrides env.ground_material=tdmpc2`.

---

## Still different from TDMPC2

**Plant / control:**

| | here | TDMPC2 |
|---|---|---|
| armature | 0.1 | 0.0 — held back deliberately |
| action latency | 0.02 s (1 step) | none |
| stance | hip 0, F-thigh 0.8 / R-thigh 1.0, calf −1.5 | hip 0, all thigh 0.9, calf −1.8 |
| spawn height | 0.42 m | 0.29017 m |
| friction DR | ×U(0.2, 1.5) multiplicative on the robot | U(0.2, 1.25) **absolute** |
| motor strength DR | off | on, per-joint U(0.9, 1.1) |
| motor offset DR | on, ±0.02 rad | none |
| Kp/Kd DR | on, ±20% each | none |
| COM DR | ±0.01 m | ±0.05 m |
| restitution DR | none | U(0.0, 0.15) |

**Task side — still entirely Genesis, nothing ported:** observations (45-dim, Genesis
scaling, batch-shared noise, ±100 clip, no history vs TDMPC2's 225-dim raw-unit 5-frame
stack), privileged obs (asymmetric 60-dim here; TDMPC2 has none), commands (±1.0 on all
three axes, 4 s resample, 0.2 deadband vs vx 0.2–0.8 / vy≡0 / ωz ±0.5, 10 s, no deadband),
rewards (12 Genesis terms vs 17 TDMPC2 terms), termination (base contact + 0.4 rad tilt vs
**disabled entirely**), reset randomisation, joint ordering in the obs/action vector.

**Already identical:** URDF bytes, import settings (merge, capsules, self-collision, solver
8/4), 31-body topology and 16.087 kg, sim dt 0.005 / decimation 4 / 50 Hz, 20 s episodes,
Kp 25 / Kd 0.5, the go2hv envelope, effective action scale 0.75 rad, joint-target clipping
off, torque law and its scale-then-clip ordering, soft joint limit factor 0.9, pushes
disabled, ground contact material.

## Next rung

Armature, stance, spawn height and termination are **one coupled group**, not four rungs.
Trying armature 0 again on its own will fail exactly as run B did. The cheapest path is to
land stance + spawn height + `enable_termination=false` together with it as a single rung.
