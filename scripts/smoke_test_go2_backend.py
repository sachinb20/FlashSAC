#!/usr/bin/env python
"""Bring-up smoke test for a Go2 simulator backend.

Builds the shared env on the chosen backend, prints the resolved topology, steps it with
zero actions, and reports whether the robot behaves sanely. Use this before launching a
50M-step run.

    # Genesis (from the genesis venv)
    .venv/bin/python scripts/smoke_test_go2_backend.py --backend genesis

    # IsaacLab (from an env with isaacsim + isaaclab)
    OMNI_KIT_ACCEPT_EULA=YES <python> scripts/smoke_test_go2_backend.py --backend isaaclab

The IsaacLab arm needs the vendored asset in place first:
``.venv/bin/python scripts/vendor_go2_asset.py`` (run from the Genesis venv).

Expected values, taken from a live Genesis build -- the index *sets* must match on both
backends even though body ordering may differ:

    n_links            17
    termination_links  [0]                          (base)
    penalized_links    [0, 5..12]                   (base + 4 thigh + 4 calf)
    feet_links         [13, 14, 15, 16]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Body *counts* differ by asset; the contact index *sets* must not. The upstream Unitree
# description imports to 31 bodies rather than 17 because Isaac Sim 5.1 only merges
# fixed-joint children that carry no mass, so the twelve 0.089 kg rotor links and the two
# 0.001 kg head links survive. The 1 / 9 / 4 contact sets are the invariant.
EXPECTED = {
    "genesis_merged": {"n_links": 17, "n_termination": 1, "n_penalized": 9, "n_feet": 4},
    "unitree_urdf": {"n_links": 31, "n_termination": 1, "n_penalized": 9, "n_feet": 4},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["genesis", "isaaclab"], required=True)
    parser.add_argument(
        "--asset-source",
        choices=["genesis_merged", "unitree_urdf"],
        default="genesis_merged",
        help="Robot description. 'unitree_urdf' is what TDMPC2 trains against (isaaclab only).",
    )
    parser.add_argument(
        "--ground-material",
        choices=["isaaclab_default", "tdmpc2"],
        default=None,
        help="Ground contact material. TDMPC2 uses 1.0/1.0 with combine=multiply.",
    )
    parser.add_argument(
        "--actuator-model",
        choices=["explicit_pd_unclipped", "dc_motor", "unitree_go2hv"],
        default=None,
        help="Torque-speed ceiling on the PD torque. TDMPC2 runs unitree_go2hv.",
    )
    parser.add_argument("--pd-stiffness", type=float, default=None, help="Genesis 30.0, TDMPC2 25.0")
    parser.add_argument("--pd-damping", type=float, default=None, help="Genesis 1.5, TDMPC2 0.5")
    parser.add_argument(
        "--dof-armature",
        type=float,
        default=None,
        help="Genesis 0.1, TDMPC2 0.0. Coupled to --pd-damping; see get_env().",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", action="store_true", help="also exercise the video path")
    parser.add_argument(
        "--save-frame",
        default=None,
        help="Where to write the rendered frame (default: smoke_frame_<backend>.png in cwd). "
        "Only written when --camera is passed.",
    )
    parser.add_argument(
        "--spawn-height",
        type=float,
        default=None,
        help="Override the reset spawn height. Set high (e.g. 2.0) to watch the joints "
        "track in free fall: that isolates the actuator/PD path from ground contact.",
    )
    args = parser.parse_args()

    import torch

    from flash_rl.envs.go2_common.go2_env import get_env

    env = get_env(
        num_envs=args.num_envs,
        eval_mode=True,
        sim_backend=args.backend,
        enable_camera=args.camera,
        asset_source=args.asset_source,
        actuator_model=args.actuator_model,
        ground_material=args.ground_material,
        pd_stiffness=args.pd_stiffness,
        pd_damping=args.pd_damping,
        dof_armature=args.dof_armature,
    )

    print(f"\n=== {args.backend} backend bring-up (asset={args.asset_source}) ===")
    print("actuator_model    :", env.env_cfg["actuator_model"])
    print("PD kp/kd          :", env.env_cfg["PD_stiffness"]["joint"], "/", env.env_cfg["PD_damping"]["joint"])
    print("dof_armature      :", env.env_cfg["dof_armature"])
    print("ground_material   :", env.env_cfg["ground_material"])
    print("n_links           :", env.sim.n_links)
    print("motor_dofs        :", env.motor_dofs)
    print("termination_links :", env.termination_contact_link_indices)
    print("penalized_links   :", env.penalized_contact_link_indices)
    print("feet_links        :", env.feet_link_indices)
    print("default_dof_pos   :", [round(float(v), 3) for v in env.default_dof_pos])
    if hasattr(env.sim, "describe_actuation"):
        for k, v in env.sim.describe_actuation().items():
            print(f"{k:<18}:", v)

    # Seed immediately before reset so both backends draw the same reset randomisation
    # (joint offsets, base tilt, yaw) and the numbers below are actually comparable.
    if args.spawn_height is not None:
        # reset_idx seeds base_pos from this tensor, so patching it moves the spawn.
        env.base_init_pos[2] = args.spawn_height
        print(f"\n[spawn height overridden to {args.spawn_height} -- free-fall isolation test]")

    torch.manual_seed(args.seed)
    env.reset()

    # Geometry check, before any dynamics run. If the URDF imported to different link
    # placements the robot cannot stand correctly no matter how good the controller is,
    # and every dynamics number downstream is meaningless. Foot offsets are expressed in
    # the base frame so the two engines are directly comparable.
    env._update_buffers()
    print("\n--- imported kinematics at reset (env 0, feet relative to base) ---")
    base = env.base_pos[0]
    for j, fi in enumerate(env.feet_link_indices):
        rel = env.foot_positions[0, j] - base
        print(f"  foot[{j}] link={fi:<3} rel_base=[{rel[0]:>7.4f},{rel[1]:>7.4f},{rel[2]:>7.4f}]")
    print(f"  base height at reset: {float(base[2]):.4f}")

    # Zero actions command the nominal stance, so a healthy robot settles: mean |joint
    # tracking error| should fall toward 0 and base height toward ~0.30. Printing the
    # trajectory distinguishes "still settling" from "diverging" -- a single end-state
    # sample cannot.
    print(f"\n--- settling under zero actions ({args.steps} steps) ---")
    print(f"{'step':>5} {'mean|q-q*|':>11} {'max|q-q*|':>10} {'mean|tau|':>10} {'height':>8} {'contact':>9}")
    for i in range(args.steps):
        obs, rew, done, _ = env.step(torch.zeros(args.num_envs, 12, device=env.device))
        if i % max(args.steps // 10, 1) == 0 or i == args.steps - 1:
            err = (env.dof_pos - env.default_dof_pos).abs()
            print(
                f"{i:>5} {float(err.mean()):>11.4f} {float(err.max()):>10.4f} "
                f"{float(env.torques.abs().mean()):>10.3f} {float(env.base_pos[:, 2].mean()):>8.4f} "
                f"{float(env.link_contact_forces.norm()):>9.2f}"
            )

    # Per-joint detail. Aggregates hide which joint is misbehaving, and a joint sitting
    # exactly on a limit (or a name/index mismatch between the requested and resolved
    # ordering) is only visible here.
    names = getattr(env.sim, "resolved_dof_names", None)
    requested = getattr(env.sim, "requested_dof_names", None)
    if names and requested:
        order_ok = list(names) == list(requested)
        print(f"\n--- per-joint state (dof order preserved: {order_ok}) ---")
        if not order_ok:
            print("  !! resolved order differs from requested -- joint tensors are permuted")
            print("     requested:", requested)
            print("     resolved :", names)
        lo, hi = env.dof_pos_limits[:, 0], env.dof_pos_limits[:, 1]
        print(f"{'idx':>3} {'joint':<16} {'q':>9} {'q*':>8} {'err':>8} {'tau':>10}  {'soft limits':>18}")
        for j, nm in enumerate(names):
            q = float(env.dof_pos[0, j])
            qs = float(env.default_dof_pos[j])
            at_limit = "  <-- AT LIMIT" if (q <= float(lo[j]) + 1e-3 or q >= float(hi[j]) - 1e-3) else ""
            print(
                f"{j:>3} {nm:<16} {q:>9.4f} {qs:>8.4f} {q - qs:>8.4f} {float(env.torques[0, j]):>10.3f}"
                f"  [{float(lo[j]):>7.3f},{float(hi[j]):>7.3f}]{at_limit}"
            )

    heights = [float(z) for z in env.base_pos[:, 2]]
    joint_err = float((env.dof_pos - env.default_dof_pos).abs().mean())
    print(f"\n--- after {args.steps} zero-action steps ---")
    print("obs shape         :", tuple(obs.shape))
    print("obs finite        :", bool(torch.isfinite(obs).all()))
    print("base height (z)   :", [round(h, 4) for h in heights])
    print("mean |q - q*|     :", round(joint_err, 4), "rad")
    print("contact force norm:", round(float(env.link_contact_forces.norm()), 3))
    print("torques[0][:3]    :", [round(float(v), 3) for v in env.torques[0][:3]])

    img = env.render()
    print("render            :", None if img is None else (img.shape, img.dtype))

    if img is not None:
        # Report colour numerically as well as saving the frame: an unlit scene renders
        # near-black and fully desaturated, which is what "the videos are black and white"
        # actually looks like. R==G==B on nearly every pixel means no light, not a codec
        # or channel-order problem.
        import numpy as np

        px = img.reshape(-1, img.shape[-1])[:, :3].astype(int)
        mono = float(((px[:, 0] == px[:, 1]) & (px[:, 1] == px[:, 2])).mean())
        spread = int((px.max(1) - px.min(1)).max())
        print(f"  colour          : {100 * (1 - mono):.1f}% of pixels carry colour, max channel spread {spread}")
        print(f"  brightness      : min={px.min()} mean={px.mean():.1f} max={px.max()}")
        if mono > 0.95 or spread < 10:
            print("  !! frame is effectively greyscale -- scene is probably unlit")

        out = args.save_frame or f"smoke_frame_{args.backend}.png"
        try:
            import imageio.v3 as iio

            iio.imwrite(out, img)
        except Exception:  # no imageio in this env -- raw array is still inspectable
            out = out.rsplit(".", 1)[0] + ".npy"
            np.save(out, img)
        print("  saved frame     :", os.path.abspath(out))

    expected = EXPECTED[args.asset_source]
    problems = []
    if env.sim.n_links != expected["n_links"]:
        problems.append(f"n_links {env.sim.n_links} != {expected['n_links']}")
    if len(env.termination_contact_link_indices) != expected["n_termination"]:
        problems.append(f"termination links: {env.termination_contact_link_indices}")
    if len(env.penalized_contact_link_indices) != expected["n_penalized"]:
        problems.append(f"penalized links: {env.penalized_contact_link_indices} (expected 9)")
    if len(env.feet_link_indices) != expected["n_feet"]:
        problems.append(f"feet links: {env.feet_link_indices} (expected 4)")
    if not torch.isfinite(obs).all():
        problems.append("non-finite observations")
    # Standing robot should be somewhere near the 0.3 m target, not sunk or launched.
    if not all(0.05 < h < 1.0 for h in heights):
        problems.append(f"implausible base heights {heights} -- suspect the torque path")

    print()
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print("  -", p)
        return 1
    print("BRING-UP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
