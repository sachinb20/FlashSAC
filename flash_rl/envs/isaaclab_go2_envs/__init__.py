from typing import Any


def get_isaaclab_go2_env(
    env_name: str,
    num_envs: int,
    seed: int,
    device: str,
    render_mode: str | None = None,
    **isaac_overrides: Any,
) -> Any:
    if env_name == "go2-vel-direct":
        from .go2_env_builder import build_go2_velocity_env_cfg
        from .isaaclab_go2_velocity_direct import UnitreeGo2VelocityDirectEnv

        env_cfg = build_go2_velocity_env_cfg(
            num_envs=num_envs,
            seed=seed,
            device=device,
            **isaac_overrides,
        )
        env = UnitreeGo2VelocityDirectEnv(cfg=env_cfg, render_mode=render_mode)

    else:
        raise ValueError(f"Unknown IsaacLab Go2 env_name {env_name!r}. Expected 'go2-vel-direct'.")

    return env
