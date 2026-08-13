from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

GO2_ASSET_SOURCE_ISAACLAB_USD = "isaaclab_usd"
GO2_ASSET_SOURCE_UNITREE_URDF = "unitree_urdf"
GO2_ASSET_SOURCES = frozenset({GO2_ASSET_SOURCE_ISAACLAB_USD, GO2_ASSET_SOURCE_UNITREE_URDF})

# This file lives at <repo_root>/flash_rl/envs/isaaclab_go2_envs/go2_urdf_asset.py,
# three directories below the FlashSAC repo root.
GO2_REPO_ROOT = Path(__file__).resolve().parents[3]
GO2_REPO_DESCRIPTION_DIR = GO2_REPO_ROOT / "assets" / "go2_description"
GO2_REPO_URDF_PATH = GO2_REPO_DESCRIPTION_DIR / "urdf" / "go2_description.urdf"
GO2_UNITREE_ROS_DESCRIPTION_DIR = GO2_REPO_DESCRIPTION_DIR
GO2_UNITREE_ROS_URDF_PATH = GO2_REPO_URDF_PATH
GO2_URDF_STAGING_DIR = Path("/tmp/flashsac_isaaclab/go2_description")

GO2_URDF_ACTUATED_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
GO2_POLICY_JOINT_NAMES = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)
GO2_EXPECTED_ACTUATED_JOINT_NAMES = GO2_URDF_ACTUATED_JOINT_NAMES


def validate_go2_asset_source(asset_source: str) -> str:
    asset_source = str(asset_source)
    if asset_source not in GO2_ASSET_SOURCES:
        raise ValueError(
            f"Invalid isaac_go2_asset_source '{asset_source}'. Expected one of {sorted(GO2_ASSET_SOURCES)}."
        )
    return asset_source


def resolve_go2_urdf_path(path: str | os.PathLike[str] | None = None) -> Path:
    candidates = _go2_urdf_path_candidates(path)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Unitree Go2 URDF not found. Checked: " + ", ".join(str(candidate) for candidate in candidates)
    )


def _go2_urdf_path_candidates(path: str | os.PathLike[str] | None = None) -> tuple[Path, ...]:
    if path is None or str(path).strip() == "":
        return (GO2_REPO_URDF_PATH,)
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return (raw,)
    candidates = []
    for candidate in (raw, GO2_REPO_ROOT / raw):
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def go2_urdf_actuated_joint_names(path: str | os.PathLike[str]) -> tuple[str, ...]:
    root = ET.parse(resolve_go2_urdf_path(path)).getroot()
    names: list[str] = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "fixed":
            names.append(str(joint.attrib["name"]))
    return tuple(names)


def validate_go2_urdf_joint_contract(path: str | os.PathLike[str]) -> tuple[str, ...]:
    names = go2_urdf_actuated_joint_names(path)
    if names != GO2_URDF_ACTUATED_JOINT_NAMES:
        raise ValueError(
            f"Unitree Go2 URDF actuated joint order mismatch. actual={names}, expected={GO2_URDF_ACTUATED_JOINT_NAMES}."
        )
    return names


def go2_policy_to_sim_joint_ids(
    sim_joint_names: tuple[str, ...] | list[str],
    policy_joint_names: tuple[str, ...] | list[str] = GO2_POLICY_JOINT_NAMES,
) -> tuple[int, ...]:
    sim_joint_names = tuple(str(name) for name in sim_joint_names)
    policy_joint_names = tuple(str(name) for name in policy_joint_names)
    missing = [name for name in policy_joint_names if name not in sim_joint_names]
    extra = [name for name in sim_joint_names if name not in policy_joint_names]
    if missing or extra or len(sim_joint_names) != len(policy_joint_names):
        raise ValueError(
            "Go2 joint-name contract mismatch. "
            f"actual={sim_joint_names}, policy={policy_joint_names}, missing={missing}, extra={extra}."
        )
    return tuple(sim_joint_names.index(name) for name in policy_joint_names)


def _replace_symlink_or_file(path: Path, target: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        if path.resolve() == target.resolve():
            return
        shutil.rmtree(path)
    os.symlink(target, path, target_is_directory=target.is_dir())


def stage_go2_urdf_for_isaaclab(
    path: str | os.PathLike[str],
    *,
    staging_dir: str | os.PathLike[str] = GO2_URDF_STAGING_DIR,
) -> Path:
    source_urdf = resolve_go2_urdf_path(path)
    validate_go2_urdf_joint_contract(source_urdf)

    source_description_dir = source_urdf.parents[1]
    staging_path = Path(staging_dir).expanduser().resolve()
    staging_path.mkdir(parents=True, exist_ok=True)
    for asset_dir_name in ("dae", "meshes"):
        source_asset_dir = source_description_dir / asset_dir_name
        if source_asset_dir.exists():
            _replace_symlink_or_file(staging_path / asset_dir_name, source_asset_dir)

    text = source_urdf.read_text(encoding="utf-8")
    text = text.replace("package://go2_description/", "")
    staged_urdf = staging_path / "go2_description.urdf"
    staged_urdf.write_text(text, encoding="utf-8")
    return staged_urdf
