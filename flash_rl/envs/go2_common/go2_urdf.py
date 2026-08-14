"""Resolve and pre-merge the Go2 URDF so both backends load identical topology.

Genesis loads ``go2.urdf`` with ``merge_fixed_links=True, links_to_keep=[the 4 feet]``,
which collapses the raw 29-link asset down to 17 links: ``base`` absorbs
``Head_upper``/``Head_lower``/``imu``/``radar``, and each ``*_calf`` absorbs its
``*_calflower``/``*_calflower1`` children. The feet survive because they are named.

That merge is not cosmetic, and getting it wrong is a silent divergence rather than an
error. The environment selects contact bodies by **substring**, and ``"calf"`` also
matches ``"calflower"``: on the unmerged asset the ``collision`` penalty would watch 12
bodies instead of 4, and ``base`` would stop being a single rigid body. IsaacLab's URDF
importer offers only all-or-nothing ``merge_fixed_joints``, so neither of its settings
reproduces Genesis's "merge everything except these four".

So we do the merge ourselves, ahead of time, and hand the *same pre-merged file* to both
engines with their own merging disabled. Topology is then identical by construction
rather than by coincidence.

Verified against a live Genesis build: 17 links, 15.019 kg total, with per-link masses
matching to 1e-6 (see ``verify_against_genesis_reference``).
"""

from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from typing import Optional, Sequence

import numpy as np

# Genesis's own build of go2.urdf, as ground truth for the merge.
GENESIS_REFERENCE_TOPOLOGY = {
    "n_links": 17,
    "total_mass": 15.019,
    "link_masses": {
        "base": 6.923,
        "FL_hip": 0.678,
        "FR_hip": 0.678,
        "RL_hip": 0.678,
        "RR_hip": 0.678,
        "FL_thigh": 1.152,
        "FR_thigh": 1.152,
        "RL_thigh": 1.152,
        "RR_thigh": 1.152,
        "FL_calf": 0.154,
        "FR_calf": 0.154,
        "RL_calf": 0.154,
        "RR_calf": 0.154,
        "FL_foot": 0.04,
        "FR_foot": 0.04,
        "RL_foot": 0.04,
        "RR_foot": 0.04,
    },
}

GO2_LINKS_TO_KEEP = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


# --------------------------------------------------------------------------- transforms


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    p = np.arcsin(sy)
    if abs(sy) < 1.0 - 1e-9:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        r = np.arctan2(-R[1, 2], R[1, 1])
        y = 0.0
    return np.array([r, p, y])


def _read_origin(elem: Optional[ET.Element]) -> tuple[np.ndarray, np.ndarray]:
    if elem is None:
        return np.zeros(3), np.eye(3)
    origin = elem.find("origin")
    if origin is None:
        return np.zeros(3), np.eye(3)
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    return xyz, _rpy_to_matrix(rpy)


def _write_origin(elem: ET.Element, xyz: np.ndarray, R: np.ndarray) -> None:
    origin = elem.find("origin")
    if origin is None:
        origin = ET.SubElement(elem, "origin")
    origin.set("xyz", " ".join(f"{v:.10g}" for v in xyz))
    origin.set("rpy", " ".join(f"{v:.10g}" for v in _matrix_to_rpy(R)))


def _inertia_matrix(elem: ET.Element) -> np.ndarray:
    i = elem.find("inertia")
    if i is None:
        return np.zeros((3, 3))
    ixx, ixy, ixz = float(i.get("ixx", 0)), float(i.get("ixy", 0)), float(i.get("ixz", 0))
    iyy, iyz, izz = float(i.get("iyy", 0)), float(i.get("iyz", 0)), float(i.get("izz", 0))
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def _set_inertia(elem: ET.Element, inertia: np.ndarray) -> None:
    i = elem.find("inertia")
    if i is None:
        i = ET.SubElement(elem, "inertia")
    i.set("ixx", f"{inertia[0, 0]:.10g}")
    i.set("ixy", f"{inertia[0, 1]:.10g}")
    i.set("ixz", f"{inertia[0, 2]:.10g}")
    i.set("iyy", f"{inertia[1, 1]:.10g}")
    i.set("iyz", f"{inertia[1, 2]:.10g}")
    i.set("izz", f"{inertia[2, 2]:.10g}")


def _parallel_axis(inertia: np.ndarray, mass: float, d: np.ndarray) -> np.ndarray:
    """Shift an inertia tensor from its COM by displacement ``d``."""
    return inertia + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


# ------------------------------------------------------------------------------- merge


def merge_fixed_links(
    src_urdf: str,
    dst_urdf: str,
    links_to_keep: Sequence[str] = GO2_LINKS_TO_KEEP,
) -> dict[str, float]:
    """Collapse every fixed joint except those leading to ``links_to_keep``.

    Reproduces Genesis's ``merge_fixed_links=True`` semantics. Child mass, inertia,
    collision and visual geometry are folded into the parent with the joint transform
    applied; any grandchild joints are reparented with their origins composed.

    Returns the resulting ``{link_name: mass}`` map.
    """
    tree = ET.parse(src_urdf)
    root = tree.getroot()

    links = {link.get("name"): link for link in root.findall("link")}
    joints = list(root.findall("joint"))
    keep = set(links_to_keep)

    def children_of(name: str) -> list[ET.Element]:
        return [j for j in joints if j.find("parent").get("link") == name]

    # Bottom-up: only ever merge a link that has no remaining children, so chains such as
    # calf <- calflower <- calflower1 collapse in the correct order.
    changed = True
    while changed:
        changed = False
        for joint in list(joints):
            if joint.get("type") != "fixed":
                continue
            child_name = joint.find("child").get("link")
            parent_name = joint.find("parent").get("link")
            if child_name in keep or child_name not in links:
                continue
            if children_of(child_name):
                continue  # not a leaf yet

            parent, child = links[parent_name], links[child_name]
            t, R = _read_origin(joint)

            p_in, c_in = parent.find("inertial"), child.find("inertial")
            if c_in is not None:
                c_mass = float(c_in.find("mass").get("value"))
                c_com, c_rot = _read_origin(c_in)
                c_I = _inertia_matrix(c_in)

                # child inertial frame -> parent frame
                c_com_p = R @ c_com + t
                c_rot_p = R @ c_rot
                c_I_p = c_rot_p @ c_I @ c_rot_p.T

                if p_in is None:
                    p_in = ET.SubElement(parent, "inertial")
                    ET.SubElement(p_in, "mass").set("value", "0")
                    p_mass, p_com, p_I = 0.0, np.zeros(3), np.zeros((3, 3))
                else:
                    p_mass = float(p_in.find("mass").get("value"))
                    p_com, p_rot = _read_origin(p_in)
                    p_I = p_rot @ _inertia_matrix(p_in) @ p_rot.T

                total = p_mass + c_mass
                new_com = (p_mass * p_com + c_mass * c_com_p) / total if total > 0 else p_com
                new_I = _parallel_axis(p_I, p_mass, p_com - new_com) + _parallel_axis(c_I_p, c_mass, c_com_p - new_com)

                p_in.find("mass").set("value", f"{total:.10g}")
                _write_origin(p_in, new_com, np.eye(3))
                _set_inertia(p_in, new_I)

            # move geometry into the parent, composing origins
            for tag in ("visual", "collision"):
                for geom in child.findall(tag):
                    g_xyz, g_rot = _read_origin(geom)
                    _write_origin(geom, R @ g_xyz + t, R @ g_rot)
                    parent.append(geom)

            root.remove(child)
            root.remove(joint)
            joints.remove(joint)
            del links[child_name]
            changed = True

    os.makedirs(os.path.dirname(dst_urdf), exist_ok=True)
    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)

    return {
        name: float(link.find("inertial").find("mass").get("value"))
        for name, link in links.items()
        if link.find("inertial") is not None
    }


def verify_against_genesis_reference(masses: dict[str, float], tol: float = 1e-6) -> None:
    """Raise if the merge did not reproduce Genesis's topology."""
    ref = GENESIS_REFERENCE_TOPOLOGY
    if len(masses) != ref["n_links"]:
        raise ValueError(f"Merged URDF has {len(masses)} links, Genesis produces {ref['n_links']}: {sorted(masses)}")
    missing = set(ref["link_masses"]) - set(masses)
    extra = set(masses) - set(ref["link_masses"])
    if missing or extra:
        raise ValueError(f"Link-name mismatch vs Genesis. missing={sorted(missing)} extra={sorted(extra)}")
    for name, expected in ref["link_masses"].items():
        if abs(masses[name] - expected) > tol:
            raise ValueError(f"Link {name!r} mass {masses[name]:.9f} != Genesis {expected:.9f}")
    total = sum(masses.values())
    if abs(total - ref["total_mass"]) > 1e-3:
        raise ValueError(f"Total mass {total:.6f} != Genesis {ref['total_mass']:.6f}")


# ----------------------------------------------------------------------------- resolve


def _genesis_asset_root() -> Optional[str]:
    try:
        import genesis  # noqa: PLC0415
    except Exception:
        return None
    return os.path.join(os.path.dirname(genesis.__file__), "assets")


def repo_asset_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "go2_genesis")


def resolve_source_urdf(urdf_path: str) -> str:
    """Locate the raw ``go2.urdf``.

    Order: ``FLASHSAC_GO2_URDF`` override, then the vendored copy under
    ``assets/go2_genesis/``, then the installed ``genesis`` package. The vendored copy is
    what makes the IsaacLab arm possible at all -- Genesis cannot be installed alongside
    IsaacLab, so its asset tree is unavailable in that virtualenv. Populate it with
    ``python scripts/vendor_go2_asset.py`` from the Genesis venv.
    """
    override = os.environ.get("FLASHSAC_GO2_URDF")
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"FLASHSAC_GO2_URDF={override!r} does not exist.")
        return override

    vendored = os.path.normpath(os.path.join(repo_asset_dir(), "urdf", "go2.urdf"))
    if os.path.isfile(vendored):
        return vendored

    root = _genesis_asset_root()
    if root is not None:
        candidate = os.path.join(root, urdf_path)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find go2.urdf. Genesis is not installed and no vendored copy exists at "
        f"{vendored!r}. Run `python scripts/vendor_go2_asset.py` from the Genesis venv first, "
        "or set FLASHSAC_GO2_URDF."
    )


def vendor_asset(urdf_path: str = "urdf/go2/urdf/go2.urdf") -> str:
    """Copy the Go2 asset out of the Genesis package into ``assets/go2_genesis/``.

    Run this once from the Genesis virtualenv. The IsaacLab virtualenv cannot import
    ``genesis`` to find the asset, so this vendored copy is what makes a shared asset --
    and therefore a meaningful comparison -- possible at all.

    Copies ~25 MB of ``.dae`` visual meshes. Collision geometry in this URDF is entirely
    primitives (5 boxes, 17 cylinders, 5 spheres), so the meshes affect rendering only,
    never contact.
    """
    src = resolve_source_urdf(urdf_path)
    dst_root = os.path.normpath(repo_asset_dir())
    dst_urdf = os.path.join(dst_root, "urdf", "go2.urdf")

    if os.path.abspath(src) == os.path.abspath(dst_urdf):
        return dst_urdf

    os.makedirs(os.path.join(dst_root, "urdf"), exist_ok=True)
    shutil.copy2(src, dst_urdf)

    src_dae = os.path.normpath(os.path.join(os.path.dirname(src), "..", "dae"))
    dst_dae = os.path.join(dst_root, "dae")
    if os.path.isdir(src_dae) and not os.path.isdir(dst_dae):
        shutil.copytree(src_dae, dst_dae)
    return dst_urdf


def get_merged_urdf(urdf_path: str, links_to_keep: Sequence[str] = GO2_LINKS_TO_KEEP, verify: bool = True) -> str:
    """Return a path to the pre-merged URDF, vendoring and building it on first use.

    Output goes into ``assets/go2_genesis/urdf/`` beside the vendored ``go2.urdf`` so both
    files share the single ``../dae`` mesh directory rather than duplicating it.
    """
    src = vendor_asset(urdf_path)
    dst = os.path.join(os.path.dirname(src), "go2_merged.urdf")

    if not os.path.isfile(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        masses = merge_fixed_links(src, dst, links_to_keep)
        if verify:
            verify_against_genesis_reference(masses)
    return dst
