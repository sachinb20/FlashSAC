#!/usr/bin/env python
"""Vendor the Genesis Go2 asset into ``assets/go2_genesis/`` and build the merged URDF.

Run once, **from the Genesis virtualenv** (it is the only one that can import ``genesis``
to locate the asset). After this, the IsaacLab virtualenv can load the identical robot:

    uv run --extra genesis python scripts/vendor_go2_asset.py

Prints the resulting topology so you can see it matches Genesis's own merge.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flash_rl.envs.go2_common.go2_urdf import (  # noqa: E402
    GENESIS_REFERENCE_TOPOLOGY,
    get_merged_urdf,
    merge_fixed_links,
    vendor_asset,
    verify_against_genesis_reference,
)


def main() -> int:
    src = vendor_asset()
    print(f"vendored source URDF -> {src}")

    merged = get_merged_urdf("urdf/go2/urdf/go2.urdf")
    print(f"merged URDF          -> {merged}")

    masses = merge_fixed_links(src, merged)
    verify_against_genesis_reference(masses)

    print(f"\nlinks: {len(masses)} (Genesis reference: {GENESIS_REFERENCE_TOPOLOGY['n_links']})")
    for name in sorted(masses):
        print(f"  {name:12s} {masses[name]:.6f} kg")
    print(f"total mass: {sum(masses.values()):.6f} kg")
    print("\nOK: topology matches the Genesis reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
