#!/usr/bin/env python3
"""Measure saved final outputs using the same code snapshot as the comparison.

Volume provenance remains explicit. These measurements support sensitivity
analysis of visually flagged objects; they do not validate spine anatomy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--sampling-frame", type=Path, required=True)
    parser.add_argument("--code-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_snapshot.resolve()))
    import neurd  # Apply compatibility shims before mesh_tools import.
    from neurd.spine_volume import estimate_mesh_volume
    from neurd.spine_refinement import _Axis, RefinementConfig

    frame = json.loads(args.sampling_frame.read_text())
    rows = []
    by_key = {}
    for record in frame["population"]:
        by_key.setdefault(record["key"], []).append(record)
    for key, population in by_key.items():
        branch = args.comparison / "branches" / key
        result = json.loads((branch / "result.json").read_text())
        axis = _Axis(np.load(result["selection"]["skeleton"]), RefinementConfig())
        for record in population:
            path = Path(record["mesh"])
            mesh = trimesh.load(path, force="mesh", process=False)
            foot, distance = axis.project([mesh.centroid])
            volume = estimate_mesh_volume(mesh)
            rows.append(dict(key=key, raw_id=record["raw_id"],
                             area_nm2=float(mesh.area),
                             centroid_axis_distance_nm=float(distance[0]),
                             volume=volume,
                             mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(json.dumps(dict(key=key, measured=len(population))), flush=True)
    assert len(rows) == frame["population_size"]
    data = dict(sampling_frame_sha256=hashlib.sha256(args.sampling_frame.read_bytes()).hexdigest(),
                code_snapshot=str(args.code_snapshot.resolve()),
                volume_module_sha256=hashlib.sha256(Path(neurd.__file__).parent.joinpath("spine_volume.py").read_bytes()).hexdigest(),
                algorithm="current_workspace_centered_strict_closure_v1",
                measured=len(rows),
                volume_methods=dict(Counter(row["volume"]["method"] for row in rows)),
                candidates=rows)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(dict(measured=len(rows), methods=data["volume_methods"])), flush=True)


if __name__ == "__main__":
    main()
