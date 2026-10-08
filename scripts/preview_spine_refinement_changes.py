#!/usr/bin/env python3
"""Render matched views of final spine changes from validate_spine_refinement.py.

Blue is the original CGAL candidate, orange is newly added parent surface,
and grey is nearby branch context. Coordinates and scale labels are in µm.
This produces scientific PNGs and a machine-readable index, not a report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def unpack(path):
    with np.load(path) as data:
        groups = np.split(data["faces"], data["offsets"][1:-1])
        return dict(zip(data["raw_ids"].tolist(), groups))


def axis_samples(skeleton):
    segments = np.asarray(skeleton).reshape(-1, 2, 3)
    samples = []
    for a, b in segments:
        length = np.linalg.norm(b - a)
        if np.isfinite(length) and length > 0:
            samples.append(a + np.linspace(0, 1, int(np.ceil(length / 100)) + 1)[:, None] * (b - a))
    return cKDTree(np.concatenate(samples)), segments


def local_view(parent, centers, center_tree, axis_tree, skeleton, original, proposal):
    union = np.union1d(original, proposal)
    candidate_vertices = parent.vertices[np.unique(parent.faces[union])]
    seed_center = centers[original].mean(axis=0)
    _, nearest = axis_tree.query(seed_center)
    anchor = axis_tree.data[nearest]
    local_axis = axis_tree.data[axis_tree.query_ball_point(anchor, 800)]
    if len(local_axis) < 3:
        local_axis = axis_tree.data[np.atleast_1d(axis_tree.query(anchor, k=min(8, len(axis_tree.data)))[1])]
    _, basis = np.linalg.eigh(np.cov(local_axis.T))
    tangent = basis[:, -1]
    if tangent[np.argmax(abs(tangent))] < 0:
        tangent *= -1
    radial = seed_center - anchor
    radial -= tangent * np.dot(radial, tangent)
    if np.linalg.norm(radial) < 1:
        radial = np.cross(tangent, np.eye(3)[np.argmin(abs(tangent))])
    radial /= np.linalg.norm(radial)
    frame = np.column_stack((radial, np.cross(tangent, radial), tangent))
    focus_points = np.vstack((candidate_vertices, anchor))
    origin = (focus_points.min(axis=0) + focus_points.max(axis=0)) / 2
    radius = np.linalg.norm(focus_points - origin, axis=1).max() + 750
    nearby = np.asarray(center_tree.query_ball_point(origin, radius), dtype=int)
    transformed = (parent.vertices - origin) @ frame / 1000
    extent = transformed[np.unique(parent.faces[union])]
    anchor_local = (anchor - origin) @ frame / 1000
    focus = np.vstack((extent, anchor_local))
    middle = (focus.max(axis=0) + focus.min(axis=0)) / 2
    span = max(1.2, np.ptp(focus, axis=0).max() + 1.2)
    skeleton_local = (skeleton - origin) @ frame / 1000
    axis_near = np.linalg.norm(skeleton.mean(axis=1) - origin, axis=1) < radius
    return transformed, nearby, skeleton_local[axis_near], middle, span


def render_panel(ax, vertices, nearby, seeds, added, skeleton, middle, span, angle, title):
    highlighted = np.union1d(seeds, added)
    context = np.setdiff1d(nearby, highlighted)
    for ids, color, alpha in ((context, "#c4c8cc", .25), (seeds, "#137ca3", 1), (added, "#f29432", 1)):
        if len(ids):
            surface = Poly3DCollection(vertices[ids], facecolor=color, edgecolor="none", alpha=alpha)
            ax.add_collection3d(surface)
    for segment in skeleton:
        ax.plot(*segment.T, color="#4f5963", alpha=.55, linewidth=.7)
    for i, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(middle[i] - span / 2, middle[i] + span / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=angle[0], azim=angle[1])
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--key", action="append")
    parser.add_argument("--raw-ids", type=int, nargs="+")
    parser.add_argument("--previous-comparison", type=Path,
                        help="Render only candidates whose final masks differ from a previous refinement")
    parser.add_argument("--controls", type=int, default=0)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()
    output = args.output or args.comparison / "previews"
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for result_path in sorted((args.comparison / "branches").glob("*/result.json")):
        result = json.loads(result_path.read_text())
        key = result["key"]
        if args.key and key not in args.key:
            continue
        parent = trimesh.load(result["selection"]["source"], process=False, force="mesh")
        centers = parent.triangles_center
        center_tree = cKDTree(centers)
        axis_tree, skeleton = axis_samples(np.load(result["selection"]["skeleton"]))
        raw = unpack(result_path.parent / "raw_faces.npz")
        before, after = (unpack(result_path.parent / (name + "_faces.npz")) for name in ("baseline", "refined"))
        rows = [] if args.controls_only else list(result["changes"])
        if args.previous_comparison:
            previous = unpack(args.previous_comparison / "branches" / key / "refined_faces.npz")
            rows = [row for row in rows if
                    (row["raw_id"] in previous) != (row["raw_id"] in after)
                    or (row["raw_id"] in previous and row["raw_id"] in after
                        and not np.array_equal(np.sort(previous[row["raw_id"]]), np.sort(after[row["raw_id"]])))]
        if args.raw_ids:
            rows = [row for row in rows if row["raw_id"] in args.raw_ids]
            selected = {row["raw_id"] for row in rows}
            rows += [dict(raw_id=i, action="unchanged_control") for i in args.raw_ids
                     if i not in selected and i in before and i in after
                     and np.array_equal(before[i], after[i])]
        if args.controls:
            unchanged = sorted(i for i in before if i in after and np.array_equal(before[i], after[i]))
            # Evenly spaced original output order; fixed and independent of the proxy.
            if unchanged:
                take = np.unique(np.linspace(0, len(unchanged) - 1, min(args.controls, len(unchanged)), dtype=int))
                rows += [dict(raw_id=unchanged[i], action="unchanged_control") for i in take]
        if not rows:
            continue
        for page, start in enumerate(range(0, len(rows), args.rows)):
            block = rows[start:start + args.rows]
            fig = plt.figure(figsize=(16, 3.5 * len(block) + .5), layout="constrained")
            fig.suptitle(key + "  |  blue: original; orange: added; grey: parent branch", fontsize=12)
            for row_index, row in enumerate(block):
                raw_id = row["raw_id"]
                original = raw[raw_id]
                proposed_path = result_path.parent / "changes" / f"raw_{raw_id:04d}_proposed_faces.npy"
                proposal = after.get(raw_id, np.load(proposed_path) if proposed_path.exists() else original)
                local, nearby, skel, middle, span = local_view(
                    parent, centers, center_tree, axis_tree, skeleton, original, proposal)
                triangles = local[parent.faces]
                added = np.setdiff1d(proposal, original)
                next_label = "After" if raw_id in after else ("Proposal FILTERED OUT" if proposed_path.exists() else "REMOVED")
                for col, (angle, enabled) in enumerate((((18, -65), False), ((18, -65), True), ((18, 65), False), ((18, 65), True))):
                    ax = fig.add_subplot(len(block), 4, row_index * 4 + col + 1, projection="3d")
                    show = original if not enabled else (original if raw_id in after or proposed_path.exists() else [])
                    added_here = added if enabled and (raw_id in after or proposed_path.exists()) else []
                    count = len(original) if not enabled else (len(proposal) if raw_id in after or proposed_path.exists() else 0)
                    label = "Before" if not enabled else next_label
                    title = f"raw {raw_id} · {row['action']} · {label}: {count} faces\nview width {span:.2f} µm"
                    render_panel(ax, triangles, nearby, show, added_here, skel, middle, span, angle, title)
                original_file = result_path.parent / "changes" / f"raw_{raw_id:04d}_before.off"
                final_file = result_path.parent / "changes" / f"raw_{raw_id:04d}_after.off"
                if row["action"] == "unchanged_control":
                    original_file = result_path.parent / "baseline" / f"spine_{list(before).index(raw_id):03d}.off"
                    final_file = result_path.parent / "refined" / f"spine_{list(after).index(raw_id):03d}.off"
                index.append(dict(key=key, raw_id=raw_id, action=row["action"], page=page,
                                  image=str(output / f"{key}_page_{page:02d}.png"),
                                  before=str(original_file), after=str(final_file) if raw_id in after else None))
            destination = output / f"{key}_page_{page:02d}.png"
            fig.savefig(destination, dpi=140)
            plt.close(fig)
        print(json.dumps(dict(key=key, rows=len(rows))), flush=True)
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
