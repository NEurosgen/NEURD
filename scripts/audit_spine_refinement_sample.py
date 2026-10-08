#!/usr/bin/env python3
"""Draw an unbiased survivor sample and a separate size-tail inspection set.

This measures visual QC on a fixed comparison pool, not anatomical ground truth
or the error rate of an entire dataset. The sampling frame is saved before any
visual decisions. The size-tail set must never be included in prevalence counts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from preview_spine_refinement_changes import axis_samples, local_view, render_panel, unpack


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inventory(comparison):
    rows = []
    for path in sorted((comparison / "branches").glob("*/result.json")):
        result = json.loads(path.read_text())
        assert result["status"] == "passed"
        parent = trimesh.load(result["selection"]["source"], force="mesh", process=False)
        groups = unpack(path.parent / "refined_faces.npz")
        decisions = {d["raw_input_index"]: d for d in result["refinement"]["candidates"]}
        for final_index, (raw_id, ids) in enumerate(groups.items()):
            points = parent.vertices[np.unique(parent.faces[ids])]
            centered = points - points.mean(axis=0)
            try:
                from scipy.spatial import ConvexHull
                hull_volume = float(ConvexHull(centered).volume)
            except Exception:
                hull_volume = None
            decision = decisions[raw_id]
            assert decision["final_output_index"] == final_index
            rows.append(dict(key=result["key"], raw_id=raw_id, final_index=final_index,
                             stratum=result["selection"]["stratum"],
                             faces=len(ids), area_nm2=float(parent.area_faces[ids].sum()),
                             convex_hull_volume_nm3=hull_volume,
                             extent_nm=float(np.ptp(points, axis=0).max()),
                             refinement_action=decision["action"], reason=decision.get("reason"),
                             mesh=str(path.parent / "refined" / f"spine_{final_index:03d}.off")))
    return rows


def render(comparison, output, records, per_page, extra_views=False):
    index = []
    by_key = {}
    for row in records:
        by_key.setdefault(row["key"], []).append(row)
    for key, selected in by_key.items():
        branch = comparison / "branches" / key
        result = json.loads((branch / "result.json").read_text())
        parent = trimesh.load(result["selection"]["source"], force="mesh", process=False)
        groups = unpack(branch / "refined_faces.npz")
        raw = unpack(branch / "raw_faces.npz")
        centers = parent.triangles_center
        center_tree = cKDTree(centers)
        axis_tree, skeleton = axis_samples(np.load(result["selection"]["skeleton"]))
        for page, start in enumerate(range(0, len(selected), per_page)):
            block = selected[start:start + per_page]
            ncols = 4 if extra_views else 2
            fig = plt.figure(figsize=(5 * ncols, 3.0 * len(block) + .5), layout="constrained")
            fig.suptitle(key + " | final surface; grey: parent branch", fontsize=11)
            for row_index, row in enumerate(block):
                raw_id = row["raw_id"]
                final = groups[raw_id]
                local, nearby, skel, middle, span = local_view(
                    parent, centers, center_tree, axis_tree, skeleton, raw[raw_id], final)
                final_vertices = np.unique(parent.faces[final])
                final_local = local[final_vertices]
                candidate_span = float(np.ptp(final_local, axis=0).max())
                center = centers[final].mean(axis=0)
                axis_distance = float(axis_tree.query(center)[0])
                # A stored skeleton can fail to cover part of an exported
                # branch. Including a far-away axis makes the object too small
                # to inspect; crop around the surface while retaining context.
                if axis_distance > 2000 and span > candidate_span + 3:
                    middle = (final_local.min(axis=0) + final_local.max(axis=0)) / 2
                    span = max(2.4, candidate_span + 2.4)
                    points = parent.vertices[final_vertices]
                    origin = (points.min(axis=0) + points.max(axis=0)) / 2
                    radius = np.linalg.norm(points - origin, axis=1).max() + 1500
                    nearby = np.asarray(center_tree.query_ball_point(origin, radius), dtype=int)
                    skel = skel[np.linalg.norm(skel.mean(axis=1) - middle, axis=1) < span]
                triangles = local[parent.faces]
                views = ((18, -65), (18, 65), (55, 0), (5, 180)) if extra_views else ((18, -65), (18, 65))
                for col, angle in enumerate(views):
                    ax = fig.add_subplot(len(block), ncols, ncols * row_index + col + 1, projection="3d")
                    title = (f"audit {row['audit_id']} | raw {raw_id} | {row['faces']} faces\n"
                             f"width {span:.2f} µm | {row['sample_basis']}")
                    context = [] if extra_views and col == 2 else nearby
                    render_panel(ax, triangles, context, final, [], skel, middle, span, angle, title)
                image = output / f"{key}_page_{page:02d}.png"
                index.append(dict(audit_id=row["audit_id"], key=key, raw_id=raw_id,
                                  sample_basis=row["sample_basis"], image=str(image)))
            fig.savefig(image, dpi=120)
            plt.close(fig)
            print(json.dumps(dict(key=key, page=page, candidates=len(block))), flush=True)
    write_json(output / "index.json", index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--size-tail", type=int, default=24)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--render-key", action="append")
    parser.add_argument("--render-ids", type=int, nargs="+")
    parser.add_argument("--preview-subdir", default="previews")
    parser.add_argument("--extra-views", action="store_true", help="Add an isolated surface view and another parent-context angle")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_path = args.output / "sampling_frame.json"
    if frame_path.exists():
        frame = json.loads(frame_path.read_text())
        assert frame["seed"] == args.seed and frame["random_sample_size"] == args.sample
        assert frame["area_tail_size"] == args.size_tail
        assert frame["comparison"] == str(args.comparison.resolve())
        rows, selected = frame["population"], frame["selected"]
    else:
        rows = inventory(args.comparison)
        rng = np.random.default_rng(args.seed)
        random_indices = set(rng.choice(len(rows), size=min(args.sample, len(rows)), replace=False).tolist())
        tail_indices = set(sorted(range(len(rows)), key=lambda i: rows[i]["area_nm2"], reverse=True)[:args.size_tail])
        known = {("neuron_3519995546_limb_000_branch_054", 1),
                 ("neuron_2047644309_limb_002_branch_030", 107)}
        selected = []
        for i, row in enumerate(rows):
            if i in random_indices or i in tail_indices or (row["key"], row["raw_id"]) in known:
                selected.append(dict(row, audit_id=len(selected), population_index=i,
                                     sample_basis="random" if i in random_indices else
                                     ("area_tail" if i in tail_indices else "known_residual"),
                                     in_area_tail=i in tail_indices))
        frame = dict(comparison=str(args.comparison.resolve()), seed=args.seed,
                     sampling="Simple random sampling without replacement from all final outputs; separate largest-area checks.",
                     random_sample_size=len(random_indices), population_size=len(rows),
                     area_tail_size=args.size_tail,
                     anatomical_ground_truth=False,
                     script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     population=rows, selected=selected)
        write_json(frame_path, frame)
        with (args.output / "population.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.render_key:
        selected = [row for row in selected if row["key"] in args.render_key]
    if args.render_ids:
        selected = [row for row in selected if row["audit_id"] in args.render_ids]
    previews = args.output / args.preview_subdir
    previews.mkdir(exist_ok=True)
    render(args.comparison, previews, selected, args.rows, args.extra_views)


if __name__ == "__main__":
    main()
