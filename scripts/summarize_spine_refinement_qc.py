#!/usr/bin/env python3
"""Summarize a frozen random visual-QC sample without mixing targeted checks.

Dropping flagged objects is a sensitivity calculation, not a correction: merged
objects include real spines. Volume methods include validated convex-hull
approximations, which are especially sensitive to segmentation contamination.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DEFINITE = {"shaft_surface_patch", "spine_shaft_merge", "multiple_spines_with_shaft"}
UNCERTAIN = {"short_protrusion_uncertain", "possible_spine_shaft_merge",
             "possible_multiple_spines_with_shaft"}


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    delta = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [float(center - delta), float(center + delta)]


def sensitivity(rows, classes):
    bad = [r for r in rows if r["classification"] in classes]
    result = dict(flagged=len(bad), total=len(rows))
    for field in ("area_nm2", "volume_nm3"):
        usable = [r for r in rows if r[field] is not None and np.isfinite(r[field])]
        flagged = [r for r in usable if r["classification"] in classes]
        total = sum(r[field] for r in usable)
        bad_total = sum(r[field] for r in flagged)
        retained = len(usable) - len(flagged)
        result[field] = dict(valid_measurements=len(usable),
                             flagged_measurement_fraction=bad_total / total,
                             relative_mean_change_if_flagged_objects_omitted=
                             ((total - bad_total) / retained) / (total / len(usable)) - 1)
    return result


def stats(rows):
    counts = Counter(r["classification"] for r in rows)
    k = sum(counts[x] for x in DEFINITE)
    uncertain = sum(counts[x] for x in UNCERTAIN)
    return dict(n=len(rows), classifications=dict(counts), definite_visual_artifacts=k,
                definite_fraction=k / len(rows),
                descriptive_wilson_95=wilson(k, len(rows)),
                anatomically_or_geometrically_uncertain=uncertain,
                uncertain_fraction=uncertain / len(rows),
                pure_surface_patch_sensitivity=sensitivity(rows, {"shaft_surface_patch"}),
                all_definite_artifact_sensitivity=sensitivity(rows, DEFINITE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    frame = json.loads((args.audit / "sampling_frame.json").read_text())
    annotations = json.loads((args.audit / "visual_annotations.json").read_text())
    measurements = json.loads((args.audit / "measurements.json").read_text())
    assert hashlib.sha256((args.audit / "sampling_frame.json").read_bytes()).hexdigest() == measurements["sampling_frame_sha256"]
    selected = {r["audit_id"]: r for r in frame["selected"]}
    review = {r["audit_id"]: r for r in annotations["reviews"]}
    measured = {(r["key"], r["raw_id"]): r for r in measurements["candidates"]}
    assert len(selected) == len(review) == len(frame["selected"])
    assert set(selected) == set(review)
    # Reconstruct the sampling draw independently from its stored RNG seed.
    drawn = set(np.random.default_rng(frame["seed"]).choice(
        frame["population_size"], frame["random_sample_size"], replace=False).tolist())
    assert drawn == {r["population_index"] for r in selected.values() if r["sample_basis"] == "random"}
    rows = []
    for i, record in selected.items():
        m = measured[record["key"], record["raw_id"]]
        assert np.isclose(record["area_nm2"], m["area_nm2"], rtol=1e-8)
        volume = m["volume"]
        rows.append(dict(audit_id=i, key=record["key"], raw_id=record["raw_id"],
                         stratum=record["stratum"], sample_basis=record["sample_basis"],
                         in_area_tail=record["in_area_tail"],
                         classification=review[i]["classification"],
                         additional_views_reviewed=review[i].get("additional_views_reviewed", False),
                         area_nm2=m["area_nm2"], volume_nm3=volume["volume_nm3"],
                         volume_method=volume["method"], centroid_axis_distance_nm=m["centroid_axis_distance_nm"],
                         mesh=record["mesh"]))
    random = [r for r in rows if r["sample_basis"] == "random"]
    assert len(random) == frame["random_sample_size"]
    source = json.loads((args.audit / "source_inventory.json").read_text())
    axis_coverage = []
    for key in sorted({r["key"] for r in measurements["candidates"]}):
        branch = [r for r in measurements["candidates"] if r["key"] == key]
        distance = np.array([r["centroid_axis_distance_nm"] for r in branch])
        axis_coverage.append(dict(key=key, spines=len(branch),
                                  median_centroid_distance_nm=float(np.median(distance)),
                                  p95_centroid_distance_nm=float(np.quantile(distance, .95)),
                                  centroids_farther_than_5000nm=int((distance > 5000).sum())))
    summary = dict(
        population_size=frame["population_size"],
        branches=len({r["key"] for r in frame["population"]}),
        neurons=len({r["key"].split("_limb_")[0] for r in frame["population"]}),
        sampling="Simple random sample without replacement from final survivors in a fixed branch comparison; branch-selection provenance remains in the comparison's selection.json.",
        seed=frame["seed"], random=stats(random),
        strata={s: stats([r for r in random if r["stratum"] == s])
                for s in sorted({r["stratum"] for r in random})},
        separate_targeted_checks=dict(
            extra_candidates=len(rows) - len(random),
            size_tail_candidates=sum(r["in_area_tail"] for r in rows),
            size_tail_classifications=dict(Counter(r["classification"] for r in rows if r["in_area_tail"])),
            known_residual_candidates=sum(r["sample_basis"] == "known_residual" for r in rows)),
        all_reviewed=len(rows), additional_view_reviews=sum(r["additional_views_reviewed"] for r in rows),
        volume_methods_in_pool=measurements["volume_methods"],
        axis_coverage=axis_coverage,
        source_inventory=source,
        limitations=[
            "Visual mesh classifications are not independent anatomical/EM ground truth.",
            "The high-count branch selection is not a random sample of all H01 branches; percentages must not be extrapolated to the complete dataset.",
            "Wilson intervals describe sample variability within the fixed pool and do not account for observer error or branch selection bias; sampling without replacement makes this approximation conservative.",
            "Flagged volume/area fractions include entire merged objects, including their real spine parts. They are not the excess volume/area attributable to shaft contamination.",
            "Mean changes after omission are sensitivity calculations, not estimates of corrected biological measurements.",
            "Possible incomplete heads/necks were noted incidentally and are not a complete head/neck accuracy audit.",
        ],
        bulk_h01_export_started=False,
        bulk_decision="Deferred: the condition of rare residual artifacts with negligible morphometric influence is not supported by this sample.",
    )
    (args.audit / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.audit / "reviewed_candidates.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(random=summary["random"], strata=summary["strata"]), indent=2))


if __name__ == "__main__":
    main()
