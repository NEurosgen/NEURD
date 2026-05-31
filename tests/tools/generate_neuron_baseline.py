"""Generate the golden Neuron-output baseline JSON from the current pipeline.

Runs the slim segmentation pipeline on the committed fixture mesh and writes the
extracted metrics to tests/fixtures/neuron_baseline.json. Re-run this only when an
output change is DELIBERATELY accepted as the new reference (and review the diff).

    python tests/tools/generate_neuron_baseline.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "tools"))

import neurd  # noqa: F401
from neurd import parameters
from neurd.segmentation_pipeline import segmentation_pipeline
from mesh_tools import trimesh_utils as tu

from neuron_metrics import extract_metrics

_FIXTURE = REPO / "tests" / "fixtures" / "864691135510518224.off"
_OUT = REPO / "tests" / "fixtures" / "neuron_baseline.json"


def main():
    os.chdir(tempfile.mkdtemp(prefix="neuron_baseline_"))
    parameters.params.use("microns")
    mesh = tu.load_mesh_no_processing(str(_FIXTURE))
    neuron_obj = segmentation_pipeline(mesh, verbose=False)
    metrics = extract_metrics(neuron_obj)
    _OUT.write_text(json.dumps(metrics, indent=2))
    print("BASELINE METRICS:")
    print(json.dumps(metrics, indent=2))
    print(f"\nwritten -> {_OUT}")


if __name__ == "__main__":
    main()
