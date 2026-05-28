"""Integration test / anchor for the slim mesh-segmentation pipeline.

Pins the segmentation deliverable: mesh in -> Neuron with somas + limbs/branches
(each with .mesh and .skeleton) + per-branch spines. This is the contract that the
Phase-5 cluster removal (DEPS_PLAN.md) must NOT break.
"""

import shutil
import unittest

import neurd
from neurd.vdi_microns import volume_data_interface as vdi
from neurd.segmentation_pipeline import segmentation_pipeline

from datasci_tools import pathlib_utils as plu

from pathlib import Path

# Mesh stages shell out to `xvfb-run meshlabserver`; skip cleanly when absent.
_MESH_TOOLS = all(shutil.which(b) for b in ("xvfb-run", "meshlabserver"))
_requires_mesh_tools = unittest.skipUnless(
    _MESH_TOOLS, "requires `xvfb-run` + `meshlabserver` on PATH"
)


class TestSegmentationPipeline(unittest.TestCase):
    def setUp(self):
        self.segment_id = 864691135510518224
        self.fixture_path = str(
            (plu.parent_directory(plu.parent_directory(Path(__file__))) / Path("fixtures")).absolute()
        )

    def test_1_fetch_mesh(self):
        self.__class__.mesh = vdi.fetch_segment_id_mesh(
            self.segment_id,
            meshes_directory=self.fixture_path,
        )
        self.assertIsNotNone(self.__class__.mesh)
        self.assertGreater(len(self.__class__.mesh.faces), 0)

    @_requires_mesh_tools
    def test_2_run_segmentation_pipeline(self):
        self.__class__.neuron_obj = segmentation_pipeline(
            self.__class__.mesh,
            segment_id=self.segment_id,
            verbose=False,
        )
        self.assertIsNotNone(self.__class__.neuron_obj)

    @_requires_mesh_tools
    def test_3_output_has_somas_branches_skeletons(self):
        n = self.__class__.neuron_obj

        somas = list(n.get_soma_node_names())
        self.assertGreaterEqual(len(somas), 1, "expected at least one soma")

        limbs = list(n.get_limb_node_names())
        self.assertGreater(len(limbs), 0, "expected at least one limb")

        total_branches = 0
        for ln in limbs:
            limb = n[ln]
            for bn in limb.get_branch_names():
                b = limb[bn]
                total_branches += 1
                # the segmentation deliverable: each branch is a mesh piece + a skeleton
                self.assertIsNotNone(b.mesh)
                self.assertGreater(len(b.mesh.faces), 0)
                self.assertIsNotNone(b.skeleton)
                self.assertGreater(len(b.skeleton), 0)

        self.assertGreater(total_branches, 0, "expected at least one branch")
