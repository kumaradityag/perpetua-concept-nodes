from typing import Dict, List
import numpy as np
import viser

import open3d as o3d
from scipy.spatial.transform import Rotation as Rsc

from concept_graphs.utils import split_camel_preserve_acronyms
from concept_graphs.inference.toolbox.ObjectMapToolbox import ObjectMapToolbox
from concept_graphs.viz.server.ObjectMapServer import ObjectMapServer

import logging

log = logging.getLogger(__name__)

COLORS = {
    "receptacle": np.array([0, 255, 255]),
    "pickupable_present": np.array([50, 255, 50]),
    "pickupable_absent": np.array([255, 40, 40]),
    "map_receptacle": np.array([180, 50, 255]),
    "map_pickupable": np.array([255, 165, 0]),
}


class VerifierServer(ObjectMapServer):
    def __init__(
        self,
        pickupables_bbox: Dict,
        receptacles_bbox: Dict,
        receptacle_map_ids: Dict,
        pickupable_existence: Dict,
        assignments: Dict,
        toolbox: ObjectMapToolbox,
        point_shape: str = "circle",
    ):
        super().__init__(toolbox, point_shape)

        # Data
        self.pickupables_bbox = pickupables_bbox
        self.receptacles_bbox = receptacles_bbox
        self.receptacle_map_ids = receptacle_map_ids
        self.pickupables_existence = pickupable_existence
        self.assignments = assignments

        # Handles
        self.gt_receptacles_handles: Dict[str, viser.BoxHandle] = {}
        self.gt_pickupables_handles: Dict[str, viser.BoxHandle] = {}
        self.pickupables_handles: Dict[str, viser.BoxHandle] = {}
        self.receptacles_handles: Dict[str, viser.BoxHandle] = {}
        self.r2r_associations_handles: Dict[str, viser.LineHandle] = {}
        self.p2p_associations_handles: Dict[str, viser.LineHandle] = {}
        self.p2r_associations_handles: Dict[str, viser.LineHandle] = {}
        # GUI
        with self.tab_group.add_tab("Verifier"):
            with self.server.gui.add_folder("Structure (Receptacles)"):
                # Group GT, Pred, and Assoc together so you can toggle them in sequence
                self.gt_receptacle_checkbox = self.server.gui.add_checkbox(
                    "Show GT (🟦)", 
                    initial_value=False
                )
                self.receptacles_checkbox = self.server.gui.add_checkbox(
                    "Show Pred (🟪)", 
                    initial_value=False
                )
                # Indent or separate the association to show it connects the two above
                self.r2r_associations_checkbox = self.server.gui.add_checkbox(
                    "Show Match (🟦 → 🟪)", 
                    initial_value=False
                )

                # Wire up callbacks
                self.gt_receptacle_checkbox.on_update(self.on_gt_receptacles_checkbox_change)
                self.receptacles_checkbox.on_update(self.on_receptacles_checkbox_change)
                self.r2r_associations_checkbox.on_update(self.on_r2r_associations_checkbox_change)

            with self.server.gui.add_folder("Objects (Pickupables)"):
                self.gt_pickupables_checkbox = self.server.gui.add_checkbox(
                    "Show GT (🟩/🟥)", 
                    initial_value=False
                )
                self.pickupables_checkbox = self.server.gui.add_checkbox(
                    "Show Pred (🟧)", 
                    initial_value=False
                )
                self.p2p_associations_checkbox = self.server.gui.add_checkbox(
                    "Show Match (🟩 → 🟧)", 
                    initial_value=False
                )

                # Wire up callbacks
                self.gt_pickupables_checkbox.on_update(self.on_gt_pickupables_checkbox_change)
                self.pickupables_checkbox.on_update(self.on_pickupables_checkbox_change)
                self.p2p_associations_checkbox.on_update(self.on_p2p_associations_checkbox_change)

            # --- GROUP 3: RELATIONSHIPS  ---
            with self.server.gui.add_folder("Pickupable to Receptacle"):
                # This is high-level logic, so it gets its own section
                self.p2r_associations_checkbox = self.server.gui.add_checkbox(
                    "Show Match (🟧 → 🟪)", 
                    initial_value=False
                )
            
                # Wire up callback
                self.p2r_associations_checkbox.on_update(self.on_p2r_associations_checkbox_change)

    def on_gt_receptacles_checkbox_change(self, data):
        show_boxes = self.gt_receptacle_checkbox.value
        if show_boxes:
            self.display_gt_receptacles()
        else:
            self.clear_gt_receptacles()

    def on_gt_pickupables_checkbox_change(self, data):
        show_boxes = self.gt_pickupables_checkbox.value
        if show_boxes:
            self.display_gt_pickupables()
        else:
            self.clear_gt_pickupables()

    def on_receptacles_checkbox_change(self, data):
        show_boxes = self.receptacles_checkbox.value
        if show_boxes:
            self.display_receptacles()
        else:
            self.clear_receptacles()

    def on_pickupables_checkbox_change(self, data):
        show_boxes = self.pickupables_checkbox.value
        if show_boxes:
            self.display_pickupables()
        else:
            self.clear_pickupables()

    def on_r2r_associations_checkbox_change(self, data):
        show_associations = self.r2r_associations_checkbox.value
        if show_associations:
            self.display_r2r_associations()
        else:
            self.clear_r2r_associations()

    def on_p2p_associations_checkbox_change(self, data):
        show_associations = self.p2p_associations_checkbox.value
        if show_associations:
            self.display_p2p_associations()
        else:
            self.clear_p2p_associations()

    def on_p2r_associations_checkbox_change(self, data):
        show_associations = self.p2r_associations_checkbox.value
        if show_associations:
            self.display_p2r_associations()
        else:
            self.clear_p2r_associations()

    # Clearing functions
    def clear_gt_receptacles(self):
        for bbox in self.gt_receptacles_handles.values():
            bbox.remove()
        self.gt_receptacles_handles = {}

    def clear_gt_pickupables(self):
        for bbox in self.gt_pickupables_handles.values():
            bbox.remove()
        self.gt_pickupables_handles = {}

    def clear_pickupables(self):
        for bbox in self.pickupables_handles.values():
            bbox.remove()
        self.pickupables_handles = {}

    def clear_receptacles(self):
        for bbox in self.receptacles_handles.values():
            bbox.remove()
        self.receptacles_handles = {}

    def clear_r2r_associations(self):
        for line in self.r2r_associations_handles.values():
            line.remove()
        self.r2r_associations_handles = {}

    def clear_p2p_associations(self):
        for line in self.p2p_associations_handles.values():
            line.remove()
        self.p2p_associations_handles = {}

    def clear_p2r_associations(self):
        for line in self.p2r_associations_handles.values():
            line.remove()
        self.p2r_associations_handles = {}

    # Engine
    def _create_oobb_from_corners(
        self, corners: np.ndarray, name: str, color: np.ndarray
    ) -> viser.BoxHandle:
        """Create an OrientedBoundingBox from 8 corner points and return a VisER BoxHandle."""
        # create_from_points computes the tightest box around these 8 points
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(corners)
        )

        # Extract box parameters
        center = bbox.center
        extent = bbox.extent
        R = np.array(bbox.R, copy=True)
        wxyz = Rsc.from_matrix(R).as_quat(scalar_first=True)

        # Create and return VisER BoxHandle
        box_handle = self.server.scene.add_box(
            name=name,
            color=color,
            dimensions=extent,
            position=center,
            wxyz=wxyz,
            wireframe=True,
        )
        return box_handle

    # Display functions
    def display_gt_receptacles(self):
        self.clear_gt_receptacles()

        for box_name, r_data in self.receptacles_bbox.items():
            corners = np.array(r_data["cornerPoints"], dtype=np.float64)
            box = self._create_oobb_from_corners(
                corners, f"gt/{box_name}", color=COLORS["receptacle"]
            )
            self.gt_receptacles_handles[f"gt/{box_name}"] = box

    def display_gt_pickupables(self):
        self.clear_gt_pickupables()

        for p_name, p_data in self.pickupables_bbox.items():
            corners = np.array(p_data["cornerPoints"], dtype=np.float64)
            color = (
                COLORS["pickupable_present"]
                if self.pickupables_existence[p_name]
                else COLORS["pickupable_absent"]
            )
            box = self._create_oobb_from_corners(corners, f"gt/{p_name}", color=color)
            self.gt_pickupables_handles[f"gt/{p_name}"] = box

    def display_receptacles(self):
        self.clear_receptacles()
        for r_name, map_id in self.receptacle_map_ids.items():
            if map_id is None:
                continue
            bbox = self.object_map[map_id].pcd.get_oriented_bounding_box()
            box = self._create_oobb_from_corners(
                bbox.get_box_points(),
                f"association/{r_name}",
                color=COLORS["map_receptacle"],
            )
            self.receptacles_handles[f"association/{r_name}"] = box

    def display_pickupables(self):
        self.clear_pickupables()
        for p_name, data in self.assignments.items():
            if data["present"]:
                p_id = data["map_object_id"]
                bbox = self.object_map[p_id].pcd.get_oriented_bounding_box()
                box = self._create_oobb_from_corners(
                    bbox.get_box_points(),
                    f"association/{p_name}",
                    color=COLORS["map_pickupable"],
                )
                self.pickupables_handles[f"association/{p_name}"] = box

    def display_r2r_associations(self):
        self.clear_r2r_associations()
        for r_name, map_id in self.receptacle_map_ids.items():
            if map_id is None:
                continue
            r_bbox = self.object_map[map_id].pcd.get_oriented_bounding_box()
            r_center = r_bbox.center
            gt_corners = self.receptacles_bbox.get(r_name, None)["cornerPoints"]
            gt_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(np.array(gt_corners, dtype=np.float64))
            )
            gt_center = gt_bbox.center
            line = self.server.scene.add_line_segments(
                name=f"r2r/association/{r_name}_to_{map_id}",
                points=np.array([[gt_center, r_center]]),
                colors=np.array([[COLORS["receptacle"], COLORS["map_receptacle"]]]),
                line_width=4.0,
            )
            self.r2r_associations_handles[f"r2r/association/{r_name}_to_{map_id}"] = line
            # Add label
            midpoint = (gt_center + r_center) / 2

            self.server.scene.add_label(
                name=f"r2r/association/{r_name}_to_{map_id}/label",
                text=f"{split_camel_preserve_acronyms(r_name.split('|')[0])} → {map_id}",
                position=midpoint,
            )

    def display_p2p_associations(self):
        self.clear_p2p_associations()
        for p_name, data in self.assignments.items():
            if data["present"]:
                p_id = data["map_object_id"]
                bbox = self.object_map[p_id].pcd.get_oriented_bounding_box()
                p_center = bbox.center
                gt_corners = self.pickupables_bbox.get(p_name, None)["cornerPoints"]
                gt_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(np.array(gt_corners, dtype=np.float64))
                )
                gt_center = gt_bbox.center
                line = self.server.scene.add_line_segments(
                    name=f"p2p/association/{p_name}_to_{p_id}",
                    points=np.array([[gt_center, p_center]]),
                    colors=np.array(
                        [[COLORS["pickupable_present"], COLORS["map_pickupable"]]]
                    ),
                    line_width=4.0,
                )
                self.p2p_associations_handles[f"p2p/association/{p_name}_to_{p_id}"] = line
                # Add label
                midpoint = (gt_center + p_center) / 2

                self.server.scene.add_label(
                    name=f"p2p/association/{p_name}_to_{p_id}/label",
                    text=f"{split_camel_preserve_acronyms(p_name.split('|')[0])} → {p_id}",
                    position=midpoint,
                )

    def display_p2r_associations(self):
        self.clear_p2r_associations()
        for p_name, data in self.assignments.items():
            if data["present"]:
                p_id = data["map_object_id"]
                p_bbox = self.object_map[p_id].pcd.get_oriented_bounding_box()
                p_center = p_bbox.center
                r_name = data["present_receptacle_name"]
                r_id = self.receptacle_map_ids.get(r_name, None)
                if r_id is None:
                    continue
                r_bbox = self.object_map[r_id].pcd.get_oriented_bounding_box()
                r_center = r_bbox.center

                line = self.server.scene.add_line_segments(
                    name=f"p2r/association/{p_name}_to_{r_name}",
                    points=np.array([[p_center, r_center]]),
                    colors=np.array(
                        [[COLORS["map_pickupable"], COLORS["map_receptacle"]]]
                    ),
                    line_width=4.0,
                )
                self.p2r_associations_handles[f"p2r/association/{p_name}_to_{r_name}"] = line
                # Add label
                midpoint = (p_center + r_center) / 2

                self.server.scene.add_label(
                    name=f"p2r/association/{p_name}_to_{r_name}/label",
                    text=f"{split_camel_preserve_acronyms(p_name.split('|')[0])} → {split_camel_preserve_acronyms(r_name.split('|')[0])}",
                    position=midpoint,
                )
