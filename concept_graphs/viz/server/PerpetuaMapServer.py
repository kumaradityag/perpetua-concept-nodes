from typing import List, Optional, Dict
from scipy.spatial.transform import Rotation as Rsc
import numpy as np
import viser

from concept_graphs.inference.toolbox.PerpetuaMapToolbox import PerpetuaMapToolbox
from concept_graphs.viz.server.ObjectMapServer import ObjectMapServer

import logging

log = logging.getLogger(__name__)


class PerpetuaMapServer(ObjectMapServer):
    def __init__(self, toolbox: PerpetuaMapToolbox, point_shape: str = "circle"):
        super().__init__(toolbox, point_shape)

        # Containers
        self.map_query_time = None
        self.object_query_time = None
        self.selected_object_id = None
        self.reset_edges = None

        self.object_names = self.object_map.get_pickupables_name()
        self.vector_handles: Dict[str, viser.VectorHandle] = {}

        # GUI
        with self.server.gui.add_folder("Temporal Queries"):
            self.map_time_tracker_gui_number = self.server.gui.add_number(
                "Map Time (hours)",
                initial_value=0,
                disabled=True,
            )
            self.debug_vectors_gui_checkbox = self.server.gui.add_checkbox(
                "Debug Vectors",
                initial_value=False,
                disabled=False,
            )
            self.debug_vectors_gui_checkbox.on_update(
                self.on_debug_vectors_checkbox_update
            )
            with self.server.gui.add_folder("Map"):
                self.map_time_gui_number = self.server.gui.add_number(
                    "Time (hours)", initial_value=0, min=0
                )
                self.map_time_gui_button = self.server.gui.add_button(
                    "Predict at Time",
                    icon=viser.Icon.MOUSE,
                )
                self.map_time_gui_button.on_click(self.on_map_time_query_submit)

            with self.server.gui.add_folder("Object"):
                self.object_dropdown = self.server.gui.add_dropdown(
                    "Pickupable",
                    options=self.object_names,
                    initial_value=self.object_names[0],
                )
                self.object_time_gui_number = self.server.gui.add_number(
                    "Time (hours)", initial_value=0, min=0
                )
                self.object_time_gui_button = self.server.gui.add_button(
                    "Predict at Time",
                    icon=viser.Icon.MOUSE,
                )
                self.object_time_gui_button.on_click(self.on_object_time_query_submit)
            self.map_gui_reset_button = self.server.gui.add_button(
                "Reset Edges",
                icon=viser.Icon.MOUSE,
            )
            self.map_gui_reset_button.on_click(self.on_map_time_reset_click)

    def reset(self):
        super().reset()
        if hasattr(self, "map_time_tracker_gui_number"):
            self.map_time_tracker_gui_number.value = self.object_map.time
        if hasattr(self, "vector_handles"):
            self.debug_vectors_gui_checkbox.value = False
            self.clear_canonical_vectors()

    # Register here all callbacks that are resource intenseful and need to be called in the main loop
    def _callbacks(self):
        super()._callbacks()
        # Map query
        if self.map_query_time is not None:
            self.toolbox.temporal_map_query(self.map_query_time)
            self._update_server_state()
            self.map_query_time = None
        # Object query
        if self.object_query_time is not None and self.selected_object_id is not None:
            self.toolbox.temporal_object_query(
                self.selected_object_id, self.object_query_time
            )
            self.display_query_object(
                self.selected_object_id, color=np.array([255, 0, 255])
            )
            self.object_query_time = None
            self.selected_object_id = None
        # Reset edges
        if self.reset_edges:
            self.toolbox.reset_temporal_edges()
            self._update_server_state()
            self.reset_edges = None

    def on_map_time_query_submit(self, data):
        self.map_query_time = self.map_time_gui_number.value

    def on_object_time_query_submit(self, data):
        self.object_query_time = self.object_time_gui_number.value
        self.selected_object_id = self.object_dropdown.value

    def on_debug_vectors_checkbox_update(self, data):
        debug_vectors = self.debug_vectors_gui_checkbox.value
        if debug_vectors:
            self.display_canonical_vectors()
        else:
            self.clear_canonical_vectors()

    def on_map_time_reset_click(self, data):
        self.reset_edges = self.map_gui_reset_button.value

    # Scene clearing methods
    def clear_canonical_vectors(self):
        for vector in self.vector_handles.values():
            vector.remove()
        self.vector_handles = {}

    def display_canonical_vectors(self):
        self.clear_canonical_vectors()
        vectors_dict = self.toolbox.build_canonical_vectors()
        for r_name, vectors in vectors_dict.items():
            for i, offset in enumerate(vectors):
                start_point = self.object_map.get_receptacle(r_name).centroid
                end_point = start_point + offset
                line = self.server.scene.add_line_segments(
                    name=f"vectors/{r_name}_{i}",
                    points=np.array([[start_point, end_point]]),
                    colors=np.array([[[255, 0, 0], [0, 255, 0]]]),
                    line_width=4.0,
                )
                self.vector_handles[f"vectors/{r_name}_{i}"] = line

    def display_query_object(self, name: str, color: Optional[List[int]] = None):
        obj = self.object_map.get_pickupable(name)
        # Get handles of object if they exist
        object_handle = self.object_handles.get(f"objects/{name}", None)
        hitbox_handle = self.hitbox_handles.get(f"hitbox/{name}", None)
        centroid_handle = self.centroid_handles.get(f"centroid/{name}", None)
        box_handle = self.box_handles.get(f"box/{name}", None)
        label_handle = self.label_handles.get(f"label/{name}", None)

        # Common data
        points = np.asarray(obj.pcd.points)
        bbox = obj.pcd.get_oriented_bounding_box()
        wxyz = Rsc.from_matrix(np.array(bbox.R, copy=True)).as_quat(scalar_first=True)
        visibility = obj.visibility

        # Update object point cloud
        if object_handle:
            object_handle.points = points
            object_handle.colors = color
            object_handle.visible = visibility
        else:
            obj_handle = self.server.scene.add_point_cloud(
                f"objects/{name}",
                points,
                color=color,
                point_size=self.pcd_size_gui_slider.value,
                point_shape=self.point_shape,
                visible=visibility
            )
            self.object_handles.append(obj_handle)
        # Update hitbox position
        if hitbox_handle:
            hitbox_handle.position = bbox.center
            hitbox_handle.dimensions = bbox.extent.tolist()
            hitbox_handle.wxyz = wxyz
            hitbox_handle.visible = visibility
        else:
            hitbox = self.server.scene.add_box(
                name=f"hitbox/{name}",
                dimensions=bbox.extent.tolist(),
                position=bbox.center,
                wxyz=wxyz,
                color=(255, 255, 255),
                opacity=0.0,
                visible=visibility
            )
            self.hitbox_handles[f"hitbox/{name}"] = hitbox
        # Update centroid if visible
        if centroid_handle:
            centroid_handle.position = obj.centroid
            centroid_handle.visible = visibility
        # Update box if visible
        if box_handle:
            box_handle.position = bbox.center
            box_handle.dimensions = bbox.extent.tolist()
            box_handle.wxyz = wxyz
            box_handle.visible = visibility
        # Update label if visible
        if label_handle:
            label_handle.position = obj.centroid
            label_handle.text = name
            label_handle.visible = visibility