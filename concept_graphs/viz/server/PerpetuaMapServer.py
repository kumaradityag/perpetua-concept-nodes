from typing import List, Optional
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

        self.object_names = self.object_map.get_pickupables_name()

        # GUI
        with self.server.gui.add_folder("Temporal Queries"):
            self.map_time_tracker_gui_number = self.server.gui.add_number(
                "Map Time (hours)",
                initial_value=0,
                disabled=True,
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
                "Reset Map Time (TODO)",
                icon=viser.Icon.MOUSE,
            )
            self.map_gui_reset_button.on_click(self.on_map_time_reset_click)

    def reset(self):
        super().reset()
        if hasattr(self, "map_time_tracker_gui_number"):
            self.map_time_tracker_gui_number.value = self.object_map.time

    # Register here all callbacks that are resource intenseful and need to be called in the main loop
    def _callbacks(self):
        super()._callbacks()
        # Map query
        if self.map_query_time is not None:
            self.toolbox.temporal_map_query(self.map_query_time)
            self._update_server_state()
            self.display_object_rgb()
            self.map_query_time = None
        # Object query
        if self.object_query_time is not None and self.selected_object_id is not None:
            self.toolbox.temporal_object_query(
                self.selected_object_id, self.object_query_time
            )
            # self._update_server_state()
            self.display_query_object(self.selected_object_id, color=np.array([255, 0, 255]))
            self.object_query_time = None
            self.selected_object_id = None

    def on_map_time_query_submit(self, data):
        self.map_query_time = self.map_time_gui_number.value

    def on_object_time_query_submit(self, data):
        self.object_query_time = self.object_time_gui_number.value
        self.selected_object_id = self.object_dropdown.value

    def on_map_time_reset_click(self, data):
        # TODO: implement me
        pass

    def display_query_object(self, name: str, color: Optional[List[int]] = None):
        obj = self.object_map.get_pickupable(name)
        # Get handles of object if they exist
        object_handle = self._get_handle(self.object_handles, f"objects/{name}")
        hitbox_handle = self._get_handle(self.hitbox_handles, f"hitbox/{name}")
        centroid_handle = self._get_handle(self.centroid_handles, f"centroid/{name}")
        box_handle = self._get_handle(self.box_handles, f"box/{name}")
        label_handle = self._get_handle(self.label_handles, f"label/{name}")

        # Common data
        points = np.asarray(obj.pcd.points)
        bbox = obj.pcd.get_oriented_bounding_box()
        wxyz = Rsc.from_matrix(np.array(bbox.R, copy=True)).as_quat(scalar_first=True)

        # Update object point cloud
        if object_handle:
            object_handle.points = points
            object_handle.colors = color
        else:
            obj_handle = self.server.scene.add_point_cloud(
                f"objects/{name}",
                points,
                color=color,
                point_size=self.pcd_size_gui_slider.value,
                point_shape=self.point_shape,
            )
            self.object_handles.append(obj_handle)
        # Update hitbox position
        if hitbox_handle:
            hitbox_handle.position = bbox.center
            hitbox_handle.dimensions = bbox.extent.tolist()
            hitbox_handle.wxyz = wxyz
        else:
            hitbox = self.server.scene.add_box(
                name=f"hitbox/{name}",
                dimensions=bbox.extent.tolist(),
                position=bbox.center,
                wxyz=wxyz,
                color=(255, 255, 255),
                opacity=0.0,
            )
            self.hitbox_handles.append(hitbox)
        # Update centroid if visible
        if centroid_handle:
            centroid_handle.position = obj.centroid
        # Update box if visible
        if box_handle:
            box_handle.position = bbox.center
            box_handle.dimensions = bbox.extent.tolist()
            box_handle.wxyz = wxyz
        # Update label if visible
        if label_handle:
            label_handle.position = obj.centroid
            label_handle.text = name
