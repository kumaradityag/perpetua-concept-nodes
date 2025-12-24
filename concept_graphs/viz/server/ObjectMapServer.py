from typing import List, Optional, Union, Dict
from scipy.spatial.transform import Rotation as Rsc
import numpy as np
import viser
import matplotlib.pyplot as plt

from concept_graphs.viz.utils import similarities_to_rgb
from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.mapping.Object import Object
from concept_graphs.inference.toolbox.ObjectMapToolbox import ObjectMapToolbox

import logging

log = logging.getLogger(__name__)


class ObjectMapServer:
    def __init__(self, toolbox: ObjectMapToolbox, point_shape: str = "circle"):
        self.server = viser.ViserServer()
        self.point_shape = point_shape
        self.toolbox = toolbox

        # Handles
        self.object_handles: Dict[str, viser.PointCloudHandle] = {}
        self.hitbox_handles: Dict[str, viser.PointCloudHandle] = {}
        self.centroid_handles: Dict[str, viser.SphereHandle] = {}
        self.box_handles: Dict[str, viser.BoxHandle] = {}
        self.label_handles: Dict[str, viser.LabelHandle] = {}
        self.current_image_handle: Optional[viser.GuiImageHandle] = None

        # Containers
        self.clip_query: str = ""

        # GUI
        self.tab_group = self.server.gui.add_tab_group()
        with self.tab_group.add_tab("Object Map"):
            with self.server.gui.add_folder("Point Cloud"):
                self.obj_counter_gui_button = self.server.gui.add_number(
                    "# Objects",
                    initial_value=0,
                    disabled=True,
                )
                self.pcd_rgb_gui_button = self.server.gui.add_button(
                    "RGB",
                    icon=viser.Icon.MOUSE,
                )
                self.pcd_rgb_gui_button.on_click(self.on_rgb_button_click)

                self.pcd_segmentation_gui_button = self.server.gui.add_button(
                    "Segmentation",
                    icon=viser.Icon.MOUSE,
                )
                self.pcd_segmentation_gui_button.on_click(self.on_segmentation_button_click)

                self.pcd_centroid_gui_checkbox = self.server.gui.add_checkbox(
                    "Centroid",
                    initial_value=False,
                )
                self.pcd_centroid_gui_checkbox.on_update(self.on_centroid_checkbox_change)

                self.pcd_boxes_gui_checkbox = self.server.gui.add_checkbox(
                    "Boxes",
                    initial_value=False,
                )
                self.pcd_boxes_gui_checkbox.on_update(self.on_boxes_checkbox_change)

                self.pcd_labels_gui_checkbox = self.server.gui.add_checkbox(
                    "Labels",
                    initial_value=False,
                )
                self.pcd_labels_gui_checkbox.on_update(self.on_labels_checkbox_change)

                self.pcd_size_gui_slider = self.server.gui.add_slider(
                    "Point size",
                    min=0.001,
                    max=0.030,
                    step=0.001,
                    initial_value=0.010,
                    disabled=False,
                )
                self.pcd_size_gui_slider.on_update(self.on_point_size_change)

            with self.server.gui.add_folder("CLIP Query"):
                self.clip_gui_text = self.server.gui.add_text(
                    "Query",
                    initial_value="",
                )
                self.clip_gui_button = self.server.gui.add_button(
                    "CLIP Similarities",
                    icon=viser.Icon.MOUSE,
                )
                self.clip_gui_button.on_click(self.on_clip_query_submit)

        self._update_server_state()

    @property
    def object_map(self) -> ObjectMap:
        return self.toolbox.object_map

    def update_object_map(self, object_map: ObjectMap):
        self.toolbox.update_object_map(object_map)
        self._update_server_state()

    def _update_server_state(self):
        rng = np.random.default_rng(42)
        self.segmentation_colors = rng.random((len(self.object_map), 3))
        self.reset()

    def reset(self):
        # self.server.scene.reset()
        self.obj_counter_gui_button.value = len(self.object_map)
        self.pcd_centroid_gui_checkbox.value = False
        self.pcd_boxes_gui_checkbox.value = False
        self.pcd_labels_gui_checkbox.value = False
        # self.hitbox_handles = {}
        # self.object_handles = {}
        self.centroid_handles = {}
        self.box_handles = {}
        self.label_handles = {}
        self.clear_image()
        self.display_object_rgb()

    def spin(self):
        while True:
            self._callbacks()

    # Gui callbacks and inputs
    def on_clip_query_submit(self, data):
        self.clip_query = self.clip_gui_text.value

    def on_rgb_button_click(self, data):
        self.display_object_rgb()

    def on_segmentation_button_click(self, data):
        self.display_object_segmentation()

    def on_centroid_checkbox_change(self, data):
        show_centroid = self.pcd_centroid_gui_checkbox.value
        if show_centroid:
            self.display_object_centroids()
        else:
            self.clear_centroids()

    def on_boxes_checkbox_change(self, data):
        show_boxes = self.pcd_boxes_gui_checkbox.value
        if show_boxes:
            self.display_boxes()
        else:
            self.clear_boxes()

    def on_labels_checkbox_change(self, data):
        show_labels = self.pcd_labels_gui_checkbox.value
        if show_labels:
            self.display_labels()
        else:
            self.clear_labels()

    def on_point_size_change(self, data):
        point_size = self.pcd_size_gui_slider.value

        for handle in self.object_handles.values():
            handle.point_size = point_size

    # Scene clearing methods
    def clear_main_objects(self):
        for hitbox, pcd in zip(
            self.hitbox_handles.values(), self.object_handles.values()
        ):
            hitbox.remove()
            pcd.remove()
        self.hitbox_handles = {}
        self.object_handles = {}

    def clear_centroids(self):
        for sphere in self.centroid_handles.values():
            sphere.remove()
        self.centroid_handles = {}

    def clear_boxes(self):
        for box in self.box_handles.values():
            box.remove()
        self.box_handles = {}

    def clear_labels(self):
        for label in self.label_handles.values():
            label.remove()
        self.label_handles = {}

    def clear_image(self):
        if self.current_image_handle is not None:
            self.current_image_handle.remove()
            self.current_image_handle = None

    # Scene display methods
    def display_object_point_clouds(self, colors: np.ndarray = None):
        for i, obj in enumerate(self.object_map):
            name = obj.name if obj.name is not None else i
            obj_key = f"objects/{name}"
            hit_key = f"hitbox/{name}"

            pcd = obj.pcd
            pcd_points = np.asarray(pcd.points)
            visibility = obj.visibility

            # Color handling
            if colors is not None:
                # Broadcast single color to all points
                pcd_colors = np.tile(colors[i], (pcd_points.shape[0], 1))
            else:
                pcd_colors = np.asarray(pcd.colors)

            bbox = pcd.get_oriented_bounding_box()
            wxyz = Rsc.from_matrix(np.array(bbox.R, copy=True)).as_quat(
                scalar_first=True
            )

            # If handles exist
            if obj_key in self.object_handles:
                # Update Point Cloud
                self.object_handles[obj_key].points = pcd_points
                self.object_handles[obj_key].colors = pcd_colors

                # Update Hitbox
                hit_handle = self.hitbox_handles[hit_key]
                hit_handle.position = bbox.center
                hit_handle.dimensions = bbox.extent.tolist()
                hit_handle.wxyz = wxyz
                hit_handle.visible = visibility
            else:
                # Add Point Cloud
                self.object_handles[obj_key] = self.server.scene.add_point_cloud(
                    obj_key,
                    pcd_points,
                    pcd_colors,
                    point_size=self.pcd_size_gui_slider.value,
                    point_shape=self.point_shape,
                    visible=visibility,
                )
                # Add Hitbox
                hitbox_handle = self.server.scene.add_box(
                    name=hit_key,
                    position=bbox.center,
                    dimensions=bbox.extent,
                    wxyz=wxyz,
                    color=(255, 255, 255),
                    opacity=0.0,
                    visible=visibility,
                )
                # Add Click Event
                hitbox_handle.on_click(lambda _, idx=name: self.on_object_clicked(idx))
                self.hitbox_handles[hit_key] = hitbox_handle

    def display_object_centroids(self):
        # Clear previous spheres
        self.clear_centroids()

        for i, obj in enumerate(self.object_map):
            name = obj.name if obj.name is not None else i
            centroid = obj.centroid
            sphere = self.server.scene.add_icosphere(
                name=f"centroid/{name}",
                color=(255, 0, 0),
                radius=0.03,
                position=centroid,
                visible=obj.visibility
            )
            self.centroid_handles[f"centroid/{name}"] = sphere

    def display_boxes(self):
        # Clear existing boxes
        self.clear_boxes()

        for i, obj in enumerate(self.object_map):
            name = obj.name if obj.name is not None else i
            bbox = obj.pcd.get_oriented_bounding_box()
            center = bbox.center
            extent = bbox.extent
            wxyz = Rsc.from_matrix(np.array(bbox.R, copy=True)).as_quat(
                scalar_first=True
            )
            box = self.server.scene.add_box(
                name=f"box/{name}",
                color=(0, 1, 0),
                dimensions=extent,
                position=center,
                wxyz=wxyz,
                wireframe=True,
                visible=obj.visibility
            )
            self.box_handles[f"box/{name}"] = box

    def display_labels(self):
        self.clear_labels()

        for i, obj in enumerate(self.object_map):
            name = obj.name if obj.name is not None else i
            centroid = obj.centroid
            label_text = obj.name if obj.name is not None else f"Object {i}"
            label_handle = self.server.scene.add_label(
                name=f"label/{name}",
                text=label_text,
                position=centroid,
                visible=obj.visibility
            )
            self.label_handles[f"label/{name}"] = label_handle

    def on_object_clicked(self, index: Union[int, str]):
        """Callback to display the image associated with the clicked object."""
        self.clear_image()

        obj = self.object_map[index]
        # Generate the image from the object's views
        obj.view_images_caption()
        fig = plt.gcf()
        fig.canvas.draw()
        # Convert the Matplotlib figure to a NumPy array
        image_data = np.asarray(fig.canvas.renderer.buffer_rgba())
        label_text = (
            f"{obj.name} View" if obj.name is not None else f"Object {index} View"
        )

        # Add the new image to the top of the GUI sidebar
        self.current_image_handle = self.server.gui.add_image(
            image=image_data,
            label=label_text,
        )
        plt.close(fig)

    def display_object_rgb(self):
        self.display_object_point_clouds()

    def display_object_segmentation(self):
        self.display_object_point_clouds(colors=self.segmentation_colors)

    def display_object_similarity(self, similarity_scores: np.ndarray):
        self.display_object_point_clouds(
            colors=similarities_to_rgb(similarity_scores, cmap_name="viridis")
        )

    # Register here all callbacks that are resource intenseful and need to be called in the main loop
    def _callbacks(self):
        if self.clip_query != "":
            sim_query = self.toolbox.clip_query(self.clip_query)
            self.display_object_similarity(sim_query)
            self.clip_query = ""
