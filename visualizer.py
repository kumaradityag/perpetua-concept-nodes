from typing import List, Optional
import hydra
import torch
from omegaconf import DictConfig
import logging
from scipy.spatial.transform import Rotation as Rsc
import numpy as np
import viser
from natsort import natsorted

from pathlib import Path
import numpy as np
import open3d as o3d
import cv2

from concept_graphs.utils import set_seed, load_point_cloud
from concept_graphs.viz.utils import similarities_to_rgb
from concept_graphs.mapping.similarity.semantic import CosineSimilarity01

from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap

# A logger for this file
log = logging.getLogger(__name__)


class ViserServer:
    def __init__(
        self, pcd_o3d, clip_ft, ft_extractor, imgs=None, point_shape: str = "circle"
    ):
        self.server = viser.ViserServer()
        self.point_shape = point_shape
        self.ft_extractor = ft_extractor

        # Geometries
        self.objects: List[o3d.geometry.PointCloud] = pcd_o3d
        self.labels = [f"{i}" for i in range(len(self.objects))]
        self.bbox: List[o3d.geometry.OrientedBoundingBox] = [
            pcd.get_oriented_bounding_box() for pcd in self.objects
        ]
        self.centroid = [np.mean(np.asarray(p.points), axis=0) for p in self.objects]

        # Semantics
        self.imgs = imgs

        # Handles
        self.object_names: List[str] = []
        self.object_handles: List[viser.PointCloudHandle] = []
        self.centroid_handles: List[viser.SphereHandle] = []
        self.box_handles: List[viser.BoxHandle] = []
        self.label_handles: List[viser.LabelHandle] = []
        self.current_image_handle: Optional[viser.GuiImageHandle] = None

        # Similarities
        device = ft_extractor.device if ft_extractor is not None else "cpu"
        self.semantic_tensor = torch.from_numpy(clip_ft).to(device)
        self.semantic_sim = CosineSimilarity01()

        # Containers
        self.query_time = None

        # GUI
        with self.server.gui.add_folder("Point Cloud"):
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

        with self.server.gui.add_folder("Temporal Query"):
            self.time_gui_number = self.server.gui.add_number(
                "Time (hours)", initial_value=0, min=0
            )
            self.time_gui_button = self.server.gui.add_button(
                "Predict at Time",
                icon=viser.Icon.MOUSE,
            )
            self.time_gui_button.on_click(self.on_time_query_submit)

    def start(self):
        rng = np.random.default_rng(42)
        self.segmentation_colors = rng.random((len(self.objects), 3))
        self.display_object_rgb()

    # Gui callbacks and inputs
    def on_clip_query_submit(self, data):
        self.clip_query = self.clip_gui_text.value
        sim_query = self.query(self.clip_gui_text.value)
        self.display_object_similarity(sim_query)

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

        for handle in self.object_handles:
            handle.point_size = point_size

    def on_time_query_submit(self, data):
        self.query_time = self.time_gui_number.value

    # scene manipulation
    def reset(self):
        self.server.scene.reset()
        self.display_object_rgb()

    def load_object_map(self, object_map: PerpetuaObjectMap):
        self.objects = [obj.pcd for obj in object_map]
        self.labels = [f"{i}" for i in range(len(self.objects))]
        self.bbox: List[o3d.geometry.OrientedBoundingBox] = [
            pcd.get_oriented_bounding_box() for pcd in self.objects
        ]
        self.centroid = [np.mean(np.asarray(p.points), axis=0) for p in self.objects]
        self.semantic_tensor = object_map.semantic_tensor.to(self.semantic_tensor.device)
        self.imgs = [obj.segments[0].rgb for obj in object_map]
        self.reset()

    def clear_main_objects(self):
        for name in self.object_names:
            self.server.scene.remove_by_name(name)
            self.server.scene.remove_by_name(f"{name}_hitbox")
        self.object_names = []
        self.object_handles = []

    def clear_centroids(self):
        for sphere in self.centroid_handles:
            sphere.remove()
        self.centroid_handles = []

    def clear_boxes(self):
        for box in self.box_handles:
            box.remove()
        self.box_handles = []

    def clear_labels(self):
        for label in self.label_handles:
            label.remove()
        self.label_handles = []

    def display_object_point_clouds(self, colors: np.ndarray = None):
        self.clear_main_objects()

        for i, obj in enumerate(self.objects):
            name = f"object_{i}"
            # Downsample point clouds to make life easier for viser
            pcd = obj.voxel_down_sample(voxel_size=0.005)
            pcd_points = np.asarray(pcd.points)
            if colors is not None:
                pcd_colors = np.tile(colors[i], (pcd_points.shape[0], 1))
            else:
                # Original rgb color
                pcd_colors = np.asarray(pcd.colors)
            handle = self.server.scene.add_point_cloud(
                name,
                pcd_points,
                pcd_colors,
                point_size=self.pcd_size_gui_slider.value,
                point_shape=self.point_shape,
            )
            # Add invisible hitbox for clicking
            bbox = self.bbox[i]
            wxyz = Rsc.from_matrix(np.array(bbox.R, copy=True)).as_quat(
                scalar_first=True
            )
            hitbox_handle = self.server.scene.add_box(
                name=f"{name}_hitbox",
                position=bbox.center,
                dimensions=bbox.extent,
                wxyz=wxyz,
                color=(255, 255, 255),
                opacity=0.0,
            )
            hitbox_handle.on_click(lambda _, idx=i: self.on_object_clicked(idx))

            self.object_names.append(name)
            self.object_handles.append(handle)

    def display_object_centroids(self):
        # Clear previous spheres
        self.clear_centroids()

        for i, center in enumerate(self.centroid):
            name = f"centroid_{i}"
            sphere = self.server.scene.add_icosphere(
                name=name,
                color=(255, 0, 0),
                radius=0.03,
                position=center,
            )
            sphere.on_click(lambda _, idx=i: self.on_object_clicked(idx))
            self.centroid_handles.append(sphere)

    def display_boxes(self):
        # Clear existing boxes
        self.clear_boxes()

        for i, box in enumerate(self.bbox):
            box_name = f"box_{i}"
            center = box.center
            extent = box.extent
            R = np.array(box.R, copy=True)
            wxyz = Rsc.from_matrix(R).as_quat(scalar_first=True)

            box = self.server.scene.add_box(
                name=box_name,
                color=(0, 1, 0),
                dimensions=extent,
                position=center,
                wxyz=wxyz,
                wireframe=True,
            )
            self.box_handles.append(box)

    def display_labels(self):
        self.clear_labels()

        for i, (center, label_text) in enumerate(zip(self.centroid, self.labels)):
            name = f"label_{i}"
            label_handle = self.server.scene.add_label(
                name=name,
                text=label_text,
                position=center,
            )
            self.label_handles.append(label_handle)

    def on_object_clicked(self, index: int):
        """Callback to display the image associated with the clicked object."""
        # Remove previous handle if exists
        if self.current_image_handle is not None:
            self.current_image_handle.remove()
            self.current_image_handle = None

        if self.imgs is None:
            log.warning("No images were provided to display.")
            return

        # Check if we have an image for this index
        if index >= len(self.imgs):
            print(f"No image found for object {index}")
            return

        image_data = self.imgs[index]

        # 3. Add the new image to the top of the GUI sidebar
        # We use a high 'order' or put it in a folder if you prefer organization
        self.current_image_handle = self.server.gui.add_image(
            image=image_data,
            label=f"Object {index} View",
        )

    def display_object_rgb(self):
        self.display_object_point_clouds()

    def display_object_segmentation(self):
        self.display_object_point_clouds(colors=self.segmentation_colors)

    def display_object_similarity(self, similarity_scores: np.ndarray):
        self.display_object_point_clouds(
            colors=similarities_to_rgb(similarity_scores, cmap_name="viridis")
        )

    def query(self, query: str):
        if self.ft_extractor is None:
            log.warning("No feature extractor provided.")
            return
        query_ft = self.ft_extractor.encode_text([query])
        sim_query = self.semantic_sim(query_ft, self.semantic_tensor)
        return sim_query.squeeze().cpu().numpy()


def load_imgs_from_folder(folder_path: Path) -> List[np.ndarray]:
    imgs = []
    for img_path in natsorted(folder_path.glob("*.png")):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs


@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    path = Path(cfg.map_path)

    if cfg.map_type == "concept_nodes":
        clip_ft = np.load(path / "clip_features.npy")
        pcd_o3d = load_point_cloud(path)
        imgs = load_imgs_from_folder(path / "object_viz")
    elif cfg.map_type == "perpetua":
        perpetua_map: PerpetuaObjectMap = PerpetuaObjectMap.load(path)
        clip_ft = perpetua_map.semantic_tensor.numpy()
        pcd_o3d = [obj.pcd for obj in perpetua_map]
        imgs = [obj.segments[0].rgb for obj in perpetua_map]

    ft_extractor = (
        hydra.utils.instantiate(cfg.ft_extraction)
        if hasattr(cfg, "ft_extraction")
        else None
    )

    log.info(f"Loading map with a total of {len(pcd_o3d)} objects")

    viser_server = ViserServer(
        pcd_o3d=pcd_o3d, clip_ft=clip_ft, ft_extractor=ft_extractor, imgs=imgs
    )
    viser_server.start()
    while True:
        if viser_server.query_time is not None and cfg.map_type == "perpetua":
            print(f"The old edges were: \n {perpetua_map.edges}")
            perpetua_map.predict(viser_server.query_time)
            print(f"The new edges are: \n {perpetua_map.edges}")
            viser_server.query_time = None
            viser_server.load_object_map(perpetua_map)


if __name__ == "__main__":
    main()
