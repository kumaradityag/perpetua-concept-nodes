from typing import Union, Self, List, Optional
from enum import Enum
import numpy as np
import jax.numpy as jnp
import open3d as o3d
from .Segment import Segment
from .SegmentHeap import SegmentHeap
from .pcd_callbacks.PointCloudCallback import PointCloudCallback
from perpetua2.utils.filter_state import Object as Estimator
from perpetua2.filters.BayesianPerpetua import object_predict
import uuid


class ObjectType(Enum):
    RECEPTACLE = "Receptacle"
    PICKUPABLE = "Pickupable"
    BACKGROUND = "Background"


class Object:
    def __init__(
        self,
        score: float,
        rgb: np.ndarray,
        mask: np.ndarray,
        point_map: np.ndarray,
        semantic_ft: np.ndarray,
        camera_pose: np.ndarray,
        label: np.ndarray,
        segment_heap_size: int,
        semantic_mode: str,
        timestep_created: int,
        timestamp: float = None,
        max_points_pcd: int = 1200,
        denoising_callback: Union[PointCloudCallback, None] = None,
        downsampling_callback: Union[PointCloudCallback, None] = None,
    ):
        self.segment_heap_size = segment_heap_size
        self.semantic_mode = semantic_mode
        self.timestep_created = timestep_created
        self.max_points_pcd = max_points_pcd
        self.denoising_callback = denoising_callback
        self.downsampling_callback = downsampling_callback

        self.pcd = None
        self.pcd_np = None
        self.vertices = None
        self.centroid = None
        self.semantic_ft = None
        self.segments = SegmentHeap(max_size=segment_heap_size)
        self.n_segments = 1
        self.caption = "empty"
        self.tag = "empty"

        # Create first segment and push to heap
        segment = Segment(
            rgb=rgb,
            mask=mask,
            point_map=point_map,
            semantic_ft=semantic_ft,
            camera_pose=camera_pose,
            score=score,
            label=label,
        )
        self.segments.push(segment)

        self.timestamps = [timestamp]

        # Set our first object-level point cloud
        pcd_rgb = segment.pcd_rgb
        pcd_points = segment.pcd_points

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        self.pcd.colors = o3d.utility.Vector3dVector(pcd_rgb / 255.0)
        self.downsample()
        self.denoise()

        self.update_geometry_np()
        self.update_semantic_ft()
        self.is_collated = True

        self.id = uuid.uuid4()

        # Perpetua attributes
        self.name = None
        self.obj_type = ObjectType.BACKGROUND
        self.estimator: Estimator = None
        self.visibility = True
        self.canonical_pos_vectors = []
        self._perpetua_map = None

    @property
    def labels(self):
        return [s.label for s in self.segments]

    @property
    def receptacles(self):
        return self.estimator.receptacle_names

    def __repr__(self):
        return f"Object with {len(self.segments)} segments. Detected a total of {self.n_segments} times."

    def update(self):
        raise NotImplementedError

    def predict(
        self, t: Union[float, jnp.array], receptacle_names: Optional[List[str]] = None
    ) -> np.ndarray:
        if isinstance(t, float):
            t = jnp.array([t])
        out = object_predict(self.estimator, t, receptacle_names=receptacle_names)
        return out

    def update_geometry(self):
        """Pull segment point clouds into one object-level point cloud"""
        points = np.concatenate([s.pcd_points for s in self.segments], axis=0)
        colors = np.concatenate([s.pcd_rgb for s in self.segments], axis=0)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.downsample()
        self.update_geometry_np()

    def update_geometry_np(self):
        self.pcd_np = np.asarray(self.pcd.points, dtype=np.float32)  # No copy
        self.centroid = np.mean(self.pcd_np, axis=0)
        aabb = self.pcd.get_axis_aligned_bounding_box()
        self.vertices = np.asarray(aabb.get_box_points(), dtype=np.float32)

        if len(self.pcd_np) > self.max_points_pcd:
            sub = np.random.choice(
                len(self.pcd_np), size=self.max_points_pcd, replace=False
            )
            self.pcd_np = self.pcd_np[sub]

    def update_semantic_ft(self):
        """Pick the representative semantic vector from the segments."""
        ft = np.stack([v.semantic_ft for v in self.segments], axis=0)

        if self.semantic_mode == "mean":
            mean = np.mean(ft, axis=0)
            norm = np.linalg.norm(mean)
            if norm > 0:
                self.semantic_ft = mean / norm
            else:
                self.semantic_ft = mean
        elif self.semantic_mode == "multi":
            if len(ft) < self.segment_heap_size:
                multiply = self.segment_heap_size // len(ft) + 1
                self.semantic_ft = np.concatenate([ft] * multiply, axis=0)[
                    : self.segment_heap_size
                ]
            else:
                self.semantic_ft = ft

    def collate(self):
        if not self.is_collated:
            self.update_geometry()
            self.update_semantic_ft()

            self.is_collated = True

    def denoise(self):
        if self.denoising_callback is not None:
            self.pcd = self.denoising_callback(self.pcd)

    def downsample(self):
        if self.downsampling_callback is not None:
            self.pcd = self.downsampling_callback(self.pcd)

    def downsample_pcd(self, voxel_size: float):
        self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)

    def cluster_top_k(self, k: int):
        ft = [v.semantic_ft for v in self.segments]
        ft = np.stack(ft, axis=0)
        mean = np.mean(ft, axis=0, keepdims=True)
        mean /= np.linalg.norm(mean, axis=1, keepdims=True)
        sim = ft @ mean.T
        idx = sim[:, 0].argsort()[-k:]

        new_heap = SegmentHeap(max_size=self.segments.max_size)
        for i in idx:
            new_heap.push(self.segments[i])
        self.segments = new_heap
        self.is_collated = False

    def __iadd__(self, other):
        if self.id == other.id:
            raise Exception("Trying to merge object with self.")

        segment_added = self.segments.extend(other.segments)
        self.n_segments += other.n_segments
        self.timestep_created = min(self.timestep_created, other.timestep_created)

        if segment_added:
            self.is_collated = False

        return self

    def pcd_to_np(self):
        # Make object pickable
        pcd_points = np.array(self.pcd.points, dtype=np.float32)
        pcd_colors = np.array(self.pcd.colors, dtype=np.float32)
        self.pcd = {"points": pcd_points, "colors": pcd_colors}

    def pcd_to_o3d(self):
        pcd_dict = self.pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_dict["points"])
        pcd.colors = o3d.utility.Vector3dVector(pcd_dict["colors"])
        self.pcd = pcd

    def view_images_caption(self):
        from ..viz.segmentation import plot_grid_images

        rgb_crops = [v.rgb / 255.0 for v in self.segments]
        plot_grid_images(
            rgb_crops, None, grid_width=3, tag=self.tag, caption=self.caption
        )

    def set_parent_map(self, parent_map: Self):
        self._perpetua_map = parent_map

    def move(self, receptacle_name: str):
        if self._perpetua_map is None:
            raise RuntimeError("Object is not attached to a Perpetua map.")

        receptacle = self._perpetua_map.get_receptacle(receptacle_name)
        vector_id = np.random.choice(len(receptacle.canonical_pos_vectors))
        vector = receptacle.canonical_pos_vectors[vector_id]
        bbox_min = receptacle.vertices.min(axis=0)
        bbox_max = receptacle.vertices.max(axis=0)
        bbox_extent = bbox_max - bbox_min

        # Just translate x-y with some noise
        noise_xy = np.random.uniform(-0.05, 0.05, size=2) * bbox_extent[:2]
        noise = np.array([noise_xy[0], noise_xy[1], 0.0])
        candidate_point = receptacle.centroid + vector + noise

        # Clip only x/y; keep z as is
        target_point = candidate_point.copy()
        target_point[:2] = np.clip(candidate_point[:2], bbox_min[:2], bbox_max[:2])

        translation = target_point - self.centroid

        self.pcd.translate(translation)
        self.pcd_np = self.pcd_np + translation
        self.vertices = self.vertices + translation
        self.centroid = self.centroid + translation


class RunningAverageObject(Object):
    """CG object from the original paper. Semantic feature average. Append pcd.

    We still use the segment heap to store images and masks, but nothing else."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.semantic_ft = self.segments[0].semantic_ft

        # We track an object-level point cloud so we don't need the segment point clouds
        # self.segments[0].pcd_points = None
        # self.segments[0].pcd_rgb = None
        # self.segments[0].semantic_ft = None

    def update_geometry(self):
        self.downsample()
        self.update_geometry_np()

    def update_semantic_ft(self):
        pass

    def cluster_top_k(self, k: int):
        pass

    def __iadd__(self, other):
        if self.id == other.id:
            raise Exception("Trying to merge object with self.")

        self.segments.extend(other.segments)
        self_ratio = self.n_segments / (self.n_segments + other.n_segments)
        other_ratio = other.n_segments / (self.n_segments + other.n_segments)
        self.semantic_ft = (
            self_ratio * self.semantic_ft + other_ratio * other.semantic_ft
        )
        self.semantic_ft = self.semantic_ft / np.linalg.norm(self.semantic_ft, 2)

        self.pcd += other.pcd

        self.timestamps.extend(other.timestamps)
        self.timestamps = list(set(self.timestamps))  # remove duplicates

        self.n_segments += other.n_segments
        self.timestep_created = min(self.timestep_created, other.timestep_created)
        self.is_collated = False

        return self


class ObjectFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs) -> Object:
        return Object(**kwargs, **self.kwargs)


class RunningAverageObjectFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs) -> Object:
        return RunningAverageObject(**kwargs, **self.kwargs)
