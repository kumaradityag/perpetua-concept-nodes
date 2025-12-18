import copy
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Self

import dill as pickle
import numpy as np
import torch
import jax.numpy as jnp

from .Object import Object, ObjectType
from perpetua2.utils.filter_state import Object as Estimator
import logging

log = logging.getLogger(__name__)


class MapState(Enum):
    COMPLETE = "Complete"
    INCOMPLETE = "Incomplete"


class PerpetuaObjectMap:
    def __init__(
        self,
        pickupable_names: Optional[Iterable[str]] = None,
        receptacle_names: Optional[Iterable[str]] = None,
        time: float = 0.0,
        device: str = "cpu",
    ):
        self.pickupable_names = list(pickupable_names or [])
        self.receptacle_names = [
            name for name in (receptacle_names or []) if name != "OOB_FAKE_RECEPTACLE"
        ]
        self.time = time
        self.device = device

        self.objects: Dict[str, Object] = {}
        self._pickupables: Dict[str, Object] = {}
        self._receptacles: Dict[str, Object] = {}

        self.edges: Dict[str, List[str]] = {}
        self.state = MapState.INCOMPLETE

        self.semantic_tensor: Optional[torch.Tensor] = None
        self.pcd_tensors: Optional[List[torch.Tensor]] = None
        self.vertices_tensor: Optional[torch.Tensor] = None
        self.centroid_tensor: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects.values())

    def has_object(self, name: str) -> bool:
        return name in self.objects

    def get_object(self, name: str) -> Object:
        return self.objects[name]

    def get_receptacle(self, name: str) -> Object:
        return self._receptacles[name]

    def get_pickupable(self, name: str) -> Object:
        return self._pickupables[name]

    def _add_object(
        self,
        name: str,
        obj: Object,
        obj_type: ObjectType,
        target_dict: Optional[Dict[str, Object]] = None,
        visibility: bool = True,
        **kwargs,
    ) -> Object:
        """Internal helper to handle common object registration logic."""
        new_obj = self._clone_object(obj)
        new_obj.obj_type = obj_type
        new_obj.name = name
        new_obj.visibility = visibility

        # Set any extra attributes passed (e.g., estimator)
        for key, value in kwargs.items():
            setattr(new_obj, key, value)

        self._register_object(name, new_obj)

        # Only add to a specific dictionary if one is provided
        if target_dict is not None:
            target_dict[name] = new_obj

        self._refresh_state()
        self._refresh_geometry_cache()
        return new_obj

    def add_pickupable(
        self, name: str, obj: Object, estimator: Estimator, visibility: bool
    ) -> Object:
        return self._add_object(
            name,
            obj,
            ObjectType.PICKUPABLE,
            target_dict=self._pickupables,
            visibility=visibility,
            estimator=estimator,  # Passed via **kwargs
        )

    def add_receptacle(self, name: str, obj: Object) -> Object:
        return self._add_object(
            name, obj, ObjectType.RECEPTACLE, target_dict=self._receptacles
        )

    def add_background(self, name: str, obj: Object) -> Object:
        return self._add_object(name, obj, ObjectType.BACKGROUND)

    def set_edges(self, edges: Dict[str, List[str]], move_pickupables: bool = True):
        attached = {}
        for receptacle_name, pickupables in edges.items():
            for pickupable_name in pickupables:
                if pickupable_name in self._pickupables:
                    attached[pickupable_name] = receptacle_name
        self.edges = edges

        for name, pickupable in self._pickupables.items():
            if name in attached:
                pickupable.visibility = True
                if move_pickupables:
                    pickupable.move(attached[name])
            else:
                pickupable.visibility = False

        self._refresh_geometry_cache()

    def predict(self, timestep: float, threshold: float = 0.5):
        if isinstance(timestep, float):
            timestep = jnp.array([timestep])

        # Predict edges based on current estimators
        inferred_edges: Dict[str, List[str]] = defaultdict(list)

        for name, pickupable in self._pickupables.items():
            if pickupable.estimator is None:
                raise ValueError(f"Pickupable {name} does not have an associated estimator.")
            belief = pickupable.predict(timestep)
            index = jnp.argmax(belief).item()
            receptacle_name = pickupable.receptacles[index] if belief[index] >= threshold else None
            if receptacle_name is None or receptacle_name not in self._receptacles:
                continue
            inferred_edges[receptacle_name].append(name)

        # FIXME: if predicting, this needs to be a temporal change, not an absolute set
        self.set_edges(inferred_edges)
        self.time = timestep

    def update_canonical_vectors(self, canonical_vectors: Dict[str, List[np.ndarray]]):
        for receptacle_name, vectors in canonical_vectors.items():
            if receptacle_name not in self._receptacles:
                continue
            rec = self._receptacles[receptacle_name]
            rec.canonical_pos_vectors = [
                np.asarray(vec, dtype=np.float32) for vec in vectors
            ]

        for rec in self._receptacles.values():
            if rec.canonical_pos_vectors:
                continue
            rec.canonical_pos_vectors = [self._default_canonical_vector(rec)]

    def save(self, map_dir: Path):
        log.info(f"Saving PerpetuaObjectMap to {map_dir}")
        map_dir = Path(map_dir)
        map_dir.mkdir(parents=True, exist_ok=True)
        map_file = map_dir / "perpetua_map.pkl"

        for obj in self.objects.values():
            obj.pcd_to_np()

        with open(map_file, "wb") as f:
            pickle.dump(self, f)

        for obj in self.objects.values():
            obj.pcd_to_o3d()

    @classmethod
    def load(cls, path: Path) -> Self:
        log.info(f"Loading PerpetuaObjectMap from {path}")
        map_file = path / "perpetua_map.pkl"
        with open(map_file, "rb") as f:
            loaded_map: "PerpetuaObjectMap" = pickle.load(f)

        for obj in loaded_map.objects.values():
            if isinstance(obj.pcd, dict):
                obj.pcd_to_o3d()
            obj.set_parent_map(loaded_map)

        loaded_map._rebuild_groups()
        loaded_map._refresh_geometry_cache()
        loaded_map._refresh_state()
        return loaded_map

    def _register_object(self, name: str, obj: Object):
        obj.set_parent_map(self)
        self.objects[name] = obj

    def _clone_object(self, obj: Object) -> Object:
        return copy.deepcopy(obj)

    def _rebuild_groups(self):
        self._pickupables = {
            name: obj
            for name, obj in self.objects.items()
            if obj.obj_type == ObjectType.PICKUPABLE
        }
        self._receptacles = {
            name: obj
            for name, obj in self.objects.items()
            if obj.obj_type == ObjectType.RECEPTACLE
        }

    def _refresh_state(self):
        pickupable_target = len(self.pickupable_names)
        receptacle_target = len(self.receptacle_names)
        if (
            len(self._pickupables) == pickupable_target
            and len(self._receptacles) == receptacle_target
        ):
            self.state = MapState.COMPLETE
        else:
            self.state = MapState.INCOMPLETE

    def refresh_state(self):
        self._refresh_state()

    def _refresh_geometry_cache(self):
        if len(self.objects) == 0:
            self.semantic_tensor = None
            self.pcd_tensors = None
            self.vertices_tensor = None
            self.centroid_tensor = None
            return

        semantic = []
        centroids = []
        vertices = []
        pcds = []

        for obj in self.objects.values():
            semantic.append(obj.semantic_ft)
            centroids.append(obj.centroid)
            vertices.append(obj.vertices)
            pcds.append(torch.from_numpy(obj.pcd_np).to(self.device))

        self.semantic_tensor = torch.from_numpy(np.stack(semantic, axis=0)).to(
            self.device
        )
        self.centroid_tensor = torch.from_numpy(np.stack(centroids, axis=0)).to(
            self.device
        )
        self.vertices_tensor = torch.from_numpy(np.stack(vertices, axis=0)).to(
            self.device
        )
        self.pcd_tensors = pcds

    def _default_canonical_vector(self, receptacle: Object) -> np.ndarray:
        # TODO: Verify that this is the desired default behavior
        vertices = receptacle.vertices
        min_x = vertices[:, 0].min()
        max_x = vertices[:, 0].max()
        min_y = vertices[:, 1].min()
        max_y = vertices[:, 1].max()
        min_z = vertices[:, 2].min()
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        target = np.array(
            [center_x, center_y, min_z], dtype=np.float32
        )  # z axis is flipped in these scenes
        return target - receptacle.centroid
