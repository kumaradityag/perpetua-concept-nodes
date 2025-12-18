import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from natsort import natsorted
import glob

import numpy as np
import torch
import cv2
import dill as pickle

from concept_graphs.mapping.similarity.semantic import CosineSimilarity01
from concept_graphs.perception.ft_extraction.FeatureExtractor import FeatureExtractor
from concept_graphs.utils import load_point_cloud

from concept_graphs.utils import load_map
from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap

log = logging.getLogger(__name__)


class BaseMapEngine:
    def __init__(
        self,
        map_path: str,
        perpetua_estimators_path: str,
        perpetua_map_path: str,
        perpetua_map_full_overwrite: bool,
        ft_extractor: FeatureExtractor,
        semantic_sim_metric: CosineSimilarity01,
        device: str = "cuda",
    ):
        self.map_path = Path(map_path)
        self.perpetua_estimators_path = Path(perpetua_estimators_path)
        self.perpetua_map_path = Path(perpetua_map_path)
        self.perpetua_map_path.mkdir(parents=True, exist_ok=True)
        self.perpetua_map_full_overwrite = perpetua_map_full_overwrite

        self.ft_extractor = ft_extractor
        self.semantic_sim = semantic_sim_metric

        self.annotations = self._load_annotations()
        self.features = self._load_features()
        self.pcd = self._load_point_clouds()
        self.bbox = [pcd.get_oriented_bounding_box() for pcd in self.pcd]
        self.device = device

        log.info(
            f"Map loaded from {self.map_path}. Found {len(self.annotations)} objects."
        )

    def _load_annotations(self) -> List[Dict]:
        anno_path = self.map_path / "segments_anno.json"

        with open(anno_path, "r") as f:
            data = json.load(f)
        return data.get("segGroups", [])

    def _load_features(self) -> torch.Tensor:
        feat_path = self.map_path / "clip_features.npy"

        # Load numpy and convert to torch for the similarity class
        feats_np = np.load(feat_path)
        return torch.from_numpy(feats_np).float()

    def _load_point_clouds(self) -> List[np.ndarray]:
        return load_point_cloud(self.map_path)

    def get_object_images(self, object_id: int, limit: int = 5) -> List[np.ndarray]:
        """
        Loads RGB images for a specific object ID from the directory structure.
        Structure: map_dir/segments/{id}/rgb/*.png
        """
        rgb_path = self.map_path / "segments" / str(object_id) / "rgb" / "*.png"

        # Get all pngs, sort them to ensure consistent order
        image_files = natsorted(glob.glob(str(rgb_path)))

        images = []
        for img_p in image_files[:limit]:
            img = cv2.imread(str(img_p))
            images.append(img)

        return images

    def process_queries(self, queries: List[str], **kwargs) -> Dict:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def save_results(self, results, output_path, **kwargs):
        """Save results to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    def _load_estimator(self, pickupable_name: str):
        "Load the Perpetua estimator for a given pickupable object."
        est_path = self.perpetua_estimators_path / f"{pickupable_name}_object.pkl"
        try:
            with open(est_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.error(f"Error loading estimator for {pickupable_name}: {e}")
            return None

    def _build_edges(
        self,
        current_p2r_mapping: Dict[str, Optional[str]],
        perpetua_map: PerpetuaObjectMap,
        skip_pickupables: Optional[Set[str]] = None,
    ) -> Dict[str, List[str]]:
        skip = skip_pickupables or set()
        edges: Dict[str, List[str]] = {}
        for pickupable_name, receptacle_name in current_p2r_mapping.items():
            if receptacle_name is None or pickupable_name in skip:
                continue
            if not perpetua_map.has_object(pickupable_name):
                continue
            if not perpetua_map.has_object(receptacle_name):
                continue
            if receptacle_name not in edges:
                edges[receptacle_name] = []
            edges[receptacle_name].append(pickupable_name)
        return edges

    def _build_canonical_vectors(
        self,
        concept_nodes_map: ObjectMap,
        current_p2r_mapping: Dict[str, Optional[str]],
        pickupable_map_ids: Dict[str, Optional[int]],
        receptacle_map_ids: Dict[str, Optional[int]],
    ) -> Dict[str, List]:
        canonical_vectors: Dict[str, List] = {}
        for pickupable_name, receptacle_name in current_p2r_mapping.items():
            if receptacle_name is None:
                continue
            pickupable_id = pickupable_map_ids.get(pickupable_name)
            receptacle_id = receptacle_map_ids.get(receptacle_name)
            if pickupable_id is None or receptacle_id is None:
                continue
            pickupable_obj = concept_nodes_map[pickupable_id]
            receptacle_obj = concept_nodes_map[receptacle_id]
            vector = pickupable_obj.centroid - receptacle_obj.centroid
            if receptacle_name not in canonical_vectors:
                canonical_vectors[receptacle_name] = []
            canonical_vectors[receptacle_name].append(vector)
        return canonical_vectors

    def update_perpetua_map(
        self,
        time: float,
        concept_nodes_map: ObjectMap,
        pickupable_names: List[str],
        receptacle_names: List[str],
        pickupable_map_ids: dict,
        receptacle_map_ids: dict,
        current_p2r_mapping: dict,
    ) -> PerpetuaObjectMap:
        # remove OOB_FAKE_RECEPTACLE from receptacle_map_ids
        if "OOB_FAKE_RECEPTACLE" in receptacle_map_ids:
            del receptacle_map_ids["OOB_FAKE_RECEPTACLE"]
        if "OOB_FAKE_RECEPTACLE" in receptacle_names:
            receptacle_names.remove("OOB_FAKE_RECEPTACLE")

        map_file = self.perpetua_map_path / "perpetua_map.pkl"

        canonical_vectors = self._build_canonical_vectors(
            concept_nodes_map,
            current_p2r_mapping,
            pickupable_map_ids,
            receptacle_map_ids,
        )

        if map_file.exists() and not self.perpetua_map_full_overwrite:
            perpetua_map = PerpetuaObjectMap.load(self.perpetua_map_path)

            for pickupable_name in pickupable_names:
                pickupable_id = pickupable_map_ids.get(pickupable_name)
                if pickupable_id is None or perpetua_map.has_object(pickupable_name):
                    continue
                estimator = self._load_estimator(pickupable_name)
                pickupable_obj = concept_nodes_map[pickupable_id]
                perpetua_map.add_pickupable(
                    pickupable_name,
                    pickupable_obj,
                    estimator,
                    visibility=False,
                )

            for receptacle_name in receptacle_names:
                receptacle_id = receptacle_map_ids.get(receptacle_name)
                if receptacle_id is None or perpetua_map.has_object(receptacle_name):
                    continue
                receptacle_obj = concept_nodes_map[receptacle_id]
                perpetua_map.add_receptacle(receptacle_name, receptacle_obj)

            perpetua_map.update_canonical_vectors(canonical_vectors)
            perpetua_map.refresh_state()

        else:
            perpetua_map = PerpetuaObjectMap(pickupable_names, receptacle_names, time)

            for receptacle_name in receptacle_names:
                receptacle_id = receptacle_map_ids.get(receptacle_name)
                if receptacle_id is None:
                    continue
                receptacle_obj = concept_nodes_map[receptacle_id]
                perpetua_map.add_receptacle(receptacle_name, receptacle_obj)

            for pickupable_name in pickupable_names:
                pickupable_id = pickupable_map_ids.get(pickupable_name)
                if pickupable_id is None:
                    continue
                estimator = self._load_estimator(pickupable_name)
                pickupable_obj = concept_nodes_map[pickupable_id]
                perpetua_map.add_pickupable(
                    pickupable_name,
                    pickupable_obj,
                    estimator,
                    visibility=True,
                )

            # TODO: Add background objects to the perpetua map

            perpetua_map.update_canonical_vectors(canonical_vectors)
            initial_edges = self._build_edges(current_p2r_mapping, perpetua_map)
            perpetua_map.set_edges(initial_edges, move_pickupables=True)
            perpetua_map.refresh_state()

        perpetua_map.save(self.perpetua_map_path)
        return perpetua_map
