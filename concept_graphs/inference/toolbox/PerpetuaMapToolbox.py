import logging
import torch

from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap
from concept_graphs.inference.toolbox.ObjectMapToolbox import ObjectMapToolbox
from concept_graphs.perception.ft_extraction.CLIP import CLIP

log = logging.getLogger(__name__)


class PerpetuaMapToolbox(ObjectMapToolbox):
    def __init__(
        self,
        map_path: str,
        ft_extractor: CLIP,
    ):
        self.object_map: PerpetuaObjectMap = None
        super().__init__(map_path, ft_extractor)

    def load_object_map(self, map_path: str = None):
        object_map = PerpetuaObjectMap.load(map_path)
        object_map.to(self.device)
        object_map.downsample_objects(voxel_size=0.005)
        return object_map

    def update_object_map(self, object_map):
        self.object_map = object_map
        self.object_map.downsample_objects(voxel_size=0.01)
        self.object_map.refresh_state()
        self.reset()

    def temporal_map_query(self, query_time: float):
        self.object_map.predict(query_time)
        self.object_map.refresh_state()

    def temporal_object_query(self, object_id: str, query_time: float):
        self.object_map.object_predict(object_id, query_time)
        self.object_map.refresh_state()

    def reset_temporal_edges(self):
        self.object_map.reset()

    # This is a debug method to visualize the canonical vectors
    def build_canonical_vectors(self):
        vectors_dict = {}
        for r_name, receptacle in self.object_map._receptacles.items():
            # All receptacles are assumed to have canonical vectors
            vectors_dict[r_name] = receptacle.canonical_pos_vectors
        return vectors_dict
