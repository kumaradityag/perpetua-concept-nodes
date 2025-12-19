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
        return object_map

    def update_object_map(self, object_map):
        self.object_map = object_map
        self.object_map.to(self.device)
        self.object_map.refresh_state()
        self.reset()

    def temporal_map_query(self, query_time: float):
        self.object_map.predict(query_time)
        self.object_map.refresh_state()

        

