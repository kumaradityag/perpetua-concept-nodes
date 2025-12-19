import logging
import torch

from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.perception.ft_extraction.CLIP import CLIP
from concept_graphs.mapping.similarity.semantic import CosineSimilarity01

log = logging.getLogger(__name__)


class ObjectMapToolbox:
    def __init__(
        self,
        map_path: str,
        ft_extractor: CLIP,
    ):

        self.ft_extractor = ft_extractor
        self.device = ft_extractor.device
        self.semantic_sim = CosineSimilarity01()
        self.object_map: ObjectMap = self.load_object_map(map_path)

    def load_object_map(self, map_path: str = None):
        object_map = ObjectMap.load(map_path)
        object_map.to(self.device)
        return object_map

    def update_object_map(self, object_map):
        self.object_map = object_map
        self.object_map.to(self.device)
        self.reset()

    def reset(self):
        pass

    def clip_query(self, query: str) -> torch.Tensor:
        query_ft = self.ft_extractor.encode_text([query])
        query_ft = query_ft.to(self.device)
        sim_objects = self.semantic_sim(query_ft, self.object_map.semantic_tensor)

        return sim_objects.squeeze().cpu().numpy()
