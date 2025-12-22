from typing import Tuple, List, Optional, Dict
import numpy as np
from PIL import Image

import warnings

# Filter out user warnings from a specific package
warnings.filterwarnings("ignore", category=UserWarning, module="sam3")

import torch
from concept_graphs.utils import split_camel_preserve_acronyms
from .SegmentationModel import SegmentationModel


class GTProcthor(SegmentationModel):
    def __init__(
        self,
        id_to_color_path: str,
        classes_path: str,
        device: str = "cuda",
    ):
        self.device = device
        self.classes = self.load_text_prompts(classes_path)
        id_to_color = self.load_id_to_color(id_to_color_path)
        self.color_to_class_idx = self._build_color_to_class_idx(id_to_color)

    def load_text_prompts(self, queries_path: str) -> List[str]:
        """Load text prompts from a file each line is a prompt"""
        with open(queries_path, "r") as f:
            prompts = f.read().splitlines()
        return prompts

    def load_id_to_color(self, id_to_color_path: str) -> dict:
        """Load id to color mapping from a json file"""
        import json

        with open(id_to_color_path, "r") as f:
            id_to_color = json.load(f)
        return id_to_color

    def _build_color_to_class_idx(self, id_to_color: Dict) -> Dict[Tuple[int, int, int], int]:
        """
        Creates a lookup table from RGB tuple to Class Index.
        Example: (12, 45, 255) -> 3 (Index of 'Apple')
        """
        lookup = {}
        for obj_id, color_list in id_to_color.items():
            # Normalize to lowercase
            category = split_camel_preserve_acronyms(obj_id.split("|")[0])
            
            # Check if this category exists in the target class list
            try:
                # Find index of category in the classes list
                class_idx = self.classes.index(category)
                lookup[tuple(color_list)] = class_idx
            except ValueError:
                # This object is not in our list of classes (e.g., Wall, Floor), ignore it
                continue
        return lookup

    def __call__(self, img: np.ndarray, semantics: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ensure semantics is numpy
        if torch.is_tensor(semantics):
            semantics = semantics.cpu().numpy()

        # Lists to store the results
        masks_list = []
        bboxes_list = []
        labels_list = []

        # Identify all unique colors in the current frame
        unique_colors = np.unique(semantics.reshape(-1, 3), axis=0)

        for color in unique_colors:
            color_tuple = tuple(color)

            # Check if this color corresponds to a class we care about
            if color_tuple not in self.color_to_class_idx:
                continue

            class_idx = self.color_to_class_idx[color_tuple]

            # Create Binary Mask for this instance
            instance_mask = np.all(semantics == color, axis=-1)

            # Extract Bounding Box (XYXY)
            y_indices, x_indices = np.where(instance_mask)
            if len(y_indices) == 0:
                continue
                
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()

            masks_list.append(instance_mask)
            bboxes_list.append([x_min, y_min, x_max, y_max])
            labels_list.append(class_idx)

        # Handle case with no detections
        if not bboxes_list:
            return None, None, None, None

        masks = torch.from_numpy(np.stack(masks_list)).to(self.device).bool() # (N, H, W)
        bbox = torch.tensor(bboxes_list, device=self.device, dtype=torch.int) # (N, 4)
        labels = torch.tensor(labels_list, device=self.device, dtype=torch.int) # (N,)
        
        # Scores are 1.0 for Ground Truth
        scores = torch.ones(len(labels_list), device=self.device, dtype=torch.float32) # (N,)

        return masks, bbox, scores, labels
