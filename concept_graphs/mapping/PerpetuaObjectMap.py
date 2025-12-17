import os
from typing import List, Dict, Tuple
import numpy as np
import torch
import open3d as o3d
from pathlib import Path
import json
import cv2
from .Object import Object
from .ObjectMap import ObjectMap


class PerpetuaObjectMap:
    def __init__(self):
        self.objects: Dict[str, Object] = dict()
        # TODO: Implement a new PerpetuaObjectMap class to handle the specific requirements of Perpetua concept nodes map
