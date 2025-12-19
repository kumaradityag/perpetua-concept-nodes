from typing import List, Optional, Union
import hydra
from hydra.utils import instantiate
import torch
from omegaconf import DictConfig
import logging
from scipy.spatial.transform import Rotation as Rsc
import numpy as np
from natsort import natsorted

from pathlib import Path
import numpy as np
import cv2

from concept_graphs.utils import set_seed


from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap
from concept_graphs.viz.server.ObjectMapServer import ObjectMapServer
from concept_graphs.inference.toolbox.ObjectMapToolbox import ObjectMapToolbox

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="map_server")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    toolbox: Union[ObjectMapToolbox] = instantiate(cfg.toolbox)
    server: Union[ObjectMapServer] = instantiate(cfg.server, toolbox=toolbox)
    log.info(f"Loading map with a total of {len(server.object_map)} objects")

    # Spin the server
    server.spin()

    # viser_server = ViserServer(ft_extractor=ft_extractor)
    # viser_server.load_object_map(scene_map)
    # viser_server.start()
    # while True:
    #     pass
        # if viser_server.query_time is not None and cfg.map_type == "perpetua":
        #     print(f"The old edges were: \n {perpetua_map.edges}")
        #     perpetua_map.predict(viser_server.query_time)
        #     print(f"The new edges are: \n {perpetua_map.edges}")
        #     viser_server.query_time = None
        #     viser_server.load_object_map(perpetua_map)


if __name__ == "__main__":
    main()
