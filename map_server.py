from typing import Union
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging
# Disable GPU memory pre-allocation to avoid OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

from concept_graphs.utils import set_seed


from concept_graphs.viz.server.ObjectMapServer import ObjectMapServer
from concept_graphs.viz.server.PerpetuaMapServer import PerpetuaMapServer

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="map_server")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    server: Union[ObjectMapServer | PerpetuaMapServer] = instantiate(cfg.server)
    log.info(f"Loading map with a total of {len(server.object_map)} objects")
    # Start and spin the server
    server.spin()


if __name__ == "__main__":
    main()
