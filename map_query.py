from pathlib import Path
import os
# Disable GPU memory pre-allocation to avoid OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

from concept_graphs.utils import set_seed

import hydra
from omegaconf import DictConfig
import logging

from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="map_query")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # Instantiate hydra objects
    dataset = hydra.utils.instantiate(cfg.dataset)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction, device=cfg.device)
    verifier = hydra.utils.instantiate(cfg.vlm_verifier)
    semantic_sim = hydra.utils.instantiate(cfg.semantic_similarity)

    # Get inputs
    pickupable_names = dataset.get_pickupable_names()
    receptacle_names = dataset.get_receptacles_names()
    p2r_map = dataset.get_pickupable_to_receptacles()

    concept_nodes_map_path = Path(cfg.map_path)

    # Initialize Engine
    engine = hydra.utils.instantiate(
        cfg.inference,
        ft_extractor=ft_extractor,
        semantic_sim_metric=semantic_sim,
        verifier=verifier,
        receptacles_bbox=dataset.get_receptacles_bbox(),
        pickupable_bbox=dataset.get_pickupables_bbox(),
        pickupable_existence=dataset.get_assignment(),
        pickupable_to_receptacles=p2r_map,
    )

    # Run Query Logic
    results = engine.process_queries(queries=pickupable_names)
    if cfg.debug:
        # Clean a bit of memory before visualization
        log.info("Visualizing map objects...")
        engine.visualize(concept_nodes_map_path, results)

    # Save Results
    output_path = concept_nodes_map_path / "assignments"
    engine.save_results(results, output_path=output_path, sssd_data=dataset.sssd_data)
    log.info(f"Results saved to {output_path}")

    # Update Perpetua Map
    pickupable_map_ids, receptacle_map_ids = engine.get_map_object_ids()
    concept_nodes_map = ObjectMap.load(concept_nodes_map_path)
    current_p2r_mapping = engine.parse_assignments(results, pickupable_names)

    perpetua_map: PerpetuaObjectMap = engine.update_perpetua_map(
        0.0,
        concept_nodes_map,
        pickupable_names,
        receptacle_names,
        pickupable_map_ids,
        receptacle_map_ids,
        current_p2r_mapping,
    )

    if cfg.debug:
        # Clean a bit of memory before visualization
        del perpetua_map
        del concept_nodes_map
        log.info("Visualizing map objects...")
        engine.visualize(concept_nodes_map_path, results)


if __name__ == "__main__":
    main()
