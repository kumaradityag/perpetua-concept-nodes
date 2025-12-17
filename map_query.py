import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from typing import List, Dict, Tuple

from concept_graphs.utils import load_map
from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap

log = logging.getLogger(__name__)


def update_perpetua_map(
    perpetua_map_path: Path,
    perpetua_estimators_path: Path,
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

    # TODO: Implement the logic to create a new or update an existing the PerpetuaObjectMap based on the provided parameters

    return


@hydra.main(version_base=None, config_path="conf", config_name="map_query")
def main(cfg: DictConfig):

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
    """
        Returns:
        {
          "pickupable_name": {
              "present": bool,
              "map_object_id": int | None,
              "query_timestamp": List[float],
              "present_receptacle_name": str | None,
              "receptacles": [
                    {
                        "receptacle_name": str,
                        "map_object_id": int,
                        "receptacle_timestamps": List[float],
                    },
                    ...
                ]
            }
        }
    """
    results = engine.process_queries(queries=pickupable_names)

    output_path = concept_nodes_map_path / "assignments"
    engine.save_results(results, output_path=output_path, sssd_data=dataset.sssd_data)
    log.info(f"Results saved to {output_path}")

    perpetua_map_path = Path(cfg.perpetua_map_path)
    perpetua_estimators_path = Path(cfg.perpetua_estimators_path)
    pickupable_map_ids, receptacle_map_ids = engine.get_map_object_ids()
    concept_nodes_map: ObjectMap = load_map(concept_nodes_map_path / "map.pkl")
    current_p2r_mapping = {
        p: (
            results[p]["present_receptacle_name"]
            if p in results and results[p].get("present", False)
            else None
        )
        for p in pickupable_names
    }

    new_perpetua_map = update_perpetua_map(
        perpetua_map_path,
        perpetua_estimators_path,
        concept_nodes_map,
        pickupable_names,
        receptacle_names,
        pickupable_map_ids,
        receptacle_map_ids,
        current_p2r_mapping,
    )

    if cfg.debug:
        log.info("Visualizing map objects...")
        engine.visualize(output_path)


if __name__ == "__main__":
    main()
