import datetime
import os
import time
import json
import logging
from pathlib import Path
from threading import Thread, Lock, Event
from queue import Queue

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from concept_graphs.utils import set_seed
from concept_graphs.mapping.utils import test_unique_segments


# A logger for this file
log = logging.getLogger(__name__)

def perception_worker(dataloader, perception_pipeline, perception_queue, stop_event):
    """
    Thread 1: Runs the perception stack. Produces segment data.
    """
    log.info("Perception thread started.")
    try:
        # The main loop now runs in this worker to process dataloader output
        for obs in dataloader:
            if stop_event.is_set():
                break
                
            # 1. PERCEPTION (Fastest)
            segments = perception_pipeline(
                obs["rgb"], obs["depth"], obs["intrinsics"], obs["camera_pose"]
            )
            timestamp = obs.get("timestamp", None)
            
            # Put result into the queue for the next stage
            perception_queue.put({
                "segments": segments, 
                "timestamp": timestamp, 
            })
            
    except Exception as e:
        log.error(f"Error in perception thread: {e}")
    finally:
        # Signal the end of processing to the next consumer thread
        perception_queue.put(None) 
        log.info("Perception thread finished.")


def local_map_worker(perception_queue, cfg_mapping, local_map_queue):
    """
    Thread 2: Creates the local map from segment data.
    """
    log.info("Local Map thread started.")
    try:
        while True:
            # Wait for data from the perception queue
            item = perception_queue.get()
            
            # Check for the sentinel object (None)
            if item is None:
                perception_queue.put(None) # Pass sentinel to next thread
                break

            segments = item["segments"]
            timestamp = item["timestamp"]

            if segments is None:
                continue

            # 2. LOCAL MAP CREATION
            local_map = hydra.utils.instantiate(cfg_mapping)
            local_map.from_perception(**segments, timestamp=timestamp)

            # Put local map into the queue for the next stage
            local_map_queue.put(local_map)
            
    except Exception as e:
        log.error(f"Error in local map thread: {e}")
    finally:
        # Signal the end of processing to the next consumer thread
        local_map_queue.put(None) 
        log.info("Local Map thread finished.")


def map_merging_worker(local_map_queue, main_map, map_stats_lock, progress_bar):
    """
    Thread 3: Merges the local map into the main map (Slowest part).
    """
    log.info("Map Merging thread started.")
    n_segments = 0
    try:
        while True:
            # Wait for data from the local map queue
            local_map = local_map_queue.get()
            
            # Check for the sentinel object (None)
            if local_map is None:
                break
            
            # 3. MAP MERGING (Slowest)
            # Use a lock when accessing the shared main_map resource
            with map_stats_lock:
                main_map[0] += local_map
                n_segments += len(local_map)

            progress_bar.update(1)
            n_segments += len(local_map)
            progress_bar.set_postfix(
                objects=len(main_map[0]),
                map_segments=main_map[0].n_segments,
                detected_segments=n_segments,
            )

                
    except Exception as e:
        log.error(f"Error in map merging thread: {e}")
    finally:
        # This thread is the last one, no need to pass sentinel
        log.info("Map Merging thread finished.")
        # Store final stats in the shared main_map/lock structure if needed, 
        # but for simplicity, we'll calculate final stats in the main thread.


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    log.info(f"Running algo {cfg.name}...")

    log.info("Loading data and models...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)
    log.info(f"Loaded dataset {dataset.name}.")

    segmentation_model = hydra.utils.instantiate(cfg.segmentation)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)
    perception_pipeline = hydra.utils.instantiate(
        cfg.perception, segmentation_model=segmentation_model, ft_extractor=ft_extractor
    )

    log.info("Mapping...")
    progress_bar = tqdm(total=len(dataset))
    progress_bar.set_description(f"Mapping")
    start = time.time()
    main_map = [hydra.utils.instantiate(cfg.mapping)]

    # Queues for data transfer between stages (Producer-Consumer)
    perception_queue = Queue(maxsize=10) # Bounded queue to prevent memory blow-up
    local_map_queue = Queue(maxsize=10) 
    
    # Lock for protecting shared resources (main_map updates)
    map_stats_lock = Lock()
    # Event to signal an emergency stop (optional but good practice)
    stop_event = Event()

    # 1. Perception Thread
    perception_thread = Thread(
        target=perception_worker, 
        args=(dataloader, perception_pipeline, perception_queue, stop_event),
        daemon=True
    )

    # 2. Local Map Creation Thread
    local_map_thread = Thread(
        target=local_map_worker, 
        args=(perception_queue, cfg.mapping, local_map_queue),
        daemon=True
    )
    
    # 3. Map Merging Thread (Slowest)
    merging_thread = Thread(
        target=map_merging_worker, 
        args=(local_map_queue, main_map, map_stats_lock, progress_bar),
        daemon=True
    )

    # Start all threads
    perception_thread.start()
    local_map_thread.start()
    merging_thread.start()

    # Wait for the perception thread to finish all its work (i.e., iterating over the dataloader)
    perception_thread.join()
    log.info("Perception stage finished. Waiting for local map and merge stages.")
    
    # Wait for the remaining threads to process the queues and finish
    local_map_thread.join()
    merging_thread.join()
    main_map = main_map[0]
    
    # Postprocessing
    main_map.filter_min_segments(n_min_segments=cfg.final_min_segments, grace=False)
    main_map.downsample_objects()
    for _ in range(2):
        main_map.denoise_objects()
        main_map.self_merge()
    main_map.downsample_objects()
    main_map.filter_min_points_pcd()

    stop = time.time()
    mapping_time = stop - start
    progress_bar.close()
    n_objects = len(main_map)
    fps = len(dataset) / (mapping_time)
    test_unique_segments(main_map)
    log.info("Objects in final map: %d" % n_objects)
    log.info(f"fps: {fps:.2f}")

    if cfg.caption and hasattr(cfg, "vlm_caption"):
        log.info("Captioning objects...")
        captioner = hydra.utils.instantiate(cfg.vlm_caption)
        captioner.caption_map(main_map)

    if cfg.tag and hasattr(cfg, "vlm_tag"):
        log.info("Tagging objects...")
        tagger = hydra.utils.instantiate(cfg.vlm_tag)
        tagger.caption_map(main_map)

    # Save visualizations and map
    if not cfg.save_map:
        return

    output_dir = Path(cfg.output_dir)
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S.%f")
    output_dir_map = output_dir / f"{dataset.name}_{cfg.name}_{date_time}"

    log.info(f"Saving map, images and config to {output_dir_map}...")
    grid_image_path = output_dir_map / "object_viz"
    os.makedirs(grid_image_path, exist_ok=False)
    main_map.save_object_grids(grid_image_path)

    # Also export some data to standard files
    main_map.export(output_dir_map)

    # Hydra config
    OmegaConf.save(cfg, output_dir_map / "config.yaml")

    # Few more stats
    stats = dict(
        fps=fps, mapping_time=mapping_time, n_objects=n_objects, n_frames=len(dataset)
    )
    json.dump(stats, open(output_dir_map / "stats.json", "w"))

    # Create symlink to latest map
    symlink = output_dir / "latest_map"
    symlink.unlink(missing_ok=True)
    os.symlink(output_dir_map, symlink)
    log.info(f"Created symlink to latest map at {symlink}")

    # Move debug directory if it exists
    if os.path.exists(output_dir / "debug"):
        os.rename(output_dir / "debug", output_dir_map / "debug")


if __name__ == "__main__":
    main()
