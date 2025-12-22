import logging
import shutil
import os
import glob
from natsort import natsorted
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

import torch
from tqdm import tqdm
import open3d as o3d

from concept_graphs.vlm.OpenAIVerifier import OpenAIVerifier
from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.utils import split_camel_preserve_acronyms, aabb_iou

from .BaseMapEngine import BaseMapEngine

from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)


class QueryObjects(BaseMapEngine):
    def __init__(
        self,
        verifier: OpenAIVerifier,
        receptacles_bbox: Dict,
        pickupable_to_receptacles: Dict,
        pickupable_bbox: Dict,
        pickupable_existence: Dict,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verifier = verifier
        self.receptacles_bbox = receptacles_bbox
        self.pickupable_to_receptacles = pickupable_to_receptacles
        self.top_k = top_k

        # Only used in GT Class, but defined here so things don't break
        self.pickupable_bbox = pickupable_bbox
        self.pickupable_existence = pickupable_existence

        self.receptacle_map_ids = self.get_receptacle_map_ids()
        self.gt_receptacles_aabb = self.compute_receptacles_aabb()

    def compute_receptacles_aabb(self):
        result = dict()
        for r_name, r_data in self.receptacles_bbox.items():
            # These are the cornerPoints of OOBB
            corners = np.array(r_data["cornerPoints"], dtype=np.float32)
            oobb = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(corners)
            )
            aabb = oobb.get_axis_aligned_bounding_box()
            result[r_name] = aabb
        return result

    def _canonical_receptacle_key(self, key: str) -> str:
        # Anything before ___ is the "physical object id"
        return key.split("___", 1)[0]

    def get_receptacle_map_ids(self) -> Dict[str, Optional[int]]:
        """
        Assign each receptacle to a map bbox index.
        """

        BREAKAWAY_MARGIN = (
            0.05  # how much better IoU must be to switch away from group assignment
        )

        receptacle_to_map: Dict[str, Optional[int]] = {}

        # Collect receptacles (skip fake)
        receptacle_keys = [
            k for k in self.receptacles_bbox.keys() if k != "OOB_FAKE_RECEPTACLE"
        ]
        receptacle_to_map["OOB_FAKE_RECEPTACLE"] = None

        # Precompute receptacle corners
        rec_corners_list: List[np.ndarray] = []
        for k in receptacle_keys:
            rec_corners = np.array(
                self.receptacles_bbox[k]["cornerPoints"], dtype=np.float32
            )
            rec_corners_list.append(rec_corners)

        # Precompute bbox corners
        bbox_corners_list: List[np.ndarray] = [
            np.asarray(b.get_box_points(), dtype=np.float32) for b in self.bbox
        ]

        R = len(receptacle_keys)
        B = len(bbox_corners_list)

        # IoU matrix: (R x B)
        iou_rb = np.zeros((R, B), dtype=np.float32)
        for i in range(R):
            rc = rec_corners_list[i]
            for j in range(B):
                iou_rb[i, j] = aabb_iou(rc, bbox_corners_list[j])

        # Group receptacles by canonical key
        canon_to_indices: Dict[str, List[int]] = {}
        for i, k in enumerate(receptacle_keys):
            ck = self._canonical_receptacle_key(k)
            canon_to_indices.setdefault(ck, []).append(i)

        canon_keys = list(canon_to_indices.keys())
        G = len(canon_keys)

        # Score matrix for groups vs bboxes: take best IoU among members of the group
        score_gb = np.zeros((G, B), dtype=np.float32)
        for gi, ck in enumerate(canon_keys):
            idxs = canon_to_indices[ck]
            score_gb[gi, :] = np.max(iou_rb[idxs, :], axis=0)

        # Allow "unassigned" by adding dummy columns
        n_cols = max(B, G)
        cost = np.ones(
            (G, n_cols), dtype=np.float32
        )  # default is dummy columns (score 0 => cost 1)
        cost[:, :B] = 1.0 - score_gb

        # Solve assignment
        assignment: List[Tuple[int, int]] = []

        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = list(zip(row_ind.tolist(), col_ind.tolist()))

        # Determine each canonical group's primary bbox, and mark bbox "ownership"
        # Ownership constraint: a bbox can be owned by at most one canonical group.
        bbox_owner: Dict[int, str] = {}
        canon_primary_bbox: Dict[str, Optional[int]] = {}

        for gi, cj in assignment:
            ck = canon_keys[gi]
            if cj >= B:
                canon_primary_bbox[ck] = None
                continue
            # If score is basically zero, treat as unassigned
            if score_gb[gi, cj] < 1e-3:
                canon_primary_bbox[ck] = None
                continue
            canon_primary_bbox[ck] = cj
            bbox_owner[cj] = ck

        # Initial per-receptacle mapping: everyone gets their group's primary bbox (may be None)
        for i, k in enumerate(receptacle_keys):
            ck = self._canonical_receptacle_key(k)
            receptacle_to_map[k] = canon_primary_bbox.get(ck, None)

        # Refinement: allow a receptacle to switch to a different bbox if:
        #  - it is clearly better (margin), and
        #  - that bbox is not owned by another canonical group (or is owned by same group),
        for i, k in enumerate(receptacle_keys):
            ck = self._canonical_receptacle_key(k)

            current = receptacle_to_map[k]
            current_iou = iou_rb[i, current] if (current is not None) else 0.0

            best_j = int(np.argmax(iou_rb[i, :]))
            best_iou = float(iou_rb[i, best_j])

            if best_iou < 1e-3:
                continue

            owner = bbox_owner.get(best_j, None)
            if owner is not None and owner != ck:
                # would steal from another canonical object group -> not allowed
                continue

            if (best_iou - current_iou) >= BREAKAWAY_MARGIN:
                receptacle_to_map[k] = best_j
                # claim it for this canonical group (safe because either unclaimed or same owner)
                bbox_owner[best_j] = ck

        return receptacle_to_map

    def point_to_aabb_distance(self, point: np.ndarray, corners: np.ndarray) -> float:
        """
        Computes distance from a point to an axis-aligned bounding box.
        Returns 0 if inside the box.
        """
        b_min = np.min(corners, axis=0)
        b_max = np.max(corners, axis=0)

        d_vars = np.maximum(0, b_min - point) + np.maximum(0, point - b_max)

        dist_sq = np.sum(d_vars**2)

        return np.sqrt(dist_sq)

    def find_receptacle(
        self, object_centroid: List[float], receptacles: Dict
    ) -> Optional[str]:
        """
        Finds the closest receptacle to the object centroid based on Euclidean distance
        to the receptacle's center.
        """
        if not receptacles:
            return None

        map_point = np.array(object_centroid)

        closest_rec = None
        min_dist_sq = (
            1.0  # Pickupable has to be at least a meter away from the receptacle
        )

        for rec_name, rec_data in receptacles.items():
            corners = np.asarray(rec_data.get_box_points(), dtype=np.float32)
            dist_sq = self.point_to_aabb_distance(map_point, corners)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_rec = rec_name

        return closest_rec

    def process_queries(self, queries: List[str], **kwargs) -> Dict:
        """
        Returns:
        {
          "query_text": {
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
        results: Dict[str, Dict] = {}
        pickupable_to_map: Dict[str, int] = {}

        log.info("Encoding queries...")
        # Encode queries -> (num_queries, feature_dim)
        queries_cleaned = [
            split_camel_preserve_acronyms(q.split("|")[0]) for q in queries
        ]
        text_features = self.ft_extractor.encode_text(queries_cleaned).cpu()

        # 2. Calculate Similarity Matrix: (num_queries, num_map_objects)
        for i, query_text in tqdm(
            enumerate(queries), total=len(queries), desc="Processing Queries"
        ):
            query_feat = text_features[i].unsqueeze(0)  # (1, dim)

            # Calculate similarity against all map objects
            query_feat = query_feat.to(self.device)
            features = self.features.to(self.device)
            sim_scores = self.semantic_sim(query_feat, features).squeeze().cpu()

            # Get Top K indices
            top_k_indices = torch.argsort(sim_scores, descending=True)[: self.top_k]
            top_k_indices = top_k_indices.cpu().numpy()

            # Initialize output structure for this query
            result_entry = {
                "present": False,
                "map_object_id": None,
                "query_timestamp": [],
                "present_receptacle_name": None,
                "receptacles": [],
            }

            # build receptacles list from mapping
            receptacle_names = self.pickupable_to_receptacles[query_text]
            for rec_name in receptacle_names:
                if rec_name == "OOB_FAKE_RECEPTACLE":
                    continue

                # Map name -> map object id
                rec_map_id = self.receptacle_map_ids[rec_name]
                if rec_map_id is None:
                    # In case the receptacle was not mapped properly
                    # We still include the receptacle but without timestamps
                    result_entry["receptacles"].append(
                        {
                            "receptacle_name": rec_name,
                            "map_object_id": None,
                            "receptacle_timestamps": [],
                        }
                    )
                    continue

                rec_timestamps = self.annotations[rec_map_id]["timestamps"]

                result_entry["receptacles"].append(
                    {
                        "receptacle_name": rec_name,
                        "map_object_id": rec_map_id,
                        "receptacle_timestamps": rec_timestamps,
                    }
                )

            # 3. Verify candidates to find where the pickupable currently is
            for idx in top_k_indices:
                idx = int(idx)  # Ensure python int
                obj_data = self.annotations[idx]

                # Get images
                images = self.get_object_images(idx, limit=self.verifier.max_images)
                if len(images) == 0:
                    log.warning(f"No images found for object ID {idx}. Skipping.")
                    continue

                # Verify
                query_text_cleaned = split_camel_preserve_acronyms(
                    query_text.split("|")[0]
                )
                is_match = self.verifier(images, query_text_cleaned)

                if is_match and idx in self.receptacle_map_ids.values():
                    log.warning(
                        f"Verified object ID {idx} for query '{query_text}' is actually a receptacle. Skipping."
                    )
                    is_match = False

                if is_match:
                    # 4. Spatial Association: which receptacle does it belong to now?
                    # NOTE from @kumaradityag: It is possible that the rec_name is not the correct receptacle, if the incorrect object was retrieved
                    centroid = obj_data["centroid"]
                    rec_name = self.find_receptacle(centroid, self.gt_receptacles_aabb)

                    result_entry["present"] = True
                    result_entry["map_object_id"] = idx
                    result_entry["query_timestamp"] = obj_data.get("timestamps", [])
                    result_entry["present_receptacle_name"] = rec_name

                    pickupable_to_map[query_text] = idx
                    break

            results[query_text] = result_entry

        self.pickupable_map_ids = pickupable_to_map
        return results

    def save_results(self, results, output_path, sssd_data: object):
        """Save results using SSSD class."""
        import polars as pl

        # TODO: @miguel: Ask charlie how to save this using her thing.
        timestamps = np.concatenate(
            [data._timestamp for data in sssd_data.get_generator_of_selves()]
        )
        priviliged_assignments = np.concatenate(
            [data._assignment for data in sssd_data.get_generator_of_selves()]
        )
        receptacle_to_int = {
            name: i for i, name in enumerate(sssd_data.receptacles_in_scene)
        }
        pickupable_to_int = {
            name: i for i, name in enumerate(sssd_data.pickupables_in_scene)
        }

        # -2 is not valid receptacle, -1 is not observed, 0 is absent and 1 is present
        assignments = np.ones_like(priviliged_assignments, dtype=int) * -2

        for p_name, res_data in results.items():
            p_idx = pickupable_to_int[p_name]
            # Populate receptacles that are either non-observed (-1) or absent (0)
            for receptacle in res_data["receptacles"]:
                rec_name = receptacle["receptacle_name"]
                if rec_name == "OOB_FAKE_RECEPTACLE":
                    continue
                # Populating non-observed for all timestamps (-1)
                r_idx = receptacle_to_int[rec_name]
                assignments[:, p_idx, r_idx] = -1
                # Populate absent for all receptacles (0)
                for t in receptacle["receptacle_timestamps"]:
                    t_idx = np.argmin(np.abs(timestamps - t))
                    assignments[t_idx, p_idx, r_idx] = 0
            # Populate presence (1) if it occurs
            if res_data["present"]:
                rec_name = res_data["present_receptacle_name"]
                r_idx = receptacle_to_int[rec_name]
                # Timestamps where the pickupable was observed
                for t in res_data["query_timestamp"]:
                    t_idx = np.argmin(np.abs(timestamps - t))
                    assignments[t_idx, p_idx, r_idx] = 1
                # Timestamps where the receptacle was observed with the pickupable
                for rec_entry in res_data["receptacles"]:
                    if rec_entry["receptacle_name"] != rec_name:
                        continue
                    for t in rec_entry["receptacle_timestamps"]:
                        t_idx = np.argmin(np.abs(timestamps - t))
                        assignments[t_idx, p_idx, r_idx] = 1

        # Create res directory if it doesn't exist
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save skeleton, delete old parquets and add new one
        sssd_data.dump_to_parquet(output_path, dump_leftover=True, verbose=False)
        parquet_paths = natsorted(glob.glob(str(output_path / "*.parquet")))
        for pq_path in parquet_paths:
            os.remove(pq_path)

        # Create dummy data for non-used keys
        data = pl.DataFrame({"timestamp": timestamps, "assignment": assignments})
        # Memory efficient way to add empty columns
        data = data.with_columns(
            [
                pl.lit(None, dtype=pl.List(pl.Float64)).alias("position"),
                pl.lit(None, dtype=pl.List(pl.Float64)).alias("rotation"),
                pl.lit(None, dtype=pl.List(pl.Float64)).alias("aabb_center"),
                pl.lit(None, dtype=pl.List(pl.Float64)).alias("aabb_size"),
                pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias(
                    "aabb_cornerPoints"
                ),
                pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias(
                    "oobb_cornerPoints"
                ),
            ]
        )
        data.write_parquet(output_path / "scan_0.parquet")

        # Save receptacles map ids
        with open(output_path / "receptacle_map_ids.json", "w") as f:
            json.dump(self.receptacle_map_ids, f, indent=4)

        # Save pickupable map ids
        pickupable_map_ids = {}
        for p_name, res_data in results.items():
            if res_data["present"]:
                pickupable_map_ids[p_name] = res_data["map_object_id"]
        with open(output_path / "pickupable_map_ids.json", "w") as f:
            json.dump(pickupable_map_ids, f, indent=4)

        # Save results in original format in case we want to plot
        with open(output_path / "query_results.json", "w") as f:
            json.dump(results, f, indent=4)

    def visualize(self, object_map_path: str, results: Dict):
        """Visualize all object point clouds with labels using O3DVisualizer."""
        from concept_graphs.inference.toolbox.ObjectMapToolbox import ObjectMapToolbox
        from concept_graphs.viz.server.VerifierServer import VerifierServer

        toolbox = ObjectMapToolbox(object_map_path, self.ft_extractor)
        server = VerifierServer(
            self.pickupable_bbox,
            self.receptacles_bbox,
            self.receptacle_map_ids,
            self.pickupable_existence,
            results,
            toolbox,
        )
        server.spin()


class QueryObjectsGT(QueryObjects):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pickupable_map_ids = self.get_pickupable_map_ids()

    def get_pickupable_map_ids(self) -> Dict[str, Optional[int]]:
        """
        One-to-one assignment of pickupables -> map bbox indices using Hungarian matching (max IoU).
        """
        pickupable_to_map: Dict[str, Optional[int]] = {}

        # 1) Keep only existing pickupables
        p_keys_all = list(self.pickupable_bbox.keys())
        p_keys = [k for k in p_keys_all if self.pickupable_existence[k]]

        # Assign None for non-existing upfront
        for k in p_keys_all:
            if not self.pickupable_existence[k]:
                pickupable_to_map[k] = None

        # Early exit if no pickupables in the scene
        if len(p_keys) == 0 or len(self.bbox) == 0:
            for k in p_keys:
                pickupable_to_map[k] = None
            return pickupable_to_map

        # 2) Exclude bboxes already claimed by receptacles
        forbidden = {v for v in self.receptacle_map_ids.values() if v is not None}
        allowed_bbox_indices = [j for j in range(len(self.bbox)) if j not in forbidden]

        if len(allowed_bbox_indices) == 0:
            for k in p_keys:
                pickupable_to_map[k] = None
            return pickupable_to_map

        # 3) Build IoU matrix: (#pickupables x #allowed_bboxes)
        P = len(p_keys)
        A = len(allowed_bbox_indices)

        # Precompute bbox corners for allowed bboxes
        allowed_bbox_corners: List[np.ndarray] = []
        for j in allowed_bbox_indices:
            allowed_bbox_corners.append(
                np.asarray(self.bbox[j].get_box_points(), dtype=np.float32)
            )

        iou_pa = np.zeros((P, A), dtype=np.float32)
        for i, p_key in enumerate(p_keys):
            p_corners = np.array(
                self.pickupable_bbox[p_key]["cornerPoints"], dtype=np.float32
            )
            for a, box_pts in enumerate(allowed_bbox_corners):
                iou_pa[i, a] = aabb_iou(p_corners, box_pts)

        # 4) Hungarian solves min-cost; we want max IoU => cost = 1 - iou
        # Add dummy columns so some pickupables can go unmatched.
        n_cols = max(A, P)
        cost = np.ones(
            (P, n_cols), dtype=np.float32
        )  # dummy columns => iou=0 => cost=1
        cost[:, :A] = 1.0 - iou_pa

        row_ind, col_ind = linear_sum_assignment(cost)

        # 5) Decode assignment
        for r, c in zip(row_ind, col_ind):
            p_key = p_keys[r]
            if c >= A:
                # matched to dummy => no map id
                pickupable_to_map[p_key] = None
                continue

            chosen_allowed_col = c
            chosen_bbox_idx = allowed_bbox_indices[chosen_allowed_col]
            chosen_iou = float(iou_pa[r, chosen_allowed_col])

            if chosen_iou < 1e-3:
                pickupable_to_map[p_key] = None
            else:
                pickupable_to_map[p_key] = chosen_bbox_idx

        return pickupable_to_map

    def process_queries(self, queries: List[str]) -> Dict:
        """
        GT-based query processing.
        Skips retrieval and verification and directly uses ground-truth
        pickupable to map-object association.
        """

        results: Dict[str, Dict] = {}

        for query_text in queries:
            result_entry = {
                "present": False,
                "map_object_id": None,
                "query_timestamp": [],
                "present_receptacle_name": None,
                "receptacles": [],
            }

            # 1. Build receptacles list (same as base class)
            receptacle_names = self.pickupable_to_receptacles.get(query_text, [])

            for rec_name in receptacle_names:
                if rec_name == "OOB_FAKE_RECEPTACLE":
                    continue

                rec_map_id = self.receptacle_map_ids.get(rec_name)

                if rec_map_id is None:
                    result_entry["receptacles"].append(
                        {
                            "receptacle_name": rec_name,
                            "map_object_id": None,
                            "receptacle_timestamps": [],
                        }
                    )
                    continue

                result_entry["receptacles"].append(
                    {
                        "receptacle_name": rec_name,
                        "map_object_id": rec_map_id,
                        "receptacle_timestamps": self.annotations[rec_map_id].get(
                            "timestamps", []
                        ),
                    }
                )

            # 2. GT pickupable existence check
            exists = self.pickupable_existence.get(query_text, False)
            map_id = self.pickupable_map_ids.get(query_text)

            if not exists or map_id is None:
                # Object is not present — still return receptacle candidates
                results[query_text] = result_entry
                continue

            # 3. Fill in GT object info
            obj_data = self.annotations[map_id]
            centroid = obj_data["centroid"]

            present_receptacle = self.find_receptacle(
                centroid, self.gt_receptacles_aabb
            )

            result_entry["present"] = True
            result_entry["map_object_id"] = map_id
            result_entry["query_timestamp"] = obj_data.get("timestamps", [])
            result_entry["present_receptacle_name"] = present_receptacle

            results[query_text] = result_entry

        return results
