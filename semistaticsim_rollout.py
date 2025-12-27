"""

EVENTUALLY, THIS FILE WILL GO INTO SSS REPO

"""

from abc import ABC
from typing import List, Dict, Union, Self, Any, Tuple

import flax
import numpy as np
from flax import struct

from semistaticsim.datawrangling.sssd import GeneratedSemiStaticData
from semistaticsim.keyboardcontrol.main_skillsim import ROBOTS
from semistaticsim.rendering.simulation.skill_simulator import Simulator

class Agent(ABC):
    def __init__(self, sim: Simulator):
        self.sim = sim

    @property
    def current_time(self):
        return self.sim.sss_data.pickupable_selves_at_current_time._timestamp[0]

    def resolve_query_into_pickupable(self, query: str) -> str:
        # assumes query is a cleaned up name
        for p in self.sim.sss_data.pickupable_names:
            if p.startswith(query):
                return p

    def _predict_object_receptacle(self, pickupable: str, current_time: float = None) -> Tuple[str, Dict[str, float]]:
        """

        Args:
            pickupable: A requested pickupable.
            current_time:

        Returns: A probability weight for each receptacle for the current location of the pickupable.

        """
        raise NotImplementedError()

    def predict_object_receptacle(self, pickupable: str, current_time: float=None) -> Tuple[str, Dict[str, float]]:
        """

        Args:
            pickupable: A pickupable object's id. Must match exactly.

        Returns: Tuple[Most likely receptacle, Dict of probability weights for each receptacle]

        """

        if current_time is None:
            current_time = self.current_time

        return self._predict_object_receptacle(pickupable, current_time)

    def update(self, observation: Dict[str, Any]) -> None:
        """
        Updates self given the observation of the environment

        Args:
            observation: An observation of the current state.

        Returns:

        """
        raise NotImplementedError()

    def found_pickupable(self, receptacle_name: str, pickupable_name: str, observation: Dict[str, Any]) -> bool:
        """

        Args:
            observation: An observation of the current state.

        Returns: True if the current state contains the pickupable, False otherwise.

        """
        raise NotImplementedError()

    def go_to_receptacle(self, receptacle_name: str, looking_for_pickupable: str) -> Tuple[Dict, bool]:
        """

        Args:
            receptacle_name: The name of the target receptacle.
            looking_for_pickupable: The name of the pickupable we are looking for.

        Returns: True if the current state contains the pickupable, False otherwise.

        """
        while True:
            is_pathing_finished = self.sim.GoToObject(ROBOTS[0], receptacle_name, max_path_length=2)
            obs = self.sim.render()
            self.sim.privileged_apn = None

            self.update(obs)

            found_it = self.found_pickupable(receptacle_name, looking_for_pickupable, obs)
            if found_it is not None:
                return obs, found_it
            if is_pathing_finished:
                return obs, False

    def goto_query(self, weighted_receptacles=None, pickupable=None, query: str=None, current_time: float=None, custom_traversal=None) -> Tuple[Dict, bool]:
        if pickupable is None:
            assert query is not None

            pickupables = self.resolve_query_into_pickupable(query, current_time)
            pickupable = pickupables[0]

        if weighted_receptacles is None:
            weighted_receptacles = self.resolve_pickupable_into_weighted_receptacles(pickupable, current_time)

        sorted_keys = sorted(weighted_receptacles, key=weighted_receptacles.get, reverse=True)

        if custom_traversal is not None:
            return custom_traversal(sorted_keys)

        for k in sorted_keys:
            last_obs, found_obj = self.go_to_receptacle(k, pickupable)
            if found_obj:
                return last_obs, True
        return last_obs, False

"""

EVENTUALLY, THIS FILE WILL GO INTO SSS REPO

"""