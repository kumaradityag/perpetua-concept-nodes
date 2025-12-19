from typing import List, Optional
from scipy.spatial.transform import Rotation as Rsc
import numpy as np
import viser

from concept_graphs.inference.toolbox.PerpetuaMapToolbox import PerpetuaMapToolbox
from concept_graphs.viz.server.ObjectMapServer import ObjectMapServer

import logging

log = logging.getLogger(__name__)


class PerpetuaMapServer(ObjectMapServer):
    def __init__(self, toolbox: PerpetuaMapToolbox, point_shape: str = "circle"):
        super().__init__(toolbox, point_shape)

        # Containers
        self.query_time = None

        # GUI
        with self.server.gui.add_folder("Temporal Query"):
            self.time_gui_number = self.server.gui.add_number(
                "Time (hours)", initial_value=0, min=0
            )
            self.time_gui_button = self.server.gui.add_button(
                "Predict at Time",
                icon=viser.Icon.MOUSE,
            )
            self.time_gui_button.on_click(self.on_time_query_submit)

    def reset(self):
        super().reset()
        # Add extra reset logic if needed

    def spin(self):
        while True:
            pass

    def on_time_query_submit(self, data):
        query_time = self.time_gui_number.value
        self.toolbox.temporal_map_query(query_time)
        self.update_object_map(self.object_map)
