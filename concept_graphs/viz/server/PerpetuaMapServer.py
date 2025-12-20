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

    # Register here all callbacks that are resource intenseful and need to be called in the main loop
    def _callbacks(self):
        super()._callbacks()
        if self.query_time is not None:
            self.toolbox.temporal_map_query(self.query_time)
            self._update_server_state()
            self.display_object_rgb()
            self.query_time = None

    def on_time_query_submit(self, data):
        self.query_time = self.time_gui_number.value
