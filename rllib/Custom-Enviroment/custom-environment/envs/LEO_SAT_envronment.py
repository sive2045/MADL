import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv

class LEOSATEnv(ParallelEnv):
    def __init__(self) -> None:
        self.area_max_x = 100    # km
        self.area_max_y = 100    # km
        self.area_max_z = 1_000  # km

        self.GS_x = None # km
        self.GS_y = None # km
        self.GS_z = 0    # km

        self.SBS_x = None # km
        self.SBS_y = None # km
        self.SBS_z = 540  # km

        self.timestep = None

        self.possible_agents = ["groud_station", "satellite_basesation"]