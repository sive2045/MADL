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
    
    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)

        self.timestep = 0

        self.GS_x = np.random.randint(0,100 + 1) # (0~101]
        self.GS_y = np.random.randint(0,100 + 1)
        self.GS_z = 0

        self.SBS_x = 0 + np.random.normal(0,1)
        self.SBS_y = 0 + np.random.normal(0,1)
        self.SBS_z = 540 + np.random.normal(0,1)

        observation = (
            [self.GS_x, self.GS_y, self.GS_z],
            [self.SBS_x, self.SBS_y, self.SBS_z]
        )
        observations = {
            "ground_stations": observation,
            "satellite_basestation": observation
        }

        return observations
        
    def step(self, actions):
        # Execute actions
        GS_action = actions["ground_station"]
        SBS_action = actions["satellite_basesations"]
