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

        self.GSs_x = None # km
        self.GSs_y = None # km
        self.GSs_z = 0    # km
        self.request = None
        self.predicted_distance = None # km

        self.SBS_x = None # km
        self.SBS_y = None # km
        self.SBS_z = 540  # km
        self.connected_GSs = None
        self.SBS_power = None

        self.timestep = None

        self.possible_agents = ["groud_station_01", "groud_station_02", "satellite_basesation"]
    
    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)

        self.timestep = 0

        self.GSs_x = np.random.randint(0,100 + 1, 2) # (0~101]
        self.GSs_y = np.random.randint(0,100 + 1, 2)
        self.GSs_z = np.zeros(2)
        self.request = np.zeros(2)
        self.predicted_distance = np.zeros(2)

        self.SBS_x = 0 + np.random.normal(0,1)
        self.SBS_y = 0 + np.random.normal(0,1)
        self.SBS_z = 540 + np.random.normal(0,1)
        self.connected_GSs = [False, False]
        self.allocated_power = [0, 0]
        self.SBS_power = 50 # dB


        GS_observation_info = {
            "coordinate": [self.GSs_x, self.GSs_y, self.GSs_z],
            "request": self.request,
            "predicted distance": self.predicted_distance,
        }
        SBS_observation_info = {
            "coordinate": [self.SBS_x, self.SBS_y, self.SBS_z],
            "connected_gs": self.connected_GSs,
            "allocated_power": self.allocated_power,
        }
        observations = {
            "ground_stations": GS_observation_info,
            "satellite_basestation": SBS_observation_info
        }

        return observations
    
    def _distance(self, GS, SBS) -> float:
        """
        input: GS and SBS ndarray
        """
        return np.linalg.norm(GS-SBS)

    def step(self, actions):
        # Execute actions
        GS_actions = actions["ground_stations"]          # 1: connect rq 0: backoff
        SBS_actions = actions["satellite_basesations"]   # decision power

        for i in range(len(GS_actions)):
            self.connected_GSs = GS_actions[i]
            self.allocated_power = SBS_actions[i]

            if self.connected_GSs:
                pass
            else:
                if self.allocated_power > 0:
                    SBS_reward = -1
                else:
                    SBS_reward = 0
        


        
