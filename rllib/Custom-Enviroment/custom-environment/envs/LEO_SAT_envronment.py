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

        self.GS_size = 10
        self.GS = np.zeros((self.GS_size, 3)) # coordinate (x, y, z) of GS

        self.SAT_len = 130
        self.SAT_coverage_radius = 55 # km
        self.SAT_speed = 7.9 # km/s
        self.theta = np.linspace(0, 2 * np.pi, 150)
        self.SAT_0 = np.zeros((self.SAT_len, 3, 150)) # coordinate (x, y, z) of SAT (plane 1)
        self.SAT_1 = np.zeros((self.SAT_len, 3, 150)) # coordinate (x, y, z) of SAT (plane 2)
        self.SAT_0[:,2,:] = 500 # km, SAT height 
        self.SAT_1[:,2,:] = 500 # km, SAT height 

        self.timestep = None

        self.possible_agents = [
            "groud_station_00", "groud_station_01", "groud_station_02",
            "groud_station_03", "groud_station_04", "groud_station_05",
            "groud_station_06", "groud_station_07", "groud_station_08",
            "groud_station_09"
            ]
    
    def _SAT_location(self, SAT_0, SAT_1, SAT_len, time, speed, radius, theta):
        for i in range(SAT_len):
            SAT_0[i][0][:] = 65*i -speed * time + radius * np.cos(theta)
            SAT_0[i][1][:] =  10                + radius * np.sin(theta)

            SAT_1[i][0][:] = -25 + 65*i -speed * time + radius * np.cos(theta)
            SAT_1[i][1][:] =  10 +   65               + radius * np.sin(theta)

        return SAT_0, SAT_1

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)

        self.timestep = 0

        # set GS position
        for i in range(self.GS_size):
            self.GS[i][0] = np.random.randint(0,100 + 1)
            self.GS[i][1] = np.random.randint(0,100 + 1)

        # set SAT position
        self.SAT_0, self.SAT_1 = self._SAT_location(self.SAT_0, self.SAT_1, self.SAT_len, self.timestep, 
                                                    self.SAT_speed, self.SAT_coverage_radius, self.theta)

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
        


        
