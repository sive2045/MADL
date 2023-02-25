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

        self.SAT_len = 22
        self.SAT_plane = 2 # of plane
        self.SAT_coverage_radius = 55 # km
        self.SAT_speed = 7.9 # km/s
        self.theta = np.linspace(0, 2 * np.pi, 150)
        self.SAT_point = np.zeros((self.SAT_len * self.SAT_plane, 3)) # coordinate (x, y, z) of SAT center point
        self.SAT_coverage = np.zeros((self.SAT_len * self.SAT_plane, 3, 150)) # coordinate (x, y, z) of SAT coverage
        self.SAT_point[:,2,:] = 500 # km, SAT height 
        self.SAT_coverage[:,2,:] = 500 # km, SAT height

        self.timestep = None
        self.terminal_time = 155 # s

        self.possible_agents = [
            "groud_station_00", "groud_station_01", "groud_station_02",
            "groud_station_03", "groud_station_04", "groud_station_05",
            "groud_station_06", "groud_station_07", "groud_station_08",
            "groud_station_09"
            ]
    
    def _SAT_coordinate(self, SAT, SAT_len, time, speed):
        """
        return real-time SAT center point
        """
        for i in range(SAT_len):
            SAT[i,0] = 65*i -speed * time
            SAT[i,1] = 10

            SAT[i + SAT_len,0] = -25 + 65*i -speed * time
            SAT[i+ SAT_len,1] =  10 + 65
        
        return SAT

    def _SAT_coverage_position(self, SAT_coverage, SAT_len, time, speed, radius, theta):
        """
        return real-time SAT coverage position
        for render
        """
        for i in range(SAT_len):
            SAT_coverage[i,0,:] = 65*i -speed * time + radius * np.cos(theta)
            SAT_coverage[i,1,:] =  10                + radius * np.sin(theta)

            SAT_coverage[i + SAT_len,0,:] = -25 + 65*i -speed * time + radius * np.cos(theta)
            SAT_coverage[i + SAT_len,1,:] =  10 +   65               + radius * np.sin(theta)

        return SAT_coverage

    def _is_in_coverage(self, SAT, GS):
        pass

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)

        self.timestep = 0

        # set GS position
        for i in range(self.GS_size):
            self.GS[i][0] = np.random.randint(0,100 + 1)
            self.GS[i][1] = np.random.randint(0,100 + 1)

        # set SAT position
        self.SAT_point = self._SAT_coordinate(self.SAT_point, self.SAT_len, self.timestep, self.SAT_speed)

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
        GS_actions = actions["ground_stations"]          # one-hot encoding
        

        
        


        
