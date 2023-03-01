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

        self.timestep = None
        self.terminal_time = 155 # s

        self.SAT_len = 22
        self.SAT_plane = 2 # of plane
        self.SAT_coverage_radius = 55 # km
        self.SAT_speed = 7.9 # km/s, caution!! direction
        self.theta = np.linspace(0, 2 * np.pi, 150)
        self.SAT_point = np.zeros((self.SAT_len * self.SAT_plane, 3)) # coordinate (x, y, z) of SAT center point
        self.SAT_coverage = np.zeros((self.SAT_len * self.SAT_plane, 3, 150)) # coordinate (x, y, z) of SAT coverage
        self.SAT_point[:,2] = 500 # km, SAT height 
        self.SAT_coverage[:,2,:] = 500 # km, SAT height
        self.SAT_Load = np.full(self.SAT_len*self.SAT_plane, 5) # the available channels of SAT
        self.SAT_W = 10 # MHz BW budget of SAT

        self.service_indicator = np.zeros((self.GS_size, self.SAT_len*2)) # indicator: users are served by SAT (one-hot vector)

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

    def _is_in_coverage(self, SAT, GS, coverage_radius):
        """
        return coverage indicator (one-hot vector)
        """
        dist = np.zeros((len(GS), len(SAT)))
        coverage_indicator = np.zeros((len(GS), len(SAT)))

        for i in range(len(GS)):
            for j in range(len(SAT)):
                dist[i][j] = np.linalg.norm(GS[i] - SAT[j])
        
        coverage_index = np.where(dist <= coverage_radius)
        coverage_indicator[coverage_index[:][0], coverage_index[:][1]] = 1

        return coverage_indicator        

    def _get_visible_time(self, SAT_point, SAT_speed, coverage_radius, GS):
        """
        return visible time btw SAT and GS
        """
        visible_time = (np.sqrt(coverage_radius ** 2 - (GS[1]-SAT_point[1]) ** 2) - GS[0] + SAT_point[0]) / SAT_speed
        visible_time = np.max((visible_time, 0))
        
        return visible_time        


    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)

        self.timestep = 0

        # set GS position
        for i in range(self.GS_size):
            self.GS[i][0] = np.random.randint(0,100 + 1)
            self.GS[i][1] = np.random.randint(0,100 + 1)

        # set SAT position
        self.SAT_point = self._SAT_coordinate(self.SAT_point, self.SAT_len, self.timestep, self.SAT_speed)
        # coverage indicator
        self.coverage_indicator = self._is_in_coverage(self.SAT_point, self.GS, self.SAT_coverage_radius)
        # visible time
        self.visible_time = np.zeros((self.GS_size,self.SAT_len*2))
        for i in range(self.GS_size):
            for j in range(self.SAT_len*2):
                self.visible_time[i][j] = self._get_visible_time(self.SAT_point[j], self.SAT_speed, self.SAT_coverage_radius, self.GS[i])

        # observations
        observations = {}
        for i in range(self.GS_size):
            observation = (
                self.coverage_indicator[i],
                self.SAT_Load,
                self.visible_time[i]
            )
            observations[f"groud_station_0{i}"] = observation

        return observations
    

    def step(self, actions):
        # Execute actions
        GS_actions = actions["ground_stations"]          # one-hot encoding

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        if self.timestep == self.terminal_time:
            terminations = {a: True for a in self.agents}
            self.agents = []

        # Get obersvations

        # Rewards
        

        
        


        
