import functools
import random
import matplotlib.pyplot as plt
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

        self.possible_agents = [f"groud_station_{i}" for i in range(self.GS_size)]
    
    def _SAT_coordinate(self, SAT, SAT_len, time, speed):
        """
        return real-time SAT center point
        """
        _SAT = np.copy(SAT)
        for i in range(SAT_len):
            _SAT[i,0] = 65*i -speed * time
            _SAT[i,1] = 10

            _SAT[i + SAT_len,0] = -25 + 65*i -speed * time
            _SAT[i+ SAT_len,1] =  10 + 65
        
        return _SAT

    def _SAT_coverage_position(self, SAT_coverage, SAT_len, time, speed, radius, theta):
        """
        return real-time SAT coverage position
        for render
        """
        _SAT_coverage = np.copy(SAT_coverage)
        for i in range(SAT_len):
            _SAT_coverage[i,0,:] = 65*i -speed * time + radius * np.cos(theta)
            _SAT_coverage[i,1,:] =  10                + radius * np.sin(theta)

            _SAT_coverage[i + SAT_len,0,:] = -25 + 65*i -speed * time + radius * np.cos(theta)
            _SAT_coverage[i + SAT_len,1,:] =  10 +   65               + radius * np.sin(theta)

        return _SAT_coverage

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
            observations[self.agents[i]] = observation

        return observations
    

    def step(self, actions):
        # Execute actions and Get Rewards
        # Action must select a covering SAT
        GS_actions = actions["ground_stations"] 

        rewards = {a: 0 for a in self.agents}
        for i in range(self.GS_size):
            reward = 0
            # HO occur
            if self.service_indicator[GS_actions[i]] == 0:
                reward = -10
            else:
                if GS_actions.count(GS_actions[i]) > self.SAT_Load:
                    reward = -5
                else:
                    reward = self.visible_time[i][GS_actions[i]]
            reward[self.agents[i]] = reward
        
        # Check termination conditions
        terminations = {a: False for a in self.agents}        

        if self.timestep == self.terminal_time:
            terminations = {a: True for a in self.agents}
            self.agents = []

        # Get info
        infos = {}
        for i in range(self.GS_size):
            infos[self.agents[i]] = {"time step":self.timestep, "selected SAT":GS_actions[i], "status":rewards[i]}

        # Get obersvations
        self.timestep += 1
        
        # Get SAT position
        self.SAT_point = self._SAT_coordinate(self.SAT_point, self.SAT_len, self.timestep, self.SAT_speed)
        # Get coverage indicator
        self.coverage_indicator = self._is_in_coverage(self.SAT_point, self.GS, self.SAT_coverage_radius)
        # Get visible time        
        for i in range(self.GS_size):
            for j in range(self.SAT_len*2):
                self.visible_time[i][j] = self._get_visible_time(self.SAT_point[j], self.SAT_speed, self.SAT_coverage_radius, self.GS[i])
        
        observations = {}
        for i in range(self.GS_size):
            observation = (
                self.coverage_indicator[i],
                self.SAT_Load,
                self.visible_time[i]
            )
            observations[f"groud_station_{i}"] = observation

        # Get truncations: this senario does not need
        truncations = {self.agents[i]: False for i in range(self.GS_size)}

        return observations, rewards, terminations, truncations, infos
        
    def render(self):
        """
        Caution time step !!
        -> execute this func before step func
        """
        figure, axes = plt.subplots(1)
        
        SAT_area = self._SAT_coverage_position(self.SAT_coverage, self.SAT_len, self.timestep, self.SAT_speed, self.SAT_coverage_radius, self.theta)        

        for i in range(self.SAT_len * self.SAT_plane):
            axes.plot(SAT_area[i,0,:], SAT_area[i,1,:])
        
        axes.plot([0,100,100,0,0], [0,0,100,100,0])
        axes.plot(self.GS[:,0], self.GS[:,1], '*')

        axes.set_aspect(1)
        axes.axis([-50, 200, -50, 150])
        axes.plot(self.SAT_point[:,0], self.SAT_point[:,1], 'o')

        plt.show()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([self.SAT_len * self.SAT_plane, self.SAT_len * self.SAT_plane, self.SAT_len * self.SAT_plane])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.GS_size)


 # test       
if __name__ == "__main__":
    env = LEOSATEnv()
    env.reset()
    env.render()
    actions = np.random.randint(0,4, (100,2))
    for i in range(100):
        _actions = {
            "prisoner": actions[i][0],
            "guard": actions[i][1]
        }
        (observations, rewards, terminations, truncations, infos) = env.step(_actions)
        env.render()