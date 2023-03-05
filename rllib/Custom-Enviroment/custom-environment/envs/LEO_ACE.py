import functools
import random
import matplotlib.pyplot as plt
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium import spaces

from pettingzoo.utils import agent_selector, wrappers
from pettingzoo import AECEnv

class LEOSATEnv(AECEnv):
    def __init__(self, render_mode=None) -> None:
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

        self.agents = [f"groud_station_{i}" for i in range(self.GS_size)]
        self.possible_agents = self.agents[:]

        self._none = self.SAT_len * self.SAT_plane
        self.action_spaces = {i: spaces.Discrete(self.SAT_len * self.SAT_plane) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(self.SAT_len * self.SAT_plane, self.SAT_len * self.SAT_plane, self.SAT_len * self.SAT_plane), dtype=np.int8
                    ),
                }
            )
            for i in self.agents
        }

        self.render_mode = render_mode        

    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]


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
                dist[i][j] = np.linalg.norm(GS[i,0:2] - SAT[j,0:2]) # 2-dim 
        #print(f"debuging dist: {dist}")
        coverage_index = np.where(dist <= coverage_radius)
        #print(f"debuging index: {coverage_index}")
        coverage_indicator[coverage_index[:][0], coverage_index[:][1]] = 1
        return coverage_indicator        

    def _get_visible_time(self, SAT_point, SAT_speed, coverage_radius, GS):
        """
        return visible time btw SAT and GS
        """
        _num = np.max((coverage_radius ** 2 - (GS[1]-SAT_point[1]) ** 2, 0))
        visible_time = (np.sqrt(_num) - GS[0] + SAT_point[0]) / SAT_speed
        visible_time = np.max((visible_time, 0))
        
        return visible_time        


    def reset(self, seed=None, return_info=False, options=None):
        self.timestep = 0

        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.state = {agent: self._none for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        #self.observations = {agent: self._none for agent in self.agents}

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
        # ====================================================== 수정해야함!
        self.observations = {}
        for i in range(self.GS_size):
            observation = (
                self.coverage_indicator[i],
                self.SAT_Load,
                self.visible_time[i]
            )
            self.observations[self.agents[i]] = observation

        #return self.observations
    
    def observe(self, agent):
        # observation of one agent is the previous state of the other
        print({f"observe: {np.array(self.observations[agent]).shape}"})
        return np.array(self.observations[agent])

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        # Execute actions and Get Rewards
        # Action must select a covering SAT
        agent = self.agent_selection
        print(f"agent: {agent}")
        self.state[self.agent_selection] = action
        print(f"state : {self.state[agent]}")
        if self._agent_selector.is_last():
            # rewards
            for i in range(self.GS_size):
                reward = 0

                # non-coverage area
                if self.coverage_indicator[i][self.state[self.agents[i]]] == 0:
                    reward = -20
                # HO occur
                elif self.service_indicator[i][self.state[self.agents[i]]] == 0:
                    reward = -10
                else:
                # Overload
                    if np.count_nonzero(self.state[self.agents[:]] == self.state[self.agents[i]]) > self.SAT_Load[i]:
                        reward = -5
                    else:
                        reward = self.visible_time[i][self.state[self.agents[i]]]
                self.rewards[self.agents[i]] = reward
        
            # Update service indicator
            self.service_indicator = np.zeros((self.GS_size, self.SAT_len*self.SAT_plane))
            for i in range(self.GS_size):
                self.service_indicator[i][self.state[self.agents[i]]] = 1
    
            # Check termination conditions
            if self.timestep == self.terminal_time:
                self.terminations = {agent: True for agent in self.agents}

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
            
            for i in range(self.GS_size):
                observation = (
                    self.coverage_indicator[i],
                    self.SAT_Load,
                    self.visible_time[i]
                )
                self.observations[f"groud_station_{i}"] = observation
            
            if self.render_mode == "human":
                self.render()
        else:
            self._clear_rewards()
        
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        

        #return self.observations, self.rewards, self.terminations, self.truncations, self.infos
        
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

    


# test       
if __name__ == "__main__":
    env = LEOSATEnv()
    observation = env.reset()
    print(f"init observation: {observation}")
    actions = np.random.randint(0,44, (155,10))
    actions = np.array([
        [1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0]
    ])
    for i in range(3):
        env.render()
        _actions = actions[i]
        print(f"{i}-step selected actions: {_actions}")
        (observations, rewards, terminations, truncations, infos) = env.step(_actions)
        print(f"observations: {observations}")
        print(f"rewards: {rewards}")