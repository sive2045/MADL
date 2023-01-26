from __future__ import division
import numpy as np
import time
import random
import math


class Environment:
    def __init__(self):
        self.Q_HAP_initial, self.V_HAP_initial = [2285e3, 2380e3, 50e3], [0, 0, 0]                      ### Need to Set Carefully
        self.Q_HAP, self.V_HAP = np.array(self.Q_HAP_initial), np.array(self.V_HAP_initial)
        self.A_HAP_list = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]

        # self.association_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.association_list = [[1, 0], [0, 1]]
        self.CurrentTimeSlot = 0
        self.TimeSlotSize = 10
        self.Q_Source, self.Q_Destination = np.array([0, 0, 0]), np.array([4000e3, 4000e3, 0])

        self.ConstantVelocity_LEO = np.array([0, 7.8e3, 0])
        # self.Num_Sat_Constel = 3
        self.Num_Sat_Constel = 2
        Num_Sat = 22
        Radius_LEO = 6371e3
        Circumstance_LEO = Radius_LEO * 2 * np.pi
        self.Num_TimeSlot_Constel = math.floor(
            Circumstance_LEO / np.linalg.norm(self.ConstantVelocity_LEO) / self.TimeSlotSize)
        Distance_LEO = Circumstance_LEO / Num_Sat                       # same distance between LEO in same orbit lane
        self.Num_TimeSlot_LEO_LEO = math.floor(
            Distance_LEO / np.linalg.norm(self.ConstantVelocity_LEO) / self.TimeSlotSize)
        self.Q_LEO_initial = np.array([500e3, 0, 550e3])
        self.Q_LEO_Constel_initial = np.zeros([self.Num_Sat_Constel, 3])
        for ii in range(self.Num_Sat_Constel):
            self.Q_LEO_Constel_initial[ii, :] = self.Q_LEO_initial + \
                                                (ii * self.ConstantVelocity_LEO * self.Num_TimeSlot_LEO_LEO * self.TimeSlotSize)
        self.Q_LEO_Constel = np.array(self.Q_LEO_Constel_initial)
        self.Coefficient_Cost_Energy, self.Coefficient_Reward_Rate = 1e-8, 1e0
        self.Done = False

    def Update_LEO(self):
        """" Updates and presets the position of LEO """
        """" Input """
        CurrentTimeSlot_LEO = self.CurrentTimeSlot
        TimeSlotSize_LEO = self.TimeSlotSize
        ConstantVelocity_LEO = self.ConstantVelocity_LEO
        Num_TimeSlot_LEO_LEO = self.Num_TimeSlot_LEO_LEO
        """" Find LEO position over time slot with given initial position of LEO """
        self.Q_LEO_Constel = np.zeros([self.Num_Sat_Constel, 3])
        for ii in range(self.Num_Sat_Constel):
            self.Q_LEO_Constel[ii] = self.Q_LEO_Constel_initial[ii, :] + \
                                     (ConstantVelocity_LEO * (CurrentTimeSlot_LEO % Num_TimeSlot_LEO_LEO) * TimeSlotSize_LEO)
        return self.Q_LEO_Constel

    def Update_HAP(self):
        """ Updates and presets the position and velocity of LEO """
        """ Input """
        TimeSlotSize_HAP = self.TimeSlotSize
        CurrentPosition_HAP = self.Q_HAP
        CurrentVelocity_HAP = self.V_HAP
        Acceleration_HAP = self.A_HAP
        """ Find HAP position over time slot with given acceleration of HAP """
        self.Q_HAP = CurrentPosition_HAP + (CurrentVelocity_HAP * TimeSlotSize_HAP) + (
                    0.5 * Acceleration_HAP * TimeSlotSize_HAP ** 2)
        self.V_HAP = CurrentVelocity_HAP + (Acceleration_HAP * TimeSlotSize_HAP)
        return np.array(self.Q_HAP), np.array(self.V_HAP)

    def Update_Done(self):
        if self.CurrentTimeSlot == self.Num_TimeSlot_Constel:
            self.Done = True
        return self.Done

    def Calc_Distance(self, position1, position2):
        distance = np.linalg.norm(position1 - position2)
        return distance

    def Calc_AchievableRate(self, Distance_Source_LEO, Distance_LEO_HAP, Distance_HAP_Destination):
        """ Calculate and presets the achievable rate for Destination """
        """ Hyper Parameter """
        B_RF = 1e9
        SNR_Ref = 1e10
        pathLossExponent = 2
        # B_FSO = 1e9
        # k1 = 4.325886184958972e+04  # alpha=1/10 and ASNR=10^(2.5)
        # # k2 = 2.412584675656626e-05                        # kim's model with V=10, which is moderate
        # # k2 = 7.021276870408771e-06                        # kim's model with V=15, which is good
        # k2 = 2.298805118092270e-06  # kim's model with V=20, which is very good

        """ Calculate the capacity of RF and FSO """
        def Commun_RF(B_RF, SNR_Ref, pathLossExponent, distance):
            Capacity_RF = B_RF * (np.log2(1 + SNR_Ref / ((distance) ** pathLossExponent)))
            return Capacity_RF
        # def Commun_FSO(B_FSO, k1, k2, distance):
        #     Capacity_FSO = B_FSO * np.log2(1 + k1 * np.exp(-k2 * (distance)))
        #     return Capacity_FSO

        """ Find the capacity for each link """
        Capacity_Source_LEO =       Commun_RF(B_RF, SNR_Ref, pathLossExponent, Distance_Source_LEO)
        Capacity_LEO_HAP =          Commun_RF(B_RF, SNR_Ref, pathLossExponent, Distance_LEO_HAP)
        Capacity_HAP_Destination =  Commun_RF(B_RF, SNR_Ref, pathLossExponent, Distance_HAP_Destination)
        AchievableRate =            min(Capacity_Source_LEO, Capacity_LEO_HAP, Capacity_HAP_Destination)
        return Capacity_Source_LEO, Capacity_LEO_HAP, Capacity_HAP_Destination, AchievableRate

    # def Cal_Energy_HAP(self, Velocity, Acceleration, TimeSlotSize):
    #     """ Calculate and presets the consumption energy in HAP """
    #     """ Hyper Parameter """
    #     c1 = 9.26e-04
    #     c2 = 2250
    #     # weight_HAP = 400
    #     g = 9.81  # acceleration of gravity (m/sec)
    #     normVelocity_HAP = max(1e-4, np.linalg.norm(Velocity))
    #     normAcceleration_HAP = np.linalg.norm(Acceleration)
    #     self.ConsumptionEnergy = TimeSlotSize * (
    #                 (c1 * normVelocity_HAP ** 3) + (c2 / normVelocity_HAP) * (1 + (
    #                     normAcceleration_HAP ** 2 - np.dot(Acceleration, Velocity) / normVelocity_HAP ** 2) / (g ** 2))) \
    #         # + ((0.5 * weight_HAP * np.linalg.norm(Velocity)**2 ) - (0.5 * weight * np.linalg.norm(self.Velocity_previous)**2 ))
    #     return self.ConsumptionEnergy

    def Find_Reward(self, AchievableRate, Q_HAP):
        """ Calculate the Reward of Rate """
        AchievableRate_Centroid = 2287885
        # print("AchievableRate:", AchievableRate)
        AchievableRate_Normal = self.Normalization(AchievableRate, AchievableRate_Centroid, 1e6)
        # print("AchievableRate_Normal:", AchievableRate_Normal)
        def sigmoidFunction(rewardInput):
            rewardOutput = 1 / (1+np.exp(-rewardInput))
            return rewardOutput
        # print("sigmoidFunction_Rate:", sigmoidFunction(AchieavableRate_Normal)-0.5)

        """ Calculate the Penalty of Distance from Centroid """
        Centroid = self.Q_HAP_initial
        Distance_HAP_Centroid = np.linalg.norm(Q_HAP - Centroid, 2)
        # print("Distance:", Distance_HAP_Centroid*1e-6)

        reward = (2*sigmoidFunction(AchievableRate_Normal)-1) - (Distance_HAP_Centroid*1e-5)
        # reward = (2 * sigmoidFunction(AchievableRate_Normal) - 1)
        # print("reward:", reward)
        return reward

    def step(self, actions):
        """ 'action' is matching with the association_LEO_HAP and acceleration_HAP """
        num_association_LEO_HAP = int(np.floor(actions / len(self.A_HAP_list)))
        # print(num_association_LEO_HAP)
        num_acceleration_HAP = int(actions % len(self.A_HAP_list))
        # print(num_acceleration_HAP)
        association_LEO_HAP = self.association_list[num_association_LEO_HAP]
        # print(association_LEO_HAP)
        acceleration_HAP = self.A_HAP_list[num_acceleration_HAP]
        # print(acceleration_HAP)
        self.association_LEO_HAP = association_LEO_HAP
        self.A_HAP = np.array(acceleration_HAP)

        """ Update current time slot """
        self.CurrentTimeSlot += 1
        # print("CurrentTimeSlot:", self.CurrentTimeSlot)
        # self.CurrentTimeSlot = 0
        # print("Warning: Now Current Time Slot is set to a certain constant")

        """ Update the position of LEOs """
        Q_LEO_Constel = self.Update_LEO()
        # print(Q_LEO_Constel)

        """ Select the LEO in constellation, and Find the position of selected LEO (association) """
        Q_LEO_selected = np.dot(self.association_LEO_HAP, self.Q_LEO_Constel)
        # print(Q_LEO_selected)

        """ Update the position and velocity of HAP with given acceleration """
        # Q_HAP_prev, V_HAP_prev = self.Q_HAP, self.V_HAP
        Q_HAP, V_HAP = self.Update_HAP()
        # print("Q_HAP, V_HAP:", Q_HAP, V_HAP)

        """ Update distances between Source-selected LEO and selected LEO-HAP and HAP-Destination """
        Distance_Source_LEO = self.Calc_Distance(self.Q_Source, Q_LEO_selected)
        Distance_LEO_HAP = self.Calc_Distance(Q_LEO_selected, Q_HAP)
        Distance_HAP_Destination = self.Calc_Distance(Q_HAP, self.Q_Destination)
        # print("List of Distance:", Distance_Source_LEO, Distance_LEO_HAP, Distance_HAP_Destination)

        """ Find reward """
        Capacity_Source_LEO, Capacity_LEO_HAP, Capacity_HAP_Destination, AchievableRate = self.Calc_AchievableRate(Distance_Source_LEO, Distance_LEO_HAP, Distance_HAP_Destination)
        # print("List of Rate:", Capacity_Source_LEO, Capacity_LEO_HAP, Capacity_HAP_Destination, AchievableRate)
        # ConsumptionEnergy = self.Cal_Energy_HAP(V_HAP_prev, self.A_HAP, self.TimeSlotSize)
        # Reward = self.Find_Reward(AchievableRate, ConsumptionEnergy)
        Reward = self.Find_Reward(AchievableRate, Q_HAP)

        """ Update 'Done' """
        Done = self.Update_Done()
        # print("Done:", Done)
        observations = np.array([Q_LEO_Constel[0][1], Q_LEO_Constel[1][1], Q_HAP[0], Q_HAP[1],
                                 Distance_Source_LEO, Distance_LEO_HAP, Distance_HAP_Destination,
                                 Capacity_Source_LEO, Capacity_LEO_HAP, Capacity_HAP_Destination, AchievableRate])
        # print("observations:", list(observations))

        state_Q_LEO_y_Normal = self.Normalization(observations[0:2], 2000e3, 4000e3)
        state_Q_HAP_Normal = self.Normalization(observations[2:4], 2000e3, 4000e3)
        state_Distance_Normal = self.Normalization(observations[4:7], 0, 1e7)
        state_Rate_Normal = self.Normalization(observations[7:11], 2287885, 1e6)
        # print("state_Distance:", observations[4:7])
        # print("state_Rate:", observations[7:11])
        state = np.concatenate([state_Q_LEO_y_Normal, state_Q_HAP_Normal, state_Distance_Normal, state_Rate_Normal])
        # print("state:", list(state))

        reward = Reward
        # print("reward:", reward)
        done = Done
        info = np.concatenate([observations, [self.CurrentTimeSlot]])
        # print("info:", list(info))
        return state, reward, done, info

    def reset(self):
        self.Q_HAP, self.V_HAP = np.array(self.Q_HAP_initial), np.array(self.V_HAP_initial)
        self.Q_LEO_Constel = self.Q_LEO_Constel_initial
        self.CurrentTimeSlot = 0
        self.Done = False
        # return np.array([self.Q_LEO_Constel_initial[0], self.Q_LEO_Constel_initial[1], self.Q_LEO_Constel_initial[2], self.Q_HAP, self.V_HAP]).flatten()
        state_reset = np.zeros(self.Find_NumState())
        return state_reset

    def Find_NumState(self):
        # num_state = len(np.concatenate([state_Q_LEO_y_Normal, state_Q_HAP_Normal, state_Distance_Normal, state_Rate_Normal]))
        """ Need to Set Manually, Please Be Careful!"""
        num_state = 2 + 2 + 3 + 4                                                       ##### Be Careful!!!!!!!!!!!
        return num_state

    def Find_NumAction(self):
        num_action = self.Num_Sat_Constel * len(self.A_HAP_list)
        return num_action

    def Sample_action_space(self):
            RandomAction = random.randrange(self.Find_NumAction())
            return RandomAction

    def Normalization(self, Value, Average, TotalRange):
        NormalizedValue = (Value - Average) / (TotalRange / 2)
        return NormalizedValue




# if __name__ == "__main__":
#     Env = Environment()
#
#     action = 0
#     experiences = []
#     ep_reward = 0
#
#     # print("Find_NumAction:", Env.Find_NumAction())
#     # print("Find_NumState:", Env.Find_NumState())
#     # print("reset():", Env.reset())
#
#     for i in range(40):
#         observations, reward, done, info = Env.step(action)
#         ep_reward += reward
#         experiences.append(np.concatenate([[action], [reward], [ep_reward], info, [done]]).tolist())
#         print(reward)
        # print(len(np.concatenate([[action], [reward], [ep_reward], info, [done]]).tolist()))

    # print(experiences)
    # experiences_export = np.reshape(experiences, [len(experiences), len(experiences[0])])
    # # print(experiences_export.tolist())
    # # print(experiences_export.tolist()[0])
    # """ Export the Experiences and Rewards and Info """
    # np.savetxt('test.txt', experiences_export.tolist(), fmt='%s', newline='\n', delimiter='; ')


