import math
import random
import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon, num_Qaction):
        self.EPSILON = epsilon
        self.__num_Qaction = num_Qaction
        self.__first_action_list = [i for i in range(num_Qaction)]
        self.__action_list = [1 for i in range(num_Qaction)]
        self.__reward_list = [0 for i in range(num_Qaction)]
        self.__reward_ave  = [0 for i in range(num_Qaction)]
        self.__use_action = []

    def __first_get_action(self):
        action = random.choice(self.__first_action_list)
        self.__use_action.append(action)
        self.__first_action_list = list(set(self.__first_action_list) - set([action]))
        return action

    def get_action(self):
        if len(self.__first_action_list) != 0:
            return self.__first_get_action()
        else:
            return np.argmax(self.__reward_ave) if random.random() > self.EPSILON else random.choice([i for i in range(self.__num_Qaction)])

    def reserve_reward(self, action, reward):
        self.__reward_list[action] += reward
        self.__reward_ave[action] = self.__reward_list[action] / self.__action_list[action]

class UpperConfidenceBound:
    def __init__(self, num_Qaction):
        self.__num_Qaction = num_Qaction
        self.__arm_reward  = [0 for i in range(num_Qaction)]
        self.__arm_count   = [1 for i in range(num_Qaction)]
        self.__play_count  = 2
        self.__first_action= [i for i in range(num_Qaction)]
        self.__ucb_action  = [0 for i in range(num_Qaction)]
        self.__now_action  = None

    def __update_ucb(self, k):
        self.__ucb_action[k] = (self.__arm_reward[k]/self.__arm_count[k]) + math.sqrt((1/self.__arm_count[k]) * math.log(self.__play_count))

    def get_action(self):
        self.__play_count += 1
        if len(self.__first_action) > 0:
            self.__now_action = self.__first_action[0]
            self.__first_action.pop(0)
        else:
            self.__now_action = random.choice([i for i, x in enumerate(self.__ucb_action) if x == max(self.__ucb_action)])
        return self.__now_action

    def reserve_reward(self, reward):
        self.__arm_count[self.__now_action] += 1
        self.__update_ucb(self.__now_action)
        self.__arm_reward[self.__now_action] += reward
