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

    def serve_reward(self, action, reward):
        self.__reward_list[action] += reward
        self.__reward_ave[action] = self.__reward_list[action] / self.__action_list[action]
