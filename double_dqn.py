#coding:utf-8
import chainer
from chainer import Function, Variable, optimizers, serializers, cuda
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from collections import deque

class DeepQNetwork:
    def __init__(self, num_in, num_hid1, num_hid2, num_hid3, num_out, pu, epsil, gamma, batch_size, model_name):
        self.Q_network = Chain(
                            hid_layer1 = L.Linear(num_in, num_hid1),
                            hid_layer2 = L.Linear(num_hid1, num_hid2),
                            hid_layer3 = L.Linear(num_hid2, num_hid3),
                            out_layer  = L.Linear(num_hid3, num_out, initialW=np.zeros((num_out, num_hid3), dtype=np.float32)))
        self.Target_network = Chain(
                            hid_layer1 = L.Linear(num_in, num_hid1),
                            hid_layer2 = L.Linear(num_hid1, num_hid2),
                            hid_layer3 = L.Linear(num_hid2, num_hid3),
                            out_layer  = L.Linear(num_hid3, num_out)
                        )
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.Q_network)

        self.xp         = cuda.cupy if pu == "GPU" else np
        self.EPSIL      = epsil
        self.GAMMA      = gamma
        self.BATCH_SIZE = batch_size
        self.record     = deque()
        self.model_name = model_name
        self.__action_num = num_out

    def __forward(self, flg, x, model, t = None):
        _x = Variable(x)
        if flg == 1: _t = Variable(t)
        h1  = F.leaky_relu(model.hid_layer1(_x))
        h2  = F.leaky_relu(model.hid_layer2(h1))
        h3  = F.leaky_relu(model.hid_layer3(h2))
        u3  = model.out_layer(h3)
        return F.mean_squared_error(u3, _t) if flg else u3

    def __backpropagation(self, loss):
        loss.backward()
        self.optimizer.update()

    def __init_grads(self):
        self.optimizer.zero_grads()

    def __get_state_vec(self, state, flg):
        vec_list = [val for val in state]
        vec = self.xp.array(vec_list, dtype=self.xp.float32)
        if flg == 1: return vec
        return self.xp.array([vec], dtype=self.xp.float32)

    def __make_target(self, state, action, reward, next_state, terminal):
        tmp = self.EPSIL
        self.EPSIL = 0
        action, _, action_list = self.policy_egreedy(state, self.Q_network)
        y_target = copy.deepcopy(action_list)
        _, _, action_list_tar = self.policy_egreedy(next_state, self.Target_network)
        self.EPSIL = tmp
        y_target[action] = reward if terminal else reward + self.GAMMA * action_list_tar[action]
        return self.xp.array(y_target, dtype=self.xp.float32)

    def __neural_network(self, state_vec, y_target):
        y_target = self.xp.array(y_target, dtype=self.xp.float32)
        self.__init_grads()
        loss = self.__forward(1, state_vec, self.Q_network, y_target)
        self.__backpropagation(loss)

    def __transelate(self):
        state_vecs  = []
        actions     = []
        rewards     = []
        terminals   = []
        next_state_vecs = []
        for data in self.record:
            state_vecs.append(self.__get_state_vec(list(data[0]), 1))
            actions.append(data[1])
            next_state_vecs.append(self.__get_state_vec(list(data[2]), 1))
            rewards.append(data[3])
            terminals.append(data[4])
        return self.xp.array(state_vecs, dtype=self.xp.float32), self.xp.array(actions), self.xp.array(rewards), self.xp.array(terminals), self.xp.array(next_state_vecs, dtype=self.xp.float32)

    def init_record(self):
        self.record.popleft()

    def stock_record(self, now_state, action, next_state, reward, terminal, action_list):
        self.record.append([tuple(now_state), action, tuple(next_state), reward, terminal, action_list])

    def reduce_epsil(self, episode):
        self.EPSIL = self.EPSIL

    def update_target_network(self):
        self.Target_network = copy.deepcopy(self.Q_network)

    def save_weight(self):
        serializers.save_npz(self.model_name, self.Q_network)

    def load_weight(self):
        import os.path
        if os.path.exists(self.model_name) == False:
            print("Nothing")
            return
        serializers.load_npz(self.model_name, self.Q_network)

    def deep_lean(self, now_state, action, next_state, reward, terminal, action_list):
        state_vec = self.__get_state_vec(now_state, 2)
        target = [self.__make_target(action, reward, next_state, terminal, action_list)]
        self.__neural_network(state_vec, target)

    def deep_learn(self, now_state, action, next_state, reward, terminal, action_list):
        state_vec = self.__get_state_vec(now_state, 2)
        target = [self.__make_target(action, reward, next_state, terminal, action_list)]

    def policy_egreedy_ver2(self, state, model):
        state_vec = self.__get_state_vec(state, 2)
        import scipy.spatial.distance
        qvalue_list = []
        tmp = []
        qvalue_list.append(self.__forward(0, state_vec, model).data[0])
        qvalue_list.append(self.__forward(0, state_vec, model).data[0])
        qvalue_vec = np.array(self.__forward(0, state_vec, model).data[0])
        for qvalue in qvalue_list:
            sim = 1 - scipy.spatial.distance.cosine(self.xp.array(qvalue), qvalue_vec)
            tmp.append(sim)
        if tmp[0] < tmp[1]:
            return (list(qvalue_list[1]).index(max(qvalue_list[1])) if random.random()>self.EPSIL else random.choice([i for i in range(self.__action_num)])), max(qvalue_list[1]), qvalue_list[1]
        else:
            return (list(qvalue_list[0]).index(max(qvalue_list[0])) if random.random()>self.EPSIL else random.choice([i for i in range(self.__action_num)])), max(qvalue_list[0]), qvalue_list[0]

    def policy_egreedy(self, state, model):
        state_vec = self.__get_state_vec(state, 2)
        qvalue = self.__forward(0, state_vec, model).data[0]
        return (np.argmax(qvalue) if random.random() > self.EPSIL else random.choice([i for i in range(self.__action_num)])), max(qvalue), qvalue

    def experience_replay(self):
        state_vecs, actions, rewards, terminals, next_state_vecs = self.__transelate()
        perm = self.xp.random.permutation(len(self.record))
        perm_re = copy.deepcopy(perm)
        for start in perm[::self.BATCH_SIZE]:
            x_batch_state_vecs   = state_vecs[perm[start:start+self.BATCH_SIZE]]
            x_batch_action       = actions[perm[start:start+self.BATCH_SIZE]]
            x_batch_rewards      = rewards[perm[start:start+self.BATCH_SIZE]]
            x_batch_terminals    = terminals[perm[start:start+self.BATCH_SIZE]]
            y_batch_targets      = []
            x_batch_next_state_vecs = next_state_vecs[perm[start:start+self.BATCH_SIZE]]
            for index in range(len(x_batch_action)):
                y_batch_targets.append(self.__make_target(x_batch_state_vecs[index], x_batch_action[index], x_batch_rewards[index], x_batch_next_state_vecs[index], x_batch_terminals[index]))
            self.__neural_network(x_batch_state_vecs, y_batch_targets)
