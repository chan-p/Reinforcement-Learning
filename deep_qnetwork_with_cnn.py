import cv2
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

class DeepQNetworkWithCNN:
    def __init__(self, num_hid1, num_hid2, num_out, pu, epsil, gamma, batch_size, model_name):
        self.xp         = cuda.cupy if pu == "GPU" else np
        self.__STATE_LENGTH = 4  # 状態を構成するフレーム数
        self.__FRAME_WIDTH = 128  # リサイズ後のフレーム幅
        self.__FRAME_HEIGHT = 128  # リサイズ後のフレーム高さ

        self.Q_network = Chain(
            conv1= L.Convolution2D(self.__STATE_LENGTH, 32, 8, stride=4),
            conv2= L.Convolution2D(32, 256,  5, pad=2),
            conv3= L.Convolution2D(256, 32,  3, pad=1),
            nn4  = L.Linear(num_hid1, num_hid2),
            nn5  = L.Linear(num_hid2, 100),
            nn6  = L.Linear(100, num_out, initialW=self.xp.zeros((num_out, num_hid2), dtype=self.xp.float32))
        )
        self.Target_network = copy.deepcopy(self.Q_network)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.Q_network)
        self.EPSIL      = epsil
        self.GAMMA      = gamma
        self.BATCH_SIZE = batch_size
        self.record     = deque()
        self.model_name = model_name
        self.__action_num = num_out
        self.__ww = self.xp.array([[255 for _ in range(self.__FRAME_WIDTH)] for _ in range(self.__FRAME_HEIGHT)])
        self.__repeat_action = None
        self.__repeat_max_action = None
        self.__repeat_action_list = None
        self.__step = 0
        self.action_interval = 4
        self.__action_list   = [i for i in range(self.__action_num)]
        print(self.__action_list)

    def __forward(self, flg, x, model, t = None):
        _x = Variable(x)
        if flg == 1: _t = Variable(t)
        h1 = F.max_pooling_2d(F.leaky_relu(self.Q_network.conv1(_x)), 3, stride=2)
        h2 = F.max_pooling_2d(F.leaky_relu(self.Q_network.conv2(h1)), 3, stride=2)
        h3 = F.max_pooling_2d(F.leaky_relu(self.Q_network.conv3(h2)), 3, stride=2)
        h4 = F.dropout(F.leaky_relu(self.Q_network.nn4(h3)))
        h5 = F.dropout(F.leaky_relu(self.Q_network.nn5(h4)))
        u3 = self.Q_network.nn6(h5)
        return F.mean_squared_error(u3, _t) if flg else u3

    def __rgb2gry(self, state):
        return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (self.__FRAME_WIDTH, self.__FRAME_HEIGHT))

    def pre_proccesing(self, now_state, last_state):
        process_state = self.xp.maximum(now_state, last_state)
        gray_state = self.__rgb2gry(process_state)
        process_state = self.xp.array(gray_state, dtype = self.xp.float32)
        process_state = process_state / self.__ww
        return process_state

    def get_initial_state(self, now_state, last_state):
        process_state = self.xp.maximum(now_state, last_state)
        gray_state = self.__rgb2gry(process_state)
        process_state = self.xp.array(gray_state, dtype = self.xp.float32)
        process_state = process_state / self.__ww
        # 4フレーム分コピーする
        state = [process_state for _ in range(self.__STATE_LENGTH)]
        return self.xp.array(state, dtype=self.xp.float32)

    def __backpropagation(self, loss):
        loss.backward()
        self.optimizer.update()

    def __init_grads(self):
        self.optimizer.zero_grads()

    def __get_state_vec(self, state, flg):
        vec = self.xp.array(state, dtype=self.xp.float32)
        if flg == 1: return vec
        return self.xp.array([vec], dtype=self.xp.float32)

    def __make_target(self, state, action, reward, next_state, terminal):
        tmp = self.EPSIL
        self.EPSIL = 0
        action, _, action_list = self.policy_greedy(state, self.Q_network)
        y_target = copy.deepcopy(action_list)
        _, _, action_list_tar = self.policy_greedy(next_state, self.Target_network)
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

    def __transelate_ver2(self):
        state_vecs  = []
        actions     = []
        rewards     = []
        terminals   = []
        next_state_vecs = []
        record = random.sample(self.record, self.BATCH_SIZE)
        for data in record:
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

    def policy_egreedy(self, state, model, step):
        if step % self.action_interval != 0:
            return self.__repeat_action, self.__repeat_max_action, self.__repeat_action_list
        state_vec = self.__get_state_vec(state, 2)
        qvalue = self.__forward(0, state_vec, model).data[0]
        self.__repeat_action = self.xp.argmax(qvalue)
        self.__repeat_max_action = max(qvalue)
        self.__repeat_action_list = qvalue
        return (self.xp.argmax(qvalue) if random.random() > self.EPSIL else random.choice(self.__action_list)), max(qvalue), qvalue

    def policy_greedy(self, state, model):
        state_vec = self.__get_state_vec(state, 2)
        qvalue = self.__forward(0, state_vec, model).data[0]
        return self.xp.argmax(qvalue), max(qvalue), qvalue

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

    def experience_replay_ver2(self):
        state_vecs, actions, rewards, terminals, next_state_vecs = self.__transelate_ver2()
        start = 0
        x_batch_state_vecs   = state_vecs[start:start+self.BATCH_SIZE]
        x_batch_action       = actions[start:start+self.BATCH_SIZE]
        x_batch_rewards      = rewards[start:start+self.BATCH_SIZE]
        x_batch_terminals    = terminals[start:start+self.BATCH_SIZE]
        y_batch_targets      = []
        x_batch_next_state_vecs = next_state_vecs[start:start+self.BATCH_SIZE]
        for index in range(len(x_batch_action)):
            y_batch_targets.append(self.__make_target(x_batch_state_vecs[index], x_batch_action[index], x_batch_rewards[index], x_batch_next_state_vecs[index], x_batch_terminals[index]))
        self.__neural_network(x_batch_state_vecs, y_batch_targets)
