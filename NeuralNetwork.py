# coding:utf-8
import chainer
from chainer import Function, Variable, optimizers, serializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork:
    def __init__(self, num_in, num_hid1, num_hid2, num_hid3, num_out):
        self.model = Chain(hid_layer1 = L.Linear(num_in, num_hid1),
                           hid_layer2 = L.Linear(num_hid1, num_hid2),
                           hid_layer3 = L.Linear(num_hid2, num_hid3),
                           out_layer  = L.Linear(num_hid3, num_out))
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
    
    def forward(self, flg, x, t = None):
        _x = Variable(x)
        if flg == 1: _t = Variable(t)
        h1  = F.dropout(F.relu(self.model.hid_layer1(_x)))
        h2  = F.dropout(F.relu(self.model.hid_layer2(h1)))
        h3  = F.dropout(F.relu(self.model.hid_layer3(h2)))
        u3  = self.model.out_layer(h3)
        # return F.softmax_cross_entropy(u2, _t) if flg else F.softmax(u2)
        # return F.mean_squared_error(self.policy_greedy(u3), _t) if flg else u3
        return F.mean_squared_error(u3, _t) if flg else u3

    def backpropagation(self, loss):
        loss.backward()
        self.optimizer.update()

    def init_grads(self):
        self.optimizer.zero_grads()

    def save_weight(self):
        serializers.save_npz("my.model", self.model)

    def load_weight(self):
        serializers.load_npz("my.model", self.model)

    def policy_greedy(self, actions):
        return np.max(actions.data, axis = 1)

class Gragh:
    def gragh(self, result, epoch):
        x = np.arange(0, epoch, 1)
        left = np.array(x)
        count = 0
        count_1 = 0
        parcent = []
        for i in result:
            count += 1
            if i == 1:
                count_1 += 1
            parcent.append(count_1/count)
        height = np.array(parcent)
        plt.plot(left, height)
    
