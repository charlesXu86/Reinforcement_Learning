# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   dqn.py
 
@Time    :   2019-09-28 16:52
 
@Desc    :
 
'''

import tensorflow as tf
import numpy as np

class DeepQNetwork():

    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False
                 ):
        '''

        :param n_actions:
        :param n_features:
        :param learning_rate:
        :param reward_decay:
        :param e_greedy:
        :param replace_target_iter:
        :param memory_size:
        :param batch_size:
        :param e_greedy_increment:
        :param output_graph:
        '''
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    def _build_net(self):
        pass
        