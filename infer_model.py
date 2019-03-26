import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

import rnn
from reactions import QuadraticEval, ConstraintQuadraticEval, RealReaction
from logger import get_handlers
from collections import namedtuple

logging.basicConfig(level=logging.INFO, handlers=get_handlers())
logger = logging.getLogger()


class StepOptimizer:
    def __init__(self, cell, func, ndim, nsteps, ckpt_path, logger, constraints):
        self.logger = logger
        self.cell = cell
        self.func = func
        self.ndim = ndim
        self.nsteps = nsteps
        self.ckpt_path = ckpt_path
        self.constraints = constraints
        self.init_state = self.cell.get_initial_state(1, tf.float32)
        self.results = self.build_graph()

        self.saver = tf.train.Saver(tf.global_variables())

    def get_state_shapes(self):
        return [(s[0].get_shape().as_list(), s[1].get_shape().as_list())
                for s in self.init_state]

    def step(self, sess, x, y, state):
        feed_dict = {'input_x:0':x, 'input_y:0':y}
        for i in range(len(self.init_state)):
            feed_dict['state_l{0}_c:0'.format(i)] = state[i][0]
            feed_dict['state_l{0}_h:0'.format(i)] = state[i][1]
        new_x, new_state = sess.run(self.results, feed_dict=feed_dict)
        return new_x, new_state

    def build_graph(self):
        x = tf.placeholder(tf.float32, shape=[1, self.ndim], name='input_x')
        y = tf.placeholder(tf.float32, shape=[1, 1], name='input_y')
        state = []
        for i in range(len(self.init_state)):
            state.append((tf.placeholder(
                              tf.float32, shape=self.init_state[i][0].get_shape(),
                              name='state_l{0}_c'.format(i)),
                          tf.placeholder(
                              tf.float32, shape=self.init_state[i][1].get_shape(),
                              name='state_l{0}_h'.format(i))))

        with tf.name_scope('opt_cell'):
            new_x, new_state = self.cell(x, y, state)
            if self.constraints:
                new_x = tf.clip_by_value(new_x, 0.01, 0.99)
        return new_x, new_state

    def load(self, sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            logger.info('Reading model parameters from {}.'.format(
                ckpt.model_checkpoint_path))
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('No checkpoint available')

    def get_init(self):
        x = np.random.normal(loc=0.5, scale=0.2, size=(1, 3))
        x = np.maximum(np.minimum(x, 0.9), 0.1)
        y = np.array(self.func(x)).reshape(1, 1)
        init_state = [(np.zeros(s[0]), np.zeros(s[1]))
                      for s in self.get_state_shapes()]
        return x, y, init_state

    def run(self):
        with tf.Session() as sess:
            self.load(sess, self.ckpt_path)
            x, y, state = self.get_init()
            x_array = np.zeros((self.nsteps + 1, self.ndim))
            y_array = np.zeros((self.nsteps + 1, 1))
            x_array[0, :] = x
            y_array[0] = y
            for i in range(self.nsteps):
                x, state = self.step(sess, x, y, state)
                y = np.array(self.func(x)).reshape(1, 1)
                x_array[i+1, :] = x
                y_array[i+1] = y

        return x_array, y_array

def main():
    config_file = open('./config.json')
    config = json.load(config_file,
                       object_hook=lambda d:namedtuple('x', d.keys())(*d.values()))

    if config.opt_direction is 'max':
        problem_type = 'concave'
    else:
        problem_type = 'convex'

    if config.constraints:
        func = ConstraintQuadraticEval(num_dim=config.num_params,
                                       random=config.instrument_error,
                                       ptype=problem_type)
    else:
        func = QuadraticEval(num_dim=config.num_params,
                             random=config.instrument_error,
                             ptype=problem_type)

    if config.policy == 'srnn':
        cell = rnn.StochasticRNNCell(cell=rnn.LSTM,
                                 kwargs=
                                 {'hidden_size':config.hidden_size,
                                  'use_batch_norm_h':config.batch_norm,
                                  'use_batch_norm_x':config.batch_norm,
                                  'use_batch_norm_c':config.batch_norm,},
                                 nlayers=config.num_layers,
                                 reuse=config.reuse)
    if config.policy == 'rnn':
        cell = rnn.MultiInputRNNCell(cell=rnn.LSTM,
                                 kwargs=
                                 {'hidden_size':config.hidden_size,
                                  'use_batch_norm_h':config.batch_norm,
                                  'use_batch_norm_x':config.batch_norm,
                                  'use_batch_norm_c':config.batch_norm,},
                                 nlayers=config.num_layers,
                                 reuse=config.reuse)

    optimizer = StepOptimizer(cell=cell, func=func, ndim=config.num_params,
                              nsteps=config.num_steps,
                              ckpt_path=config.save_path, logger=logger,
                              constraints=config.constraints)
    x_array, y_array = optimizer.run()

    # np.savetxt('./scratch/nn_y.csv', y_array, delimiter=',')
    # np.save('./scratch/nn_x.npy', y_array)
    plt.figure(1)
    plt.plot(y_array)
    plt.show()
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(x_array[:, 0], x_array[:, 1], x_array[:, 2])
    fig2.show()
    plt.show()
    
if __name__ == '__main__':
    main()
