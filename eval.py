import util
import logging
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import Optimizer
from rnn import MultiInputLSTM
from logger import get_handlers
from collections import namedtuple

logging.basicConfig(level=logging.INFO, handlers=get_handlers(False))
logger = logging.getLogger()


def main():


    config_file = open('./config.json')
    config = json.load(config_file,
                       object_hook=lambda d:namedtuple('x', d.keys())(*d.values()))
    num_unrolls = config.num_steps // config.unroll_length
    with tf.Session() as sess:
        model = util.load_model(sess, config, logger)
        all_y = []
        for i in range(10):
            print(i)
            _, loss, reset, fx_array, x_array = model.step()
            cost, others = util.run_epoch(sess, loss, [fx_array, x_array],
                reset, num_unrolls)
            Y, X = others
            all_y.append(Y)

    all_y = np.hstack(all_y)
    np.save('srnn.npy', all_y)
    plt.figure(1)
    y_mean = np.mean(all_y, axis=1)
    plt.plot(y_mean)
    print(min(y_mean))
    plt.show()


if __name__ == '__main__':
    main()
