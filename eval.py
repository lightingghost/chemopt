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
        _, loss, reset, fx_array, x_array = model.step()
        cost, others = util.run_epoch(sess, loss, [fx_array, x_array],
            reset, num_unrolls)
        Y, X = others
        Yp = []
        for i in range(config.batch_size):
            arr = np.squeeze(Y[:, i])
            if arr[-1] <= 0.3:
                Yp.append(arr)

        np.save('./scratch/gmm0.npy', np.array(Yp))
        plt.figure(1)
        plt.plot(np.squeeze(Y))
        plt.show()


if __name__ == '__main__':
    main()
