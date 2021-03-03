import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import util
import numpy as np
import logging
import json

from model import Optimizer
from rnn import MultiInputLSTM
from logger import get_handlers
from collections import namedtuple

logging.basicConfig(level=logging.INFO, handlers=get_handlers())
logger = logging.getLogger()

def main():
    config_file = open('./config.json')
    config = json.load(config_file,
                       object_hook=lambda d:namedtuple('x', d.keys())(*d.values()))
    config_file.seek(0)
    logger.info(str(json.load(config_file)))
    config_file.close()
    num_unrolls = config.num_steps // config.unroll_length
    with tf.compat.v1.Session() as sess:
        # tf.get_default_graph().finalize()
        model = util.create_model(sess, config, logger)
        step, loss, reset, fx_array, x_array = model.step()

        best_cost = [float('inf')] * 3
        epoch_cost = 0
        total_cost = 0

        for e in range(config.num_epochs):
            cost, _ = util.run_epoch(sess, loss, [step], reset, num_unrolls)
            epoch_cost += cost
            total_cost += cost

            if (e + 1) % config.log_period == 0:
                lm_e = epoch_cost / config.log_period
                logger.info('Epoch {}, Mean Error: {:.3f}'.format(e, lm_e))
                epoch_cost = 0

            if (e + 1) % config.evaluation_period == 0:
                elm_e = total_cost / config.evaluation_period
                logger.info('Current {} epochs, Mean Error: {:.3f}'.format(config.evaluation_period, elm_e))

                mbc = max(best_cost)
                if config.save_path is not None and total_cost < mbc:
                    best_cost.remove(mbc)
                    best_cost.append(total_cost)
                    logger.info('Save current model ...')
                    model.saver.save(sess, config.save_path, global_step=e)

                total_cost = 0


if __name__ == '__main__':
    main()
