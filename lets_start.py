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
    with tf.Session() as sess:
        # tf.get_default_graph().finalize()
        model = util.create_model(sess, config, logger)
        step, loss, reset, fx_array, x_array = model.step()

        best_evaluation = float('inf')
        total_cost = 0
        for e in range(config.num_epochs):
            cost, _ = util.run_epoch(sess, loss, [step], reset, num_unrolls)
            total_cost += cost

            if (e + 1) % config.log_period == 0:
                lm_e = total_cost / config.log_period
                logger.info('Epoch {}, Mean Error: {:.3f}'.format(e, lm_e))
                total_cost = 0

            if (e + 1) % config.evaluation_period == 0:
                eval_cost = 0
                for _ in range(config.evaluation_epochs):
                    cost, _ = util.run_epoch(sess, loss, [step, ], reset,
                                      num_unrolls)
                    eval_cost += cost
                elm_e = eval_cost / config.evaluation_epochs
                logger.info('EVALUATION, Mean Error: {:.3f}'.format(elm_e))

                if config.save_path is not None and eval_cost < best_evaluation:
                    logger.info('Save current model ...')
                    model.saver.save(sess, config.save_path, global_step=e)
                    best_evaluation = eval_cost

if __name__ == '__main__':
    main()
