import os
import math
import reactions
import tensorflow as tf

import rnn
from model import Optimizer
from shutil import copyfile


def run_epoch(sess, cost_op, ops, reset, num_unrolls):
    """Runs one optimization epoch."""
    sess.run(reset)
    for _ in range(num_unrolls):
        results = sess.run([cost_op] + ops)
    return results[0], results[1:]

def create_model(sess, config, logger):
    if not config.save_path == None:
        if not os.path.exists(config.save_path):
            os.mkdir(config.save_path)
    copyfile('config.json', os.path.join(config.save_path, 'config.json'))

    if config.opt_direction == 'max':
        problem_type = 'concave'
    else:
        problem_type = 'convex'

    if config.reaction_type == 'quad' and config.constraints == False:
        rxn_yeild = reactions.Quadratic(
            batch_size=config.batch_size,
            num_dims=config.num_params,
            ptype=problem_type,
            random=config.instrument_error)
    elif config.reaction_type == 'quad' and config.constraints == True:
        rxn_yeild = reactions.ConstraintQuadratic(
            batch_size=config.batch_size,
            num_dims=config.num_params,
            ptype=problem_type,
            random=config.instrument_error)
    elif config.reaction_type == 'gmm':
        rxn_yeild = reactions.GMM(
            batch_size=config.batch_size,
            num_dims=config.num_params,
            random=config.instrument_error,
            cov=config.norm_cov)
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
    model = Optimizer(cell=cell, logger=logger, func=rxn_yeild,
        ndim=config.num_params, batch_size=config.batch_size,
        unroll_len=config.unroll_length, lr=config.learning_rate,
        loss_type=config.loss_type, optimizer=config.optimizer,
        trainable_init=config.trainable_init,
        direction=config.opt_direction, constraints=config.constraints,
        discount_factor=config.discount_factor)

    ckpt = tf.train.get_checkpoint_state(config.save_path)
    if ckpt and ckpt.model_checkpoint_path:
        logger.info('Reading model parameters from {}.'.format(
            ckpt.model_checkpoint_path))
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logger.info('Creating Model with fresh parameters.')
        sess.run(tf.compat.v1.global_variables_initializer())
    return model

def load_model(sess, config, logger):
    assert(os.path.exists(config.save_path))

    if config.opt_direction == 'max':
        problem_type = 'concave'
    else:
        problem_type = 'convex'

    if config.reaction_type == 'quad' and config.constraints == False:
        rxn_yeild = reactions.Quadratic(
            batch_size=config.batch_size,
            num_dims=config.num_params,
            ptype=problem_type,
            random=config.instrument_error)
    elif config.reaction_type == 'quad' and config.constraints == True:
        rxn_yeild = reactions.ConstraintQuadratic(
            batch_size=config.batch_size,
            num_dims=config.num_params,
            ptype=problem_type,
            random=config.instrument_error)
    elif config.reaction_type == 'gmm':
        rxn_yeild = reactions.GMM(
            batch_size=config.batch_size,
            num_dims=config.num_params,
            random=config.instrument_error,
            cov=config.norm_cov)

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
    model = Optimizer(cell=cell, logger=logger, func=rxn_yeild,
        ndim=config.num_params, batch_size=config.batch_size,
        unroll_len=config.unroll_length, lr=config.learning_rate,
        loss_type=config.loss_type, optimizer=config.optimizer,
        trainable_init=config.trainable_init,
        direction=config.opt_direction, constraints=config.constraints,
        discount_factor=config.discount_factor)


    ckpt = tf.train.get_checkpoint_state(config.save_path)
    if ckpt and ckpt.model_checkpoint_path:
        logger.info('Reading model parameters from {}.'.format(
            ckpt.model_checkpoint_path))
        model.saver.restore(sess, ckpt.model_checkpoint_path)

    return model


def check_initializers(initializers, keys):
    if initializers is None:
        return {}
    keys = set(keys)

    if not issubclass(type(initializers), dict):
        raise TypeError("A dict of initializers was expected, but not "
                    "given. You should double-check that you've nested the "
                    "initializers for any sub-modules correctly.")

    if not set(initializers) <= keys:
        extra_keys = set(initializers) - keys
        raise KeyError(
            "Invalid initializer keys {}, initializers can only "
            "be provided for {}".format(
                ", ".join("'{}'".format(key) for key in extra_keys),
                ", ".join("'{}'".format(key) for key in keys)))

    def check_nested_callables(dictionary):
        for key, entry in dictionary.items():
            if isinstance(entry, dict):
                check_nested_callables(entry)
            elif not callable(entry):
                raise TypeError(
                    "Initializer for '{}' is not a callable function "
                    "or dictionary".format(key))
    check_nested_callables(initializers)
    return dict(initializers)

def create_linear_initializer(input_size):
    """Returns a default initializer for weights or bias of a linear module."""
    stddev = 1 / math.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev)

def trainable_initial_state(batch_size, state_size, dtype, initializers=None):
    flat_state_size = nest.flatten(state_size)

    if not initializers:
        flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
    else:
        nest.assert_same_structure(initializers, state_size)
        flat_initializer = nest.flatten(initializers)
        if not all([callable(init) for init in flat_initializer]):
            raise ValueError("Not all the passed initializers are callable objects.")

  # Produce names for the variables. In the case of a tuple or nested tuple,
  # this is just a sequence of numbers, but for a flat `namedtuple`, we use
  # the field names. NOTE: this could be extended to nested `namedtuple`s,
  # but for now that's extra complexity that's not used anywhere.
    try:
        names = ["init_{}".format(state_size._fields[i])
                    for i in range(len(flat_state_size))]
    except (AttributeError, IndexError):
        names = ["init_state_{}".format(i) for i in range(len(flat_state_size))]

    flat_initial_state = []

    for name, size, init in zip(names, flat_state_size, flat_initializer):
        shape_with_batch_dim = [1] + tensor_shape.as_shape(size).as_list()
        initial_state_variable = tf.compat.v1.get_variable(
            name, shape=shape_with_batch_dim, dtype=dtype, initializer=init)

        initial_state_variable_dims = initial_state_variable.get_shape().ndims
        tile_dims = [batch_size] + [1] * (initial_state_variable_dims - 1)
        flat_initial_state.append(
            tf.tile(initial_state_variable, tile_dims, name=(name + "_tiled")))

    return nest.pack_sequence_as(structure=state_size,
                                 flat_sequence=flat_initial_state)
