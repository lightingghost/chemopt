import tensorflow as tf
import ops
import pdb

from tensorflow.python.util import nest

class Optimizer:
    def __init__(self, cell, logger, func, ndim, batch_size, unroll_len,
                 lr=0.01, loss_type='naive', optimizer='Adam', trainable_init=False,
                 direction='max', constraints=False, discount_factor=1.0):
        self.batch_size = batch_size
        self.constraints = constraints
        self.logger = logger
        self.cell = cell
        self.trainable_init = trainable_init
        self.df = self.make_discount(discount_factor, unroll_len)
        self.make_loss(func, ndim, batch_size, unroll_len)
        loss_func = self.get_loss_func(loss_type, direction)
        self.loss = loss_func(self.fx_array)
        optimizer = getattr(tf.train, optimizer + 'Optimizer')(lr)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
        self.opt = optimizer.apply_gradients(capped_gvs)

        # self.opt = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
        logger.info('model variable:')
        logger.info(str([var.name for var in tf.global_variables()]))
        logger.info('trainable variables:')
        logger.info(str([var.name for var in tf.trainable_variables()]))
        self.fx_array = self.fx_array.stack()
        self.x_array = self.x_array.stack()


    def make_discount(self, gamma, unroll_len):
        df = [(gamma ** (unroll_len - i)) for i in range(unroll_len + 1)]
        return tf.constant(df, shape=[unroll_len + 1, 1], dtype=tf.float32)
    

    def make_loss(self, func, ndim, batch_size, unroll_len):
        self.unroll_len = unroll_len
        x = tf.get_variable('x', shape=[batch_size, ndim],
                            initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.2),
                            trainable=self.trainable_init)
        constants = func.get_parameters()
        state = self.cell.get_initial_state(batch_size, tf.float32)
        self.fx_array = tf.TensorArray(tf.float32,
            size=unroll_len+1, clear_after_read=False)
        self.x_array = tf.TensorArray(tf.float32,
            size=unroll_len+1, clear_after_read=False)

        def step(t, x, state, fx_array, x_array):
            with tf.name_scope('fx'):
                fx = func(x)
                fx_array = fx_array.write(t, fx)
                x_array = x_array.write(t, x)
            with tf.name_scope('opt_cell'):
                new_x, new_state = self.cell(x, fx, state)
                if self.constraints:
                    new_x = tf.clip_by_value(new_x, 0.01, 0.99)

            with tf.name_scope('t_next'):
                t_next = t + 1

            return t_next, new_x, new_state, fx_array, x_array

        _, x_final, s_final, self.fx_array, self.x_array = tf.while_loop(
            cond=lambda t, *_: t < unroll_len,
            body=step, loop_vars=(0, x, state, self.fx_array, self.x_array),
            parallel_iterations=1,
            swap_memory=True
        )

        with tf.name_scope('fx'):
            fx_final = func(x_final)
            self.fx_array = self.fx_array.write(unroll_len, fx_final)
            self.x_array = self.x_array.write(unroll_len, x)

        # Reset the state; should be called at the beginning of an epoch.
        with tf.name_scope('reset'):

            variables = [x,] + constants
            # Empty array as part of the reset process.
            self.reset = [tf.variables_initializer(variables),
                self.fx_array.close(), self.x_array.close()]

        return self.fx_array, self.x_array

    def get_loss_func(self, loss_type='naive', direction='max'):
        def loss_func(fx):
            if loss_type == 'naive':
                loss = tf.reduce_sum(
                    tf.matmul(tf.reshape(fx.stack(), [self.batch_size, -1]),
                              self.df, name='loss'))
            elif loss_type == 'oi' and direction == 'max':
                loss = tf.reduce_sum(
                    [fx.read(i) - tf.reduce_max(
                            fx.gather(list(range(i))), axis=0)
                        for i in range(1, self.unroll_len + 1)],
                    name='loss')
            elif loss_type == 'oi' and direction == 'min':
                loss = tf.reduce_sum(
                    [fx.read(i) - tf.reduce_min(
                            fx.gather(list(range(i))), axis=0)
                        for i in range(1, self.unroll_len + 1)],
                    name='loss')
            if direction == 'max':
                loss = - loss
            return loss / self.batch_size
        return loss_func

    def step(self):
        return self.opt, self.loss, self.reset, self.fx_array, self.x_array
