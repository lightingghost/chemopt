import tensorflow as tf
import tensorflow_probability as tfp
import batch_norm
import util
import pdb

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.ops.math_ops import tanh


class MultiInputLSTM(LSTMCell):
    def __init__(self, nlayers, num_units, input_size=None,
                 use_peepholes=False, cell_clip=None, initializer=None,
                 num_proj=None, proj_clip=None, num_unit_shards=1,
                 num_proj_shards=1, forget_bias=1.0, state_is_tuple=True,
                 activation=tanh):

        super(MultiInputLSTM, self).__init__(num_units, input_size=None,
            use_peepholes=False,cell_clip=None, initializer=None, num_proj=None,
            proj_clip=None, num_unit_shards=1, num_proj_shards=1,
            forget_bias=1.0,state_is_tuple=True, activation=tanh)

        self.cell = super(MultiInputLSTM, self).__call__

        if nlayers > 1:
            self.cell = MultiRNNCell([self.cell] * nlayers)
        self.nlayers = nlayers

    def __call__(self, x, y, state, scope=None):
        x_dim = x.get_shape()[1]
        inputs = tf.concat([x, tf.reshape(y, (-1, 1))],
                           axis=1, name='inputs')
        output, nstate = self.cell(inputs, state)
        with tf.compat.v1.variable_scope('proj'):
            w = tf.compat.v1.get_variable('proj_weight', [self._num_units, x_dim])
            x = tf.matmul(output, w)
        return x, nstate

    def get_initial_state(self, batch_size, dtype):
        if self.nlayers == 1:
            return super(MultiInputLSTM, self).zero_state(batch_size, dtype)
        else:
            return tuple(
                [super(MultiInputLSTM, self).zero_state(batch_size, dtype)] * self.nlayers)


class MultiInputRNNCell(RNNCell):
    def __init__(self, cell, kwargs, nlayers=1, reuse=False):
        self.cell = cell(**kwargs, name="lstm")
        self.nlayers = nlayers
        self.rnncell = self.cell
        if nlayers > 1:
            if reuse:
                self.rnncell = MultiRNNCell([self.cell] * nlayers)
            else:
                self.rnncell = MultiRNNCell([cell(**kwargs, name='lstm_{}'.format(i))
                                             for i in range(nlayers)])

    def __call__(self, x, y, state, scope=None):
        with tf.compat.v1.variable_scope(scope or 'multi_input_rnn'):
            x_dim = int(x.get_shape()[1])
            y = tf.tile(tf.reshape(y, [-1, 1]), [1, x_dim])
            inputs = tf.concat([x, y], axis=1, name='inputs')
            output, nstate = self.rnncell(inputs, state)
            with tf.compat.v1.variable_scope('proj'):
                w = tf.compat.v1.get_variable('proj_weight',
                    [self.cell.output_size.as_list()[0], x_dim])
                b = tf.compat.v1.get_variable('proj_bias', [x_dim])
                x = tf.matmul(output, w) + b
            return x, nstate

    def get_initial_state(self, batch_size, dtype=tf.float32):
        try:
            state = self.cell.get_initial_state(batch_size)
        except:
            state = self.cell.zero_state(batch_size, dtype)
        if self.nlayers == 1:
            return state
        else:
            return tuple([state] * self.nlayers)


class StochasticRNNCell(RNNCell):
    def __init__(self, cell, kwargs, nlayers=1, reuse=False):
        self.cell = cell(**kwargs, name="lstm")
        self.nlayers = nlayers
        self.rnncell = self.cell
        if nlayers > 1:
            if reuse:
                self.rnncell = MultiRNNCell([self.cell] * nlayers)
            else:
                self.rnncell = MultiRNNCell([cell(**kwargs, name='lstm_{}'.format(i))
                                             for i in range(nlayers)])

    def __call__(self, x, y, state, scope=None):
        hidden_size = self.cell.output_size.as_list()[0]
        batch_size = x.get_shape().as_list()[0]
        with tf.compat.v1.variable_scope(scope or 'multi_input_rnn'):
            x_dim = int(x.get_shape()[1])
            y = tf.tile(tf.reshape(y, [-1, 1]), [1, x_dim])
            inputs = tf.concat([x, y], axis=1, name='inputs')
            output, nstate = self.rnncell(inputs, state)
            tot_dim = x_dim * (x_dim + 1)
            with tf.compat.v1.variable_scope('proj'):
                w = tf.compat.v1.get_variable('proj_weight', [hidden_size, tot_dim])
                b = tf.compat.v1.get_variable('proj_bias', [tot_dim])
                out = tf.matmul(output, w) + b
                mean, var = tf.split(out, [x_dim, x_dim ** 2], axis=1)
                var = tf.reshape(var, [batch_size, x_dim, x_dim])
                dist = tfp.distributions.MultivariateNormalTriL(
                    mean, var, name='x_dist')
                x = dist.sample()

            return x, nstate

    def get_initial_state(self, batch_size, dtype=tf.float32):
        try:
            state = self.cell.get_initial_state(batch_size)
        except:
            state = self.cell.zero_state(batch_size, dtype)
        if self.nlayers == 1:
            return state
        else:
            return tuple([state] * self.nlayers)



class LSTM(RNNCell):
    # Keys that may be provided for parameter initializers.
    W_GATES = "w_gates"  # weight for gates
    B_GATES = "b_gates"  # bias of gates
    W_F_DIAG = "w_f_diag"  # weight for prev_cell -> forget gate peephole
    W_I_DIAG = "w_i_diag"  # weight for prev_cell -> input gate peephole
    W_O_DIAG = "w_o_diag"  # weight for prev_cell -> output gate peephole
    GAMMA_H = "gamma_h"  # batch norm scaling for previous_hidden -> gates
    GAMMA_X = "gamma_x"  # batch norm scaling for input -> gates
    GAMMA_C = "gamma_c"  # batch norm scaling for cell -> output
    BETA_C = "beta_c"  # (batch norm) bias for cell -> output
    POSSIBLE_KEYS = {W_GATES, B_GATES, W_F_DIAG, W_I_DIAG, W_O_DIAG, GAMMA_H,
                   GAMMA_X, GAMMA_C, BETA_C}

    def __init__(self,
               hidden_size,
               forget_bias=1.0,
               initializers=None,
               use_peepholes=False,
               use_batch_norm_h=False,
               use_batch_norm_x=False,
               use_batch_norm_c=False,
               max_unique_stats=1,
               name="lstm"):
        super(LSTM, self).__init__()
        self.name_ = name
        self._template = tf.compat.v1.make_template(self.name_, self._build,
                                          create_scope_now_=True)
        self._hidden_size = hidden_size
        self._forget_bias = forget_bias
        self._use_peepholes = use_peepholes
        self._max_unique_stats = max_unique_stats
        self._use_batch_norm_h = use_batch_norm_h
        self._use_batch_norm_x = use_batch_norm_x
        self._use_batch_norm_c = use_batch_norm_c
        self.possible_keys = self.get_possible_initializer_keys(use_peepholes=use_peepholes, use_batch_norm_h=use_batch_norm_h,
            use_batch_norm_x=use_batch_norm_x, use_batch_norm_c=use_batch_norm_c)
        self._initializers = util.check_initializers(initializers,
                                                     self.possible_keys)
        if max_unique_stats < 1:
            raise ValueError("max_unique_stats must be >= 1")
        if max_unique_stats != 1 and not (
            use_batch_norm_h or use_batch_norm_x or use_batch_norm_c):
            raise ValueError("max_unique_stats specified but batch norm disabled")

        if use_batch_norm_h:
            self._batch_norm_h = LSTM.IndexedStatsBatchNorm(max_unique_stats,
                                                          "batch_norm_h")
        if use_batch_norm_x:
            self._batch_norm_x = LSTM.IndexedStatsBatchNorm(max_unique_stats,
                                                          "batch_norm_x")
        if use_batch_norm_c:
            self._batch_norm_c = LSTM.IndexedStatsBatchNorm(max_unique_stats,
                                                          "batch_norm_c")

    def with_batch_norm_control(self, is_training=True, test_local_stats=True):
        return LSTM.CellWithExtraInput(self,
                                       is_training=is_training,
                                       test_local_stats=test_local_stats)

    @classmethod
    def get_possible_initializer_keys(
      cls, use_peepholes=False, use_batch_norm_h=False, use_batch_norm_x=False,
      use_batch_norm_c=False):
        possible_keys = cls.POSSIBLE_KEYS.copy()
        if not use_peepholes:
            possible_keys.difference_update(
                {cls.W_F_DIAG, cls.W_I_DIAG, cls.W_O_DIAG})
        if not use_batch_norm_h:
            possible_keys.remove(cls.GAMMA_H)
        if not use_batch_norm_x:
            possible_keys.remove(cls.GAMMA_X)
        if not use_batch_norm_c:
            possible_keys.difference_update({cls.GAMMA_C, cls.BETA_C})
        return possible_keys

    def _build(self, inputs, prev_state, is_training=True, test_local_stats=True):
        if self._max_unique_stats == 1:
            prev_hidden, prev_cell = prev_state
            time_step = None
        else:
            prev_hidden, prev_cell, time_step = prev_state

        self._create_gate_variables(inputs.get_shape(), inputs.dtype)
        self._create_batch_norm_variables(inputs.dtype)

        if self._use_batch_norm_h or self._use_batch_norm_x:
            gates_h = tf.matmul(prev_hidden, self._w_h)
            gates_x = tf.matmul(inputs, self._w_x)
            if self._use_batch_norm_h:
                gates_h = self._gamma_h * self._batch_norm_h(gates_h,
                                                             time_step,
                                                             is_training,
                                                             test_local_stats)
            if self._use_batch_norm_x:
                gates_x = self._gamma_x * self._batch_norm_x(gates_x,
                                                             time_step,
                                                             is_training,
                                                             test_local_stats)
            gates = gates_h + gates_x + self._b
        else:
        # Parameters of gates are concatenated into one multiply for efficiency.
            inputs_and_hidden = tf.concat([inputs, prev_hidden], axis=1)
            gates = tf.matmul(inputs_and_hidden, self._w_xh) + self._b

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(gates, 4, axis=1)

        if self._use_peepholes:  # diagonal connections
            self._create_peephole_variables(inputs.dtype)
            f += self._w_f_diag * prev_cell
            i += self._w_i_diag * prev_cell

        forget_mask = tf.sigmoid(f + self._forget_bias)
        new_cell = forget_mask * prev_cell + tf.sigmoid(i) * tf.tanh(j)
        cell_output = new_cell
        if self._use_batch_norm_c:
            cell_output = (self._beta_c
                     + self._gamma_c * self._batch_norm_c(cell_output,
                                                          time_step,
                                                          is_training,
                                                          test_local_stats))
        if self._use_peepholes:
            cell_output += self._w_o_diag * cell_output
        new_hidden = tf.tanh(cell_output) * tf.sigmoid(o)

        if self._max_unique_stats == 1:
            return new_hidden, (new_hidden, new_cell)
        else:
            return new_hidden, (new_hidden, new_cell, time_step + 1)

    def __call__(self, input, prev_state,
                 is_training=True, test_local_stats=True):
        return self._template(input, prev_state, is_training, test_local_stats)

    def _create_batch_norm_variables(self, dtype):
        """Initialize the variables used for the `BatchNorm`s (if any)."""
        gamma_initializer = tf.constant_initializer(0.1)

        if self._use_batch_norm_h:
            self._gamma_h = tf.compat.v1.get_variable(
              LSTM.GAMMA_H,
              shape=[4 * self._hidden_size],
              dtype=dtype,
              initializer=(self._initializers.get(LSTM.GAMMA_H, gamma_initializer)))
        if self._use_batch_norm_x:
            self._gamma_x = tf.compat.v1.get_variable(
              LSTM.GAMMA_X,
              shape=[4 * self._hidden_size],
              dtype=dtype,
              initializer=(self._initializers.get(LSTM.GAMMA_X, gamma_initializer)))
        if self._use_batch_norm_c:
            self._gamma_c = tf.compat.v1.get_variable(
                LSTM.GAMMA_C,
                shape=[self._hidden_size],
                dtype=dtype,
                initializer=(
                    self._initializers.get(LSTM.GAMMA_C, gamma_initializer)))
            self._beta_c = tf.compat.v1.get_variable(
                LSTM.BETA_C,
                shape=[self._hidden_size],
                dtype=dtype,
                initializer=self._initializers.get(LSTM.BETA_C))

    def _create_gate_variables(self, input_shape, dtype):
        """Initialize the variables used for the gates."""
        if len(input_shape) != 2:
            raise ValueError(
            "Rank of shape must be {} not: {}".format(2, len(input_shape)))
        input_size = input_shape.dims[1].value

        b_shape = [4 * self._hidden_size]

        equiv_input_size = self._hidden_size + input_size
        initializer = util.create_linear_initializer(equiv_input_size)

        if self._use_batch_norm_h or self._use_batch_norm_x:
            self._w_h = tf.compat.v1.get_variable(
                LSTM.W_GATES + "_H",
                shape=[self._hidden_size, 4 * self._hidden_size],
                dtype=dtype,
                initializer=self._initializers.get(LSTM.W_GATES, initializer))
            self._w_x = tf.compat.v1.get_variable(
                LSTM.W_GATES + "_X",
                shape=[input_size, 4 * self._hidden_size],
                dtype=dtype,
                initializer=self._initializers.get(LSTM.W_GATES, initializer))
        else:
            self._w_xh = tf.compat.v1.get_variable(
                LSTM.W_GATES,
                shape=[self._hidden_size + input_size, 4 * self._hidden_size],
                dtype=dtype,
                initializer=self._initializers.get(LSTM.W_GATES, initializer))
            self._b = tf.compat.v1.get_variable(
                LSTM.B_GATES,
                shape=b_shape,
                dtype=dtype,
                initializer=self._initializers.get(LSTM.B_GATES, initializer))

    def _create_peephole_variables(self, dtype):
        """Initialize the variables used for the peephole connections."""
        self._w_f_diag = tf.compat.v1.get_variable(
            LSTM.W_F_DIAG,
            shape=[self._hidden_size],
            dtype=dtype,
            initializer=self._initializers.get(LSTM.W_F_DIAG))
        self._w_i_diag = tf.compat.v1.get_variable(
            LSTM.W_I_DIAG,
            shape=[self._hidden_size],
            dtype=dtype,
            initializer=self._initializers.get(LSTM.W_I_DIAG))
        self._w_o_diag = tf.compat.v1.get_variable(
            LSTM.W_O_DIAG,
            shape=[self._hidden_size],
            dtype=dtype,
            initializer=self._initializers.get(LSTM.W_O_DIAG))

    def get_initial_state(self, batch_size, dtype=tf.float32, trainable=False,
                    trainable_initializers=None):

        if self._max_unique_stats == 1:
            return super(LSTM, self).initial_state(
                batch_size, dtype, trainable, trainable_initializers)
        else:
            if not trainable:
                state = super(rnn_core.RNNCore, self).zero_state(batch_size, dtype)
            else:
        # We have to manually create the state ourselves so we don't create a
        # variable that never gets used for the third entry.
                state = util.trainable_initial_state(
                    batch_size,
                    (tf.TensorShape([self._hidden_size]),
                     tf.TensorShape([self._hidden_size])),
                    dtype,
                    trainable_initializers)
            return (state[0], state[1], tf.constant(0, dtype=tf.int32))

    @property
    def state_size(self):
        """Tuple of `tf.TensorShape`s indicating the size of state tensors."""
        if self._max_unique_stats == 1:
            return (tf.TensorShape([self._hidden_size]),
                    tf.TensorShape([self._hidden_size]))
        else:
            return (tf.TensorShape([self._hidden_size]),
                    tf.TensorShape([self._hidden_size]),
                    tf.TensorShape(1))

    @property
    def output_size(self):
        """`tf.TensorShape` indicating the size of the core output."""
        return tf.TensorShape([self._hidden_size])

    @property
    def use_peepholes(self):
        """Boolean indicating whether peephole connections are used."""
        return self._use_peepholes

    @property
    def use_batch_norm_h(self):
        """Boolean indicating whether batch norm for hidden -> gates is enabled."""
        return self._use_batch_norm_h

    @property
    def use_batch_norm_x(self):
        """Boolean indicating whether batch norm for input -> gates is enabled."""
        return self._use_batch_norm_x

    @property
    def use_batch_norm_c(self):
        """Boolean indicating whether batch norm for cell -> output is enabled."""
        return self._use_batch_norm_c

    class IndexedStatsBatchNorm(object):
        def __init__(self, max_unique_stats, name=None):
            # super(LSTM.IndexedStatsBatchNorm, self).__init__()
            self._max_unique_stats = max_unique_stats

        def _build(self, inputs, index, is_training, test_local_stats):
            def create_batch_norm():
                return batch_norm.BatchNorm(offset=False, scale=False)(
                    inputs, is_training, test_local_stats)

            if self._max_unique_stats > 1:
                pred_fn_pairs = [(tf.equal(i, index), create_batch_norm)
                         for i in range(self._max_unique_stats - 1)]
                out = tf.case(pred_fn_pairs, create_batch_norm)
                out.set_shape(inputs.get_shape())  # needed for tf.case shape inference
                return out
            else:
                return create_batch_norm()

    class CellWithExtraInput(RNNCell):
        def __init__(self, cell, *args, **kwargs):
            self._cell = cell
            self._args = args
            self._kwargs = kwargs

        def __call__(self, inputs, state):
            return self._cell(inputs, state, *self._args, **self._kwargs)

        @property
        def state_size(self):
            """Tuple indicating the size of nested state tensors."""
            return self._cell.state_size

        @property
        def output_size(self):
            """`tf.TensorShape` indicating the size of the core output."""
            return self._cell.output_size
