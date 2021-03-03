import tensorflow as tf

import base

from tensorflow.contrib.layers.python.layers import utils

class BatchNorm():
    GAMMA = "gamma"
    BETA = "beta"
    POSSIBLE_INITIALIZER_KEYS = {GAMMA, BETA}

    def __init__(self, reduction_indices=None, offset=True, scale=False,
        decay_rate=0.999, eps=1e-3, initializers=None,
        use_legacy_moving_second_moment=False,
        name="batch_norm"):

        self._reduction_indices = reduction_indices
        self._offset = offset
        self._scale = scale
        self._decay_rate = decay_rate
        self._eps = eps
        self._use_legacy_moving_second_moment = use_legacy_moving_second_moment
        self._initializers = util.check_initializers(
            initializers, self.POSSIBLE_INITIALIZER_KEYS)

    def _set_default_initializer(self, var_name):
        if var_name not in self._initializers:
            if var_name == self.GAMMA:
                self._initializers[self.GAMMA] = tf.ones_initializer()
            elif var_name == self.BETA:
                self._initializers[self.BETA] = tf.zeros_initializer

    def _build_statistics_variance(self, input_batch,
        reduction_indices, use_batch_stats):
        self._moving_mean = tf.compat.v1.get_variable(
            "moving_mean",
            shape=self._mean_shape,
            collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                         tf.GraphKeys.VARIABLES],
            initializer=tf.zeros_initializer,
            trainable=False)

        self._moving_variance = tf.compat.v1.get_variable(
            "moving_variance",
            shape=self._mean_shape,
            collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                         tf.GraphKeys.VARIABLES],
            initializer=tf.ones_initializer(),
            trainable=False)

        def build_batch_stats():
            """Builds the batch statistics calculation ops."""
            shift = tf.add(self._moving_mean, 0)
            counts, shifted_sum_x, shifted_sum_x2, _ = tf.nn.sufficient_statistics(
                input_batch,
                reduction_indices,
                keep_dims=True,
                shift=shift,
                name="batch_norm_ss")

            mean, variance = tf.nn.normalize_moments(counts,
                                               shifted_sum_x,
                                               shifted_sum_x2,
                                               shift,
                                               name="normalize_moments")

            return mean, variance

        def build_moving_stats():
            return (
                tf.identity(self._moving_mean),
                tf.identity(self._moving_variance),)

        mean, variance = utils.smart_cond(
            use_batch_stats,
            build_batch_stats,
            build_moving_stats,
        )

        return mean, variance

    def _build_statistics_second_moment(self, input_batch,
        reduction_indices, use_batch_stats):
        self._moving_mean = tf.compat.v1.get_variable(
            "moving_mean",
            shape=self._mean_shape,
            collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                         tf.GraphKeys.VARIABLES],
            initializer=tf.zeros_initializer,
            trainable=False)

        self._moving_second_moment = tf.compat.v1.get_variable(
            "moving_second_moment",
            shape=self._mean_shape,
            collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                         tf.GraphKeys.VARIABLES],
            initializer=tf.ones_initializer(),
            trainable=False)

        self._moving_variance = tf.sub(self._moving_second_moment,
                                       tf.square(self._moving_mean),
                                       name="moving_variance")

        def build_batch_stats():
            shift = tf.add(self._moving_mean, 0)
            counts, shifted_sum_x, shifted_sum_x2, _ = tf.nn.sufficient_statistics(
                input_batch,
                reduction_indices,
                keep_dims=True,
                shift=shift,
                name="batch_norm_ss")

            mean, variance = tf.nn.normalize_moments(counts,
                                               shifted_sum_x,
                                               shifted_sum_x2,
                                               shift,
                                               name="normalize_moments")
            second_moment = variance + tf.square(mean)

            return mean, variance, second_moment

        def build_moving_stats():
            return (
                tf.identity(self._moving_mean),
                tf.identity(self._moving_variance),
                tf.identity(self._moving_second_moment),
            )

        mean, variance, second_moment = utils.smart_cond(
            use_batch_stats,
            build_batch_stats,
            build_moving_stats,
        )

        return mean, variance, second_moment

    def _build_update_ops_variance(self, mean, variance, is_training):
        def build_update_ops():
            update_mean_op = moving_averages.assign_moving_average(
              variable=self._moving_mean,
              value=mean,
              decay=self._decay_rate,
              name="update_moving_mean").op

            update_variance_op = moving_averages.assign_moving_average(
              variable=self._moving_variance,
              value=variance,
              decay=self._decay_rate,
              name="update_moving_variance").op

            return update_mean_op, update_variance_op

        def build_no_ops():
            return (tf.no_op(), tf.no_op())

            # Only make the ops if we know that `is_training=True`, or the
            # value of `is_training` is unknown.
        is_training_const = utils.constant_value(is_training)
        if is_training_const is None or is_training_const:
            update_mean_op, update_variance_op = utils.smart_cond(
                is_training,
                build_update_ops,
                build_no_ops,
            )

          # Every new connection creates a new op which adds its contribution
          # to the running average when ran.
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance_op)

    def _build_update_ops_second_moment(self, mean, second_moment, is_training):
        def build_update_ops():
            update_mean_op = moving_averages.assign_moving_average(
              variable=self._moving_mean,
              value=mean,
              decay=self._decay_rate,
              name="update_moving_mean").op

            update_second_moment_op = moving_averages.assign_moving_average(
              variable=self._moving_second_moment,
              value=second_moment,
              decay=self._decay_rate,
              name="update_moving_second_moment").op

            return update_mean_op, update_second_moment_op

        def build_no_ops():
            return (tf.no_op(), tf.no_op())

        is_training_const = utils.constant_value(is_training)
        if is_training_const is None or is_training_const:
            update_mean_op, update_second_moment_op = utils.smart_cond(
                is_training,
                build_update_ops,
                build_no_ops,
                )

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_second_moment_op)

    def _build(self, input_batch, is_training=True, test_local_stats=True):

        input_shape = input_batch.get_shape()

        if self._reduction_indices is not None:
            if len(self._reduction_indices) > len(input_shape):
                raise base.IncompatibleShapeError(
                    "Too many reduction indices specified.")

            if max(self._reduction_indices) >= len(input_shape):
                raise base.IncompatibleShapeError(
                    "Reduction index too large for input shape.")

            if min(self._reduction_indices) < 0:
                raise base.IncompatibleShapeError(
                    "Reduction indeces must be non-negative.")

            reduction_indices = self._reduction_indices
        else:
            reduction_indices = range(len(input_shape))[:-1]

        if input_batch.dtype == tf.float16:
            raise base.NotSupportedError(
                "BatchNorm does not support `tf.float16`, insufficient "
                "precision for calculating sufficient statistics.")

        self._mean_shape = input_batch.get_shape().as_list()
        for index in reduction_indices:
            self._mean_shape[index] = 1

        use_batch_stats = is_training | test_local_stats

        # Use the legacy moving second moment if the flag is set.
        if self._use_legacy_moving_second_moment:
            tf.logging.warning(
                "nn.BatchNorm `use_legacy_second_moment=True` is deprecated.")

            mean, variance, second_moment = self._build_statistics_second_moment(
                    input_batch,
                    reduction_indices,
                    use_batch_stats)

            self._build_update_ops_second_moment(mean, second_moment, is_training)
        else:
            mean, variance = self._build_statistics_variance(
                input_batch,
                reduction_indices,
                use_batch_stats)

            self._build_update_ops_variance(mean, variance, is_training)

            # Set up optional scale and offset factors.
        if self._offset:
            self._set_default_initializer(self.BETA)
            self._beta = tf.compat.v1.get_variable(
                self.BETA,
                shape=self._mean_shape,
                initializer=self._initializers[self.BETA])
        else:
            self._beta = None

        if self._scale:
            self._set_default_initializer(self.GAMMA)
            self._gamma = tf.compat.v1.get_variable(
                self.GAMMA,
                shape=self._mean_shape,
                initializer=self._initializers[self.GAMMA])
        else:
            self._gamma = None

        out = tf.nn.batch_normalization(
            input_batch,
            mean,
            variance,
            self._beta,
            self._gamma,
            self._eps,
            name="batch_norm")

        return out

    @property
    def moving_mean(self):
        self._ensure_is_connected()
        return self._moving_mean

    @property
    def moving_second_moment(self):
        self._ensure_is_connected()
        return self._moving_second_moment

    @property
    def moving_variance(self):
        self._ensure_is_connected()
        return self._moving_variance

    @property
    def beta(self):
        self._ensure_is_connected()

        if self._beta is None:
            raise base.Error(
                "Batch normalization doesn't have an offset, so no beta")
        else:
            return self._beta

    @property
    def gamma(self):
        self._ensure_is_connected()

        if self._gamma is None:
            raise base.Error(
                "Batch normalization doesn't have a scale, so no gamma")
        else:
            return self._gamma
