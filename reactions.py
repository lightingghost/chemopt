import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np

class ConstraintQuadratic:
    """Quadratic problem: f(x) = ||Wx - y||."""
    def __init__(self, batch_size=128, num_dims=3, ptype='convex',
                 random=0.05, dtype=tf.float32):
        self.ptype = ptype
        self.w = tf.get_variable('w', shape=[batch_size, num_dims, num_dims],
            dtype=dtype, initializer=tf.random_normal_initializer(),
            trainable=False)

        self.a = tf.get_variable('y', shape=[batch_size, num_dims],
            dtype=dtype, initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.99),
            trainable=False)

        self.y = tf.squeeze(tf.matmul(self.w, tf.expand_dims(self.a, -1)))

        self.normalizer = tf.maximum(
            self._func(tf.zeros([batch_size, num_dims])),
            self._func(tf.ones([batch_size, num_dims])))

        if random is not None:
            self.e = tf.random_normal(shape=[batch_size,], stddev=random,
                dtype=dtype, name='e')
        else:
            self.e = 0.0

    def get_parameters(self):
        return [self.w, self.a]

    def _func(self, var):
        product = tf.squeeze(tf.matmul(self.w, tf.expand_dims(var, -1)))
        norm = tf.reduce_sum((product - self.y) ** 2, 1)
        return norm

    def _barrier(self, var):
        return -tf.reduce_sum(tf.log(var) + tf.log(1 - var), 1) / 1e10

    def __call__(self, x):
        '''
        x = tf.get_variable('x', shape=[batch_size, num_dims],
            dtype=dtype, initializer=tf.random_normal_initializer(stddev=stdev))
        '''
        res = (self._func(x) / self.normalizer + self.e + self._barrier(x))
        if self.ptype == 'concave':
            res = 1 - res
        return res

class GMM:
    def __init__(self, batch_size=128, ncoef=6, num_dims=3, random=None,
                 cov=0.1, dtype=tf.float32):
        self.ncoef = ncoef
        self.num_dim = num_dims
        self.batch_size = batch_size
        self.dtype = dtype
        with tf.variable_scope('func_gmm'):
            self.m = [tf.get_variable('mu_{}'.format(i), shape=[batch_size, num_dims],
                dtype=dtype,
                initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.99),
                trainable=False)
                      for i in range(ncoef)]

            self.cov = [tf.get_variable('cov_{}'.format(i), shape=[batch_size, num_dims],
                dtype=dtype,
                initializer=tf.truncated_normal_initializer(
                    mean=cov, stddev=cov/5),
                trainable=False)
                      for i in range(ncoef)]

            self.coef = tf.get_variable('coef', shape=[ncoef, 1], dtype=dtype,
                initializer=tf.random_normal_initializer(stddev=0.2),
                trainable=False)

            self.random = random
            # if random is not None:
                # self.e = tf.random_normal(shape=[batch_size, ], stddev=random,
                                          # dtype=dtype, name='error')
            # else:
                # self.e = 0.0

            self.cst = (2 * 3.14159) ** (- self.num_dim / 2)
            modes = tf.concat([(1 / tf.reduce_prod(cov, axis=1, keep_dims=True))
                               for cov in self.cov], axis=1) * tf.transpose(self.coef)
            self.tops = tf.reduce_max(modes, axis=1, keep_dims=True)
            self.bots = tf.reduce_min(modes, axis=1, keep_dims=True)

    def get_parameters(self):
        return self.m + self.cov + [self.coef]

    def __call__(self, x):
        dist = [tf.contrib.distributions.MultivariateNormalDiag(
                    self.m[i], self.cov[i], name='MultVarNorm_{}'.format(i))
                for i in range(self.ncoef)]
        p = tf.concat([tf.reshape(dist[i].prob(x), [-1, 1])
                       for i in range(self.ncoef)], axis=1)

        fx = tf.matmul(p, self.coef)
        result = (fx / self.cst - self.bots) / (self.tops - self.bots)
        # import pdb; pdb.set_trace()
        if self.random:
            result = result + tf.random_normal(shape=[self.batch_size, 1], 
                    stddev=self.random,
                    dtype=self.dtype, name='error')
        return result



class Quadratic:
    """Quadratic problem: f(x) = ||Wx - y||."""
    def __init__(self, batch_size=128, num_dims=3, ptype='convex',
                 random=0.05, dtype=tf.float32):
        self.ptype = ptype
        self.w = tf.get_variable('w', shape=[batch_size, num_dims, num_dims],
            dtype=dtype, initializer=tf.random_normal_initializer(),
            trainable=False)

        self.a = tf.get_variable('y', shape=[batch_size, num_dims],
            dtype=dtype, initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.2),
            trainable=False)

        self.y = tf.squeeze(tf.matmul(self.w, tf.expand_dims(self.a, -1)))


        self.normalizer = tf.maximum(
            self._func(tf.zeros([batch_size, num_dims])),
            self._func(tf.ones([batch_size, num_dims])))

        if random is not None:
            self.e = tf.random_normal(shape=[batch_size,], stddev=random,
                dtype=dtype, name='e')
        else:
            self.e = 0.0

    def get_parameters(self):
        return [self.w, self.a]

    def _func(self, var):
        product = tf.squeeze(tf.matmul(self.w, tf.expand_dims(var, -1)))
        norm = tf.reduce_sum((product - self.y) ** 2, 1)
        return norm

    def __call__(self, x):
        '''
        x = tf.get_variable('x', shape=[batch_size, num_dims],
            dtype=dtype, initializer=tf.random_normal_initializer(stddev=stdev))
        '''
        res = (self._func(x) / self.normalizer + self.e)
        if self.ptype == 'concave':
            res = 1 - res
        return res


class QuadraticEval:
    def __init__(self, num_dim=3, random=0.5, ptype='convex',
                 dtype=np.float32, ifprint=False, record=False):
        self.ndim = num_dim
        self.dtype = dtype
        if random is not None:
            self.e = np.random.normal(scale=random)
        else:
            self.e = 0.0
        self.record = record
        self.refresh()
        self.normalizer = np.maximum(
            self._func(np.zeros([1, self.ndim], dtype=self.dtype)),
            self._func(np.ones([1, self.ndim], dtype=self.dtype)))
        self.ptype = ptype
        self.ifprint = ifprint

    def refresh(self):
        self.w = np.random.normal(size=(self.ndim, self.ndim))
        self.a = np.random.uniform(low=0.01, high=0.99, size=(1, self.ndim))
        self.y = np.dot(self.a, self.w)
        if self.record:
            self.history = {'x':[], 'y':[]}

    def _func(self, x):
        product = np.squeeze(np.dot(x, self.w))
        norm = np.sum((product - self.y) ** 2)
        return norm

    def __call__(self, x):
        if self.ifprint:
            print('Input:')
            print(x)
        res = np.asscalar(self._func(x) / self.normalizer + self.e)
        if self.ptype == 'concave':
            res = 1 - res
        if self.ifprint:
            print('Output:')
            print(res)
        if self.record:
            self.history['x'].append(x)
            self.history['y'].append(res)
        return res 

class ConstraintQuadraticEval:
    def __init__(self, num_dim=3, random=0.5, ptype='convex',
                 dtype=np.float32):
        self.ndim = num_dim
        self.dtype = dtype
        if random is not None:
            self.e = np.random.normal(scale=random)
        else:
            self.e = 0.0
        self.refresh()
        self.normalizer = np.maximum(
            self._func(np.zeros([1, self.ndim], dtype=self.dtype)),
            self._func(np.ones([1, self.ndim], dtype=self.dtype)))
        self.ptype = ptype

    def refresh(self):
        self.w = np.random.normal(size=(self.ndim, self.ndim))
        self.a = np.maximum(np.minimum(np.random.normal(size=[self.ndim]), 0.8), 0.2)
        self.y = np.dot(self.a, self.w)

    def _func(self, x):
        product = np.squeeze(np.dot(x, self.w))
        norm = np.sum((product - self.y) ** 2)
        return norm

    def _barrier(self, x):
        return - np.sum(np.log(x) + np.log(1-x), 1) / 1e10

    def __call__(self, x):
        print('Input:')
        print(x)
        res = np.asscalar(self._func(x) / self.normalizer + self.e + self._barrier(x))
        if self.ptype == 'concave':
            res = 1 - res
        print('Output:')
        print(res)
        return res 


class RealReaction:
    def __init__(self, num_dim, param_range, param_names=['x1', 'x2', 'x3'],
                 direction='max', logger=None):
        self.ndim = num_dim
        self.param_range = param_range
        self.param_names = param_names
        self.direction = direction

    def x_convert(self, x):
        real_x = np.zeros([self.ndim])
        for i in range(self.ndim):
            a, b = self.param_range[i]
            real_x[i] = x[i] * (b - a) + a
        return real_x

    def y_convert(self, y):
        if self.direction == 'max':
            return 1 - y
        return y

    def __call__(self, x):
        print('Set Reaction Condition:')
        real_x = self.x_convert(np.squeeze(x))
        for i in range(self.ndim):
            print('{0}: {1:.3f}'.format(self.param_names[i], real_x[i]))
        result = float(input('Input the reaction yield:'))
        return self.y_convert(result)

    
