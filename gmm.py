import numpy as np
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
from reactions import GMM as tf_GMM

class GMM:
    def __init__(self, n=6, ndim=3, cov=0.15, record=False):
        self.n = n
        self.ndim = ndim
        self.record = record
        self.cov = cov
        self.refresh()

    def refresh(self):
        self.m = [np.random.rand(self.ndim) for _ in range(self.n)]
        self.cov = [np.random.normal(self.cov, self.cov/5, size=self.ndim)
                    for _ in range(self.n)]
        self.param = np.random.normal(loc=0, scale=0.2, size=self.n)
        self.param /= np.sum(np.abs(self.param))
        if self.record:
            self.history = {'x':[], 'y':[]}

        self.cst = (2 * 3.14159) ** (- self.ndim / 2)
        modes = np.array([1/np.prod(cov) for cov in self.cov])
        modes = modes * self.param
        self.tops = np.max(modes)
        self.bots = np.min(modes)

    def __call__(self, x):
        y = [normal.pdf(x, self.m[i], self.cov[i]) for i in range(self.n)]

        fx = np.asscalar(
            np.dot(
                self.param.reshape((1, -1)),
                np.array(y).reshape((-1, 1)))/self.n)
        result = (fx / self.cst - self.bots) / (self.tops - self.bots)
        if self.record:
            self.history['x'].append(x)
            self.history['y'].append(result)
        return result



def test_1d():
    gmm = GMM(ndim=1)
    x = np.arange(0, 1, 0.01)
    y = [gmm(i) for i in x]
    plt.figure(1)
    plt.plot(x, y)
    plt.show()

def test_2d():
    gmm = GMM(ndim=2)
    xr = list(np.arange(0, 1, 0.02))
    X = np.array(list(product(xr, repeat=2)))
    Y = [gmm(i) for i in X]
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X[:, 0], X[:, 1], Y)
    fig.show()
    plt.show()

def test_tf():
    xr = list(np.arange(0, 1, 0.02))
    X = np.array(list(product(xr, repeat=2)))
    Y = []
    with tf.compat.v1.Session() as sess:
        gmm = tf_GMM(batch_size=1, ncoef=6, num_dims=2, cov=0.5)
        y = gmm(tf.placeholder(tf.float32, shape=[1, 2], name='x'))
        sess.run(tf.compat.v1.global_variables_initializer())
        for x in X:
            Y.append(sess.run(y, feed_dict={'x:0':x.reshape((1, 2))}))

    cmap = cm.get_cmap('rainbow')
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X[:, 0], X[:, 1], np.squeeze(Y),
                    linewidth=0.0, antialiased=True,
                    cmap=cmap)
    fig.show()
    plt.show()

if __name__ == '__main__':
    test_tf()
