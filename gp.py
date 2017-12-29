import GPy
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from reactions import QuadraticEval

def gp(prior_X, prior_Y, variance=0.1, lengthscale=0.1, X=None, nsamples=1):
    nd = prior_X.shape[1]
    kernel = GPy.kern.RBF(input_dim=nd, variance=variance, lengthscale=lengthscale)
    m = GPy.models.GPRegression(prior_X, prior_Y, kernel)
    if X is None:
        X = np.arange(0, 1, 0.01).reshape((-1, 1))
    Y = m.posterior_samples_f(X, size=nsamples)
    return Y

def plot_1d(X, Y):
    plt.figure(1)
    plt.plot(X, Y)
    plt.show()

def test_1d():
    prior_X = np.array([[0], [1]])
    prior_Y = np.array([[10], [10]])
    X = np.arange(0, 1, 0.01)
    Y = gp(prior_X, prior_Y, nsamples=1)
    plot_1d(X, Y)

def plot_2d(X, Y, Z):
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, np.squeeze(Z))
    plt.show()

def test_2d():
    prior_X = np.array([[0, 0]])
    prior_Y = np.array([[0]])
    xr = list(np.arange(0, 1, 0.02))
    X = np.array(list(product(xr, repeat=2))) 
    Y = gp(prior_X, prior_Y, X = X, nsamples=1, lengthscale=0.3)
    plot_2d(X[:, 0], X[:, 1], Y)


class SquaredDistanceKernel():
    def __init__(self, param=0.1):
        self.param = param

    def __call__(self, a, b):
        sq_dist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
        return np.exp(- sq_dist / self.param / 2)

class GaussianProcess(object):
    def __init__(self, kernel=SquaredDistanceKernel(), noise=0.0):
        self.kernel = kernel
        self.noise = noise
        self.X = None
        self.Y = None

    def prior(self, x, y):
        self.X = x
        self.Y = y

    def predict(self, x):
        k2 = self.kernel(x, x)
        self.cov = None
        if self.X and self.Y:
            self.cov = self.kernel(X)
        if self.cov is None:
            mu = np.zeros(x.shape)
            cov_posterior = k2 + (self.noise * np.eye(k2.shape[0]))

        else:
            l = np.linalg.cholesky(self.cov + self.noise * np.eye(self.cov.shape[0]))
            k = self.kernel(X, x)
            ldk = np.linalg.solve(l, k)

            mu = np.dot(ldk.T, np.linalg.solve(1, self.Y))
            cov_posterior = k2 + self.noise * np.eye(k2.shape[0]) - np.dot(ldk.T, ldk)

        return mu, cov_posterior

def t1d():
    gp = GaussianProcess()
    np.random.seed(1)
    x = np.arange(0, 1, 0.01).reshape(-1, 1)
    mu, cov = gp.predict(x)
    y = np.random.multivariate_normal(np.squeeze(mu), cov)
    plot_1d(np.squeeze(x), np.squeeze(y))

def t2d():
    gp = GaussianProcess()
    np.random.seed(1)
    xr = list(np.arange(0, 1, 0.02))
    x = np.array(list(product(xr, repeat=2)))
    mu, cov = gp.predict(np.array([[0.48, 0.68], [0, 0]]))
    import pdb; pdb.set_trace()
    y = np.random.multivariate_normal(np.zeros(2), cov)
    plot_2d(x[:, 0], x[:, 1], y)

class GPOpt:
    def __init__(self, ndim, prange=[]):
        self.ndim = ndim
        self.prange = prange
        self.X = []
        self.y = []
        xr = list(np.arange(0, 1, 0.1))
        self.x = np.array(list(product(xr, repeat=ndim)))

    def update(self, X, y):
        normalized_X = [0] * self.ndim
        for i in range(self.ndim):
            a, b = self.prange[i]
            normalized_X[i] = (X[i] - a) / (b - a)
        self.X.append(normalized_X)
        self.y.append(y)

    def next(self):
        if len(self.X) == 0:
            x = np.random.rand(3)
        else:
            kernel = GPy.kern.RBF(input_dim=self.ndim,
                                  variance=1, lengthscale=1)
            X = np.array(self.X)
            y = np.array(self.y).reshape((-1, 1))
            m = GPy.models.GPRegression(X, y, kernel)
            y_pred = m.posterior_samples_f(self.x, size=1)
            x = self.x[np.argmax(y_pred)]
        real_x = [0] * self.ndim
        for i in range(self.ndim):
            a, b = self.prange[i]
            real_x[i] = x[i] * (b - a) + a
        return real_x

def test_gpopt():
    opt = GPOpt(3, prange=[(0, 2), (0, 2), (0, 2)])
    func = QuadraticEval(num_dim=3, random=None, ptype='concave')

    y_array = []
    for i in range(30):
        x = opt.next()
        y_eval = func(x)
        y_array.append(y_eval)
        opt.update(x, y_eval)

    plt.figure()
    plt.plot(y_array)
    plt.show()

test_gpopt()


