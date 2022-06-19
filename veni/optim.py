from jax.tree_util import tree_map
import jax
import jax.numpy as jnp
from .module import Optimizer, Sampler


class SGD(Optimizer):

    def __init__(self, params, momentum=0, dampening=0, eta=1e-3):
        """Implements stochastic gradient descent (optionally with momentum).

        :param params: paramters to optimize
        :type params: jax.array
        :param momentum: momentum factor, defaults to 0
        :type momentum: int, optional
        :param dampening: dampening for momentum, defaults to 0
        :type dampening: int, optional
        :param eta:  learning rate, defaults to 1e-3
        :type eta: float, optional
        """
        self.params = params
        self.momentum = momentum
        self.dampening = dampening
        self.eta = eta
        self.prev_grad = None

    def _update(self, grad, prev_grad):
        return [(self.momentum * pgw + (1 - self.dampening)*gw,
                 self.momentum * pgb + (1 - self.dampening)*gb)
                for ((gw, gb), (pgw, pgb)) in zip(grad, prev_grad)]

    def update(self, params, grad):
        """Update method for SGD

        :param params: paramters to optimize
        :type params: jax.array
        :param grad: loss gradient
        :type grad: jax.array
        :return: optimized parameters
        :rtype: jax.array
        """
        if self.momentum != 0:
            if self.prev_grad is None:
                self.prev_grad = grad
            else:
                grad = self._update(grad, self.prev_grad)
                self.prev_grad = grad

        return [(w - self.eta * dw, b - self.eta * db)
                for (w, b), (dw, db) in zip(params, grad)]


class Adam(Optimizer):

    def __init__(self, params, beta1=0.9, beta2=0.999, eta=1e-3):
        """Implements Adam algorithm

        :param params: paramters to optimize
        :type params: jax.array
        :param beta1: coefficients used for computing running
        averages of gradient, defaults to 0.9
        :type beta1: float, optional
        :param beta2: coefficients used for computing running
        averages of gradient square, defaults to 0.999
        :type beta2: float, optional
        :param eta: learning rate, defaults to 1e-3
        :type eta: float, optional
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.tolerance = 1e-8
        self.t = 0
        self.m = [[0, 0] for _ in range(len(params))]
        self.v = [[0, 0] for _ in range(len(params))]

    def _single_update(self, grad, m, v):
        m = self.beta1 * m + (1. - self.beta1) * grad
        v = self.beta2 * v + (1. - self.beta2) * grad ** 2
        m_hat = m / (1. - self.beta1 ** self.t)
        v_hat = v / (1. - self.beta2 ** self.t)
        update_coeff = m_hat / (jnp.sqrt(v_hat) + self.tolerance)
        return update_coeff, m, v

    def update(self, params, grads):
        """Update method for Adam

        :param params: paramters to optimize
        :type params: jax.array
        :param grad: loss gradient
        :type grad: jax.array
        :return: optimized parameters
        :rtype: jax.array
        """
        self.t += 1
        update_list = []
        for i, ((w, b), (dw, db)) in enumerate(zip(params, grads)):

            dw, self.m[i][0], self.v[i][0] = self._single_update(
                dw, self.m[i][0], self.v[i][0])

            db, self.m[i][1], self.v[i][1] = self._single_update(
                db, self.m[i][1], self.v[i][1])

            update_list.append((w - self.eta * dw, b - self.eta * db))

        return update_list


def plist_reduce(vs, js):
    res = []
    how_many_vs = len(vs)
    len_v = len(vs[0])

    for j in range(len_v):
        w, b = jnp.zeros_like(vs[0][j][0]), jnp.zeros_like(vs[0][j][1])
        for i in range(how_many_vs):
            w += js[i] * vs[i][j][0]
            b += js[i] * vs[i][j][1]

        res.append((w, b))

    return res


class NormalLikeSampler(Sampler):
    def __init__(self, key=None):
        """Sampler for sampling from a N(0,1) distribution

        :param key: jax prng key, defaults to None. Acts like the seed for prng sampling initialization if key is None it is initialized using the internal clock
        :type key: jax.random.PRNGKey, optional
        """
        super(NormalLikeSampler, self).__init__(key)

    def forward(self, arr):
        sample = jax.random.normal(self._key, arr.shape)
        self._key, _ = jax.random.split(self._key)
        return sample


class RademacherLikeSampler(Sampler):
    def __init__(self, key=None):
        """Sampler for sampling from a Rademacher distribution

        :param key: jax prng key, defaults to None. Acts like the seed for prng sampling initialization if key is None it is initialized using the internal clock
        :type key: jax.random.PRNGKey, optional
        """
        super().__init__(key)

    def forward(self, arr):
        sample = jax.random.rademacher(self._key, arr.shape, dtype='float32')
        self._key, _ = jax.random.split(self._key)
        return sample


class TruncatedNormalLikeSampler(Sampler):
    def __init__(self, key=None, lower=-1, upper=1):
        """Sampler for sampling from a Rademacher distribution

        :param key: jax prng key, defaults to None. Acts like the seed for prng sampling initialization if key is None it is initialized using the internal clock
        :type key: jax.random.PRNGKey, optional
        """
        super().__init__(key)
        self.lower = lower
        self.upper = upper

    def forward(self, arr):
        sample = jax.random.truncated_normal(self._key, self.lower, self.upper,
                                             arr.shape, dtype='float32')
        self._key, _ = jax.random.split(self._key)
        return sample


def grad_fwd(params, x, y, loss, dirs=1, sampler=NormalLikeSampler()):
    """Function to calculate the gradient in forward mode using 1 or more directions

    :param params: Parameters of the model
    :type params: List
    :param x: Input of the model
    :type x: jnp.array
    :param y: labels
    :type y: jnp.array
    :param loss: loss function
    :type loss: Callable
    :param dirs: Number of directions used to calculate the gradient, defaults to 1
    :type dirs: int, optional
    :param sampler: Sampler used to sample gradient direction for each layer, defaults to NormalLikeSampler()
    :type sampler: Class, optional
    :return: Gradient as list of all components for each layer
    :rtype: List
    """

    if dirs == 1:
        v = tree_map(sampler, params)
        _, j = jax.jvp(lambda p: loss(p, x, y), (params, ), (v,))
        return tree_map(lambda a: jnp.dot(a, j), v)

    else:
        vs = [tree_map(sampler, params) for _ in range(dirs)]
        js = [jax.jvp(lambda p: loss(p, x, y), (params, ), (v,))[1]
              for v in vs]

        return plist_reduce(vs, js)
