import jax
import jax.numpy as jnp
from .module import Optimizer


class SGD(Optimizer):

    def __init__(self, params, momentum=0, dampening=0, eta=1e-3, nestereov=False):
        """Implements stochastic gradient descent (optionally with momentum).

        :param params: paramters to optimize
        :type params: jax.array
        :param momentum: momentum factor, defaults to 0
        :type momentum: int, optional
        :param dampening: dampening for momentum, defaults to 0
        :type dampening: int, optional
        :param eta:  learning rate, defaults to 1e-3
        :type eta: float, optional
        :param nestereov: enables Nesterov momentum, defaults to False
        :type nestereov: bool, optional
        """
        self.params = params
        self.momentum = momentum
        self.dampening = dampening
        self.eta = eta
        self.nesterov = nestereov
        self.prev_grad = None
        self.t = 0

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
            if self.t == 0:
                self.b = grad
            else:
                self.b = [self.momentum * b + (1 - self.dampening) * g
                          for (b, g) in zip(self.b, grad)]

            if self.nesterov and self.t > 0:
                grad = [prev_grad + self.momentum * b
                        for (b, prev_grad) in zip(self.b, self.prev_grad)]
            else:
                grad = self.b

        self.prev_grad = grad
        self.t += 1
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
        self.m = [0 for _ in range(len(params)*2)]
        self.v = [0 for _ in range(len(params)*2)]

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

            dw, self.m[i], self.v[i] = self._single_update(
                self, dw, self.m[i], self.v[i])

            db, self.m[i + 1], self.v[i + 1] = self._single_update(
                self, dw, self.m[i + 1], self.v[i + 1])

            update_list.append((w - self.eta * dw, b - self.eta * db))

        return update_list
