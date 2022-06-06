import jax
import jax.numpy as jnp
from .module import Optimizer


class SGD(Optimizer):

    def __init__(self, params, momentum=0,
                damping=0, eta=1e-3, nestereov=False):
        """Implements stochastic gradient descent (optionally with momentum).

        :param params: paramters to optimize
        :type params: jax.array
        :param momentum: momentum factor, defaults to 0
        :type momentum: int, optional
        :param damping: damping for momentum, defaults to 0
        :type damping: int, optional
        :param eta:  learning rate, defaults to 1e-3
        :type eta: float, optional
        :param nestereov: enables Nesterov momentum, defaults to False
        :type nestereov: bool, optional
        """
        #self.shape = params.shape
        self.momentum = momentum
        self.damping = damping
        self.eta = eta
        self.nesterov = nestereov
        self.prev_grad = None
        self.t = 0

    def __update(self, params, grad):
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
                self.b = self.momentum * self.b + (1 - self.damping) * grad

            if self.nesterov and self.t > 0:
                grad = self.prev_grad + self.momentum * self.b
            else:
                grad = self.b

        self.prev_grad = grad
        self.t += 1
        return params - self.eta * grad

    def update(self, params, grad, scale = 1):
        return [ (self.__update(w, gw)*scale, self.__update(b, gb)*scale) for (w,b), (gw, gb) in zip(params, grad) ]



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
        self.shape = params.shape
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.t = 0
        self.m = jnp.zeros(self.shape)
        self.v = jnp.zeros(self.shape)

    def __update(self, params, grad):
        """Update method for Adam

        :param params: paramters to optimize
        :type params: jax.array
        :param grad: loss gradient
        :type grad: jax.array
        :return: optimized parameters
        :rtype: jax.array
        """
        tolerance = 1e-8
        self.t += 1
        self.m = self.beta1 * self.m + (1. - self.beta1) * grad
        self.v = self.beta2 * self.v + (1. - self.beta2) * grad ** 2
        m_hat = self.m / (1. - self.beta1 ** self.t)
        v_hat = self.v / (1. - self.beta2 ** self.t)
        return params - self.eta * m_hat / (jnp.sqrt(v_hat) + tolerance)
