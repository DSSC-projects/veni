import jax
import jax.numpy as jnp


class SGD(object):

    def __init__(self, params, momentum=0, dampening=0, eta=1e-3, nestereov=False):
        self.shape = params.shape
        self.momentum = momentum
        self.dampening = dampening
        self.eta = eta
        self.nesterov = nestereov
        self.prev_grad = None
        self.t = 0

    def __update(self, params: jnp.DeviceArray, grad: jnp.DeviceArray) -> jnp.DeviceArray:

        if self.momentum != 0:
            if self.t == 0:
                self.b = grad
            else:
                self.b = self.momentum * self.b + (1 - self.dampening) * grad

            if self.nesterov and self.t > 0:
                grad = self.prev_grad + self.momentum * self.b
            else:
                grad = self.b

        self.prev_grad = grad
        self.t += 1
        return params - self.eta * grad

    def forward_update(self, params, key, loss):
        v = jax.random.normal(key, shape=self.shape)
        _, dd = jax.jvp(loss, (params,), (v,))
        return self.__update(params, dd * v)

    def backward_update(self, params, loss):
        grad = jax.grad(loss)(params)
        return self.__update(params, grad)


class Adam(object):

    def __init__(self, params, beta1=0.9, beta2=0.999, eta=1e-3):
        self.shape = params.shape
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.t = 0
        self.m = jnp.zeros(self.shape)
        self.v = jnp.zeros(self.shape)

    def __update(self, params: jnp.DeviceArray, grad: jnp.DeviceArray) -> jnp.DeviceArray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.eta * m_hat / (jnp.sqrt(v_hat) + 1e-8)

    def forward_update(self, params, key, loss):
        v = jax.random.normal(key, shape=self.shape)
        _, dd = jax.jvp(loss, (params,), (v,))
        return self.__update(params, dd * v)

    def backward_update(self, params, loss):
        grad = jax.grad(loss)(params)
        return self.__update(params, grad)
