import abc
import jax.numpy as jnp
import jax


class Module(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, x, params=None):
        pass

    def __call__(self, x, params=None):
        return self.forward(x, params)


class Activation(abc.ABC):
    def __init__(self, f):
        self._f = f

    @abc.abstractmethod
    def forward(self, x, params=None):
        pass

    def __call__(self, x, params=None):
        return self.forward(x, params)

    @abc.abstractmethod
    def generate_parameters(self):
        pass


class Optimizer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def update(self, params, grad):
        pass

    def __call__(self, params, grad):
        return self.update(params, grad)


class Sampler(abc.ABC):
    def __init__(self, key=None):
        if key is None:
            from time import time_ns
            self._key = jax.random.PRNGKey(time_ns())

    @abc.abstractmethod
    def forward(self, arr):
        pass

    def __call__(self, arr):
        return self.forward(arr)
