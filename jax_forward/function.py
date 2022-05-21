import jax.numpy as jnp
from jax import vmap
from .functiontools import *
from .module import Activation


class ReLU(Activation):

    def __init__(self):
        """Base class for relu activation function"""
        super().__init__(relu)

    def forward(self, x, params=None):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([])


class LeakyReLu(Activation):

    def __init__(self):
        """Base class for leaky relu activation function"""
        super().__init__(leaky_relu)

    def forward(self, x, params=None):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([])


class Tanh(Activation):

    def __init__(self):
        """Base class for tanh activation function"""
        super().__init__(tanh)

    def forward(self, x, params=None):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([])


class Sigmoid(Activation):
    def __init__(self):
        """Base class for sigmoid activation function"""
        super().__init__(sigmoid)

    def forward(self, x, params=None):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([])


class LogSigmoid(Activation):

    def __init__(self):
        """Base class for log sigmoid activation function"""
        super().__init__(log_sigmoid)

    def forward(self, x, params=None):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([])


class Softplus(Activation):

    def __init__(self):
        """Base class for softplus activation function"""
        super().__init__(softplus)

    def forward(self, x, params=None):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([])


class Softmax(Activation):

    def __init__(self):
        """Base class for softmax activation function"""
        super().__init__(softmax)

    def forward(self, x, params=None):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([])


class LogSoftmax(Activation):

    def __init__(self):
        """Base class for log softmax activation function"""
        super().__init__(log_softmax)
    def forward(self, x, params=None):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([])
