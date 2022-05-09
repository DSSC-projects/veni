import jax.numpy as jnp
from jax import vmap
import functiontools as F
from module import Activation


class ReLu(Activation):

    def __init__(self):
        """Base class for relu activation function"""
        super().__init__(vmap(F._relu))

    def forward(self, x):
        return self._f(x)


class LeakyReLu(Activation):

    def __init__(self):
        """Base class for leaky relu activation function"""
        super().__init__(vmap(F._leaky_relu))

    def forward(self, x):
        return self._f(x)


class Tanh(Activation):

    def __init__(self):
        """Base class for tanh activation function"""
        super().__init__(vmap(F._tanh))

    def forward(self, x):
        return self._f(x)


class Sigmoid(Activation):
    def __init__(self):
        """Base class for sigmoid activation function"""
        super().__init__(vmap(F._sigmoid))

    def forward(self, x):
        return self._f(x)


class LogSigmoid(Activation):

    def __init__(self):
        """Base class for log sigmoid activation function"""
        super().__init__(vmap(F._log_sigmoid))

    def forward(self, x):
        return self._f(x)


class Softplus(Activation):

    def __init__(self):
        """Base class for softplus activation function"""
        super().__init__(vmap(F._softplus))

    def forward(self, x):
        return self._f(x)


class Softmax(Activation):

    def __init__(self):
        """Base class for softmax activation function"""
        super().__init__(vmap(F._softmax))

    def forward(self, x):
        return self._f(x)


class LogSoftmax(Activation):

    def __init__(self):
        """Base class for log softmax activation function"""
        super().__init__(vmap(F._log_softmax))

    def forward(self, x):
        return self._f(x)
