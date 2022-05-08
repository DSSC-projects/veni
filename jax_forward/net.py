from msilib.schema import Class
from turtle import forward
import jax.numpy as jnp
from utils import _init_network_params


class MLP(object):

    def __init__(self, layers, func, key):
        """Multiperceptron neural network

        :param layers: list of layers
        :type layers: list[int]
        :param func: activation function
        :type func: list[functional]
        :param key: random seed
        :type key: jax.random.PRNGKey
        """
        self._key = key
        self._layers = layers
        self.params = _init_network_params(layers, key)

        if isinstance(func, list):
            self._functions = func
        else:
            self._functions = [func for _ in range(len(self._layers) - 1)]

    def __call__(self, x):
        return forward(self, x)

    # @jit
    def forward(self, x):
        for i, (w, b) in enumerate(self.params[:-1]):
            act = jnp.dot(w, x) + b
            x = self._functions[i](act)

        final_w, final_b = self.params[-1]

        return jnp.dot(final_w, x) + final_b

    @property
    def layers(self):
        return self._layers

    @property
    def key(self):
        return self._key


class Linear(object):

    def __init__(self, input, output, key):
        """Base class for linear layer

        :param input: dimension of input
        :type input: int
        :param output: dimension of output
        :type output: int
        :param key: random key seed
        :type key: jax.random.PRNGKey
        """
        self.params = _init_network_params([input, output], key)
        self._key = key
        self._input = input
        self._output = output

    def __call__(self, x):
        return forward(self, x)

    # @jit
    def forward(self, x):
        w, b = self.params[0]
        return jnp.dot(w, x) + b

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def key(self):
        return self._key


class Sequential(object):

    def __init__(self, list, func):
        """Base class for costructing sequential model

        :param list: list of models
        :type list: list
        :param func: activation functions
        :type func: functional
        """
        self._networks = list

        if isinstance(func, list):
            self._functions = func
        else:
            self._functions = [func for _ in range(len(self._networks) - 1)]

    def __call__(self, x):
        return forward(self, x)

    # @jit
    def forward(self, x):
        for i, net in enumerate(self._networks[:-1]):
            x = net(x)
            x = self._functions[i](x)
        return self._networks[-1](x)
