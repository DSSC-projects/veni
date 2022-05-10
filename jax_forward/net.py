import jax.numpy as jnp
from jax import vmap, jit
from utils import _init_network_params
from module import Module


class MLP(Module):

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

        self._forward = vmap(self.single_forward, in_axes=(None, 0))

    def single_forward(self, params, x):
        for i, (w, b) in enumerate(params[:-1]):
            act = jnp.dot(w, x) + b
            x = self._functions[i](act)

        final_w, final_b = params[-1]

        return jnp.dot(final_w, x) + final_b

    def forward(self, params, x):
        return self._forward(params, x)
    def generate_parameters(self):
        return None
    @property
    def layers(self):
        return self._layers

    @property
    def key(self):
        return self._key


class Linear(Module):

    def __init__(self, input, output, key):
        """Base class for linear layer

        :param input: dimension of input
        :type input: int
        :param output: dimension of output
        :type output: int
        :param key: random key seed
        :type key: jax.random.PRNGKey
        """
        self._key = key
        self._input = input
        self._output = output
        

    def _forward(self,params,x):
        return jnp.dot(x,params[0]) + params[1]

    def forward(self,params, x):
        return self._forward(params,x)
    
    def generate_parameters(self):

        params = _init_network_params([self._input, self._output],self._key)
        return params[0][0].T, params[0][1]

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def key(self):
        return self._key


class Sequential(Module):

    def __init__(self, list : list):
        """Base class for costructing sequential model

        :param list: list of models
        :type list: list
        :param func: activation functions
        :type func: functional
        """
        self._components = list

    def forward(self, params, x):
        out = x
        for p,c in zip(params,self._components):
            out = c(p,out)
        
        return out
    
    def generate_parameters(self):
        return [c.generate_parameters() for c in self._components]
