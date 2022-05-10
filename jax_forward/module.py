import abc
import jax.numpy as jnp


class Module(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, params, x):
        pass

    def __call__(self, params, x):
        return self.forward(params, x)

   



class Activation(abc.ABC):
    def __init__(self, f):
        self._f = f

    @abc.abstractmethod
    def forward(self, params, x):
        pass

    def __call__(self,params, x):
        return self.forward(params, x)

    @abc.abstractmethod
    def generate_parameters(self):
        pass