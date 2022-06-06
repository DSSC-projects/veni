import abc
import jax.numpy as jnp


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

    def update(self, params, grad):
        return [ (self.__update(w, gw), self.__update(b, gb)) for (w,b), (gw, gb) in zip(params, grad) ]


    def __call__(self, params, grad):
        return self.update(params, grad)

   # @abc.abstractmethod
   # def __update(self,params,grad):
   #     pass
