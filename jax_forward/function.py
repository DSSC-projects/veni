import jax.numpy as jnp
from jax import vmap
import functiontools as F
from module import Activation


class ReLU(Activation):

    def __init__(self):
        """Base class for relu activation function"""
        super().__init__(vmap(F._relu))

    def forward(self,params, x):
        return self._f(x)
    
    def generate_parameters(self):
        return jnp.array([]), jnp.array([]) 


class LeakyReLu(Activation):

    def __init__(self):
        """Base class for leaky relu activation function"""
        super().__init__(vmap(F._leaky_relu))

    def forward(self,params, x):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([]) 
         

class Tanh(Activation):

    def __init__(self):
        """Base class for tanh activation function"""
        super().__init__(vmap(F._tanh))

    def forward(self,params, x):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([]) 
         

class Sigmoid(Activation):
    def __init__(self):
        """Base class for sigmoid activation function"""
        super().__init__(vmap(F._sigmoid))

    def forward(self,params, x):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([]) 
         

class LogSigmoid(Activation):

    def __init__(self):
        """Base class for log sigmoid activation function"""
        super().__init__(vmap(F._log_sigmoid))

    def forward(self,params, x):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([]) 
         

class Softplus(Activation):

    def __init__(self):
        """Base class for softplus activation function"""
        super().__init__(vmap(F._softplus))

    def forward(self,params, x):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([]) 
         

class Softmax(Activation):

    def __init__(self):
        """Base class for softmax activation function"""
        super().__init__(vmap(F._softmax))

    def forward(self,params, x):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([]) 
         

class LogSoftmax(Activation):

    def __init__(self):
        """Base class for log softmax activation function"""
        super().__init__(vmap(F._log_softmax))

    def forward(self,params, x):
        return self._f(x)

    def generate_parameters(self):
        return jnp.array([]), jnp.array([]) 


         