import jax.numpy as jnp
from jax import vmap, jit
from .utils import _init_network_params
from .module import Module
import jax.random
from jax import lax


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

        self._forward = vmap(self.single_forward, in_axes=(0, None))

    def single_forward(self, x, params):
        for i, (w, b) in enumerate(params[:-1]):
            act = jnp.dot(w, x) + b
            x = self._functions[i](act)

        final_w, final_b = params[-1]

        return jnp.dot(final_w, x) + final_b

    def forward(self, x, params):
        return self._forward(x, params)

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

    def _forward(self, x, params):
        """Public forward method for Linear layer

        :param params: Parameters of the layer
        :type params: jnp.array
        :param x: Input
        :type x: jnp.array
        :return: Activation
        :rtype: jnp.array
        """
        return jnp.dot(x, params[0]) + params[1]

    def forward(self, x, params):
        """Public forward method for Linear layer

        :param params: Parameters of the layer
        :type params: jnp.array
        :param x: Input
        :type x: jnp.array
        :return: Activation
        :rtype: jnp.array
        """
        return self._forward(x, params)

    def generate_parameters(self):
        """Generate parameters for current layer

        :return: weight and bias tensors N(0,1) initialized
        :rtype: jnp.array
        """
        params = _init_network_params([self._input, self._output], self._key)
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

class Conv2D(Module):

    def __init__(self, inChannels,outChannels,kernelSize,  stride, padding, key):
        """Class for convolutional layer in 2d

        :param inChannels: number of input channels
        :type inChannels: int
        :param outChannels: number of output channels
        :type outChannels: int
        :param kernelSize: kernel radius
        :type kernelSize: int
        :param stride: stride of the convolution
        :type stride: int
        :param padding: number of pixel to pad the image
        :type padding: int
        :param key: prngKey
        :type key: jax.random.PRNGKey
        """
        self._key = key
        self._inCh = inChannels
        self._outCh = outChannels 
        self._s = stride
        self._p = padding
        self._k = kernelSize

   
    def forward(self, x, params):
        """Public forward method for Conv layer

        EXPECTS:
        x: tensor of the form NCHW (images)x(channels)x)(height)x(width)
        params[0]: tensor of the form OIHW (outputCh)x(inputCh)x(kernelHeight)x(kernelWidth)
        params[1]: bias

        :param params: Parameters of the layer
        :type params: jnp.array
        :param x: Input
        :type x: jnp.array
        :return: Activation
        :rtype: jnp.array
        """

        out = jax.lax.conv_general_dilated(x,params[0], (self._s,self._s), self._p, dimension_numbers=("NCHW","OIHW","NCHW"))
        return out + params[1]

    def generate_parameters(self):
        """Generate parameters for current layer

        :return: weight and bias tensors N(0,1) initialized
        :rtype: jnp.array
        """
        k_key, b_key = jax.random.split(self._key)
        params = _init_network_params([self._input, self._output], self._key)
        return 0.01*jax.random.normal(k_key, (self._outCh, self._inCh, self._k, self._k)), 0.01*jax.random.normal(b_key, (1, self._outCh))

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def key(self):
        return self._key

class Flatten(Module):

    def __init__(self):
        """Flatten layer

            takes a tensor of the shape (N,k1,k2,...,kn)
            and returns tensor of the shape (N, k1 * k2 * ... * kn)

        """
        

   
    def forward(self, x, params = None):

        """returns flattened tensor

        :return: _description_
        :rtype: _type_
        """
        l = 1
        for d in x.shape:
            l = l*d
        
        l = l/x.shape[0]

        return x.reshape((x.shape[0],l))

    def generate_parameters(self):
        """Generate parameters for current layer

        :return: weight and bias tensors N(0,1) initialized
        :rtype: jnp.array
        """
       
        return jnp.array([]), jnp.array([])

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

    def __init__(self, list):
        """Base class for costructing sequential model

        :param list: list of models
        :type list: list
        :param func: activation functions
        :type func: functional
        """
        self._components = list

    def forward(self, x, params):
        """Forward method for sequential object

        :param params: _description_
        :type params: jnp.array
        :param x: _description_
        :type x: jnp.array
        :return: activation
        :rtype: jnp.array
        """
        out = x
        for p, c in zip(params, self._components):
            out = c(out, p)

        return out

    def generate_parameters(self):
        """Generate parameters for layers in sequential

        :return: _description_
        :rtype: jnp.array
        """
        return [c.generate_parameters() for c in self._components]
