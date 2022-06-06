import jax.numpy as jnp
from jax import vmap, jit
from .utils import _init_network_params
from .module import Module
import jax.random
import jax
import jax.nn



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

    def __init__(self, input, output, key, bias = True):
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
        self._bias = bias

    def _forward(self, x, params):
        """Public forward method for Linear layer

        :param params: Parameters of the layer
        :type params: jnp.array
        :param x: Input
        :type x: jnp.array
        :return: Activation
        :rtype: jnp.array
        """
        if not self._bias:
            return jnp.dot(x,params[0])
        else:

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
        self._key = jax.random.split(self._key)
        """Generate parameters for current layer

        :return: weight and bias tensors N(0,1) initialized
        :rtype: jnp.array
        """
        w_shape = (self._input, self._output)
        b_shape = (1, self._output)

        w_init = jax.nn.initializers.he_uniform(in_axis=0, out_axis=1)
        b_init = jax.nn.initializers.he_uniform(in_axis = 0, out_axis=1)
        params = [ w_init(self._key[0], w_shape), b_init(self._key[1], b_shape)]
        if self._bias:
            return params[0], params[1]
        else:
            return params[0], jnp.array([]) 

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
        l = 1/jnp.sqrt(self._inCh )
        kernel_shape = (self._outCh, self._inCh, self._k, self._k)
        bias_shape = (1, self._outCh, 1, 1)
        k_init = jax.nn.initializers.he_uniform(in_axis = 1, out_axis=0)
        b_init = jax.nn.initializers.he_uniform(out_axis= 1)
        return k_init(k_key, shape = kernel_shape), b_init(b_key,shape = bias_shape)
        #return l*jax.random.normal(k_key, (self._outCh, self._inCh, self._k, self._k)), jax.random.normal(b_key, (1, self._outCh, 1, 1)) / jnp.sqrt(self._inCh)

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def key(self):
        return self._key


class MaxPool2D(Module):

    def __init__(self, kernel_size, stride = None, padding = None):
        """Class for avgPool layer in 2d

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
        self._k = kernel_size
        if stride == None:
            self._s = kernel_size
        else:
            self._s = stride
        
        self._avp = lambda x: jax.lax.reduce_window(x, -jax.numpy.inf, jax.lax.max, 
                                                    window_dimensions= (self._k, self._k),
                                                    window_strides= (self._s, self._s),
                                                    padding = 'VALID')

        self._bf = vmap(self._avp, in_axes=0, out_axes = 0)
        self._f = vmap(self._bf, in_axes= 1, out_axes= 1)

   
    def forward(self, x, params = None):
        """Public forward method for Conv layer


        :param params: Parameters of the layer
        :type params: jnp.array
        :param x: Input
        :type x: jnp.array
        :return: Activation
        :rtype: jnp.array
        """

        return self._f(x)

    def generate_parameters(self):
        """Generate parameters for current layer

        :return: weight and bias tensors N(0,1) initialized
        :rtype: jnp.array
        """
        return jnp.array([]) , jnp.array([])

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def key(self):
        return self._key

class AvgPool2D(Module):

    def __init__(self, kernel_size, stride = None, padding = None):
        """Class for avgPool layer in 2d

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
        self._k = kernel_size
        if stride == None:
            self._s = kernel_size
        else:
            self._s = stride
        
        self._avp = lambda x: jax.lax.reduce_window(x, 0, jax.lax.add, 
                                                    window_dimensions= (self._k, self._k),
                                                    window_strides= (self._s, self._s),
                                                    padding = 'VALID')

        self._bf = vmap(self._avp, in_axes=0, out_axes= 0)
        self._f = vmap(self._bf, in_axes= 1, out_axes= 1)

   
    def forward(self, x, params = None):
        """Public forward method for Conv layer


        :param params: Parameters of the layer
        :type params: jnp.array
        :param x: Input
        :type x: jnp.array
        :return: Activation
        :rtype: jnp.array
        """

        return self._f(x)

    def generate_parameters(self):
        """Generate parameters for current layer

        :return: weight and bias tensors N(0,1) initialized
        :rtype: jnp.array
        """
        return jnp.array([]) , jnp.array([])

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

        TODO: optimize that
        """
        l = 1
        for d in x.shape:
            l = l*d
        
        l = l//x.shape[0]

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
