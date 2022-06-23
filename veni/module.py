import abc
import jax.numpy as jnp
import jax
import numpy as np
import torch


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
    def __init__(self):

        self._params_shape = None
        self._params_offset = None
        self._params_len = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def _extract_vector_dim(self, arr):
        """Extracting vector dimensions for all parameters

        :param arr: list(tuple(jnp.array, jnp.array))
        :type arr: void
        """
        self._params_shape = []
        self._params_offset = []
        self._params_len = 0

        l = 0
        o = 0
        for w, b in arr:
            self._params_shape.append((w.shape, b.shape))
            wl = np.prod(w.shape)
            bl = np.prod(b.shape)

            wo = l
            l += wo
            bo = l

            self._params_offset.append((wo, bo))
            self._params_len += wl + bl
        self._params_cumulative = np.array(
            [[np.prod(ws), np.prod(wb)] for ws, wb in self._params_shape])
        self._params_cumulative[:, 0] = np.cumsum(
            self._params_cumulative[:, 0])
        self._params_cumulative[:, 1] = np.cumsum(
            self._params_cumulative[:, 1])

        return

    def _make_sample(self, arr):
        """Sampling tangent vector for jvp

        :param arr: parameters of the model
        :type arr: list(tuple(jnp.array, jnp.array))
        :return: tangent vectors
        :rtype:  list(tuple(jnp.array, jnp.array))
        """
        if self._params_shape is None or self._params_offset is None or self._params_len is None:
            self._extract_vector_dim(arr)

        v = self._sample_vect()
        vv = np.split(v, [self._params_cumulative[-1, 0]])

        vw = np.split(vv[0], self._params_cumulative[:-1, 0])
        vb = np.split(vv[1], self._params_cumulative[:-1, 1])
        v_shaped = [(w.reshape(ws), b.reshape(bs))
                    for w, b, (ws, bs) in zip(vw, vb, self._params_shape)]

        return v_shaped

    @abc.abstractmethod
    def _sample_vect(self):
        pass

    def __call__(self, arr):
        return self._make_sample(arr)
