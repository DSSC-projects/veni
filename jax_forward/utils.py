import jax.numpy as jnp
from jax import random
import numpy as np
from torch.utils import data

def _random_layer_params(m, n, key, scale=1e-2):
    """Helper function to randomly initialize weights and biases
    for a dense neural network layer

    :param m: input layer
    :type m: int
    :param n: output layer
    :type n: int
    :param key: random seed 
    :type key: jax.random.PRNGKey
    :param scale: scaling factor, defaults to 1e-2
    :type scale: float, optional
    :return: random weights inizialization
    :rtype: jax.array
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def _init_network_params(sizes, key):
    """Initialize all layers for a fully-connected neural
    network with different sizes

    :param sizes: layer sizes
    :type sizes: list[int]
    :param key: random seed
    :type key: jax.random.PRNGKey
    :return: inizialization for all layers in a neural network
    :rtype: jax.array
    """
    keys = random.split(key, len(sizes))
    return [_random_layer_params(m, n, k)
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def _get_vector(key, params):
    v_shaped = []
    for w, b in params:
        key, subkey = random.split(key)
        v_w = random.normal(key, shape=w.shape)
        v_b = random.normal(subkey, shape=b.shape)
        v_shaped.append((v_w, v_b))
    return v_shaped

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))

    
