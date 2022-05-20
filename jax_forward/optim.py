import jax.numpy as jnp
from jax import jvp, jit, grad
from utils import _get_vector


@jit
def SGD(params, loss, data, step_size=0.01):
    x, y = data
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]


@jit
def FGD(params, loss, data, key, step_size=1e-4):
    x, y = data
    v = _get_vector(key, params)
    _, dd = jvp(lambda params: loss(params, x, y), (params,), (v,))
    step = step_size * dd
    return [(w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params, v)]
