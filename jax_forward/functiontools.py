import jax.numpy as jnp
from jax import vmap


def _relu(x):
    """Applies the rectified linear unit function element-wise

    :param x: input
    :type x: jax.array
    :return: rectified linear unit on x
    :rtype: jax.array
    """
    return jnp.maximum(0, x)


def relu(x):
    """Applies the rectified linear unit function element-wise

    :param x: input
    :type x: jax.array
    :return: rectified linear unit on x
    :rtype: jax.array
    """
    f = vmap(_relu)
    return f(x)


def _leaky_relu(x, negative_slope=0.01):
    """Applies the leaky rectified linear unit function element-wise

    :param x: input
    :type x: jax.array
    :param negative_slope: negative slope, defaults to 0.01
    :type negative_slope: float, optional
    :return: leaky rectified linear unit on x
    :rtype: jax.array
    """
    return jnp.maximum(0, x) + negative_slope * jnp.minimum(0, x)


def leaky_relu(x):
    """Applies the leaky rectified linear unit function element-wise

    :param x: input
    :type x: jax.array
    :param negative_slope: negative slope, defaults to 0.01
    :type negative_slope: float, optional
    :return: leaky rectified linear unit on x
    :rtype: jax.array
    """
    f = vmap(_leaky_relu)
    return f(x)


def _tanh(x):
    """Applies the tanh function element-wise

    :param x: input
    :type x: jax.array
    :return: tanh on x
    :rtype: jax.array
    """
    return jnp.tanh(x)


def tanh(x):
    """Applies the tanh function element-wise

    :param x: input
    :type x: jax.array
    :return: tanh on x
    :rtype: jax.array
    """
    f = vmap(_tanh)
    return f(x)


def _sigmoid(x):
    """Applies the sigmoid function element-wise

    :param x: input
    :type x: jax.array
    :return: sigmoid on x
    :rtype: jax.array
    """
    return  1. / (jnp.exp(-x) + 1.)
    


def sigmoid(x):
    """Applies the sigmoid function element-wise

    :param x: input
    :type x: jax.array
    :return: sigmoid on x
    :rtype: jax.array
    """
    f = vmap(_sigmoid)
    return f(x)


def _log_sigmoid(x):
    """Applies the logarithmic sigmoid function element-wise

    :param x: input
    :type x: jax.array
    :return: logarithmic sigmoid on x
    :rtype: jax.array
    """
    return jnp.log(_sigmoid(x))


def log_sigmoid(x):
    """Applies the logarithmic sigmoid function element-wise

    :param x: input
    :type x: jax.array
    :return: logarithmic sigmoid on x
    :rtype: jax.array
    """
    f = vmap(_log_sigmoid)
    return f(x)


def _softplus(x, beta=1, threshold=20):
    """Applies the softplus function element-wise.
    For numerical stability the implementation reverts 
    to the linear function when input*beta > threshold.

    :param x: input
    :type x: jax.array
    :param beta: paramter, defaults to 1
    :type beta: int, optional
    :param threshold: threshold, defaults to 20
    :type threshold: int, optional
    :raises ValueError: beta value must be greater than zero
    :return: softplus on x
    :rtype: jax.array
    """
    if beta < 0:
        raise ValueError("beta value must be greater than zero.")

    if beta * x > threshold:
        return x

    return jnp.log(1. + jnp.exp(beta * x)) / beta


def softplus(x):
    """Applies the softplus function element-wise.
    For numerical stability the implementation reverts 
    to the linear function when input*beta > threshold.

    :param x: input
    :type x: jax.array
    :param beta: paramter, defaults to 1
    :type beta: int, optional
    :param threshold: threshold, defaults to 20
    :type threshold: int, optional
    :raises ValueError: beta value must be greater than zero
    :return: softplus on x
    :rtype: jax.array
    """
    f = vmap(_softplus)
    return f(x)


def _softmax(x):
    """Applies the softmax function element-wise.

    :param x: input
    :type x: jax.array
    :return: softmax on x
    :rtype: jax.array
    """
    return jnp.exp(x) / jnp.exp(x).sum()


def softmax(x):
    """Applies the softmax function element-wise.

    :param x: input
    :type x: jax.array
    :return: softmax on x
    :rtype: jax.array
    """
    f = vmap(_softmax)
    return f(x)


def _log_softmax(x):
    """Applies the logarithmic softmax function element-wise.

    :param x: input
    :type x: jax.array
    :return: logarithmic softmax on x
    :rtype: jax.array
    """
    return jnp.log(_softmax(x))


def log_softmax(x):
    """Applies the logarithmic softmax function element-wise.

    :param x: input
    :type x: jax.array
    :return: logarithmic softmax on x
    :rtype: jax.array
    """
    f = vmap(_log_softmax)
    return f(x)

#### LOSSES

def CrossEntropy(y,y_hat):
    """CrossEntropy loss
    EXPECTS: tensor of the shape (N, k1, k2, ..., kn)
    where N is the number of examples in the batch


    :param y: Ground truth tensor
    :type y: jnp.array
    :param y_hat: Model predictions
    :type y_hat: jnp.array
    :return: Loss for each batch
    :rtype: float
    """
    return jnp.sum(-y*jnp.log(y_hat))/y.shape[0]