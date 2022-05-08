import jax.numpy as jnp


def relu(x):
    """Applies the rectified linear unit function element-wise

    :param x: input
    :type x: jax.array
    :return: rectified linear unit on x
    :rtype: jax.array
    """
    return jnp.maximum(0, x)


def leaky_relu(x, negative_slope=0.01):
    """Applies the leaky rectified linear unit function element-wise

    :param x: input
    :type x: jax.array
    :param negative_slope: negative slope, defaults to 0.01
    :type negative_slope: float, optional
    :return: leaky rectified linear unit on x
    :rtype: jax.array
    """
    return jnp.maximum(0, x) + negative_slope * jnp.minimum(0, x)


def tanh(x):
    """Applies the tanh function element-wise

    :param x: input
    :type x: jax.array
    :return: tanh on x
    :rtype: jax.array
    """
    return jnp.tanh(x)


def sigmoid(x):
    """Applies the sigmoid function element-wise

    :param x: input
    :type x: jax.array
    :return: sigmoid on x
    :rtype: jax.array
    """
    return jnp.exp(x) / (jnp.exp(x) + 1.)


def log_sigmoid(x):
    """Applies the logarithmic sigmoid function element-wise

    :param x: input
    :type x: jax.array
    :return: logarithmic sigmoid on x
    :rtype: jax.array
    """
    return jnp.log(sigmoid(x))


def softplus(x, beta=1, threshold=20):
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


def softmax(x):
    """Applies the softmax function element-wise.

    :param x: input
    :type x: jax.array
    :return: softmax on x
    :rtype: jax.array
    """
    return jnp.exp(x) / jnp.exp(x).sum()


def log_softmax(x):
    """Applies the logarithmic softmax function element-wise.

    :param x: input
    :type x: jax.array
    :return: logarithmic softmax on x
    :rtype: jax.array
    """
    return jnp.log(softmax(x))
