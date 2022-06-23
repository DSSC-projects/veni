import jax
from jax import numpy as jnp
from datetime import datetime
import pandas as pd


@jax.jit
def beale(x: jnp.array) -> jnp.array:
    return jnp.power(1.5 - x[0] + x[0] * x[1], 2) + \
        jnp.power(2.25 - x[0] + x[0] * x[1] * x[1], 2) + \
        jnp.power(2.625 - x[0] + x[0] * x[1] * x[1] * x[1], 2)


@jax.jit
def rosenbrock(x: jnp.array) -> jnp.array:
    return jnp.power(1. - x[0], 2) + \
        100. * jnp.power(x[1] - x[0] * x[0], 2)


def fwd_grad(f, x0, key):
    t = jax.random.normal(key, shape=x0.shape)
    return jax.jvp(f, (x0,), (t,))[1] * t


def run_test(f, f_grad, x0: jnp.array, n_iter: int, n_trials: int, learning_rate: float) -> pd.DataFrame:
    data = []
    for j in range(n_trials):
        key = jax.random.PRNGKey(j)
        start = datetime.now()
        x = x0
        for i in range(n_iter):
            key, key_grad = jax.random.split(key)
            data.append(
                [j, i, (datetime.now() - start).total_seconds(), float(f(x))])
            x = x - learning_rate * f_grad(x, key_grad)
    df = pd.DataFrame(data, columns=['trial', 'iter', 'time', 'f'])
    return df


if __name__ == '__main__':

    x0 = jnp.array([0., 0.5])
    n_iter = 1000
    learning_rate = 0.01
    df_fwd = run_test(beale, lambda x, key: fwd_grad(
        beale, x0=x, key=key), x0, n_iter, 10, learning_rate)
    df_bwd = run_test(beale, lambda x, key: jax.grad(beale)
                      (x), x0, n_iter, 1, learning_rate)
    df_fwd['kind'] = 'fwd'
    df_bwd['kind'] = 'bwd'
    df = pd.concat([df_fwd, df_bwd])
    df.to_csv('./logs/fmin_beale.csv')

    x0 = jnp.array([-1., 0.])
    n_iter = 25000
    learning_rate = 5 * 10 ** -4
    df_fwd = run_test(rosenbrock, lambda x, key: fwd_grad(
        rosenbrock, x0=x, key=key), x0, n_iter, 10, learning_rate)
    df_bwd = run_test(rosenbrock, lambda x, key: jax.grad(
        rosenbrock)(x), x0, n_iter, 1, learning_rate)
    df_fwd['kind'] = 'fwd'
    df_bwd['kind'] = 'bwd'
    df = pd.concat([df_fwd, df_bwd])
    df.to_csv('./logs/fmin_rosenbrock.csv')
