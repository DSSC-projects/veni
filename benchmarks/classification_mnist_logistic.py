from jax import grad, jvp
import jax.numpy as jnp
import jax
from torchvision.datasets import MNIST
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.append('../')
from veni.net import Module, Sequential, Linear
from veni.function import Sigmoid
from veni.utils import one_hot, NumpyLoader, FlattenAndCast, _get_vector
from veni.functiontools import CrossEntropy
from veni.optim import SGD


dspath = '../../datasets'
batch_size = 64
num_runs = 10
num_epochs = 10
n_targets = 10
step_size = 1e-4
logging_freq = 200
save_path = './logs/'


class tf(object):
    def __call__(self, pic):
        return (np.ravel(np.array(pic, dtype=jnp.float32)) / 255. - 0.5) * 2


class LogisticRegressor(Module):
    def __init__(self):
        self.layers = Sequential([
            Linear(28*28, 10, key),
            Sigmoid()
        ])

        self.params = self.layers.generate_parameters()
        # eliminate the bias

    def forward(self, x, params):
        return self.layers(x, params)


def loss(params, x, y):
    y_hat = model(x, params)

    l1 = y*jnp.log(y_hat)
    l2 = (1-y)*jnp.log(1 - y_hat)
    l = -jnp.sum(l1 + l2)
    return l/y.shape[0]


def accuracy(y, y_hat):
    model_predictions = jnp.argmax(y_hat, axis=1)
    return jnp.mean(y == model_predictions)


def get_vector(params):
    v_shaped = []
    for w, b in params:
        v_w = jnp.array(np.random.normal(0, 1, w.shape))
        v_b = jnp.array(np.random.normal(0, 1, b.shape))
        v_shaped.append((v_w, v_b))
    return v_shaped


def fwd_grad(params, x, y):
    v = _get_vector(key, params)
    _, proj = jvp(lambda p: loss(p, x, y), (params, ), (v,))

    return [(proj * dw, proj * db) for dw, db in v]


if __name__ == '__main__':

    print(f"Loading into/from path {dspath}")
    train_dataset = MNIST(dspath, train=True, download=True, transform=tf())
    train_generator = NumpyLoader(train_dataset, batch_size=batch_size)

    test_dataset = MNIST(dspath, train=False, download=True, transform=tf())
    test_generator = NumpyLoader(test_dataset, batch_size=batch_size)

    results = []

    print(f"Backward training")

    for k in tqdm(range(num_runs), desc='Runs'):
        key = jax.random.PRNGKey(k)
        model = LogisticRegressor()
        params = model.params
        optimizer = SGD(params, eta=2e-4)
        iter = 0
        start = datetime.now()
        for epoch in tqdm(range(num_epochs), desc='Epoch'):
            for i, (image, label) in enumerate(train_generator):
                one_hot_label = one_hot(label, n_targets)
                g = grad(loss)(params, image, one_hot_label)
                params = optimizer.update(params, g)
                loss_item = loss(params, image, one_hot_label)
                results.append(['bwd', k, iter, (datetime.now() - start).total_seconds(), loss_item])
                iter += 1

    print(f"Forward training")

    for k in tqdm(range(num_runs), desc='Runs'):
        key = jax.random.PRNGKey(k)
        model = LogisticRegressor()
        params = model.params
        optimizer = SGD(params, eta=2e-4)
        iter = 0
        start = datetime.now()
        for epoch in tqdm(range(num_epochs), desc='Epoch'):
            for i, (image, label) in enumerate(train_generator):
                one_hot_label = one_hot(label, n_targets)
                g = fwd_grad(params, image, one_hot_label)
                params = optimizer.update(params, g)
                loss_item = loss(params, image, one_hot_label)
                results.append(['fwd', k, iter, (datetime.now() - start).total_seconds(), loss_item])
                iter += 1

    df = pd.DataFrame(results, columns=['method', 'run', 'iter', 'time', 'loss'])
    df.to_csv(f'{save_path}/logistic_MNIST.csv', index=False)

    print(f"Results saved to {save_path}/logistic_MNIST.csv")
