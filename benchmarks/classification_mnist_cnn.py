import jax
import numpy as np
from torchvision.datasets import MNIST
from tqdm import tqdm
import jax.numpy as jnp
import sys
from datetime import datetime
import pandas as pd

sys.path.append('../')
from veni.functiontools import CrossEntropy
from veni.optim import SGD, NormalLikeSampler
from veni.utils import one_hot, NumpyLoader
from veni.function import ReLU, Softmax
from veni.net import Module, Sequential, Linear, Conv2D, MaxPool2D, Flatten
from jax import grad, jvp

dspath = '../../datasets'
batch_size = 64
num_epochs = 3
n_targets = 10
step_size = 2e-4
s0 = 2e-4
logging_freq = 200
save_path = 'logs/cnn_MNIST'
num_runs = 10


class tf(object):
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32).reshape([1, 28, 28]) / 255.


class CNN(Module):
    def __init__(self, key):
        self.layers = Sequential([
            Conv2D(1, 64, 3, 1, 'VALID', key),
            ReLU(),
            Conv2D(64, 64, 3, 1, 'VALID', key),
            ReLU(),
            MaxPool2D(2),
            Conv2D(64, 64, 3, 1, 'VALID', key),
            ReLU(),
            Conv2D(64, 64, 3, 1, 'VALID', key),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Linear(1024, 1024, key),
            ReLU(),
            Linear(1024, 10, key),
            Softmax()
        ])

        self.params = self.layers.generate_parameters()

    def forward(self, x, params):
        return self.layers(x, params)


def rate_decay(i, eta_0, k=1e-4):
    return eta_0 * np.exp(-i * k)


def loss(params, x, y):
    y_hat = model(x, params)
    return CrossEntropy(y, y_hat)


def accuracy(y, y_hat):
    model_predictions = jnp.argmax(y_hat, axis=1)
    return jnp.mean(y == model_predictions)


def evaluatePerf(gen):
    acc = 0
    count = 0
    for x, y in gen:
        y_hat = model(x, params)
        acc += accuracy(y, y_hat)*x.shape[0]
        count += x.shape[0]
    return acc/count


def fwd_grad(params, x, y, sampler):
    v = sampler(params)
    _, proj = jvp(lambda p: loss(p, x, y), (params, ), (v,))
    return [(w - proj * dw, b -proj * db)
            for (w, b), (dw, db) in zip(params, v)]


if __name__ == '__main__':

    print(f"Loading into/from path {dspath}")
    train_dataset = MNIST(dspath, train=True, download=True, transform=tf())
    train_generator = NumpyLoader(train_dataset, batch_size=batch_size)

    test_dataset = MNIST(dspath, train=False, download=True, transform=tf())
    test_generator = NumpyLoader(test_dataset, batch_size=batch_size)

    results = []

    print(f"Backward training")

    for k in tqdm(range(num_runs), desc="Runs"):
        key = jax.random.PRNGKey(k)
        model = CNN(key)
        params = model.params
        optimizer = SGD(params, eta=step_size)
        iter = 0
        start = datetime.now()
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            for i, (image, label) in enumerate(train_generator):
                step_size = rate_decay(i, 0.5e-5)
                one_hot_label = one_hot(label, n_targets)
                g = grad(loss)(params, image, one_hot_label)
                params = optimizer.update(params, g)            # TODO : Update learning rate schedule
                loss_item = loss(params, image, one_hot_label)
                results.append(
                    ['bwd', k, iter, (datetime.now() - start).total_seconds(), loss_item])
                iter += 1

    print(f"Forward training")

    for k in tqdm(range(num_runs), desc="Runs"):
        key = jax.random.PRNGKey(k)
        model = CNN(key)
        params = model.params
        optimizer = SGD(params, eta=step_size)
        iter = 0
        sampler = NormalLikeSampler()
        start = datetime.now()
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            for i, (image, label) in enumerate(train_generator):
                step_size = rate_decay(i, 0.5e-5)
                one_hot_label = one_hot(label, n_targets)
                g = fwd_grad(params, image, one_hot_label, sampler)
                params = optimizer.update(params, g)            # TODO : Update learning rate schedule
                loss_item = loss(params, image, one_hot_label)
                results.append(
                    ['fwd', k, iter, (datetime.now() - start).total_seconds(), loss_item])
                iter += 1

    df = pd.DataFrame(results, columns=[
                      'method', 'run', 'iter', 'time', 'train_loss'])
    df.to_csv(f'{save_path}/mlp_MNIST.csv', index=False)

    print(f"Results saved to {save_path}/mlp_MNIST.csv")
