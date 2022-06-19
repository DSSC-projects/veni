import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import jax
from jax import jit, grad
import csv
import jax.numpy as jnp
from torch import triplet_margin_loss
from veni import MLP, ReLU, Softmax, Sequential, Linear
from veni.module import Module
from veni.utils import NumpyLoader, one_hot
from veni.optim import SGD, Adam, grad_fwd
from veni.functiontools import CrossEntropy
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
from time import time


jax.config.update('jax_platform_name', 'cpu')

PATH = "./direction_benchmarks/"
# define hyperparameters
num_epochs = 15
n_targets = 10
key = jax.random.PRNGKey(210)
batch_size = 64
eta = 0.0005


class tf(object):
    def __call__(self, pic):
        return (np.ravel(np.array(pic, dtype=jnp.float32)) / 255. - 0.5) * 2


training_dataset = MNIST('/tmp/mnist/', train=True,
                         download=True, transform=tf())
training_generator = NumpyLoader(training_dataset, batch_size=batch_size)


testing_dataset = MNIST('/tmp/mnist/', train=False,
                        download=True, transform=tf())
testing_generator = NumpyLoader(testing_dataset, batch_size=batch_size)

# define network


class MLP(Module):
    def __init__(self):
        self.layers = Sequential([
            Linear(28*28, 1024, key),
            ReLU(),
            Linear(1024, 1024, key),
            ReLU(),
            Linear(1024, 10, key),
            Softmax()
        ])

        self.params = self.layers.generate_parameters()
        # eliminate the bias

    def forward(self, x, params):
        return self.layers(x, params)


model = MLP()
params = model.params

# loss and accuracy


def loss(params, x, y):
    y_hat = model(x, params)
    return CrossEntropy(y, y_hat)


def accuracy(y, y_hat):
    model_predictions = jnp.argmax(y_hat, axis=1)
    return jnp.mean(y == model_predictions)


def grad_bwd(params, x, y, loss, key):
    grads = grad(loss)(params, x, y)
    return grads


def update(params, x, y, loss, optimizer, key, dirs=1,
           grad_type='fwd', sampler=None):
    key = jax.random.split(key)
    if grad_type == 'fwd':
        if sampler is None:
            grads = grad_fwd(params, x, y, loss, dirs)
        else:
            grads = grad_fwd(params, x, y, loss, dirs, sampler)
    elif grad_type == 'bwd':
        grads = grad_bwd(params, x, y, loss, key)
    else:
        raise ValueError(
            f"Invalid grad_type, expected 'fwd' or 'bwd' got '{grad_type}'")

    return optimizer(params, grads)


def evaluatePerf(gen, model):
    acc = 0
    count = 0
    for x, y in gen:
        y_hat = model(x, params)
        acc += accuracy(y, y_hat)*x.shape[0]
        count += x.shape[0]
    return acc / count


# =========== HERE TEST STARTS ========== #
model = MLP()
params = model.params

# ADAM test
print("ADAM test for backward")
optimizer = Adam(params, eta=eta)

final_accuracy = []
train_loss = []
train_accuracy = []
iteration = []
final_time = []
count = 0
start = time()
for epoch in range(num_epochs):
    running_loss = 0
    running_accuracy = 0
    bs = 0
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        # update parameters
        params = update(params, x, one_hot_label, loss,
                        optimizer, key, None, 'bwd')
        bs += x.shape[0]
        running_loss += loss(params, x, one_hot_label)*x.shape[0]
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator, model)))
    train_loss.append(running_loss/bs)
    train_accuracy.append(float(evaluatePerf(training_generator, model)))
    final_time.append(time()-start)

with open(PATH+'adam_bwd.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(
        zip(iteration, train_loss, train_accuracy, final_accuracy, final_time))

plt.plot(jnp.array(iteration), jnp.array(train_loss))
plt.savefig(PATH+'adam_bwd.png')
plt.close()

for d in [1, 5, 10, 50, 100]:
    print(f"ADAM test for forward {d} dir")

    optimizer = Adam(params, eta=eta)

    model = MLP()
    params = model.params
    final_accuracy = []
    train_loss = []
    train_accuracy = []
    iteration = []
    final_time = []
    count = 0
    start = time()
    for epoch in range(num_epochs):
        running_loss = 0
        running_accuracy = 0
        bs = 0
        for x, y in training_generator:
            key = jax.random.split(key)
            one_hot_label = one_hot(y, n_targets)
            # update parameters
            params = update(params, x, one_hot_label, loss,
                            optimizer, key, d, 'fwd')
            bs += x.shape[0]
            running_loss += loss(params, x, one_hot_label)*x.shape[0]
            count += 1

        iteration.append(count + epoch)
        final_accuracy.append(float(evaluatePerf(testing_generator, model)))
        train_loss.append(running_loss/bs)
        train_accuracy.append(float(evaluatePerf(training_generator, model)))
        final_time.append(time()-start)

    with open(PATH+f"adam_fwd_{d}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(
            zip(iteration, train_loss, train_accuracy, final_accuracy, final_time))

    plt.plot(jnp.array(iteration), jnp.array(train_loss))
    plt.savefig(PATH+f"adam_fwd_{d}.png")
    plt.close()
