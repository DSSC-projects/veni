from operator import getitem
import sys
sys.path.append('../')

import jax 
import jax.numpy as jnp
from jax import grad, jvp
import pandas as pd
from veni.net import Module, Sequential, Linear
from veni.function import ReLU, Softmax
from veni.utils import one_hot, NumpyLoader, _get_vector
from veni.functiontools import CrossEntropy
from veni.optim import SGD
from torchvision.datasets import MNIST
import numpy as np
from tqdm import tqdm 
from datetime import datetime

dspath = '../../datasets'
batch_size = 64
num_epochs = 5 
n_targets = 10
step_size = 2e-4
save_path = './logs'
s0 = 2e-4
num_runs = 10

def rate_decay(i,eta_0, k = 1e-4):
    return eta_0 * np.exp(-i * k)

class tf(object):
    def __call__(self, pic):
        return np.array(np.ravel(pic), dtype=jnp.float32)/ 255.

class MLP(Module):
    def __init__(self, key):
        self.layers = Sequential([
            Linear(28*28, 1024, key),
            ReLU(),
            Linear(1024, 10, key),
            Softmax()
        ])

        self.params = self.layers.generate_parameters()
    
    def forward(self,x,params):
        return self.layers(x,params)


def loss(params, x, y):
    y_hat = model(x,params)                    # TODO : Improve this
    return CrossEntropy(y,y_hat)

def accuracy(y,y_hat):
    model_predictions = jnp.argmax(y_hat, axis= 1)
    return jnp.mean(y == model_predictions)

def update_bwd(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

def get_vector(params):
    v_shaped = []
    for w, b in params:
        v_w = jnp.array(np.random.normal(0,1, w.shape))
        v_b = jnp.array(np.random.normal(0,1, b.shape))
        v_shaped.append((v_w, v_b))
    return v_shaped


def fwd_grad(params, x, y):
    v = _get_vector(key, params)
    _, proj = jvp(lambda p: loss(p, x, y), (params, ), (v,))

    return [(proj * dw, proj * db) for dw, db in v]


if __name__ == '__main__':

    print(f"Loading into/from path {dspath}")
    train_dataset = MNIST(dspath, train = True, download= True, transform= tf())
    train_generator = NumpyLoader(train_dataset, batch_size= batch_size )

    test_dataset = MNIST(dspath, train = False,download= True, transform= tf() )
    test_generator = NumpyLoader(test_dataset, batch_size= batch_size )

    results = []

    print(f"Backward training")

    for k in tqdm(range(num_runs), desc="Runs"):
        key = jax.random.PRNGKey(k)
        model = MLP(key)
        params = model.params
        optimizer = SGD(params, eta=step_size)
        iter = 0
        start = datetime.now()
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            for i, (image, label) in enumerate(train_generator):
                one_hot_label = one_hot(label, n_targets)
                g = grad(loss)(params, image, one_hot_label)
                params = optimizer.update(params, g)
                loss_item =  loss(params, image, one_hot_label)
                results.append(['bwd', k, iter, (datetime.now() - start).total_seconds(), loss_item])
                iter += 1

    print(f"Forward training")

    for k in tqdm(range(num_runs), desc="Runs"):
        key = jax.random.PRNGKey(k)
        model = MLP(key)
        params = model.params
        optimizer = SGD(params, eta=step_size)
        iter = 0
        start = datetime.now()
        for epoch in range(num_epochs):
            for i, (image, label) in enumerate(train_generator):
                key = jax.random.split(key)
                one_hot_label = one_hot(label, n_targets)
                g = fwd_grad(params, image, one_hot_label)
                params = optimizer.update(params, g)
                loss_item =  loss(params, image, one_hot_label)
                results.append(['fwd', k, iter, (datetime.now() - start).total_seconds(), loss_item])
                iter += 1

    df = pd.DataFrame(results, columns=['method', 'run', 'iter', 'time', 'loss'])
    df.to_csv(f'{save_path}/mlp_MNIST.csv', index=False)

    print(f"Results saved to {save_path}/mlp_MNIST.csv")
