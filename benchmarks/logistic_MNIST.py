#! /usr/bin/python3

import logging
import os, sys
sys.path.append('../')
import jax 
import jax.numpy as jnp
from jax import grad, jvp
jax.config.update('jax_platform_name', 'cpu')

from jax_forward.net import Module, Sequential, Linear
from jax_forward.function import Sigmoid
from jax_forward.utils import one_hot, NumpyLoader, FlattenAndCast, _get_vector
from jax_forward.functiontools import CrossEntropy
from torchvision.datasets import MNIST
import numpy as np


dspath = '../../datasets'
batch_size = 64
num_epochs = 10
n_targets = 10
step_size = 1e-4

logging_freq = 200

save_path = 'logs/logistic_MNIST'
bwd_path  = os.path.join(save_path, 'bwd') 
fwd_path  = os.path.join(save_path, 'fwd') 

os.makedirs(bwd_path, exist_ok= True)
os.makedirs(fwd_path, exist_ok= True)

bwd_run = 1
fwd_run = 1

l = os.listdir(bwd_path)
l.sort()

if len(l) != 0:
    bwd_run = int(l[-1].split('.')[-1]) + 1

l = os.listdir(fwd_path)
l.sort()
print(len(l))
if len(l) != 0:
    print(l[-1].split('.')[-1])
    fwd_run = int(l[-1].split('.')[-1]) + 1

bwd_file = open(os.path.join(bwd_path, f"run.{bwd_run}"),'w')
print(f"logging into {bwd_file.name}")
fwd_file = open(os.path.join(fwd_path, f"run.{fwd_run}"),'w')
print(f"logging into {fwd_file.name}")

bwd_file.write(f"#epoch,batch_number,running_loss\n")
fwd_file.write(f"#epoch,batch_number,running_loss\n")



key = jax.random.PRNGKey(10)

#flatten and normalize
class tf(object):
    def __call__(self, pic):
        return ( np.ravel(np.array(pic, dtype=jnp.float32)) / 255. - 0.5 ) * 2

print(f"Loading into/from path {dspath}")
train_dataset = MNIST(dspath, train = True, download= True, transform= tf())
train_generator = NumpyLoader(train_dataset, batch_size= batch_size )


test_dataset = MNIST(dspath, train = False,download= True, transform= tf() )
test_generator = NumpyLoader(test_dataset, batch_size= batch_size )
print("Loaded")


class LogisticRegressor(Module):
    def __init__(self):
        self.layers = Sequential([
            Linear(28*28, 10, key),
            Sigmoid()
        ])

        self.params = self.layers.generate_parameters()
        #eliminate the bias
    
    def forward(self,x,params):
        return self.layers(x,params)

model = LogisticRegressor()
params = model.params

def loss(params, x, y):
    y_hat = model(x,params)

    l1 = y*jnp.log(y_hat)
    l2 = (1-y)*jnp.log(1 - y_hat)
    l = -jnp.sum(l1 + l2)
    return l/y.shape[0]

def accuracy(y,y_hat):
    model_predictions = jnp.argmax(y_hat, axis= 1)
    return jnp.mean(y == model_predictions)

def update_bwd(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

print(f"Backward training")
def evaluatePerf(gen):
    acc = 0
    count = 0
    for x,y in gen:
        y_hat = model(x,params)
        acc += accuracy(y,y_hat)*x.shape[0]
        count += x.shape[0]
    return acc/count

for epoch in range(num_epochs):
    running_loss = 0
    for i, (image, label) in enumerate(train_generator):
        one_hot_label = one_hot(label, n_targets)
        params = update_bwd(params, image, one_hot_label)
        # loss info
        loss_item =  loss(params, image, one_hot_label)
        running_loss = running_loss + loss_item
        if i % logging_freq == logging_freq - 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / logging_freq:.3f}')
            bwd_file.write(f'{epoch + 1},{i + 1},{running_loss / logging_freq}\n')
            running_loss = 0.0  


#model assessment

print(f"Training set accuracy {evaluatePerf(train_generator)}")


print(f"Training set accuracy {evaluatePerf(test_generator)}")

def get_vector(params):
    v_shaped = []
    for w, b in params:
        v_w = jnp.array(np.random.normal(0,1, w.shape))
        v_b = jnp.array(np.random.normal(0,1, b.shape))
        v_shaped.append((v_w, v_b))
    return v_shaped

def update_fwd(params, x, y):
    v = _get_vector(key,params)
    _ , proj = jvp(lambda p: loss(p,x,y), (params, ), (v,) )
    return [(w - step_size * proj * dw, b - step_size * proj * db)
          for (w, b), (dw, db) in zip(params, v)]

print(f"Forward training")
model = LogisticRegressor()
params = model.params

for epoch in range(num_epochs):
    running_loss = 0
    for i, (image, label) in enumerate(train_generator):
        key, _ = jax.random.split(key)
        one_hot_label = one_hot(label, n_targets)
        params = update_fwd(params, image, one_hot_label)
        # loss info
        loss_item =  loss(params, image, one_hot_label)
        running_loss = running_loss + loss_item
        if i % logging_freq == logging_freq - 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / logging_freq:.3f}')
            fwd_file.write(f'{epoch + 1},{i + 1},{running_loss / logging_freq}\n')
            running_loss = 0.0  


print(f"Training set accuracy {evaluatePerf(train_generator)}")


print(f"Training set accuracy {evaluatePerf(test_generator)}")

fwd_file.close()
bwd_file.close()