#! /usr/bin/python3

from operator import getitem
import sys

sys.path.append('../')

import jax 
import jax.numpy as jnp
from jax import grad, jvp
jax.config.update('jax_platform_name', 'cpu')

from vina.net import Module, Sequential, Linear, Conv2D, MaxPool2D, Flatten
from vina.function import Sigmoid, ReLU, Softmax, LogSoftmax
from vina.utils import one_hot, NumpyLoader, FlattenAndCast, _get_vector
from vina.functiontools import CrossEntropy, MSE, CrossEntropyV2
from torchvision.datasets import MNIST
import numpy as np

def rate_decay(i,eta_0, k = 1e-4):
    return eta_0 * np.exp(-i * k)

dspath = '../../datasets'
batch_size = 64
num_epochs = 3 
n_targets = 10
step_size = 2e-4

s0 = 2e-4

key = jax.random.PRNGKey(111111)

logging_freq = 200

save_path = 'logs/cnn_MNIST'
bwd_path  = os.path.join(save_path, 'bwd') 
fwd_path  = os.path.join(save_path, 'fwd') 

os.makedirs(bwd_path, exist_ok= True)
os.makedirs(fwd_path, exist_ok= True)

bwd_run = 1
fwd_run = 1

l = os.listdir(bwd_path)
if len(l) != 0:
    bwd_run = int(l[-1].split('.')[-1]) + 1

l = os.listdir(fwd_path)
l.sort(key = lambda x: int(str.split(x,'.')[-1]))
if len(l) != 0:
    fwd_run = int(l[-1].split('.')[-1]) + 1
    print(l[-1].split('.')[-1])

bwd_file = open(os.path.join(bwd_path, f"run.{bwd_run}"),'w')
print(f"logging into {bwd_file.name}")
fwd_file = open(os.path.join(fwd_path, f"run.{fwd_run}"),'w')
print(f"logging into {fwd_file.name}")

bwd_file.write(f"#epoch,batch_number,running_loss\n")
fwd_file.write(f"#epoch,batch_number,running_loss\n")

#normalize
class tf(object):
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32).reshape([1, 28, 28]) / 255.

print(f"Loading into/from path {dspath}")
train_dataset = MNIST(dspath, train = True, download= True, transform= tf())
train_generator = NumpyLoader(train_dataset, batch_size= batch_size )


test_dataset = MNIST(dspath, train = False,download= True, transform= tf() )
test_generator = NumpyLoader(test_dataset, batch_size= batch_size )
print("Loaded")


class cnn(Module):
    def __init__(self):
        self.layers = Sequential([
            Conv2D(1,64,3,1,'VALID',key),
            ReLU(),
            Conv2D(64,64,3,1,'VALID', key),
            ReLU(),
            MaxPool2D(2),
            Conv2D(64,64,3,1,'VALID', key),
            ReLU(),
            Conv2D(64,64,3,1,'VALID', key),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Linear(1024, 1024, key),
            ReLU(),
            Linear(1024, 10, key),
            Softmax()
        ])

        self.params = self.layers.generate_parameters()
        #eliminate the bias
    
    def forward(self,x,params):
        return self.layers(x,params)

model = cnn()
params = model.params

def loss(params, x, y):
    y_hat = model(x,params)
    return CrossEntropy(y,y_hat)

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

        step_size = rate_decay(i,0.5e-5)
        one_hot_label = one_hot(label, n_targets)
        params = update_bwd(params, image, one_hot_label)
        # loss info
        loss_item =  loss(params, image, one_hot_label)
        running_loss = running_loss + loss_item
        if i % logging_freq == logging_freq - 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / logging_freq:.3f}')
            bwd_file.write(f'{epoch + 1},{i + 1},{running_loss / logging_freq}\n')
            running_loss = 0.0  


#print(f"Training set accuracy {evaluatePerf(train_generator)}")


#print(f"Training set accuracy {evaluatePerf(test_generator)}")

def get_vector(params):
    v_shaped = []
    for w, b in params:
        v_w = jnp.array(np.random.normal(0,1, w.shape))
        v_b = jnp.array(np.random.normal(0,1, b.shape))

        #v_w = v_w / jnp.linalg.norm(v_w)
        #v_b = v_b / jnp.linalg.norm(v_b)
        v_shaped.append((v_w, v_b))
    return v_shaped

def update_fwd(params, x, y):
    v = _get_vector(key,params)
    _ , proj = jvp(lambda p: loss(p,x,y), (params, ), (v,) )
    p = proj
    return [(w - step_size * proj * dw , b - step_size * proj * db)
          for (w, b), (dw, db) in zip(params, v)]

def grad_by_forward(key, params, loss, data, N=2):
    x, y = data
    for i in range(N):
        key, _ = jax.random.split(key)
        v = _get_vector(key, params)
        _, proj = jax.jvp(lambda p: loss(p, x, y), (params, ), (v,))
        grad_now = [(proj * dw, proj * db) for (dw, db) in v]

        if i == 0:
            grad = grad_now

        elif i > 0:
            grad = [(dw + dw_prev, db + db_prev)
                           for (dw, db), (dw_prev, db_prev) in zip(grad_now, grad)]


    if N != 1:
        grad = [(dw / N, db / N) for (dw, db) in grad]

    return grad
print(f"Forward training")

model = cnn()
params = model.params

for epoch in range(num_epochs):
    running_loss = 0
    for i, (image, label) in enumerate(train_generator):
        step_size = rate_decay(i,0.5e-5)
        key = jax.random.split(key)
        one_hot_label = one_hot(label, n_targets)

        g = grad_by_forward(key,params,loss,(image,one_hot_label), 1)
        params = [(w - step_size *  dw , b - step_size *  db)
          for (w, b), (dw, db) in zip(params, g)]
        #params = update_fwd(params, image, one_hot_label)
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
