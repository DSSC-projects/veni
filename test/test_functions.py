import sys, os
import jax
jax.config.update('jax_platform_name','cpu')
sys.path.append('../')
sys.path.append('.')
from veni.function import *

import jax.numpy as jnp
import numpy as np


x = jnp.array([[float(i*4 + j) for j in range(4)] for i in range(4)])

def test_relu():
    f = ReLU()
    y = f(x)

    y_exp = jnp.array(
        [   [ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])

    a = jnp.isclose(y,y_exp)
    assert a.all(), f"Testing arrays do not match got {y}. \n \n Reference is {y_exp}"

def test_leakyrelu():
    f = LeakyReLu()
    y = f(x)

    y_exp = jnp.array(
        [   [ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])

    a = jnp.isclose(y,y_exp)
    assert a.all(), f"Testing arrays do not match got {y}. \n \n Reference is {y_exp}"
    
def test_tanh():
    f = Tanh()
    y = f(x)

    y_exp = jnp.array([[0.        , 0.7615942 , 0.9640275 , 0.9950547 ],
             [0.9993292 , 0.99990916, 0.9999878 , 0.99999833],
             [0.99999976, 0.99999994, 0.99999994, 0.99999994],
             [0.99999994, 0.99999994, 0.99999994, 0.99999994]])  

    a = jnp.isclose(y,y_exp)
    assert a.all(), f"Testing arrays do not match got {y}. \n \n Reference is {y_exp}"

def test_sigmoid():
    f = Sigmoid()
    y = f(x)

    y_exp = jnp.array([[0.5       , 0.73105854, 0.8807971 , 0.95257413],
             [0.98201376, 0.9933071 , 0.9975274 , 0.999089  ],
             [0.99966466, 0.9998766 , 0.9999546 , 0.9999833 ],
             [0.9999938 , 0.99999774, 0.99999917, 0.99999964]])  

    a = jnp.isclose(y,y_exp)
    assert a.all(), f"Testing arrays do not match got {y}. \n \n Reference is {y_exp}"

def test_logsigmoid():
    f = LogSigmoid()
    y = f(x)

    y_exp = jnp.array([[-6.9314718e-01, -3.1326175e-01, -1.2692800e-01,
              -4.8587345e-02],
             [-1.8149957e-02, -6.7153843e-03, -2.4756414e-03,
              -9.1141259e-04],
             [-3.3539196e-04, -1.2338923e-04, -4.5419773e-05,
              -1.6689441e-05],
             [-6.1989022e-06, -2.2649790e-06, -8.3446537e-07,
              -3.5762793e-07]])  

    a = jnp.isclose(y,y_exp)
    assert a.all(), f"Testing arrays do not match got {y}. \n \n Reference is {y_exp}"

#def test_Softplus():
#    f = Softplus()
#    y = f(x)
#
#    y_exp = jnp.array([[-6.9314718e-01, -3.1326175e-01, -1.2692800e-01,
#              -4.8587345e-02],
#             [-1.8149957e-02, -6.7153843e-03, -2.4756414e-03,
#              -9.1141259e-04],
#             [-3.3539196e-04, -1.2338923e-04, -4.5419773e-05,
#              -1.6689441e-05],
#             [-6.1989022e-06, -2.2649790e-06, -8.3446537e-07,
#              -3.5762793e-07]])  
#
#    a = jnp.isclose(y,y_exp)
#    assert a.all(), f"Testing arrays do not match got {y}. \n \n Reference is {y_exp}"

def test_Softmax():
    f = Softmax()
    y = f(x)

    y_exp = jnp.array([[0.0320586 , 0.08714432, 0.23688282, 0.6439143 ],
             [0.0320586 , 0.08714432, 0.23688282, 0.6439143 ],
             [0.0320586 , 0.08714432, 0.23688282, 0.6439143 ],
             [0.0320586 , 0.08714432, 0.23688282, 0.6439142 ]])  

    a = jnp.isclose(y,y_exp)
    assert a.all(), f"Testing arrays do not match got {y}. \n \n Reference is {y_exp}"
    
def test_logsoftmax():
    f = LogSoftmax()
    y = f(x)

    y_exp = jnp.array([[-3.4401896 , -2.4401896 , -1.4401896 , -0.4401896 ],
             [-3.4401898 , -2.4401898 , -1.4401898 , -0.44018984],
             [-3.4401894 , -2.4401894 , -1.4401894 , -0.44018936],
             [-3.4401894 , -2.4401894 , -1.4401894 , -0.44018936]])  

    a = jnp.isclose(y,y_exp)
    assert a.all(), f"Testing arrays do not match got {y}. \n \n Reference is {y_exp}"
