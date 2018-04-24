def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
import random

# 为方面比较同一优化算法的从零开始实现和Gluon实现，将输出保持确定
mx.random.seed(1)
random.seed(1)

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
X = nd.random_normal(scale=1, shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(scale=1, shape=y.shape)

# 初始化模型参数
def init_params():
    w = nd.random_normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params

def linreg(X, w, b):
    return nd.dot(X,w) + b

def squared_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

def data_iter(batch_size, num_examples, random, X, y):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for batch_i, i in enumerate(range(0, num_examples, batch_size)):
        j = nd.array(idx[i:min(i+batch_size, num_examples)])
        yield batch_i, X.take(j), y.take(j)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import utils

net = linreg
squared_loss = squared_loss

def optimize(batch_size, lr, num_epochs, log_interval):
    w, b = init_params()
    y_vals = [nd.mean(squared_loss(net(X, w, b), y)).asnumpy()]
    print('batch size', batch_size)
    for epoch in range(1, num_epochs + 1):
        if epoch > 2:
            lr *= 0.1
        for batch_i, features, label in data_iter(
            batch_size, num_examples, random, X, y):
            with autograd.record():
                output = net(features, w, b)
                loss = squared_loss(output, label)
            loss.backward()
            sgd([w, b], lr, batch_size)
            if batch_i *batch_size % log_interval == 0:
                y_vals.append(
                    nd.mean(squared_loss(net(X, w, b), y)).asnumpy())
        print('epoch %d, learning rate %f, loss %.4e' %(epoch, lr, y_vals[-1]))
    print('w:', np.reshape(w.asnumpy(), (1, -1)),'b:', b.asnumpy()[0], '\n')
    x_vals = np.linspace(0, num_epochs, len(y_vals), endpoint=True)
    utils.set_fig_size(mpl)
    plt.semilogy(x_vals, y_vals)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

#optimize(batch_size=1, lr=0.2, num_epochs=3, log_interval=10)
#optimize(batch_size=1000, lr=0.999, num_epochs=3, log_interval=1000)
#optimize(batch_size=10, lr=0.2, num_epochs=3, log_interval=10)
#optimize(batch_size=10, lr=5, num_epochs=3, log_interval=10)
optimize(batch_size=10, lr=0.002, num_epochs=3, log_interval=10)
