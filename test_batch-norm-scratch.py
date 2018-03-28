from mxnet import nd

def pure_batch_norm(X, gamma, beta, eps=1e-5):
    assert len(X.shape) in (2, 4)
    if len(X.shape) == 2:
        mean = X.mean(axis=0)
        a = X - mean
        variance = ((X - mean)**2).mean(axis=0)
    else:
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)

    X_hat = (X - mean) / nd.sqrt(variance + eps)
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)
    
A = nd.arange(6).reshape((3,2))
pure_batch_norm(A, gamma=nd.array([1, 1]), beta=nd.array([0, 0]))
    
B = nd.arange(18).reshape((1,2,3,3))
pure_batch_norm(B,gamma=nd.array([1,1]), beta=nd.array([0,0]))

def batch_norm(X, gamma, beta, is_training, moving_mean, moving_variance, eps = 1e-5, moving_momentum = 0.9):
    assert len(X.shape) in (2, 4)
    if len(X.shape) == 2:
            mean = X.mean(axis=0)
            variance = ((X - mean)**2).mean(axis=0)
    else:
        mean = X.mean(axis=(0,2,3), keepdims=True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)
        
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(mean.shape)

    if is_training:
        X_hat = (X - mean) / nd.sqrt(variance + eps)
        moving_mean[:] = moving_momentum * moving_mean + (1.0 - moving_momentum) * mean
        moving_variance[:] = moving_momentum * moving_variance + (1.0 - moving_momentum) * variance
    else:
        X_hat = (X - moving_mean) / nd.sqrt(moving_variance + eps)

    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)

import sys
sys.path.append('..')
import utils
ctx = utils.try_gpu()
ctx

weight_scale = .01
c1 = 20
W1 = nd.random.normal(shape=(c1,1,5,5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(c1,ctx=ctx)

gamma1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
beta1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
moving_mean1 = nd.zeros(c1, ctx=ctx)
moving_variance1 = nd.zeros(c1, ctx=ctx)

c2 = 50
W2 = nd.random_normal(shape=(c2, c1, 3, 3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(c2, ctx=ctx)

gamma2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
beta2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
moving_mean2 = nd.zeros(c2, ctx=ctx)
moving_variance2 = nd.zeros(c2, ctx=ctx)

o3 = 128
W3 = nd.random.normal(shape=(1250, o3), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(o3, ctx=ctx)

W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(W4.shape[1], ctx=ctx)

params = [W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, W3, b3, W4, b4]

for param in params:
    param.attach_grad()

def net(X, is_training=False, verbose=False):
    X = X.as_in_context(W1.context)
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=c1)
    h1_bn = batch_norm(h1_conv, gamma1, beta1, is_training, moving_mean1, moving_variance1)
    h1_activation = nd.relu(h1_bn)
    h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2,2), stride=(2,2))
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=c2)
    h2_bn = batch_norm(h2_conv, gamma2, beta2, is_training, moving_mean2, moving_variance2)
    h2_activation = nd.relu(h2_bn)
    h2 = nd.Pooling(data=h2_activation, pool_type='max', kernel=(2,2), stride=(2,2))
    h2 = nd.flatten(h2)

    h3_linear = nd.dot(h2, W3) + b3
    h3 = nd.relu(h3_linear)
    h4_linear = nd.dot(h3, W4) + b4

    if verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear

from mxnet import autograd
from mxnet import gluon

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = 0.2

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data, is_training=True)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    print('Epoch %d. Loss:%f, Train acc %f, Test_acc %f'%(
            epoch, train_loss/len(train_data), train_acc/len(train_data),
            test_acc))


























                






