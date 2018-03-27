from mxnet import nd

w = nd.arange(4).reshape((1,1,2,2))
b = nd.array([1])
data = nd.arange(9).reshape((1,1,3,3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])


out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1], stride=(2,2), pad=(1,1))

w = nd.arange(8).reshape((1,2,2,2))
b= nd.array([1])
data = nd.arange(18).reshape((1,2,3,3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])

w = nd.arange(16).reshape((2,2,2,2))
data = nd.arange(18).reshape((1,2,3,3))
b = nd.array([1,2])
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])

data = nd.arange(18).reshape((1,2,3,3))
max_pool = nd.Pooling(data=data, pool_type='max', kernel=(2,2))
avg_pool = nd.Pooling(data=data, pool_type='avg', kernel=(2,2))

import sys
sys.path.append('..')
from utils import load_data_fashion_mnist
batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)

import mxnet as mx
try:
	ctx = mx.gpu()
	_ = nd.zeros((1,), ctx=ctx)
except:
	ctx = mx.cpu()
ctx

weight_scale = .01
W1 = nd.random_normal(shape=(20, 1, 5, 5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(W1.shape[0], ctx=ctx)

W2 = nd.random_normal(shape=(50,20,3,3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(W2.shape[0], ctx=ctx)

W3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(W3.shape[1], ctx=ctx)

W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(W4.shape[1], ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]
for param in params:
	param.attach_grad()

def net(X, verbose=False):
	X = X.as_in_context(W1.context)
	h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=W1.shape[0])
	h1_activation = nd.relu(h1_conv)
	h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2,2), stride=(2,2))
	h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=W2.shape[0])
	h2_activation = nd.relu(h2_conv)
	h2 = nd.Pooling(data=h2_activation, pool_type='max', kernel=(2,2), stride=(2,2))
	h2 = nd.flatten(h2)

	h3_linear = nd.dot(h2, W3) + b3
	h3 = nd.relu(h3_linear)

	h4_linear = nd.dot(h3,W4) + b4
	
	if verbose:
		print('1st conv block:', h1.shape)
		print('2nd conv block:', h2.shape)
		print('1st dense:', h3.shape)
		print('2nd dense:', h4_linear.shape)
		print('output:', h4_linear)
	return h4_linear

for data, _ in train_data:
	net(data, verbose=True)
	break

from mxnet import autograd as autograd
from utils import SGD, accuracy, evaluate_accuracy
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = .2

for epoch in range(5):
	train_loss = 0.
	train_acc = 0.
	for data, label in train_data:
		label = label.as_in_context(ctx)
		with autograd.record():
			output = net(data)
			loss = softmax_cross_entropy(output, label)
		loss.backward()
		SGD(params, learning_rate/batch_size)

		train_loss += nd.mean(loss).asscalar()
		train_acc += accuracy(output, label)
	test_acc = evaluate_accuracy(test_data, net, ctx)
	print("Epoch %d. Loss:%f, Train acc %f, Test acc %f" %(
				epoch, train_loss/len(train_data),
				train_acc/len(train_data), test_acc))
