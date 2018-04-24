from mxnet.gluon import nn
from mxnet import nd

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation='relu'))
        net.add(nn.Dense(2))
    return net

x = nd.random.uniform(shape=(3, 5))

import sys
try:
    net = get_net()
    net.initialize()
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))

w =net[0].weight
b = net[0].bias
print('name:', net[0].name, '\nweight: ', w, '\nbias: ', b)

print('weight:', w.data())
print('weight gradient', w.grad())
print('bias:', b.data())
print('bias gradient', b.grad())

params = net.collect_params()
print(params)
print(params['sequential0_dense0_bias'].data())
print(params.get('dense0_weight').data())

from mxnet import init
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())

params.initialize(init=init.One(), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())

net=get_net()
net.collect_params()

net.initialize()
net.collect_params()

net(x)
net.collect_params()

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation='relu'))
    net.add(nn.Dense(4, activation='relu'))
    net.add(nn.Dense(4, activation='relu', params=net[-1].params))
    net.add(nn.Dense(2))

net.initialize()
net(x)
print(net[1].weight.data())
print(net[2].weight.data())

class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit, self).__init__()
        self._verbose = True
    def _init_weight(self, _, arr):
        print('init weight', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)

net = get_net()
net.initialize(MyInit())
net(x)
net[0].weight.data()

net = get_net()
net.initialize()
net(x)

print('default weight:', net[1].weight.data())
w = net[1].weight
w.set_data(nd.ones(w.shape))
print('init to all ls:', net[1].weight.data())
