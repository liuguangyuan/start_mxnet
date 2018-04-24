from mxnet import nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x-x.mean()

layer = CenteredLayer()
layer(nd.array([1,2,3,4,5]))

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dense(10))
    net.add(CenteredLayer())

net.initialize()
y = net(nd.random.uniform(shape=(4,8)))
a = y.mean()

from mxnet import gluon
my_param = gluon.Parameter("exciting_parameter_yay", shape=(3,3))
my_param.initialize()
(my_param.data(), my_param.grad())

pd = gluon.ParameterDict(prefix='blocks_')
pd.get('exciting_parameter_yay', shape=(3, 3))
pd

class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)

dense = MyDense(5, in_units=10, prefix='o_my_dense_')
dense.params

dense.initialize()
dense(nd.random.uniform(shape=(2, 10)))

net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(32, in_units=64))
    net.add(MyDense(2, in_units=32))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))
