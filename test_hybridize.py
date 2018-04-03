def add(A, B):
    return A + B
def fancy_func(A, B, C, D):
    E = add(A, B)
    F = add(C, D)
    G = add(E, F)
    return G

print(fancy_func(1,2,3,4))

def add_str():
    return '''
def add(A, B):
    return A + B
'''
def fancy_func_str():
    return '''
def fancy_func(A, B, C, D):
    E = add(A, B)
    F = add(C, D)
    G = add(E, F)
    return G
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1,2,3,4))
'''
prog = evoke_str()
print (prog)

y = compile(prog, '', 'exec')
exec(y)

from mxnet.gluon import nn
from mxnet import nd

def get_net():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
                nn.Dense(256, activation='relu'),
                nn.Dense(128, activation='relu'),
                nn.Dense(2)
                )
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
print(net(x))

net.hybridize()
print(net(x))

from time import time

def bench(net, x):
    start = time()
    for i in range(1000):
        y = net(x)
    nd.waitall()
    return time() - start

net = get_net()
print('Befor hybridizeing: %.4f sec'%(bench(net, x)))
net.hybridize()
print('After hybridizeing: %.4f sec'%(bench(net, x)))

from mxnet import sym
x = sym.var('data')
y = net(x)
print(y)

class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(10)
            self.fc2 = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print(F)
        print(x)
        x = F.relu(self.fc1(x))
        print(x)
        return self.fc2(x)

net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
y = net(x)

y = net(x)

net.hybridize()
y = net(x)
y = net(x)
