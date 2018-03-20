from mxnet import gluon
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utils import SGD
from mxnet import autograd

batch_size = 256
num_inputs = 784
num_outputs = 10


w = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)
params = [w, b]
for param in params:
    param.attach_grad()

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(10, 10))
    for i in range(n):
        figs[i].imshow(images[i].reshape(28, 28).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()
       
def get_text_labels(label):
    text_labels = [ 
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
   ] 
    return [text_labels[int(i)] for i in label]

def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1, keepdims=True)
    ret = exp / partition
    return ret 

def net(X):
    X0 = X.reshape((-1, num_inputs))
    a = nd.dot(X0, w)
    c = a + b
    d = softmax(c)
    return d 

def cross_entropy(yhat, y):
    a = nd.log(yhat)
    b = nd.pick(a, y)
    return -b

def accuracy(output, label):
    a = output.argmax(axis=1)
    b = label
    c = nd.mean(a ==b)
    d = c.asscalar()
    #ret = nd.mean(output.argmax(axis=1)==label).asscalar()
    return d 
def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    ret = acc / len(data_iterator)
    return ret 



if __name__ == '__main__':
    mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
    '''
    data, label = mnist_train[0]
    print(data.shape, label)
    data, label = mnist_train[0:9]
    show_images(data)
    print(get_text_labels(label))
    '''
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

    '''
    X = nd.random_normal(shape=(2, 5))
    X_prob = softmax(X)
    total = X_prob.sum(axis=1)
    #XXX:why, it is believed 1/10
    eva_value = evaluate_accuracy(test_data, w, b,  net)
    '''

    learning_rate = .1

    for epoch in range(5):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = cross_entropy(output, label)
            loss.backward()
            #XXX: 
            SGD(params, learning_rate/batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)
        test_acc = evaluate_accuracy(test_data, net)

    data, label = mnist_test[0:9]
    show_images(data)
    print('true labels')
    print(get_text_labels(label))

    predicted_labels = net(data).argmax(axis=1)
    print('predicted labels')
    print(get_text_labels(predicted_labels.asnumpy()))
