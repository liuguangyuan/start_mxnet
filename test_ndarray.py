from mxnet import ndarray as nd
import numpy as np

if __name__ == '__main__':
	x = nd.arange(0,9).reshape((3, 3))
	x[1:2,1:3]=9.0
	print(x)
