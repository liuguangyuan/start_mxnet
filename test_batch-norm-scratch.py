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






