import numpy as np
from sklearn import datasets,linear_model, metrics

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

X = diabetes_X_train

y = diabetes_y_train

W = np.random.uniform(low = 0.1 , high = 0.1, size = diabetes_X_train.shape[1])

b = 0.0

learning_rate = 0.1
epochs = 100000

for i in range(epochs):

	y_predict = X.dot(W) + b

	error = y - y_predict

	mean_squared_error = np.mean(np.power(error,2))

	W_gradient = -(1.0/len(X)) * error.dot(X)
	b_gradient = -(1.0/len(X)) * np.sum(error)

	W = W -(learning_rate * W_gradient)
	b = b - (learning_rate * b_gradient)

	if i % 5000 == 0: 
		print("Epoch %d: %f"%(i,mean_squared_error))

X = diabetes_X_test
y = diabetes_y_test

y_predict = X.dot(W)+b
error = y - y_predict
mean_squared_error = np.mean(np.power(error, 2))
print("Mean squared error: %.2f" % mean_squared_error)
print("="*120)

