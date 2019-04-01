from scipy.io import loadmat
import matplotlib.pyplot as plt
from cost_function import compute_cost
import numpy as np

filename = 'ex5data1.mat'
data = loadmat(filename)

# Training set
x, y = data['X'], data['y']
# Validation set
Xval, yval = data['Xval'], data['yval']
# Test set
Xtest, ytest = data['Xtest'], data['ytest']

data = plt.scatter(x, y, c='r', s=25, marker='x')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
# plt.show()

theta = np.array([[1, 1]])

hyper_p = 1
cost = compute_cost(x, y, theta, hyper_p)
print(cost)
