from scipy.io import loadmat
import matplotlib.pyplot as plt
from cost_function import compute_cost
import numpy as np
from gradient import gradient
from scipy.optimize import minimize

filename = 'ex5data1.mat'
data = loadmat(filename)

# Training set
x, y = data['X'], data['y'].flatten()
# Validation set
Xval, yval = data['Xval'], data['yval']
# Test set
Xtest, ytest = data['Xtest'], data['ytest']

# data = plt.scatter(x, y, c='r', s=25, marker='x')
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
# plt.show()

theta = np.zeros(x.shape[1] + 1)
hyper_p = 0


def optimize(theta, x, y, hyper_p=0):
    result = minimize(compute_cost, theta, args=(x, y, hyper_p), method='CG',
                  jac=gradient, options={'maxiter': 500})
    opt_theta = result.x
    return opt_theta


opt_theta = optimize(theta, x, y, hyper_p)
print(opt_theta)
final = compute_cost(opt_theta, x, y, hyper_p)
print(final)


def predict(opt_theta, x):
    x_copy = np.insert(x, 0, 1, axis=1)
    prediction = x_copy @ opt_theta
    return prediction


def plot_prediction(opt_theta, x, y):
    x_pred = np.linspace(-50, 50, 30)[:, None]
    y_pred = predict(opt_theta, x_pred)
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.scatter(x, y, marker='x', c='r')
    ax.plot(x_pred, y_pred, c='b')


plot_prediction(opt_theta, x, y)
plt.show()
