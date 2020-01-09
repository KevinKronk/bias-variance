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
xval, yval = data['Xval'], data['yval'].flatten()
# Test set
xtest, ytest = data['Xtest'], data['ytest'].flatten()

# Plot all of the data
plt.scatter(x, y, c='r', s=25, marker='x', label="Training Data")
plt.scatter(xval, yval, c='g', s=25, label="Validation Data")
plt.scatter(xtest, ytest, c='b', s=25, label="Test Data")
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.legend()
plt.legend()
plt.show()

theta = np.zeros(x.shape[1] + 1)
hyper_p = 0


# Create an optimizer to run gradient descent on the cost function


def optimize(theta, x, y, hyper_p=0):
    result = minimize(compute_cost, theta, args=(x, y, hyper_p), method='CG',
                  jac=gradient, options={'maxiter': 500})
    opt_theta = result.x
    return opt_theta


opt_theta = optimize(theta, x, y, hyper_p)
print(opt_theta)
final = compute_cost(opt_theta, x, y, hyper_p)
print(final)


# Plot the linear prediction on the training data


def predict(opt_theta, x):
    x_copy = np.insert(x, 0, 1, axis=1)
    prediction = x_copy @ opt_theta
    return prediction


def plot_prediction(opt_theta, x, y):
    x_pred = np.linspace(-50, 50, 30)[:, None]
    y_pred = predict(opt_theta, x_pred)
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.scatter(x, y, marker='x', c='r', label="Training Data")
    ax.set_xlabel('Change in water level (x)')
    ax.set_ylabel('Water flowing out of the dam (y)')
    ax.plot(x_pred, y_pred, c='b', label="Linear Fit")
    ax.legend()


plot_prediction(opt_theta, x, y)
plt.show()


# Plot the learning curves


def plot_learning_curves(x, y, xval, yval, hyper_p=0):
    m, n = x.shape
    train_cost = np.zeros(m)
    val_cost = np.zeros(m)
    theta = np.zeros(n + 1)
    num_samples = np.arange(m)

    for i in num_samples:
        theta = optimize(theta, x[:i + 1, :], y[:i + 1], hyper_p)
        train_cost[i] = compute_cost(theta, x[:i + 1, :], y[:i + 1])
        val_cost[i] = compute_cost(theta, xval, yval)

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.plot(num_samples, train_cost, label='Training error')
    ax.plot(num_samples, val_cost, label='Cross-Validation error')
    ax.legend()
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Error')
    plt.show()


plot_learning_curves(x, y, xval, yval)

theta = np.zeros(x.shape[1] + 1)
hyper_p = 0

opt_theta = optimize(theta, x, y, hyper_p)
train_cost = compute_cost(opt_theta, x, y)
val_cost = compute_cost(opt_theta, xval, yval)
print(train_cost, val_cost)
