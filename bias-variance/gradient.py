import numpy as np


def gradient(theta, x, y, hyper_p=0):
    size = len(y)
    x_copy = np.insert(x, 0, 1, axis=1)
    h = x_copy @ theta.T
    grad = ((h - y).T @ x_copy) / size
    reg = ((hyper_p / size) * theta)

    reg_grad = grad + reg
    # reg_grad[0] = grad[0]
    return reg_grad
