import numpy as np


def compute_cost(x, y, theta, hyper_p):
    size = len(x)
    x = np.insert(x, 0, 1, axis=1)
    cost = np.sum((1 / (2 * size)) * (((np.dot(x, theta.T)) - y) ** 2))
    reg = (hyper_p / (2 * size)) + np.sum(theta[:, 1:theta.shape[1]] ** 2)
    reg_cost = cost + reg
    return reg_cost
