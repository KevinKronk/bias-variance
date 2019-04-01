import numpy as np


def compute_cost(theta, x, y, hyper_p=0):
    size = len(y)
    x_copy = np.insert(x, 0, 1, axis=1)
    cost = np.sum((1 / (2 * size)) * (((np.dot(x_copy, theta.T)) - y) ** 2))
    reg = (hyper_p / (2 * size)) * np.sum(theta ** 2)
    reg_cost = cost + reg
    return reg_cost
