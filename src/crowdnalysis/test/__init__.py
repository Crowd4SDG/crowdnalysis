import numpy as np


def distance(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a-b)


def close(a: np.ndarray, b: np.ndarray, eps=0.05):
    return distance(a, b) < eps
