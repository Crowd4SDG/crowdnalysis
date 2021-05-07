import numpy as np

TOLERANCE = 0.05


def distance(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a-b)


def close(a: np.ndarray, b: np.ndarray, eps=TOLERANCE):
    return distance(a, b) < eps
