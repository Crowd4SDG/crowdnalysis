from .. import log
from ..measures import Accuracy, LSAccuracy
import numpy as np

def test_accuracies():
    real = np.array([0, 1, 0, 1, 0])
    consensus_all = np.array([[0.8, 0.2], [0.1, 0.7], [0.6, 0.4], [0.45, 0.55], [0.99, 0.01]])
    consensus_none = np.ones((5,2))-consensus_all

    assert Accuracy.evaluate(real, consensus_all) == 1.0
    assert Accuracy.evaluate(real, consensus_none) == 0.0
    assert LSAccuracy.evaluate(real, consensus_all) == 1.0
    assert LSAccuracy.evaluate(real, consensus_none) == 1.0