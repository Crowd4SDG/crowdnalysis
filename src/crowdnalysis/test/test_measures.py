import pytest

import numpy as np

from . import TOLERANCE
from ..measures import Accuracy, AllClose, CloseRatio, LSAccuracy


@pytest.fixture(scope="module")
def ref_consensus():
    consensus = np.array([[0.61, 0.17, 0.18, 0.04],   # 0
                          [0.25, 0.17, 0.40, 0.18],   # 2
                          [0.02, 0.64, 0.16, 0.18],   # 1
                          [0.02, 0.51, 0.18, 0.29],   # 1
                          [0.23, 0.14, 0.28, 0.35]])  # 3
    return consensus


@pytest.fixture(scope="module")
def real_labels():
    return np.array([0, 2, 1, 1, 3])  # see `ref_consensus`


def test_accuracies_binary():
    real = np.array([0, 1, 0, 1, 0])
    consensus_all = np.array([[0.8, 0.2], [0.1, 0.7], [0.6, 0.4], [0.45, 0.55], [0.99, 0.01]])
    # Switch 0<->1
    consensus_none = np.ones(consensus_all.shape) - consensus_all

    assert Accuracy.evaluate(real, consensus_all) == 1.0
    assert Accuracy.evaluate(real, consensus_none) == 0.0
    assert LSAccuracy.evaluate(real, consensus_all) == 1.0
    assert LSAccuracy.evaluate(real, consensus_none) == 1.0


def test_accuracies_multilabel(real_labels, ref_consensus):
    real2 = np.array([0, 2, 1, 2, 3])  # different than the consensus
    # Switch labels 1<->3
    consensus_switched = np.array([[i[0], i[3], i[2], i[1]] for i in ref_consensus.tolist()])
    assert Accuracy.evaluate(real_labels, ref_consensus) == 1.0       # [0, 2, 1, 1, 3] vs [0, 2, 1, 1, 3]
    assert Accuracy.evaluate(real_labels, consensus_switched) == 0.4  # [0, 2, 1, 1, 3] vs [0, 2, 3, 3, 1]
    assert LSAccuracy.evaluate(real_labels, ref_consensus) == 1.0
    assert LSAccuracy.evaluate(real_labels, consensus_switched) == 1.0

    assert Accuracy.evaluate(real2, ref_consensus) == 0.8       # [0, 2, 1, 2, 3] vs [0, 2, 1, 1, 3]
    assert Accuracy.evaluate(real2, consensus_switched) == 0.4  # [0, 2, 1, 2, 3] vs [0, 2, 3, 3, 1]
    assert LSAccuracy.evaluate(real2, ref_consensus) == 0.8
    assert LSAccuracy.evaluate(real2, consensus_switched) == 0.8

    # Switch labels 0<->3, 1<->2
    consensus_switched = np.array([[i[3], i[2], i[1], i[0]] for i in ref_consensus.tolist()])
    assert Accuracy.evaluate(real_labels, ref_consensus) == 1.0       # [0, 2, 1, 1, 3] vs [0, 2, 1, 1, 3]
    assert Accuracy.evaluate(real_labels, consensus_switched) == 0.0  # [0, 2, 1, 1, 3] vs [3, 1, 2, 2, 0]
    assert LSAccuracy.evaluate(real_labels, ref_consensus) == 1.0
    assert LSAccuracy.evaluate(real_labels, consensus_switched) == 1.0

    assert Accuracy.evaluate(real2, ref_consensus) == 0.8       # [0, 2, 1, 2, 3] vs [0, 2, 1, 1, 3]
    assert Accuracy.evaluate(real2, consensus_switched) == 0.2  # [0, 2, 1, 2, 3] vs [3, 1, 2, 2, 0]
    assert LSAccuracy.evaluate(real2, ref_consensus) == 0.8
    assert LSAccuracy.evaluate(real2, consensus_switched) == 0.8


def test_closeratio(ref_consensus):
    assert CloseRatio.evaluate(ref_consensus, ref_consensus, atol=0.) == 1.0
    assert CloseRatio.evaluate(ref_consensus, ref_consensus - TOLERANCE, atol=TOLERANCE) == 1.0
    assert CloseRatio.evaluate(ref_consensus, ref_consensus - 2 * TOLERANCE, atol=TOLERANCE) == 0.0
    n_labels = ref_consensus.shape[1]
    compared = np.copy(ref_consensus)
    compared[:, 0] -= 2 * TOLERANCE
    assert CloseRatio.evaluate(ref_consensus, compared) == (n_labels - 1.) / n_labels


def test_allclose(ref_consensus):
    assert AllClose.evaluate(ref_consensus, ref_consensus, atol=0.)
    assert AllClose.evaluate(ref_consensus, ref_consensus - TOLERANCE, atol=TOLERANCE)
    assert not AllClose.evaluate(ref_consensus, ref_consensus - 2 * TOLERANCE)
