from typing import Tuple

import numpy as np
from statsmodels.stats import inter_rater

from .consensus import AbstractConsensus
from .data import Data


def get_n_and_ranges(d: Data, question: str) -> Tuple[np.ndarray, int, int, int]:
    """Return 2D (I, J) annotation count matrix `n`, and I, J, K values

    where I: # of tasks, J: # of labels, K: # of annotators
    """
    m, I, J, K = AbstractConsensus.get_question_matrix_and_ranges(d, question)
    n = AbstractConsensus.compute_counts(m, I, J)
    return n, I, J, K


def fleiss_kappa(d: Data, question: str) -> float:
    """Return Fleiss' kappa value for the annotation data"""
    n, *_ = get_n_and_ranges(d, question)
    kappa = inter_rater.fleiss_kappa(table=n, method='fleiss')
    return kappa


def full_agreement_percentage(d: Data, question: str) -> float:
    """Return the percentage of all annotations for the question"""
    n, I, *_ = get_n_and_ranges(d, question)
    best_count = np.amax(n, axis=1)
    pct = np.sum(np.sum(n, axis=1) == best_count) / I
    return pct
