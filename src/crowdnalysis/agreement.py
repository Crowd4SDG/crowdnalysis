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


def gen_fleiss_kappa(d: Data, question: str, w: np.ndarray=None) -> float:
    """Return Fleiss' kappa value for the annotation data"""
    n, *_ = get_n_and_ranges(d, question)
    kappa = _gen_fleiss_kappa(n, w)
    return kappa


def _gen_fleiss_kappa(r, w=None):
    # r[i][l] -> number of raters that assigned item i to category l
    n, q = r.shape
    if w is None:
        w = np.identity(q)

    r_star = np.dot(r, np.transpose(w))
    r_i = np.sum(r, axis=1)
    # print(r_i)
    assert(np.all(r_i>1))
    den_i = r_i * (r_i-1)
    # print(den_i)
    num_i_k= r * (r_star - 1)
    num_sum_i = np.sum(num_i_k, axis=1)
    # print(num_sum_i)
    p_0 = np.sum(num_sum_i/den_i)/n
    # print("p_0:",p_0)
    pi_k = np.sum(r/r_i[:,np.newaxis], axis=0)/n
    # print(pi_k)
    p_c = np.dot(np.dot(pi_k, w), pi_k)
    return (p_0-p_c)/(1-p_c)


def full_agreement_percentage(d: Data, question: str) -> float:
    """Return the percentage of all annotations for the question"""
    n, I, *_ = get_n_and_ranges(d, question)
    best_count = np.amax(n, axis=1)
    pct = np.sum(np.sum(n, axis=1) == best_count) / I
    return pct
