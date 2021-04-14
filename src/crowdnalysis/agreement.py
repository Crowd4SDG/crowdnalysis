from typing import Tuple

import numpy as np
from statsmodels.stats import inter_rater

from .consensus import AbstractConsensus
from .data import Data


def get_n_and_ranges(d: Data, question: str) -> Tuple[np.ndarray, int, int, int]:
    """Return 2D (n_tasks, n_labels) annotation count matrix `n`, and n_tasks, n_labels, n_annotators values

    where n_tasks: # of tasks, n_labels: # of labels, n_annotators: # of annotators
    """
    dcp = AbstractConsensus.get_problem(d, question)
    n = AbstractConsensus.compute_counts(dcp.m, dcp.n_tasks, dcp.n_labels)
    return n, dcp.n_tasks, dcp.n_labels, dcp.n_annotators


def fleiss_kappa(d: Data, question: str) -> float:
    """Return Fleiss' kappa value for the annotation data"""
    n, *_ = get_n_and_ranges(d, question)
    kappa = inter_rater.fleiss_kappa(table=n, method='fleiss')
    return kappa


def fleiss_gen_kappa(d: Data, question: str, w: np.ndarray=None) -> float:
    """Return Fleiss' generalized kappa value for the annotation data

    ref: Gwet KL. (2014) Handbook of Inter-Rater Reliability.
    """
    n, *_ = get_n_and_ranges(d, question)
    kappa = _fleiss_gen_kappa(n, w)
    return kappa


def _fleiss_gen_kappa(r, w=None):
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
    """Return the percentage of annotations for the `question` where all annotators agreed on the same answer"""
    n, n_tasks, *_ = get_n_and_ranges(d, question)
    best_count = np.amax(n, axis=1)
    pct = np.sum(np.sum(n, axis=1) == best_count) / n_tasks
    return pct
