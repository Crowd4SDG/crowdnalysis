from typing import Tuple

import numpy as np
from statsmodels.stats import inter_rater

from . import log
from .data import Data


def get_n_and_ranges(d: Data, question: str, ignore_le1_annots: bool = True) -> Tuple[np.ndarray, int]:
    """Return 2D (n_tasks, n_labels) annotation count matrix `n`, and # of unique tasks in data `d`.

    Args:
        ignore_le1_annots: Ignores tasks with # of annotations ≤ 1. This makes sense since to speak of an agreement,
            there have to be > 1 annotations for a task.
    """
    dcp = d.get_dcp(question)
    n, _ = dcp.compute_n(ignore_zero_annots=False)
    n = n.sum(axis=0)
    if ignore_le1_annots:
        le1_annots = n.sum(axis=1) <= 1
        if np.any(le1_annots):
            log.info("{} tasks with ≤ 1 annotations out of {} tasks are ignored in agreement calculation.".format(
                np.sum(le1_annots), n.shape[0]))
            # print("n[le1_annots, :]:\n", n[le1_annots, :])
            n = n[~le1_annots, :]
    return n, dcp.n_tasks


def fleiss_kappa(d: Data, question: str) -> float:
    """Return Fleiss' kappa value for the annotation data

    Raises:
        ValueError: If all tasks don't have the same number of annotations
    """
    n, *_ = get_n_and_ranges(d, question)
    n_sum = np.sum(n, axis=1)
    if np.all(n_sum == np.amax(n_sum)):
        kappa = inter_rater.fleiss_kappa(table=n, method='fleiss')
    else:
        raise ValueError("All tasks must have equal number of annotations")
    return kappa


def gen_fleiss_kappa(d: Data, question: str, w: np.ndarray = None) -> float:
    """Return generalized Fleiss' kappa value for the annotation data

    ref: Gwet KL. (2014) Handbook of Inter-Rater Reliability.
    """
    n, *_ = get_n_and_ranges(d, question)
    # print("\nn:\n", n)
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
    assert(np.all(r_i > 1))
    den_i = r_i * (r_i-1)
    # print(den_i)
    num_i_k = r * (r_star - 1)
    num_sum_i = np.sum(num_i_k, axis=1)
    # print(num_sum_i)
    p_0 = np.sum(num_sum_i/den_i)/n
    # print("p_0:",p_0)
    pi_k = np.sum(r/r_i[:, np.newaxis], axis=0)/n
    # print(pi_k)
    p_c = np.dot(np.dot(pi_k, w), pi_k)
    return (p_0-p_c)/(1-p_c)


def full_agreement_percentage(d: Data, question: str) -> float:
    """Return the percentage of annotations for the `question` where all annotators agreed on the same answer"""
    n, n_tasks = get_n_and_ranges(d, question)
    best_count = np.amax(n, axis=1)
    pct = np.sum(np.sum(n, axis=1) == best_count) / n_tasks
    return pct
