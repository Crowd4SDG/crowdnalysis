import numpy as np

from . import consensus
from .data import Data


class MajorityVoting(consensus.AbstractConsensus):
    name = "MajorityVoting"
  
    def __init__(self):
        pass

    @classmethod
    def majority_voting(cls, m, I, J):
        n = cls.compute_counts(m, I, J)
        # print(n)
        best_count = np.amax(n, axis=1)
        bool_best_candidates = (n == best_count[:, np.newaxis])  # A cell is True if the label is a/the best candidate
        num_best_candidates = np.sum(bool_best_candidates, axis=1)
        best = np.argmax(n, axis=1)
        best[num_best_candidates != 1] = -1  #
        consensus = bool_best_candidates / (num_best_candidates[:, np.newaxis])  # give probability distribution when num_best_candidates > 1
        # print("MajorityVoting.majority_voting consensus ({}) -> \n{}\nbest ({}) -> \n{}".format(consensus.shape, consensus, best.shape, best))
        return consensus, best

    @classmethod
    def fit_and_compute_consensus(cls, d: Data, question):
        m, I, J, K = cls._get_question_matrix_and_ranges(d, question)
        # print("MajorityVoting > question_matrix ({}) -> \n".format(m.shape), m)
        return cls.majority_voting(m, I, J)

    @classmethod
    def success_rate(cls, real_labels, crowd_labels, J):
        # TODO (OM, 20201203): Had to pass J as an arg since MajorityVoting needs it
        I = real_labels.shape[0]  # number of tasks
        # print("majority_success_rate > crowd_labels ({}):\n".format(crowd_labels.shape), crowd_labels)
        _, best = cls.majority_voting(crowd_labels, I, J)
        # print("majority_success_rate > majority_voting ({}):\n".format(c.shape), c)
        num_successes = np.sum(real_labels == best)
        return num_successes / I

    @classmethod
    def success_rate_with_fixed_parameters(cls, p, _pi, I, K):
        from .dawid_skene import DawidSkene
        DS = DawidSkene()
        # TODO (OM, 20201207): Using DawidSkene inside MajorityVoting to access DS.fast_sample()??
        real_labels, crowd_labels = DS.fast_sample(p=p, _pi=_pi, I=I, num_annotators=K)
        return cls.success_rate(real_labels, crowd_labels, J=len(p))

    @classmethod
    def success_rates(cls, p, _pi, I, annotators):
        from .dawid_skene import DawidSkene
        success_p = np.zeros(len(annotators))
        DS = DawidSkene()
        # TODO (OM, 20201207): Using DawidSkene inside MajorityVoting to access DS.fast_sample()??
        for K in annotators:
            real_labels, crowd_labels = DS.fast_sample(p=p, _pi=_pi, I=I, num_annotators=K)
            success_p[annotators.index(K)] = cls.success_rate(real_labels, crowd_labels, J)
        return success_p

