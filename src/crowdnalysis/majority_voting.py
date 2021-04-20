import numpy as np

from .consensus import AbstractSimpleConsensus, DiscreteConsensusProblem
from . import consensus
from .data import Data
from dataclasses import dataclass


class MajorityVoting(AbstractSimpleConsensus):

    name = "MajorityVoting"

    @dataclass
    class Parameters(AbstractSimpleConsensus.Parameters):
        pass

    def __init__(self):
        pass

    @classmethod
    def consensus_and_single_most_voted(cls, dcp: DiscreteConsensusProblem):
        n = dcp.compute_n().sum(axis=0)
        # print(n)
        best_count = np.amax(n, axis=1)
        bool_best_candidates = (n == best_count[:, np.newaxis])  # A cell is True if the label is a/the best candidate
        num_best_candidates = np.sum(bool_best_candidates, axis=1)
        single_most_voted = np.argmax(n, axis=1)
        single_most_voted[num_best_candidates != 1] = -1  #
        consensus = bool_best_candidates / (num_best_candidates[:, np.newaxis])  # give probability distribution when num_best_candidates > 1
        # print("MajorityVoting.majority_voting consensus ({}) -> \n{}\nbest ({}) -> \n{}".format(consensus.shape, consensus, best.shape, best))
        return consensus, single_most_voted

    def fit_and_compute_consensus(self, dcp: DiscreteConsensusProblem, **kwargs):
        consensus, _ = self.consensus_and_single_most_voted(dcp)
        return consensus, MajorityVoting.Parameters()
