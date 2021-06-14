from dataclasses import dataclass

import numpy as np

from crowdnalysis.consensus import AbstractSimpleConsensus
from crowdnalysis.problems import DiscreteConsensusProblem
from .factory import Factory

class Probabilistic(AbstractSimpleConsensus):
    name = "Probabilistic"

    def __init__(self):
        pass

    @classmethod
    def fit_and_compute_consensus(cls, dcp: DiscreteConsensusProblem, softening=0.1):
        n, _ = dcp.compute_n()
        n = n.sum(axis=0) + softening
        n = n[:, dcp.classes]
        consensus = n / np.sum(n, axis=1)[:, np.newaxis]
        # print("Probabilistic._probabilistic_consensus ({}) -> \n".format(consensus.shape), consensus)
        return consensus, AbstractSimpleConsensus.Parameters()


class MajorityVoting(AbstractSimpleConsensus):

    name = "MajorityVoting"

    @dataclass
    class Parameters(AbstractSimpleConsensus.Parameters):
        pass

    def __init__(self):
        pass

    @classmethod
    def consensus_and_single_most_voted(cls, dcp: DiscreteConsensusProblem):
        consensus, _ = Probabilistic.fit_and_compute_consensus(dcp)
        best_index = np.amax(consensus, axis=1)
        bool_best_candidates = (consensus == best_index[:, np.newaxis])
        # A cell is True if the label is a/the best candidate
        num_best_candidates = np.sum(bool_best_candidates, axis=1)
        single_most_voted = np.argmax(consensus, axis=1)
        # TODO: Consider using masked arrays next
        single_most_voted[num_best_candidates != 1] = -1e10  # Mark as -1e20 when there is a tie in the number of votes
        consensus = bool_best_candidates / (num_best_candidates[:, np.newaxis])
        # give probability distribution when num_best_candidates > 1
        # print("MajorityVoting.majority_voting consensus ({}) -> \n{}\n best ({}) -> \n{}".format(consensus.shape,
        # consensus, best.shape, best))
        return consensus, single_most_voted

    def fit_and_compute_consensus(self, dcp: DiscreteConsensusProblem, **kwargs):
        consensus, _ = self.consensus_and_single_most_voted(dcp)
        return consensus, MajorityVoting.Parameters()

Factory.register_consensus_algorithm(MajorityVoting)
Factory.register_consensus_algorithm(Probabilistic)