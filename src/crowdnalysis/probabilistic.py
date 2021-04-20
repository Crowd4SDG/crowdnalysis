import numpy as np

from .consensus import AbstractConsensus, DiscreteConsensusProblem
from .data import Data


class Probabilistic(AbstractConsensus):
    name = "Probabilistic"
    
    def __init__(self):
        pass

    @classmethod
    def fit_and_compute_consensus(cls, dcp: DiscreteConsensusProblem, softening=0.1):
        n = dcp.compute_n().sum(axis=0) + softening
        consensus = n / np.sum(n, axis=1)[:, np.newaxis]
        # print("Probabilistic._probabilistic_consensus ({}) -> \n".format(consensus.shape), consensus)
        return consensus, {"softening": softening}
