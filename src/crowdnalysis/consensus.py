import numpy as np

from typing import List, Tuple

from .data import Data

class AbstractConsensus:
    """ Base class for a consensus algorithm."""
    name = None
    
    @staticmethod
    def compute_counts(m, I, J):
        n = np.zeros((I, J))
        for i, k, j in m:
            n[i, j] += 1
        return n
        # print(n)

    @staticmethod
    def _get_question_matrix_and_ranges(d, question):
        m = d.get_question_matrix(question)
        I = d.n_tasks
        J = d.n_labels(question)
        K = d.n_annotators
        return m, I, J, K

    def compute_consensus(self, d: Data, question):
        """Computes consensus for question question from Data d.
        returns consensus, model parameters""" 
        raise NotImplementedError

    def success_rate(self, real_labels, crowd_labels):
        """"""
        raise NotImplementedError

class Factory:
    algorithms = {}
    @staticmethod
    def getConsensusAlgorithm(name):
        return algorithms[name]
        
    def registerConsensusAlgorithm(algorithm):
        algorithms[algorithm.name] = algorithm

import majority_voting
import probabilistic
import dawid_skene
import stan

Factory.registerConsensusAlgorithm(majority_voting.MajorityVoting)
Factory.registerConsensusAlgorithm(probabilistic.Probabilistic)
Factory.registerConsensusAlgorithm(dawid_skene.DawidSkene)
Factory.registerConsensusAlgorithm(stan.StanOptimizeConsensus)
Factory.registerConsensusAlgorithm(stan.StanDSOptimizeConsensus)
#Factory.registerConsensusAlgorithm(stan.StanDSSampleConsensus)


