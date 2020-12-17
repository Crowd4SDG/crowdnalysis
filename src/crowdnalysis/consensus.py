import numpy as np
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
    def get_question_matrix_and_ranges(d, question):
        m = d.get_question_matrix(question)
        I = d.n_tasks
        J = d.n_labels(question)
        K = d.n_annotators
        return m, I, J, K

    def fit_and_compute_consensuses(self, d, questions, **kwargs):
        consensuses = {}
        parameters = {}
        for q in questions:
            consensuses[q], parameters[q] = self.fit_and_compute_consensus(d, q)
        return consensuses, parameters

    def fit_and_compute_consensus(self, d: Data, question, **kwargs):
        """Computes consensus and fits model for question question from Data d.

        returns consensus, model parameters"""
        # TODO (OM, 20201210): A return class for model parameters instead of dictionary
        raise NotImplementedError

    def fit_many(self, d:Data, reference_consensuses):
        parameters = {}
        for q in reference_consensuses:
            parameters[q] = self.fit(d, q, reference_consensuses[q])
        return parameters

    def fit(self, d: Data, question, reference_consensus):
        """ Fits the model parameters provided that the consensus is already known.
        This is useful to determine the errors of a different set of annotators than the
        ones that were used to determine the consensus.

        returns parameters """

    def compute_consensus(self, d:Data, question, parameters):
        """ Computes the consensus with a fixed pre-determined set of parameters.

        returns consensus """

    def success_rate(self, real_labels, crowd_labels):
        """"""
        raise NotImplementedError



