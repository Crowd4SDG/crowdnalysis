import importlib.resources
from typing import Any, Dict

import numpy as np
import pystan

from . import consensus
from .data import Data

import pickle


with importlib.resources.path("crowdnalysis", "DS.stan") as path_:
    DS_STAN_PATH = str(path_)
with importlib.resources.path("crowdnalysis", "HDS-NC.stan") as path_:
    HDS_STAN_PATH = str(path_)

# TODO (OM, 20201210): Beautify above code


class AbstractStanOptimizeConsensus(consensus.AbstractConsensus):

    name = "AbstractStanOptimizeConsensus"

    def __init__(self, stan_model_filename):
        self.stan_model = pystan.StanModel(file=stan_model_filename)
    
    def map_data_to_model(self, m, I, J, K):
        """

        Args:
            m (np.ndarray): question matrix
            I (int): number of tasks
            J (int): number of labels
            K (int): number of annotators

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments to pass to the `StanModel.optimizing` method

        """
        raise NotImplementedError

    def reduce_num_annotators(self, jj, minim_an):
        """

        Args:
            jj(np.array): annotators of each annotation
            minim_an (int): minimum number of annotations for annotators to be considered independent

        Returns:
            jj (np.array): new list of annotators of each annotation

        """
        N = len(jj)
        print('old J: '+ str(len(np.unique(jj))))
        new_label = N+1

        b, c = np.unique(jj, return_counts=True)  # b is a list of single annotators and c its number of annotations
        for i in range(len(c)):
            if c[i] >= minim_an:
                jj[np.where(jj == b[i])[0]] = new_label
                new_label += 1
        for i in range(N):
            if jj[i] < N + 1:
                jj[i] = new_label
        jj[:] -= N
        print('new J: ' +str(len(np.unique(jj))))
        return jj

    def fit_and_compute_consensus(self, d: Data, question):
        """Computes consensus for question question from Data d.
        returns consensus, model parameters""" 
       
        m, I, J, K = self.get_question_matrix_and_ranges(d, question)
        stan_data, init_data, kwargs = self.map_data_to_model(m, I, J, K)
        # Here you should implement the mapping from to the input data for this Stan Model
        # This mapping should work for any question
        stan_model = pickle.load(open('DS_stan.pkl', 'rb'))
        results = stan_model.optimizing(data=stan_data, init=init_data, **kwargs)
        
        # Here you should obtain the consensus (the q's) from Stan and also return the additional parameters.
        return results["q_z"], kwargs

    def fit(self, d: Data, question, reference_consensus):
        """ Fits the model parameters provided that the consensus is already known.
        This is useful to determine the errors of a different set of annotators than the
        ones that were used to determine the consensus.

            Args:
                Data: data used for the Dawid-Skene stan model.
                question (str): label in which we are working (relevant, severity, compact_severity)
                reference_consensus (np.ndarray): real consensus for every item.

            Returns:
                results['pi'] (np.array): Prior probabilities for each label.
                results['beta'] (np.ndarray): Error-matrices for each annotator.

        """

        m, I, J, K = self.get_question_matrix_and_ranges(d, question)
        stan_data, init_data, kwargs = self.map_data_to_model(m, I, J, K)
        stan_data['q_z'] = reference_consensus

        stan_model_fit = pickle.load(open('DS_stan_fit.pkl', 'rb'))
        results = stan_model_fit.optimizing(data=stan_data, init=init_data, **kwargs)

        # Here you should obtain the consensus (the q's) from Stan and also return the additional parameters.
        return results['pi'], results['beta'], kwargs

    def compute_consensus(self, d: Data, question, pi, beta):
        """ Computes the consensus with a fixed pre-determined set of parameters.
        returns consensus

            Args:
                Data: data used for the Dawid-Skene stan model.
                question (str): label in which we are working (relevant, severity, compact_severity)
                pi (np.array): Prior probabilities for each label.
                beta (np.ndarray): Error-matrices for each annotator.

            Returns:
                results['q_z'] (np.ndarray): real consensus for every item.
        """

        m, I, J, K = self.get_question_matrix_and_ranges(d, question)
        stan_data, init_data, kwargs = self.map_data_to_model(m, I, J, K)
        stan_data['pi'] = pi
        stan_data['beta'] = beta

        stan_model_fit = pickle.load(open('DS_stan_compute_consensus.pkl', 'rb'))
        results = stan_model_fit.optimizing(data=stan_data, init=init_data, **kwargs)

        # Here you should obtain the consensus (the q's) from Stan and also return the additional parameters.
        return results['q_z'], kwargs


    #def success_rate(self, real_labels, crowd_labels):
    #    Apply this model to the crowd_labels and compare against the real_labels
    #    """"""
    #    raise NotImplementedError


class StanDSOptimizeConsensus(AbstractStanOptimizeConsensus):

    name = "StanDSOptimize"

    def __init__(self):
        # TODO (OM, 20201210): Move 'DS.stan' to an appropriate folder.
        #super().__init__(stan_model_filename=DS_STAN_PATH)
        pass

    def map_data_to_model(self, m, I, J, K, minim_an = 10):
        """

                Args:
                    m (np.ndarray): question matrix
                    I (int): number of tasks
                    J (int): number of labels
                    K (int): number of annotators
                    minim_an (int): minimum number of annotations for annotators to be considered independent

                Returns:
                    stan_data (): data in stan format
                    init_data (): initial values for error matrices in stan format

                """
        newN = len(m[:, 0])
        newii = m[:, 0] + 1
        jj = m[:, 1] + 1
        newyy = m[:, 2] + 1
        newI = I
        newJ = J
        newK = K
        newjj = self.reduce_num_annotators(jj, minim_an)

        stan_data = {'J': newK, 'K': newJ, 'N': newN,'I': newI,'ii': newii,'jj': newjj,'y': newyy, 'alpha':np.array([1,1])}

        init_beta = (np.identity(J) + 0.1) / (1 + J * 0.1)  # We suppose that the annotators tend to answer correctly the labels
        init_beta = np.tile(init_beta, (K, 1, 1))  # We need a matrix-error for each annotator
        init_data = {"beta": init_beta}
        args = {"iter":2000}
        return stan_data, init_data, args



# class StanHDSOptimizeConsensus(AbstractStanOptimizeConsensus):
#     """ Hierarchical DS model"""
#
#     name = "StanHDSOptimize"
#
#     def __init__(self):
#          super().__init__(stan_model_filename=HDS_STAN_PATH)
#
#     def map_data_to_model(m, I, J, K):
#         # TODO: Modify the line below to work properly with m, I, J, and K whatsoever
#         stan_data = {'J': newK, 'K': newJ, 'N': newN,'I': newI,'ii': newii,'jj': newjj,'y': newyy, 'alpha':np.array([1,1])}
#         # TODO: Modify the line below to define init_beta correctly for any m, I, J and K.
#         init_beta = 0
#         init_data = {"beta": init_beta}
#         args = {"iter":2000}
#         return stan_data, init_data, args
