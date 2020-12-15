import importlib.resources
from typing import Any, Dict

import numpy as np
import pystan

from . import consensus
from .data import Data


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
        
    def compute_consensus(self, d: Data, question):
        """Computes consensus for question question from Data d.
        returns consensus, model parameters""" 
       
        m, I, J, K = self.get_question_matrix_and_ranges(d, question)
        stan_data, init_data, kwargs = self.map_data_to_model(m, I, J, K)
        # Here you should implement the mapping from to the input data for this Stan Model
        # This mapping should work for any question
        results = self.stan_model.optimizing(data=stan_data, init=init_data, **kwargs)
        
        # Here you should obtain the consensus (the q's) from Stan and also return the additional parameters.
        return results["q"], kwargs

    #def success_rate(self, real_labels, crowd_labels):
    #    Apply this model to the crowd_labels and compare against the real_labels
    #    """"""
    #    raise NotImplementedError


class StanDSOptimizeConsensus(AbstractStanOptimizeConsensus):

    name = "StanDSOptimize"

    def __init__(self):
        # TODO (OM, 20201210): Move 'DS.stan' to an appropriate folder.
        super().__init__(stan_model_filename=DS_STAN_PATH)

    
    def map_data_to_model(m, I, J, K):
        # TODO: Modify the line below to work properly with m, I, J, and K whatsoever
        stan_data = {'J': newK, 'K': newJ, 'N': newN,'I': newI,'ii': newii,'jj': newjj,'y': newyy, 'alpha':np.array([1,1])}
        # TODO: Modify the line below to define init_beta correctly for any m, I, J and K.
        init_beta = 0
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
