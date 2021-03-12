from importlib.resources import path as irpath
from typing import Any, Dict, Tuple

import numpy as np
from cmdstanpy import CmdStanModel, CmdStanMLE, CmdStanMCMC

from . import consensus
from .data import Data

import pickle


def _parse_var_dims_and_index_ranges(names: Tuple[str, ...]) -> Dict:
    """
    Use Stan CSV file column names to get variable names, dimensions.
    Assumes that CSV file has been validated and column names are correct.
    """
    if names is None:
        raise ValueError('missing argument "names"')
    vars_dict = {}
    idx = 0
    while idx < len(names):
        if names[idx].endswith('__'):
            pass
        elif '[' not in names[idx]:
            vars_dict[names[idx]] = (1, slice(idx))
        else:
            vs = names[idx].split('[')
            idx_start = idx
            while idx < len(names) - 1 and names[idx + 1].split('[')[0] == vs[0]:
                idx += 1
            vs = names[idx].split('[')
            dims = [int(x) for x in vs[1][:-1].split(',')]
            vars_dict[vs[0]] = (tuple(dims), slice(idx_start, idx + 1))
        idx += 1
    return vars_dict

def _get_var_dict(results: CmdStanMLE):
    var_info = _parse_var_dims_and_index_ranges(results.column_names)
    values_array = results.optimized_params_np
    var_dict = { var_name: values_array[var_info[var_name][1]].reshape(var_info[var_name][0], order='F')
                    for var_name in var_info
                }
    return var_dict

#with importlib.resources.path("crowdnalysis", "Multinomial.sample_annotations.stan") as path_:
#    MULTINOMIAL_STAN_PATH = str(path_)
#with importlib.resources.path("crowdnalysis", "HDS-NC.stan") as path_:
#    HDS_STAN_PATH = str(path_)

# TODO (OM, 20201210): Beautify above code
with irpath(__package__, __name__.split('.')[-1]) as path_:
    resources_path = path_


def resource_filename(filename):
    return resources_path / filename


class AbstractStanOptimizeConsensus(consensus.GenerativeAbstractConsensus):

    name = "AbstractStanOptimizeConsensus"

    def __init__(self, model_name):
        self.model_name = model_name
    
    def map_data_to_model(self, m, I, J, K, **kwargs):
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

    def MLE_parameters(self, results: CmdStanMLE):
        return {var_name: results.optimized_params_dict[var_name] for var_name in self.hidden_variables}

    # def reduce_num_annotators(self, jj, minim_an):
    #     """
    #
    #     Args:
    #         jj(np.array): annotators of each annotation
    #         minim_an (int): minimum number of annotations for annotators to be considered independent
    #
    #     Returns:
    #         jj (np.array): new list of annotators of each annotation
    #
    #     """
    #     N = len(jj)
    #     print('old J: '+ str(len(np.unique(jj))))
    #     new_label = N+1
    #
    #     b, c = np.unique(jj, return_counts=True)  # b is a list of single annotators and c its number of annotations
    #     for i in range(len(c)):
    #         if c[i] >= minim_an:
    #             jj[np.where(jj == b[i])[0]] = new_label
    #             new_label += 1
    #     for i in range(N):
    #         if jj[i] < N + 1:
    #             jj[i] = new_label
    #     jj[:] -= N
    #     print('new J: ' +str(len(np.unique(jj))))
    #     return jj

    def fit_and_compute_consensus_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".fit_and_consensus.stan"))


    def m_fit_and_compute_consensus(self, m, I, J, K, **kwargs):
        """Fits the model parameters and computes the consensus.
        returns consensus, model parameters""" 

        stan_data, init_data, kwargs = self.map_data_to_model(m, I, J, K)
        model = self.fit_and_compute_consensus_model()
        results = model.optimize(data=stan_data, inits=init_data, **kwargs)
        var_dict = _get_var_dict(results)

        return var_dict["t_C"], var_dict

    def fit_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".fit.stan"))

    def m_fit(self, m, I, J, K, reference_consensus, **kwargs):
        """ Fits the model parameters provided that the consensus is already known.
        This is useful to determine the errors of a different set of annotators than the
        ones that were used to determine the consensus.
        """

        stan_data, init_data, kwargs = self.map_data_to_model(m, I, J, K)
        stan_data['t_C'] = reference_consensus

        model = self.fit_model()
        results = model.optimize(data=stan_data, inits=init_data, **kwargs)

        return _get_var_dict(results)

    def compute_consensus_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".consensus.stan"))

    def m_compute_consensus(self, m, I, J, K, **kwargs):
        #print(kwargs["data"])
        stan_data, init_data, kwargs_opt = self.map_data_to_model(m, I, J, K)
        stan_data.update(kwargs["data"])

        model = self.compute_consensus_model()
        results = model.optimize(data=stan_data, inits=init_data, **kwargs_opt)

        return _get_var_dict(results)['t_C'], kwargs_opt

    def sample_tasks_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".sample_tasks.stan"))

    def sample_annotations_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".sample_annotations.stan"))


class StanMultinomialOptimizeConsensus(AbstractStanOptimizeConsensus):

    name = "StanMultinomialOptimize"

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "Multinomial")

    def map_data_to_model(self, m, I, J, K):
        """

                Args:
                    m (np.ndarray): question matrix
                    I (int): number of tasks
                    J (int): number of labels
                    K (int): number of annotators

                Returns:
                    stan_data (): data in stan format
                    init_data (): initial values for error matrices in stan format

                """

        def tau_prior(ann, t_A, t, k, alpha=1.):
            # ct = np.zeros(t,k)
            unique, counts = np.unique(ann, return_counts=True)
            tp = np.ones(k)
            for i, x in enumerate(unique):
                tp[x - 1] += counts[i]
            tp /= np.sum(tp)
            return alpha * tp

        def pi_prior(k, alpha=1, beta=10):
            return np.ones((k, k)) * alpha + np.identity(k) * beta

        t_A =  m[:, 0] + 1
        w_A = m[:, 1] + 1
        ann = m[:, 2] + 1
        t = I
        w = K
        k = J
        a = m.shape[0]
        tau_prior_ = tau_prior(ann, t_A, t, k, 5.)
        pi_prior_ = pi_prior(k, beta=20)

        stan_data = {'w': w,
                     't': t,
                     'a': a,
                     'k': k,
                     't_A': t_A,
                     'w_A': w_A,
                     'ann': ann,
                     'tau_prior': tau_prior_,
                     'pi_prior': pi_prior_}
        print(stan_data)
        init_params = {'tau': tau_prior(ann, t_A, t, k, alpha=1.),
                       'pi' : pi_prior_/np.sum(pi_prior_[0])}
        #args = {"iter": 2000}
        args = {'algorithm': 'LBFGS',
                'init_alpha': 0.01,
                'output_dir': "."}
        return stan_data, init_params, args

    def sample_tasks(self, I, parameters=None):
        """

        Args:
            I: number of tasks

        Returns:
            numpy.ndarray:
        """
        model = self.sample_tasks_model()
        sample = model.sample(data={'t': I, 'k': len(parameters['tau'])},
                     inits=parameters,
                     fixed_param=True,
                     iter_sampling=1)
        t_C = sample.stan_variable('t_C').to_numpy(dtype=int)[0]-1
        return t_C


    def sample_annotations(self, real_labels, num_annotations_per_task, parameters=None):
        """

        Args:
            real_labels (numpy.ndarray): 1D array with dimension (I)
            num_annotations_per_task (int):number of annotations per task

        Returns:
            numpy.ndarray: 2D array with dimensions (I * num_annotations_per_task, 3)

        """
        model = self.sample_annotations_model()
        sample = model.sample(data={'w': 1,
                                    't': len(real_labels),
                                    'num_annotations_per_task': num_annotations_per_task,
                                    'k': len(parameters['tau']),
                                    't_C': real_labels+1},
                              inits=parameters,
                              fixed_param=True,
                              iter_sampling=1)
        m = np.zeros((num_annotations_per_task * len(real_labels), 3), dtype=int)
        m[:, 0] = sample.stan_variable('t_A').to_numpy(dtype=int)[0] - 1
        m[:, 1] = sample.stan_variable('w_A').to_numpy(dtype=int)[0] - 1
        m[:, 2] = sample.stan_variable('ann').to_numpy(dtype=int)[0] - 1

        return m
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
