import dataclasses
from . import log
from importlib.resources import path as irpath
from typing import Any, Dict, Tuple, Optional

import numpy as np
from cmdstanpy import CmdStanModel, CmdStanMLE

from .consensus import GenerativeAbstractConsensus, DiscreteConsensusProblem
from dataclasses import dataclass

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
    var_dict = {var_name: values_array[var_info[var_name][1]].reshape(var_info[var_name][0], order='F')
                for var_name in var_info
                }
    return var_dict


with irpath(__package__, __name__.split('.')[-1]) as path_:
    resources_path = path_


def resource_filename(filename):
    return resources_path / filename


class AbstractStanOptimizeConsensus(GenerativeAbstractConsensus):
    name = "AbstractStanOptimizeConsensus"

    def __init__(self, model_name):
        self.model_name = model_name

    def map_data_to_model(self, dcp: DiscreteConsensusProblem, **kwargs):
        d = self.map_data_to_data(dcp, **kwargs)
        return d, self.map_data_to_inits(**d, **kwargs), self.map_data_to_args(**d, **kwargs)

    def map_data_to_data(self, dcp: DiscreteConsensusProblem, **kwargs):
        stan_data = {'w': dcp.n_workers,
                     't': dcp.n_tasks,
                     'a': dcp.n_annotations,
                     'k': dcp.n_labels,
                     't_A': (dcp.t_A + 1).tolist(),
                     'w_A': (dcp.w_A + 1).tolist(),
                     'ann': (dcp.f_A + 1).flatten().tolist()}
        prior = self.map_data_to_prior(**stan_data, **kwargs)
        stan_data.update(prior)
        return stan_data

    def map_data_to_prior(self, **kwargs):
        raise NotImplementedError

    def map_data_to_inits(self, **kwargs):
        raise NotImplementedError

    def map_data_to_args(self, **kwargs):
        raise NotImplementedError

    def MLE_parameters(self, results: CmdStanMLE):
        return {var_name: results.optimized_params_dict[var_name] for var_name in self.hidden_variables}

    def fit_and_compute_consensus_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".fit_and_consensus.stan"))

    def fit_and_compute_consensus(self, dcp: DiscreteConsensusProblem, **kwargs):
        """Fits the model parameters and computes the consensus.
        returns consensus, model parameters"""

        stan_data, init_data, kwargs = self.map_data_to_model(dcp)
        model = self.fit_and_compute_consensus_model()
        log.info(stan_data.keys())
        for f in stan_data.keys():
            log.info("Type of %s is %s", f, type(stan_data[f]))
        log.info(init_data)
        #stan_data = {}
        #init_data = {}
        results = model.optimize(data=stan_data, inits=init_data, **kwargs)
        var_dict = _get_var_dict(results)
        keys = [x.name for x in dataclasses.fields(self.Parameters)]
        filtered_dict = {x: var_dict[x] for x in keys}
        return var_dict["t_C"], self.Parameters(**filtered_dict)

    def fit_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".fit.stan"))

    def fit(self, dcp: DiscreteConsensusProblem, reference_consensus, **kwargs):
        """ Fits the model parameters provided that the consensus is already known.
        This is useful to determine the errors of a different set of annotators than the
        ones that were used to determine the consensus.
        """

        stan_data, init_data, kwargs = self.map_data_to_model(dcp)
        stan_data['t_C'] = reference_consensus

        model = self.fit_model()
        results = model.optimize(data=stan_data, inits=init_data, **kwargs)
        keys = [x.name for x in dataclasses.fields(self.Parameters)]
        d_results = _get_var_dict(results)
        filtered_dict = {x: d_results[x] for x in keys}
        return self.Parameters(**filtered_dict)

    def compute_consensus_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".consensus.stan"))

    def compute_consensus(self, dcp: DiscreteConsensusProblem, **kwargs):
        # print(kwargs["data"])
        stan_data, init_data, kwargs_opt = self.map_data_to_model(dcp)
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

    @dataclass
    class Parameters(AbstractStanOptimizeConsensus.Parameters):
        tau: np.ndarray = np.array([0.5, 0.5])
        pi: np.ndarray = np.array([[0.9, 0.1], [0.2, 0.8]])

    @dataclass
    class DataGenerationParameters(GenerativeAbstractConsensus.DataGenerationParameters):
        n_tasks: int = 10
        num_annotations_per_task: int = 2

        def __post_init__(self):
            self.n_annotations = self.n_tasks * self.num_annotations_per_task

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "Multinomial")

    def tau_prior(self, ann, k, alpha=1.):
        unique, counts = np.unique(ann, return_counts=True)
        tp = np.ones(k)
        for i, x in enumerate(unique):
            tp[x - 1] += counts[i]
        tp /= np.sum(tp)
        return alpha * tp

    def pi_prior(self, k, alpha=1, beta=10):
        return np.ones((k, k)) * alpha + np.identity(k) * beta

    def map_data_to_prior(self, k, ann, **kwargs):
        tau_prior_ = self.tau_prior(ann, k, 5.)
        pi_prior_ = self.pi_prior(k)

        return {"tau_prior": tau_prior_.tolist(),
                "pi_prior": pi_prior_.tolist()}

    def map_data_to_inits(self, ann, k, **kwargs):
        pi_prior_ = self.pi_prior(k)
        return {'tau': self.tau_prior(ann, k, alpha=1.).tolist(),
                'pi': (pi_prior_ / np.sum(pi_prior_[0])).tolist()}

    def map_data_to_args(self, **kwargs):
        # args = {"iter": 2000}
        return {'algorithm': 'LBFGS',
                'init_alpha': 0.01,
                'output_dir': "."}

    def sample_tasks(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None) \
            -> Tuple[int, Optional[np.ndarray]]:
        model = self.sample_tasks_model()
        sample = model.sample(data={'t': dgp.n_tasks, 'k': parameters.tau.shape[0]},
                              inits=dataclasses.asdict(parameters),
                              fixed_param=True,
                              iter_sampling=1)
        t_C = sample.stan_variable('t_C').to_numpy(dtype=int)[0] - 1
        log.debug("Tasks type: %s", type(t_C.dtype))
        return dgp.n_tasks, t_C

    def sample_workers(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None)\
            -> Tuple[int, Optional[np.ndarray]]:
        return 1, None

    def sample_annotations(self, tasks, workers, dgp: DataGenerationParameters, parameters: Optional[Parameters]=None)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = self.sample_annotations_model()
        sample = model.sample(data={'w': 1,
                                    't': dgp.n_tasks,
                                    'num_annotations_per_task': dgp.num_annotations_per_task,
                                    'k': parameters.tau.shape[0],
                                    't_C': tasks + 1},
                              inits=dataclasses.asdict(parameters),
                              fixed_param=True,
                              iter_sampling=1)
        t_A = sample.stan_variable('t_A').to_numpy(dtype=int)[0]
        t_A -= 1
        w_A = sample.stan_variable('w_A').to_numpy(dtype=int)[0]
        log.debug(type(w_A.dtype))
        w_A -= 1
        log.debug(type(w_A.dtype))
        f_A = sample.stan_variable('ann').to_numpy(dtype=int)[0]
        f_A -= 1

        return w_A, t_A, f_A


class StanMultinomialEtaOptimizeConsensus(StanMultinomialOptimizeConsensus):
    name = "StanMultinomialEtaOptimize"

    @dataclass
    class Parameters(AbstractStanOptimizeConsensus.Parameters):
        tau: np.ndarray = np.array([0.5, 0.5])
        eta: np.ndarray = np.array([[0.9], [0.8]])

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "MultinomialEta")

    def map_data_to_prior(self, k, ann, **kwargs):
        tau_prior_ = self.tau_prior(ann, k, 5.)
        min_pi_prior_ = np.zeros((k, k - 1))
        max_pi_prior_ = np.ones((k, k - 1)) * 5
        return {'tau_prior': tau_prior_,
                'min_pi_prior': min_pi_prior_,
                'max_pi_prior': max_pi_prior_}

    def map_data_to_inits(self, ann, k, **kwargs):
        return {'tau': self.tau_prior(ann, k, alpha=1.),
                'eta': np.ones((k, k - 1))}

    # def map_data_to_args(self):
    #    #args = {"iter": 2000}
    #    return {'algorithm': 'LBFGS',
    #            'output_dir': "."}


class StanDSOptimizeConsensus(StanMultinomialOptimizeConsensus):
    name = "StanDSOptimize"

    @dataclass
    class Parameters(AbstractStanOptimizeConsensus.Parameters):
        tau: np.ndarray = np.array([0.5, 0.5])
        pi: np.ndarray = np.array([[[0.9, 0.1], [0.2, 0.8]]])

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "DS")

    def map_data_to_prior(self, k, ann, **kwargs):
        tau_prior_ = self.tau_prior(ann, k, 5.)
        pi_prior_ = self.pi_prior(k)

        return {"tau_prior": tau_prior_,
                "pi_prior": pi_prior_}

    def map_data_to_inits(self, ann, k, w, **kwargs):
        pi_prior_ = self.pi_prior(k)
        pi_param_ = np.broadcast_to(pi_prior_ / np.sum(pi_prior_[0]), (w, k, k))
        return {'tau': self.tau_prior(ann, k, alpha=1.),
                'pi': pi_param_}


    # TODO: Implement sampling from this model
    sample_tasks = 0
    '''
    @dataclass
    class DataGenerationParameters(StanMultinomialOptimizeConsensus.DataGenerationParameters):
        pass

    def sample_tasks(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None) \
            -> Tuple[int, Optional[np.ndarray]]:
        model = self.sample_tasks_model()
        sample = model.sample(data={'t': dgp.n_tasks, 'k': parameters.tau.shape[0]},
                              inits=dataclasses.asdict(parameters),
                              fixed_param=True,
                              iter_sampling=1)
        t_C = sample.stan_variable('t_C').to_numpy(dtype=int)[0] - 1
        return dgp.n_tasks, t_C

    def sample_workers(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None)\
            -> Tuple[int, Optional[np.ndarray]]:
        return 1, None

    def sample_annotations(self, tasks, workers, dgp: DataGenerationParameters, parameters: Optional[Parameters]=None)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = self.sample_annotations_model()
        sample = model.sample(data={'w': 1,
                                    't': dgp.n_tasks,
                                    'num_annotations_per_task': dgp.num_annotations_per_task,
                                    'k': parameters.tau.shape[0],
                                    't_C': tasks + 1},
                              inits=dataclasses.asdict(parameters),
                              fixed_param=True,
                              iter_sampling=1)
        t_A = sample.stan_variable('t_A').to_numpy(dtype=int)[0] - 1
        w_A = sample.stan_variable('w_A').to_numpy(dtype=int)[0] - 1
        f_A = sample.stan_variable('ann').to_numpy(dtype=int)[0] - 1
        return w_A, t_A, f_A
    '''
class StanDSEtaOptimizeConsensus(StanDSOptimizeConsensus):
    name = "StanDSEtaOptimize"

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "DSEta")

    def map_data_to_prior(self, k, ann, **kwargs):
        tau_prior_ = self.tau_prior(ann, k, 5.)
        min_pi_prior_ = np.zeros((k, k - 1))
        max_pi_prior_ = np.ones((k, k - 1)) * 5

        return {'tau_prior': tau_prior_,
                'min_pi_prior': min_pi_prior_,
                'max_pi_prior': max_pi_prior_}

    def map_data_to_inits(self, ann, k, w, **kwargs):
        return {'tau': self.tau_prior(ann, k, alpha=1.),
                'eta': np.ones((w, k, k - 1))}

    def map_data_to_args(self, **kwargs):
        # args = {"iter": 2000}
        return {'algorithm': 'LBFGS',
                'init_alpha': 0.01,
                'output_dir': "."}


class StanDSEtaHOptimizeConsensus(StanDSOptimizeConsensus):
    name = "StanDSEtaHOptimize"

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "DSEtaH")

    def map_data_to_prior(self, k, ann, **kwargs):
        tau_prior_ = self.tau_prior(ann, k, 5.)
        min_pi_prior_ = np.zeros((k, k - 1))
        max_pi_prior_ = np.ones((k, k - 1)) * 5

        return {'tau_prior': tau_prior_,
                'min_pi_prior': min_pi_prior_,
                'max_pi_prior': max_pi_prior_}

    def map_data_to_inits(self, ann, k, w, **kwargs):
        pi_prior_ = self.pi_prior(k)
        pi_param_ = np.broadcast_to(pi_prior_ / np.sum(pi_prior_[0]), (w, k, k))
        return {'tau': self.tau_prior(ann, k, alpha=1.),
                'eta': np.ones((k, k - 1)),
                'pi': pi_param_}

    def map_data_to_args(self, **kwargs):
        # args = {"iter": 2000}
        return {'algorithm': 'LBFGS',
                'init_alpha': 0.01,
                'output_dir': "."}
