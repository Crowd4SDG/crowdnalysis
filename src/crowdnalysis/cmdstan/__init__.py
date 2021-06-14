import ast
import dataclasses
from .. import log
from importlib.resources import path as irpath
from typing import Dict, Tuple

import numpy as np
from cmdstanpy import CmdStanModel, CmdStanMLE

from ..consensus import GenerativeAbstractConsensus, DiscreteConsensusProblem


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


with irpath(__package__, "__init__.py") as path_:
    resource_dir = __name__.split('.')[-1]  # directory name for the *.stan files
    resources_path = path_.parents[0] / resource_dir


def resource_filename(filename):
    return resources_path / filename


class AbstractStanOptimizeConsensus(GenerativeAbstractConsensus):
    name = "AbstractStanOptimizeConsensus"

    def __init__(self, model_name):
        self.model_name = model_name
        # self.hidden_variables = []

    def map_data_to_model(self, dcp: DiscreteConsensusProblem, **kwargs):
        d = self.map_data_to_data(dcp, **kwargs)
        init = self.map_data_to_inits(dcp, **d, **kwargs)
        args = self.map_data_to_args(dcp, **d, **kwargs)
        return d, init, args

    def map_data_to_data(self, dcp: DiscreteConsensusProblem, **kwargs):
        stan_data = {'w': dcp.n_workers,
                     't': dcp.n_tasks,
                     'a': dcp.n_annotations,
                     'k': len(dcp.classes),
                     'l': dcp.n_labels,
                     'classes': (np.array(dcp.classes) + 1),
                     't_A': (dcp.t_A + 1),
                     'w_A': (dcp.w_A + 1),
                     'ann': (dcp.f_A + 1).flatten()}
        prior = self.map_data_to_prior(dcp, **stan_data, **kwargs)
        log.debug(prior)
        stan_data.update(prior)
        return stan_data

    def map_data_to_prior(self, dcp, **kwargs):
        raise NotImplementedError

    def map_data_to_inits(self, dcp,  **kwargs):
        raise NotImplementedError

    def map_data_to_args(self, dcp, **kwargs):
        raise NotImplementedError

    # def MLE_parameters(self, results: CmdStanMLE):
    #     return {var_name: results.optimized_params_dict[var_name] for var_name in self.hidden_variables}

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
        # stan_data = {}
        # init_data = {}
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

    def compute_consensus(self, dcp: DiscreteConsensusProblem, parameters: GenerativeAbstractConsensus.Parameters,
                          **kwargs):
        # print(kwargs["data"])
        stan_data, init_data, kwargs_opt = self.map_data_to_model(dcp)
        # print("stan_data:", stan_data)
        # print("kwargs['data']:", ast.literal_eval(kwargs["data"]))
        # stan_data.update(ast.literal_eval(kwargs["data"]))
        stan_data.update(ast.literal_eval(parameters.to_json()))

        model = self.compute_consensus_model()
        results = model.optimize(data=stan_data, inits=init_data, **kwargs_opt)

        return _get_var_dict(results)['t_C'], kwargs_opt

    def sample_tasks_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".sample_tasks.stan"))

    def sample_annotations_model(self):
        return CmdStanModel(stan_file=resource_filename(self.model_name + ".sample_annotations.stan"))
