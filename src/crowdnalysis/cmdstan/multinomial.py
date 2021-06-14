import dataclasses
from .. import log
from typing import List, Tuple, Optional

import numpy as np
from cmdstanpy import CmdStanModel

from ..consensus import GenerativeAbstractConsensus
from dataclasses import dataclass
from ..factory import Factory

from . import AbstractStanOptimizeConsensus
from . import resource_filename

class StanMultinomialOptimizeConsensus(AbstractStanOptimizeConsensus):
    name = "StanMultinomialOptimize"

    @dataclass
    class Parameters(AbstractStanOptimizeConsensus.Parameters):
        tau: np.ndarray = np.array([0.5, 0.5])
        pi: np.ndarray = np.array([[0.9, 0.1], [0.2, 0.8]])

    @dataclass
    class DataGenerationParameters(GenerativeAbstractConsensus.DataGenerationParameters):
        n_tasks: int = 10
        n_annotations_per_task: int = 2

        def __post_init__(self):
            self.n_annotations = self.n_tasks * self.n_annotations_per_task

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "Multinomial")

    def tau_estimate(self, dcp, ann, k, classes):
        alg = Factory.make("MajorityVoting")
        t_C, _ = alg.fit_and_compute_consensus(dcp)
        single_most_voted = np.argmax(t_C, axis=1)
        unique, counts = np.unique(single_most_voted, return_counts=True)
        # print("u, c:", unique, counts)
        return counts / np.sum(counts)

    def tau_prior(self, dcp, ann, k, classes, alpha=5., beta=5.):
        return alpha * self.tau_estimate(dcp, ann, k, classes) + beta

    def tau_init(self, dcp, ann, k, classes, alpha=0.99):
        beta = (1 - alpha) / k
        return alpha * self.tau_estimate(dcp, ann, k, classes) + beta

    def pi_prior(self, k, l, classes, alpha=1., beta=10):
        labels_which_are_classes = np.zeros((k, l))
        for k_, l_ in enumerate(classes):
            labels_which_are_classes[k_, l_-1] = 1.
        return np.ones((k, l)) * alpha + labels_which_are_classes * beta

    def map_data_to_prior(self, dcp, k, l, classes, ann, **kwargs):
        tau_prior_ = self.tau_prior(dcp, ann, k, classes, alpha=5.)+5.
        #print("tp:", tau_prior_)
        pi_prior_ = self.pi_prior(k, l, classes, alpha=5, beta=5)

        return {"tau_prior": tau_prior_,
                "pi_prior": pi_prior_}

    def map_data_to_inits(self, dcp, ann, k, l, classes, **kwargs):
        pi_prior_ = self.pi_prior(k, l, classes)
        return {'tau': self.tau_init(dcp, ann, k, classes),
                'pi': (pi_prior_ / np.sum(pi_prior_[0]))}

    def map_data_to_args(self, dcp, **kwargs):
        # args = {"iter": 2000}
        return {'algorithm': 'LBFGS',
                'sig_figs': 10,
                #'init_alpha': 0.001
                 'output_dir': "."
                }

    def sample_tasks(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None) \
            -> Tuple[int, Optional[np.ndarray]]:
        model = self.sample_tasks_model()
        sample = model.sample(data={'t': dgp.n_tasks, 'k': parameters.tau.shape[0], 'l': parameters.pi.shape[1]},
                              inits=dataclasses.asdict(parameters),
                              fixed_param=True,
                              iter_sampling=1)
        t_C = sample.stan_variable('t_C').astype(int)[0] - 1
        log.debug("Tasks type: %s", type(t_C.dtype))
        return dgp.n_tasks, t_C

    def sample_workers(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None)\
            -> Tuple[int, Optional[np.ndarray]]:
        return 1, None

    def sample_annotations(self, tasks, workers, dgp: DataGenerationParameters,
                           parameters: Optional[Parameters] = None)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        model = self.sample_annotations_model()
        sample = model.sample(data={'w': 1,
                                    't': dgp.n_tasks,
                                    'n_annotations_per_task': dgp.n_annotations_per_task,
                                    'k': parameters.tau.shape[0],
                                    'l': parameters.pi.shape[1],
                                    't_C': tasks + 1},
                              inits=dataclasses.asdict(parameters),
                              fixed_param=True,
                              iter_sampling=1)
        t_A = sample.stan_variable('t_A').astype(int)[0]
        t_A -= 1
        w_A = sample.stan_variable('w_A').astype(int)[0]
        log.debug(type(w_A.dtype))
        w_A -= 1
        log.debug(type(w_A.dtype))
        f_A = sample.stan_variable('ann').astype(int)[0]
        f_A -= 1

        return w_A, t_A, f_A, list(range(parameters.tau.shape[0]))


class StanMultinomialEtaOptimizeConsensus(StanMultinomialOptimizeConsensus):
    name = "StanMultinomialEtaOptimize"

    @dataclass
    class Parameters(AbstractStanOptimizeConsensus.Parameters):
        tau: np.ndarray = np.array([0.5, 0.5])
        eta: np.ndarray = np.array([[0.9], [0.8]])
        pi: np.ndarray = np.array([[0.9, 0.1], [0.2, 0.8]])

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "MultinomialEta")

    def map_data_to_prior(self, dcp, k, l, ann, classes, **kwargs):
        tau_prior_ = self.tau_prior(dcp, ann, k, classes, alpha=5.)
        eta_alpha_prior_ = np.ones((k, l - 1))
        eta_beta_prior_ = np.ones((k, l - 1)) * 0.01
        return {'tau_prior': tau_prior_,
                'eta_alpha_prior': eta_alpha_prior_,
                'eta_beta_prior': eta_beta_prior_}

    def map_data_to_inits(self, dcp, ann, k, l, classes, **kwargs):

        return {'tau': self.tau_init(dcp, ann, k, classes),
                'eta': np.ones((k, l - 1))}

    def compute_consensus_model(self):
        return CmdStanModel(stan_file=resource_filename(StanMultinomialOptimizeConsensus().model_name
                                                        + ".consensus.stan"))
    # def map_data_to_args(self):
    #    #args = {"iter": 2000}
    #    return {'algorithm': 'LBFGS',
    #            'output_dir': "."}

Factory.register_consensus_algorithm(StanMultinomialOptimizeConsensus)
Factory.register_consensus_algorithm(StanMultinomialEtaOptimizeConsensus)