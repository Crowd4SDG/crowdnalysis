import numpy as np

from crowdnalysis.cmdstan import AbstractStanOptimizeConsensus, resource_filename
from crowdnalysis.cmdstan.multinomial import StanMultinomialOptimizeConsensus, StanMultinomialEtaOptimizeConsensus
from scipy.special import softmax
from cmdstanpy import CmdStanModel
from dataclasses import dataclass
from ..factory import Factory

class StanDSOptimizeConsensus(StanMultinomialOptimizeConsensus):
    name = "StanDSOptimize"

    @dataclass
    class Parameters(AbstractStanOptimizeConsensus.Parameters):
        tau: np.ndarray = np.array([0.5, 0.5])
        pi: np.ndarray = np.array([[[0.9, 0.1], [0.2, 0.8]]])

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "DS")

    def map_data_to_prior(self, dcp, k, l, classes, ann, **kwargs):
        tau_prior_ = self.tau_prior(dcp, ann, k, classes, 5.)
        pi_prior_ = self.pi_prior(k, l, classes)
        return {"tau_prior": tau_prior_,
                "pi_prior": pi_prior_}

    def map_data_to_inits(self, dcp, ann, k, w, l, classes, **kwargs):
        alg = Factory.make("StanMultinomialEtaOptimize")
        t_C, params = alg.fit_and_compute_consensus(dcp)
        tau_ = params.tau
        multinomial_pi_ = params.pi
        pi_param_ = np.broadcast_to(multinomial_pi_, (w, k, l))
        return {'tau': tau_,
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
                                    'n_annotations_per_task': dgp.n_annotations_per_task,
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

# class StanDSEtaOptimizeConsensus(StanDSOptimizeConsensus):
#     name = "StanDSEtaOptimize"
#
#     def __init__(self):
#         AbstractStanOptimizeConsensus.__init__(self, "DSEta")
#
#     def map_data_to_prior(self, k, ann, **kwargs):
#         tau_prior_ = self.tau_prior(ann, k, 5.)
#         min_pi_prior_ = np.zeros((k, k - 1))
#         max_pi_prior_ = np.ones((k, k - 1)) * 10
#
#         return {'tau_prior': tau_prior_,
#                 'min_pi_prior': min_pi_prior_,
#                 'max_pi_prior': max_pi_prior_}
#
#     def map_data_to_inits(self, ann, k, w, **kwargs):
#         return {'tau': self.tau_prior(ann, k, alpha=1.),
#                 'eta': np.ones((w, k, k - 1))}
#
#     def map_data_to_args(self, **kwargs):
#         # args = {"iter": 2000}
#         return {'algorithm': 'LBFGS',
#                 'init_alpha': 0.01,
#                 'output_dir': "."}
#

class StanDSEtaHOptimizeConsensus(StanDSOptimizeConsensus):
    name = "StanDSEtaHOptimize"

    def __init__(self):
        AbstractStanOptimizeConsensus.__init__(self, "DSEtaH")

    def map_data_to_prior(self, dcp, k, l, ann, classes, **kwargs):
        tau_prior_ = self.tau_prior(dcp, ann, k, classes, alpha=5.)
        eta_alpha_prior_ = np.ones((k, l - 1))
        eta_beta_prior_ = np.ones((k, l - 1)) * 0.01
        return {'tau_prior': tau_prior_,
                'eta_alpha_prior': eta_alpha_prior_,
                'eta_beta_prior': eta_beta_prior_}

    def compute_movements(self, eta, classes):
        k, l = eta.shape
        l += 1
        dst = np.zeros((k,l-1), dtype=np.int16)
        for _k in range(k):
            for _l in range(l-1):
                dst[_k][_l] = _l + (_l >= (classes[_k]-1));
        return dst

    def softmax_diag(self, eta, classes):
        dst = self.compute_movements(eta, classes)
        k, l = eta.shape
        l += 1
        pi = np.zeros((k, l))
        for _k in range(k):
            pi[_k][dst[_k]] = -eta[_k]
            pi[_k] = softmax(pi[_k])
        return pi

    def map_data_to_inits(self, dcp, ann, k, w, l, classes, **kwargs):
        eta = np.ones((k, l - 1))*5.
        print("eta:", eta)
        pi_prior_ = self.pi_prior(k, l, classes)
        print("old_pi_prior:", pi_prior_)
        pi_prior_ = self.softmax_diag(eta, classes)
        print("pi_prior:", pi_prior_)
        pi_param_ = np.broadcast_to(pi_prior_ / np.sum(pi_prior_[0]), (w, k, l))
        print("param:", pi_param_)
        ti = self.tau_init(dcp, ann, k, classes)
        print("tau_init =", ti)
        return {'tau': ti,
                'eta': eta,
                'pi': pi_param_}

    def map_data_to_args(self, dcp, **kwargs):
        # args = {"iter": 2000}
        return {'algorithm': 'LBFGS',
                #'init_alpha': 0.01,
                'sig_figs': 10,
                'output_dir': "."}

    def compute_consensus_model(self):
        return CmdStanModel(stan_file=resource_filename(StanDSOptimizeConsensus().model_name
                                                        + ".consensus.stan"))

Factory.register_consensus_algorithm(StanDSOptimizeConsensus)
# Factory.register_consensus_algorithm(StanDSEtaOptimizeConsensus)
Factory.register_consensus_algorithm(StanDSEtaHOptimizeConsensus)