import numpy as np

from . import log

from .consensus import GenerativeAbstractConsensus, DiscreteConsensusProblem, \
    DataGenerationParameters as DGP, Parameters as AbstractParameters
from .probabilistic import Probabilistic
from dataclasses import dataclass
from typing import Optional, Tuple

class DawidSkene(GenerativeAbstractConsensus):

    name = "DawidSkene"

    @dataclass
    class Parameters(AbstractParameters):
        tau: np.ndarray = np.array([0.5, 0.5])
        pi: np.ndarray = np.array([[0.9, 0.1], [0.2, 0.8]])

    def __init__(self):
        self.n = None
        self.T = None
        self.tau = None
        self.logpi = None

    def fit_and_compute_consensus(self, dcp: DiscreteConsensusProblem, max_iterations=10000, tolerance=1e-3, prior=1.0, verbose=False, init_params=None):
        self.n = dcp.compute_n()
        # First estimate of T_{i,j} is done by probabilistic consensus
        if init_params is not None:
            self.tau = init_params.tau
            self.logpi = np.log(init_params.pi)
            self.T, _ = Probabilistic.fit_and_compute_consensus(dcp, softening=prior)
        else:
            self.T, _ = Probabilistic.fit_and_compute_consensus(dcp, softening=prior)
            # Initialize the percentages of each label
            self.tau = self._m_step_p(self.T, prior)

            # Initialize the errors
            self.logpi = self._m_step_logpi(self.T, self.n, prior)

        has_converged = False
        num_iterations = 0
        any_nan = (np.isnan(self.T).any() or np.isnan(self.tau).any() or np.isnan(self.logpi).any())
        while (num_iterations < max_iterations and not has_converged) and not any_nan:
            # Expectation step
            old_T = self.T
            self.T = self._e_step(self.n, self.logpi, self.tau)
            # Maximization
            self.tau, self.logpi = self._m_step(self.T, self.n, prior)
            has_converged = np.allclose(old_T, self.T, atol=tolerance)
            num_iterations += 1
            any_nan = (np.isnan(self.T).any() or np.isnan(self.tau).any() or np.isnan(self.logpi).any())
        if any_nan:
            log.info("NaN values detected")
        elif has_converged:
            log.info("DS has converged in %i iterations", num_iterations)
        else:
            log.info("The maximum of %i iterations has been reached", max_iterations)

        return self.T, self.Parameters(tau=self.tau,pi=np.exp(self.logpi))

    def fit(self,  dcp:DiscreteConsensusProblem, T, prior=1.0):
        n = dcp.compute_n()
        tau, logpi = self._m_step(T, n, prior)
        return self.Parameters(tau=tau,pi=np.exp(logpi))

    def compute_consensus(self, dcp:DiscreteConsensusProblem, parameters: Parameters):
        n = dcp.compute_n()
        return self._e_step(n, np.log(parameters.pi), parameters.tau)

    #def get_dimensions(self, parameters:Parameters):
    #    return parameters.tau.shape[0], parameters.pi.shape[0], parameters.pi.shape[2]

    # Methods from GenerativeAbstractConsensus
    @dataclass
    class DataGenerationParameters(DGP):
        n_tasks: int = 10
        num_annotations_per_task: int = 2

        def __post_init__(self):
            self.n_annotations = self.n_tasks * self.num_annotations_per_task

    def sample_tasks(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None):
        if parameters is None:
            parameters = self.Parameters()
        n_real_labels = len(parameters.tau)  # number of real labels
        # Sample the real labels
        return dgp.n_tasks, np.random.choice(n_real_labels, size=dgp.n_tasks, p=parameters.tau)

    def sample_workers(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None) \
            -> Tuple[int, Optional[np.ndarray]]:
        return parameters.pi.shape[0], None

    def sample_annotations(self, tasks, workers, dgp: DataGenerationParameters, parameters: Optional[Parameters]=None):
        if parameters is None:
            parameters = self.Parameters()
        n_labels = parameters.pi.shape[2]
        n_workers = parameters.pi.shape[0]  #
        # Sample the annotators
        annotators = np.random.choice(n_workers, size=(dgp.n_tasks, dgp.num_annotations_per_task))
        labels_and_annotators = annotators + tasks[:, np.newaxis] * n_workers
        labels_and_annotators = labels_and_annotators.flatten()
        unique_la, inverse_la, counts_la = np.unique(labels_and_annotators, return_inverse=True, return_counts=True)
        w_A = np.zeros(dgp.n_annotations, dtype=np.int32)
        f_A = np.zeros(dgp.n_annotations, dtype=np.int32)
        t_A = np.arange(dgp.n_annotations, dtype=np.int32) // dgp.num_annotations_per_task
        for i_la, label_and_annotator in enumerate(unique_la):
            real_label = label_and_annotator // n_workers
            annotator_index = label_and_annotator % n_workers
            emission_p = parameters.pi[annotator_index, real_label]
            emitted_labels = np.random.choice(n_labels, size=counts_la[i_la], p=emission_p)
            ca_indexes = np.equal(inverse_la, i_la)
            w_A[ca_indexes] = annotator_index
            f_A[ca_indexes] = emitted_labels
        return w_A, t_A, f_A

    # Private methods

    def _m_step(self, T, n, prior):
        return (self._m_step_p(T, prior), self._m_step_logpi(T, n, prior))

    def _m_step_p(self, T, prior):
        p = np.sum(T, axis=0)
        p += prior
        p /= np.sum(p)
        return p

    def _m_step_logpi(self, T, n, prior):
        log.debug(T.shape)
        log.debug(n.shape)
        _pi = np.swapaxes(np.dot(T.transpose(), n), 0, 1)
        _pi += prior
        sums = np.sum(_pi, axis=2)
        _pi /= sums[:, :, np.newaxis]
        return np.log(_pi)

    def _e_step(self, n, logpi, tau):
        logT = np.tensordot(n, logpi, axes=([0, 2], [0, 2]))
        logT += np.log(tau[np.newaxis, :])
        maxLogT = np.max(logT)
        logSumT = np.log(np.sum(np.exp(logT-maxLogT), axis=1)) + maxLogT
        logT -= logSumT[:,np.newaxis]
        T = np.exp(logT)
        return T