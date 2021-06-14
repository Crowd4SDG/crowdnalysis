import numpy as np

from . import log

from .consensus import GenerativeAbstractConsensus, DiscreteConsensusProblem
from .simple import Probabilistic
from .factory import Factory
from dataclasses import dataclass
from typing import Optional, Tuple


class DawidSkene(GenerativeAbstractConsensus):

    name = "DawidSkene"

    @dataclass
    class Parameters(GenerativeAbstractConsensus.Parameters):
        tau: np.ndarray = np.array([0.5, 0.5])
        pi: np.ndarray = np.array([[[0.9, 0.1], [0.2, 0.8]]])

    @classmethod
    def fit_and_compute_consensus(cls, dcp: DiscreteConsensusProblem, max_iterations=10000,
                                  tolerance=1e-3, prior=1.0, verbose=False, init_params=None):
        n, _ = dcp.compute_n()
        # First estimate of T_{i,j} is done by probabilistic consensus
        if init_params is not None:
            tau = init_params.tau
            log_pi = np.log(init_params.pi)
            T, _ = Probabilistic.fit_and_compute_consensus(dcp, softening=prior)
        else:
            T, _ = Probabilistic.fit_and_compute_consensus(dcp, softening=prior)
            # Initialize the percentages of each label
            tau = cls._m_step_p(T, prior)

            # Initialize the errors
            log_pi = cls._m_step_log_pi(T, n, prior)

        has_converged = False
        num_iterations = 0
        any_nan = (np.isnan(T).any() or np.isnan(tau).any() or np.isnan(log_pi).any())
        while (num_iterations < max_iterations and not has_converged) and not any_nan:
            # Expectation step
            old_T = T
            T = cls._e_step(n, log_pi, tau)
            # Maximization
            tau, log_pi = cls._m_step(T, n, prior)
            has_converged = np.allclose(old_T, T, atol=tolerance)
            num_iterations += 1
            any_nan = (np.isnan(T).any() or np.isnan(tau).any() or np.isnan(log_pi).any())
        if any_nan:
            log.info("NaN values detected")
        elif has_converged:
            log.info("DS has converged in %i iterations", num_iterations)
        else:
            log.info("The maximum of %i iterations has been reached", max_iterations)

        return T, cls.Parameters(tau=tau, pi=np.exp(log_pi))

    @classmethod
    def fit(cls, dcp: DiscreteConsensusProblem, T, prior=1.0):
        n, _ = dcp.compute_n()
        tau, log_pi = cls._m_step(T, n, prior)
        return cls.Parameters(tau=tau, pi=np.exp(log_pi))

    @classmethod
    def compute_consensus(cls, dcp: DiscreteConsensusProblem, parameters: Parameters):
        n, _ = dcp.compute_n()
        return cls._e_step(n, np.log(parameters.pi), parameters.tau)

    # Methods from GenerativeAbstractConsensus
    @dataclass
    class DataGenerationParameters(GenerativeAbstractConsensus.DataGenerationParameters):
        n_tasks: int = 10
        n_annotations_per_task: int = 2

        def __post_init__(self):
            self.n_annotations = self.n_tasks * self.n_annotations_per_task

    def get_dimensions(self, parameters: Parameters):
        return parameters.tau.shape[0], parameters.pi.shape[0], parameters.pi.shape[2]

    def sample_tasks(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None):
        if parameters is None:
            parameters = self.Parameters()
        n_classes = len(parameters.tau)  # number of real labels
        # Sample the real labels
        return dgp.n_tasks, np.random.choice(n_classes, size=dgp.n_tasks, p=parameters.tau)

    def sample_workers(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None) \
            -> Tuple[int, Optional[np.ndarray]]:
        return parameters.pi.shape[0], None

    def sample_annotations(self, tasks, workers, dgp: DataGenerationParameters,
                           parameters: Optional[Parameters] = None):
        if parameters is None:
            parameters = self.Parameters()
        n_workers, n_classes, n_labels = parameters.pi.shape
        # Sample the annotators
        annotators = np.random.choice(n_workers, size=(dgp.n_tasks, dgp.n_annotations_per_task))
        labels_and_annotators = annotators + tasks[:, np.newaxis] * n_workers
        labels_and_annotators = labels_and_annotators.flatten()
        unique_la, inverse_la, counts_la = np.unique(labels_and_annotators, return_inverse=True, return_counts=True)
        w_A = np.zeros(dgp.n_annotations, dtype=np.int32)
        f_A = np.zeros(dgp.n_annotations, dtype=np.int32)
        t_A = np.arange(dgp.n_annotations, dtype=np.int32) // dgp.n_annotations_per_task
        for i_la, label_and_annotator in enumerate(unique_la):
            real_label = label_and_annotator // n_workers
            annotator_index = label_and_annotator % n_workers
            emission_p = parameters.pi[annotator_index, real_label]
            emitted_labels = np.random.choice(n_labels, size=counts_la[i_la], p=emission_p)
            ca_indexes = np.equal(inverse_la, i_la)
            w_A[ca_indexes] = annotator_index
            f_A[ca_indexes] = emitted_labels
        classes = list(range(n_classes))
        return w_A, t_A, f_A, classes

    # Private methods

    @classmethod
    def _m_step(cls, T, n, prior):
        return cls._m_step_p(T, prior), cls._m_step_log_pi(T, n, prior)

    @classmethod
    def _m_step_p(cls, T, prior):
        p = np.sum(T, axis=0)
        p += prior
        p /= np.sum(p)
        return p

    @classmethod
    def _m_step_log_pi(cls, T, n, prior):
        log.debug("_m_step_log_pi -> T.shape: {}".format(str(T.shape)))
        log.debug("_m_step_log_pi -> n.shape: {}".format(str(n.shape)))
        _pi = np.swapaxes(np.dot(T.transpose(), n), 0, 1)
        _pi += prior
        sums = np.sum(_pi, axis=2)
        _pi /= sums[:, :, np.newaxis]
        return np.log(_pi)

    @classmethod
    def _e_step(cls, n, log_pi, tau):
        log_T = np.tensordot(n, log_pi, axes=([0, 2], [0, 2]))
        log_T += np.log(tau[np.newaxis, :])
        maxLogT = np.max(log_T)
        logSumT = np.log(np.sum(np.exp(log_T-maxLogT), axis=1)) + maxLogT
        log_T -= logSumT[:, np.newaxis]
        T = np.exp(log_T)
        return T

Factory.register_consensus_algorithm(DawidSkene)