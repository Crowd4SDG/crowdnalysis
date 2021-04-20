import numpy as np

from . import log
from .consensus import GenerativeAbstractConsensus, DiscreteConsensusProblem
from .probabilistic import Probabilistic
from .common import vprint


class DawidSkene(GenerativeAbstractConsensus):

    name = "DawidSkene"

    def __init__(self):
        self.n = None
        self.T = None
        self.tau = None
        self.logpi = None

    def fit_and_compute_consensus(self, dcp: DiscreteConsensusProblem, max_iterations=10000, tolerance=1e-3, prior=1.0, verbose=False, init_params=None):

        self.n = dcp.compute_n()
        # ("n:\n{}", self.n)
        # print("First estimate of T ({}) by probabilistic consensus:\n".format(str(self.T.shape)), self.T)
        # First estimate of T_{i,j} is done by probabilistic consensus
        if init_params is not None:
            self.tau, _pi = self._get_parameters_from_dict(init_params)
            self.logpi = np.log(_pi)
            self.T, _ = Probabilistic.probabilistic_consensus(m, n_tasks, n_labels, softening=prior)
        else:
            self.T, _ = Probabilistic.probabilistic_consensus(m, n_tasks, n_labels, softening=prior)

            # Initialize the percentages of each label
            self.tau = self._m_step_p(self.T, prior)
            # print("Initial percentages ({}) of each label:\n".format(str(self.tau.shape)), self.tau)

            #print("p=", self.tau)

            # Initialize the errors
            # _pi[k,j,l] (KxJxJ)
            self.logpi = self._m_step_logpi(self.T, self.n, prior)
            #self.logpi = np.log(self.taui)
            # print("Initial errors ({}):\n".format(str(np.exp(self.logpi).shape)), np.exp(self.logpi))

            #print("pi=", np.exp(self.logpi))

        has_converged = False
        num_iterations = 0
        any_nan = (np.isnan(self.T).any() or np.isnan(self.tau).any() or np.isnan(self.logpi).any())
        while (num_iterations < max_iterations and not has_converged) and not any_nan:
            # Expectation step
            old_T = self.T
            self.T = self._e_step(self.n, self.logpi, self.tau)
            # print("T=", self.T)
            # Maximization
            self.tau, self.logpi = self._m_step(self.T, self.n, prior)
            #print("tau=", self.tau)
            #print("pi=", np.exp(self.logpi))
            has_converged = np.allclose(old_T, self.T, atol=tolerance)
            num_iterations += 1
            any_nan = (np.isnan(self.T).any() or np.isnan(self.tau).any() or np.isnan(self.logpi).any())
        if any_nan:
            vprint("NaN values detected", verbose=verbose)
        elif has_converged:
            vprint("DS has converged in", num_iterations, "iterations", verbose=verbose)
            #print("tau=", self.tau)
            #print("pi=", np.exp(self.logpi))
            #print("T=", self.T)
        else:
            vprint("The maximum of", max_iterations, "iterations has been reached", verbose=verbose     )
        # print("\np {}:\n{}, \npi {}:\n{},\nT {}:\n{}".format(self.tau.shape, self.tau, self.logpi.shape, np.exp(self.logpi), self.T.shape, self.T))
        
        return self.T, self._make_parameter_dict()

    def fit(self, m, n_tasks, n_labels, n_annotators, T, prior=1.0):
        n = self._compute_n(m, n_tasks, n_labels, n_annotators)
        tau, logpi = self._m_step(T, n, prior)
        return self._make_parameter_dict(tau, logpi)

    def compute_consensus(self, m, n_tasks, n_labels, n_annotators, parameters):
        n = self._compute_n(m, n_tasks, n_labels, n_annotators)
        tau, _pi = self._get_parameters_from_dict(parameters)
        return self._e_step(n, np.log(_pi), tau)

    def get_dimensions(self, parameters):
        tau, _pi = self._get_parameters_from_dict(parameters)
        return len(tau), _pi.shape[0], _pi.shape[2]

    # Methods from GenerativeAbstractConsensus

    def sample_tasks(self, n_tasks, parameters=None):
        """

        Args:
            n_tasks: number of tasks
            num_annotators: number of annotators

        Returns:
            Tuple[numpy.ndarray]:
        """

        p, _pi = self._get_parameters_from_dict(parameters)
        n_real_labels = len(p)  # number of real labels
        # Sample the real labels
        return np.random.choice(n_real_labels, size=n_tasks, p=p)

    def sample_annotations(self, real_labels, num_annotations_per_task, parameters=None):
        """

        Args:
            real_labels (numpy.ndarray): 1D array with dimension (n_tasks)
            num_annotations_per_task (int):number of annotations per task

        Returns:
            numpy.ndarray: 2D array with dimensions (n_tasks * num_annotations_per_task, 3)

        """
        tau, _pi = self._get_parameters_from_dict(parameters)
        #print("pi", _pi)
        n_tasks = real_labels.shape[0]  # number of tasks
        n_labels = _pi.shape[2]
        n_annotators = _pi.shape[0]  #
        # Sample the annotators
        # print("Generating crowd labels n_tasks: {}, n_labels: {}, n_annotators: {}".format(n_tasks, n_labels, n_annotators))
        annotators = np.random.choice(n_annotators, size=(n_tasks, num_annotations_per_task))
        labels_and_annotators = annotators + real_labels[:, np.newaxis] * n_annotators
        labels_and_annotators = labels_and_annotators.flatten()
        unique_la, inverse_la, counts_la = np.unique(labels_and_annotators, return_inverse=True, return_counts=True)
        # print(inverse_la.shape)
        # print(inverse_la)
        crowd_labels = np.zeros((n_tasks * num_annotations_per_task, 3), dtype=np.int32)
        crowd_labels[:, 0] = np.arange(n_tasks * num_annotations_per_task) // num_annotations_per_task
        # crowd_labels.flatten()
        for i_la, label_and_annotator in enumerate(unique_la):
            real_label = label_and_annotator // n_annotators
            annotator_index = label_and_annotator % n_annotators
            # print("Real_label:", real_label)
            # print("Annotator:", annotator_index)
            emission_p = _pi[annotator_index, real_label]
            # print("i_la:", i_la)
            # print("counts:", counts_la[i_la])
            emitted_labels = np.random.choice(n_labels, size=counts_la[i_la], p=emission_p)
            ca_indexes = np.equal(inverse_la, i_la)
            # print(ca_indexes.shape)
            # print(ca_indexes)
            crowd_labels[:, 1][ca_indexes] = annotator_index
            crowd_labels[:, 2][ca_indexes] = emitted_labels
        return crowd_labels

    # Private methods

    def _get_parameters_from_dict(self, parameters):
        if parameters is None:
            tau = self.tau
            _pi = np.exp(self.logpi)
        else:
            tau, _pi = parameters["tau"], parameters["pi"]
        return tau, _pi

    def _make_parameter_dict(self, tau=None, logpi=None):
        if tau is None:
            tau = self.tau
        if logpi is None:
            logpi = self.logpi
        return {"tau": tau, "pi": np.exp(logpi)}

    def _m_step(self, T, n, prior):
        return (self._m_step_p(T, prior), self._m_step_logpi(T, n, prior))

    def _m_step_p(self, T, prior):
        p = np.sum(T, axis=0)
        p += prior
        p /= np.sum(p)
        return p

    def _m_step_logpi(self, T, n, prior):
        _pi = np.swapaxes(np.dot(T.transpose(), n), 0, 1)
        _pi += prior
        sums = np.sum(_pi, axis=2)
        _pi /= sums[:, :, np.newaxis]
        return np.log(_pi)

    def _e_step(self, n, logpi, tau):
        # print("_e_step > n:\n{}\n logpi:\n{}\n p:\n{}".format(n, logpi, p))
        logT = np.tensordot(n, logpi, axes=([0, 2], [0, 2]))
        logT += np.log(tau[np.newaxis, :])
        maxLogT = np.max(logT)
        logSumT = np.log(np.sum(np.exp(logT-maxLogT), axis=1)) + maxLogT
        logT -= logSumT[:,np.newaxis]
        T = np.exp(logT)
        # print("T=",T)
        return T