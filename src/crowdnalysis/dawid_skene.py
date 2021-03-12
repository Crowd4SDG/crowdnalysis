import numpy as np

from . import consensus
from .probabilistic import Probabilistic
from .common import vprint


class DawidSkene(consensus.GenerativeAbstractConsensus):

    name = "DawidSkene"

    def __init__(self):
        self.n = None
        self.T = None
        self.p = None
        self.logpi = None

    def m_fit_and_compute_consensus(self, m, I, J, K, max_iterations=10000, tolerance=1e-3, prior=1.0, verbose=False):
        self.n = self._compute_n(m, I, J, K)
        # ("n:\n{}", self.n)
        # print("First estimate of T ({}) by probabilistic consensus:\n".format(str(self.T.shape)), self.T)
        # First estimate of T_{i,j} is done by probabilistic consensus
        self.T, _ = Probabilistic.probabilistic_consensus(m, I, J, softening=prior)

        # Initialize the percentages of each label
        self.p = self._m_step_p(self.T, prior)
        # print("Initial percentages ({}) of each label:\n".format(str(self.p.shape)), self.p)

        #print("p=", self.p)

        # Initialize the errors
        # _pi[k,j,l] (KxJxJ)
        self.logpi = self._m_step_logpi(self.T, self.n, prior)
        #self.logpi = np.log(self.pi)
        # print("Initial errors ({}):\n".format(str(np.exp(self.logpi).shape)), np.exp(self.logpi))

        #print("pi=", np.exp(self.logpi))

        has_converged = False
        num_iterations = 0
        any_nan = (np.isnan(self.T).any() or np.isnan(self.p).any() or np.isnan(self.logpi).any())
        while (num_iterations < max_iterations and not has_converged) and not any_nan:
            # Expectation step
            old_T = self.T
            self.T = self._e_step(self.n, self.logpi, self.p)
            # print("T=", self.T)
            # Maximization
            self.p, self.logpi = self._m_step(self.T, self.n, prior)
            # print("p=", self.p)
            # print("pi=", np.exp(self.logpi))
            has_converged = np.allclose(old_T, self.T, atol=tolerance)
            num_iterations += 1
            any_nan = (np.isnan(self.T).any() or np.isnan(self.p).any() or np.isnan(self.logpi).any())
        if any_nan:
            vprint("NaN values detected", verbose=verbose)
        elif has_converged:
            vprint("DS has converged in", num_iterations, "iterations", verbose=verbose)
        else:
            vprint("The maximum of", max_iterations, "iterations has been reached", verbose=verbose)
        # print("\np {}:\n{}, \npi {}:\n{},\nT {}:\n{}".format(self.p.shape, self.p, self.logpi.shape, np.exp(self.logpi), self.T.shape, self.T))
        
        return self.T, self._make_parameter_dict()

    def m_fit(self, m, I, J, K, T, prior=1.0):
        n = self._compute_n(m, I, J, K)
        p, logpi = self._m_step(T, n, prior)
        return self._make_parameter_dict(p, logpi)

    def m_compute_consensus(self, m, I, J, K, parameters):
        n = self._compute_n(m, I, J, K)
        p, logpi = self._get_parameters_from_dict(parameters)
        return self._e_step(n, logpi, p)

    def get_dimensions(self, parameters):
        p, _pi = self._get_parameters_from_dict(parameters)
        return len(p), _pi.shape[0]

    # Methods from GenerativeAbstractConsensus

    def sample_tasks(self, I, parameters=None):
        """

        Args:
            I: number of tasks
            num_annotators: number of annotators

        Returns:
            Tuple[numpy.ndarray]:
        """

        p, _pi = self._get_parameters_from_dict(parameters)
        J = len(p)  # number of labels
        # Sample the real labels
        return np.random.choice(J, size=I, p=p)

    def sample_annotations(self, real_labels, num_annotations_per_task, parameters=None):
        """

        Args:
            real_labels (numpy.ndarray): 1D array with dimension (I)
            num_annotations_per_task (int):number of annotations per task

        Returns:
            numpy.ndarray: 2D array with dimensions (I * num_annotations_per_task, 3)

        """
        p, _pi = self._get_parameters_from_dict(parameters)
        #print("_pi", _pi)
        I = real_labels.shape[0]  # number of tasks
        J = len(p)
        K = _pi.shape[0]  #
        # Sample the annotators
        # print("Generating crowd labels I: {}, J: {}, K: {}".format(I, J, K))
        annotators = np.random.choice(K, size=(I, num_annotations_per_task))
        labels_and_annotators = annotators + real_labels[:, np.newaxis] * K
        labels_and_annotators = labels_and_annotators.flatten()
        unique_la, inverse_la, counts_la = np.unique(labels_and_annotators, return_inverse=True, return_counts=True)
        # print(inverse_la.shape)
        # print(inverse_la)
        crowd_labels = np.zeros((I * num_annotations_per_task, 3), dtype=np.int32)
        crowd_labels[:, 0] = np.arange(I * num_annotations_per_task) // num_annotations_per_task
        # crowd_labels.flatten()
        for i_la, label_and_annotator in enumerate(unique_la):
            real_label = label_and_annotator // K
            annotator_index = label_and_annotator % K
            # print("Real_label:", real_label)
            # print("Annotator:", annotator_index)
            emission_p = _pi[annotator_index, real_label]
            # print("i_la:", i_la)
            # print("counts:", counts_la[i_la])
            emitted_labels = np.random.choice(J, size=counts_la[i_la], p=emission_p)
            ca_indexes = np.equal(inverse_la, i_la)
            # print(ca_indexes.shape)
            # print(ca_indexes)
            crowd_labels[:, 1][ca_indexes] = annotator_index
            crowd_labels[:, 2][ca_indexes] = emitted_labels
        return crowd_labels

    # Private methods

    def _get_parameters_from_dict(self, parameters):
        if parameters is None:
            p = self.p
            _pi = np.exp(self.logpi)
        else:
            p, _pi = parameters["p"], parameters["_pi"]
        return p, _pi

    def _make_parameter_dict(self, p=None, logpi=None):
        if p is None:
            p = self.p
        if logpi is None:
            logpi = self.logpi
        return {"p": p, "_pi": np.exp(logpi)}

    def _compute_n(self, m, I, J, K):
        # TODO: This should be optimized
        # print(m)
        #N = m.shape[0]

        # print("N=", N, "I=", self.I, "J=", self.J, "K=", self.K)

        # Compute the n matrix

        n = np.zeros((K, I, J))
        for i, k, j in m:
            n[k, i, j] += 1
        return n

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

    def _e_step(self, n, logpi, p):
        # print("_e_step > n:\n{}\n logpi:\n{}\n p:\n{}".format(n, logpi, p))
        T = np.exp(np.tensordot(n, logpi, axes=([0, 2], [0, 2])))  # IxJ
        # print("_e_step > T after tensordot:\n{}", T)
        T *= p[np.newaxis, :]
        # print("_e_step > T after T *= p:\n{}", T)
        T /= np.sum(T, axis=1)[:, np.newaxis]
        # print("_e_step > np.sum(T, axis=1)[:, np.newaxis]):\n{}", np.sum(T, axis=1)[:, np.newaxis])
            # Potential numerical error here.
            # Plan for using smthg similar to the logsumexp trick in the future
        # print("_e_step > T:\n", T)
        return T

