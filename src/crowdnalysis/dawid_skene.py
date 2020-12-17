import numpy as np

from . import consensus
from .data import Data
from .probabilistic import Probabilistic
from .majority_voting import MajorityVoting


class DawidSkene(consensus.AbstractConsensus):

    name = "DawidSkene"

    def __init__(self):
        self.I = None
        self.J = None
        self.K = None
        self.n = None
        self.T = None
        self.p = None
        self.logpi = None

    def fit_and_compute_consensus(self, d, question, **kwargs):
        m, I, J, K = self.get_question_matrix_and_ranges(d, question)
        return self._fit_and_compute_consensus(m , I, J, K, **kwargs)

    def _fit_and_compute_consensus(self, m, I, J, K, max_iterations=10000, tolerance=1e-7, prior=1.0):
        self.I = I
        self.J = J
        self.K = K
        self.n = self._compute_n(m)
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
            print("NaN values detected")
        elif has_converged:
            print("DS has converged in", num_iterations, "iterations")
        else:
            print("The maximum of", max_iterations, "iterations has been reached")
        # print("\np {}:\n{}, \npi {}:\n{},\nT {}:\n{}".format(self.p.shape, self.p, self.logpi.shape, np.exp(self.logpi), self.T.shape, self.T))
        
        return self.T, self._make_parameter_dict()
    
    def fit(self, d: Data, question, T, prior=1.0):
        m, self.I, self.J, self.K = self.get_question_matrix_and_ranges(d, question)
        n = self._compute_n(m)
        p, logpi = self._m_step(T, n, prior)
        return self._make_parameter_dict(p, logpi)

    def compute_consensus(self, d:Data, question, parameters):
        m, self.I, self.J, self.K = self.get_question_matrix_and_ranges(d, question)
        n = self._compute_n(m)
        p, logpi = self._get_parameters_from_dict(parameters)
        return self._e_step(n, logpi, p)

    def _get_parameters_from_dict(self, parameters):
        p, logpi = parameters["p"], parameters["_pi"]
        return p, logpi

    def _make_parameter_dict(self, p=None, logpi=None):
        if p is None:
            p = self.p
        if logpi is None:
            logpi = self.logpi
        return {"p": p, "_pi": np.exp(logpi)}

    def _compute_n(self, m):

        # print(m)
        #N = m.shape[0]

        # print("N=", N, "I=", self.I, "J=", self.J, "K=", self.K)

        # Compute the n matrix

        n = np.zeros((self.K, self.I, self.J))
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

    """
    def sample(self, p, _pi, I, num_annotators):
        # TODO: Consider using pyAgrum
        # _pi = np.exp(logpi)
        J = len(p)
        K = _pi.shape[0]
        # Sample the real labels
        real_labels = np.random.choice(J, size=I, p=p)
        # Sample the annotators
        annotators = np.random.choice(K, size=(I, num_annotators))
        crowd_labels = np.zeros((I * num_annotators, 3), dtype=np.int32)

        for i in range(I):
            for i_a in range(num_annotators):
                annotator_index = annotators[i, i_a]
                task_run_index = i * num_annotators + i_a
                crowd_labels[task_run_index, 0] = i
                crowd_labels[task_run_index, 1] = annotator_index
                crowd_labels[task_run_index, 2] = np.random.choice(J, size=1, p=_pi[annotator_index, real_labels[i]])
        return real_labels, crowd_labels

    def OLD_fast_sample(self, p, _pi, I, num_annotators):
        #print("I:", I)
        #print("Num annotators:", num_annotators)
        J = len(p)
        K = _pi.shape[0]
        # Sample the real labels
        real_labels = np.random.choice(J, size=I, p=p)
        # Sample the annotators
        annotators = np.random.choice(K, size=(I, num_annotators))
        labels_and_annotators = annotators + real_labels[:, np.newaxis] * K
        labels_and_annotators = labels_and_annotators.flatten()
        unique_la, inverse_la, counts_la = np.unique(labels_and_annotators, return_inverse=True, return_counts=True)
        #print(inverse_la.shape)
        #print(inverse_la)
        crowd_labels = np.zeros((I * num_annotators, 3), dtype=np.int32)
        crowd_labels[:, 0] = np.arange(I * num_annotators) // num_annotators
        #crowd_labels.flatten()
        for i_la, label_and_annotator in enumerate(unique_la):
            real_label = label_and_annotator // K
            annotator_index = label_and_annotator % K
            #print("Real_label:", real_label)
            #print("Annotator:", annotator_index)
            emission_p = _pi[annotator_index, real_label]
            #print("i_la:", i_la)
            #print("counts:", counts_la[i_la])
            emitted_labels = np.random.choice(J, size=counts_la[i_la], p=emission_p)
            ca_indexes = np.equal(inverse_la, i_la)
            #print(ca_indexes.shape)
            #print(ca_indexes)
            crowd_labels[:, 1][ca_indexes] = annotator_index
            crowd_labels[:, 2][ca_indexes] = emitted_labels
        return real_labels, crowd_labels

    def OLD_majority_success_rate(self, p, _pi, I, K):
        real_labels, crowd_labels = self.fast_sample(p, _pi, I, K)
        # print("majority_success_rate > crowd_labels K={} ({}):\n".format(K, crowd_labels.shape), crowd_labels)
        c = MajorityVoting.majority_voting(crowd_labels, I, len(p))
        # print("majority_success_rate > majority_voting K={} ({}):\n".format(K, c.shape), c)
        num_successes = np.sum(real_labels == c)
        return num_successes / I

    def OLD_majority_success_rates(self, p, _pi, I, annotators):
        success_p = np.zeros(len(annotators))
        for K in annotators:
            success_p[annotators.index(K)] = self.majority_success_rate(p, _pi, I, K)
        return success_p

    def OLD_DS_consensus_success_rate(self, p, _pi, I, K):
        real_labels, crowd_labels = self.fast_sample(p, _pi, I, K)
        # print(crowd_labels)
        prob_consensus, consensus = self.compute_consensus_with_fixed_parameters(p, _pi, crowd_labels, I)
        num_successes = np.sum(real_labels == consensus)
        return num_successes / I

    def OLD_DS_consensus_success_rates(self, p, _pi, I, annotators):
        success_p = np.zeros(len(annotators))
        for K in annotators:
            success_p[annotators.index(K)] = self.DS_consensus_success_rate_OLD(p, _pi, I, K)
        return success_p

    #def compute_consensus_with_fixed_parameters(self, p, _pi, labels, I):
    #    self.I = I
    #    self.J = len(p)
    #    self.K = _pi.shape[0]
    #    n = self._compute_n(labels)
    #    logpi = np.log(_pi)
    #    prob_consensus = self._e_step(n, logpi, p)
    #    consensus = np.argmax(prob_consensus, axis=1)
    #    return prob_consensus, consensus

    """


    def generate_crowd_labels(self, real_labels, num_annotators, parameters=None):
        """

        Args:
            real_labels (numpy.ndarray): 1D array with dimension (I)
            num_annotators (int): number of annotators
            J (int): number of labels
            _pi:  individual error rates

        Returns:
            numpy.ndarray: 2D array with dimensions (I * K, 3)

        """
        if parameters is None:
            p = self.p
            _pi = np.exp(self.logpi)
        else:
            p, _pi = self._get_parameters_from_dict(parameters)

        print("_pi",_pi)
        I = real_labels.shape[0]  # number of tasks
        J = len(p)
        K = _pi.shape[0]  #
        # Sample the annotators
        # print("Generating crowd labels I: {}, J: {}, K: {}".format(I, J, K))
        annotators = np.random.choice(K, size=(I, num_annotators))
        labels_and_annotators = annotators + real_labels[:, np.newaxis] * K
        labels_and_annotators = labels_and_annotators.flatten()
        unique_la, inverse_la, counts_la = np.unique(labels_and_annotators, return_inverse=True, return_counts=True)
        # print(inverse_la.shape)
        # print(inverse_la)
        crowd_labels = np.zeros((I * num_annotators, 3), dtype=np.int32)
        crowd_labels[:, 0] = np.arange(I * num_annotators) // num_annotators
        # crowd_labels.flatten()
        for i_la, label_and_annotator in enumerate(unique_la):
            real_label = label_and_annotator // K
            annotator_index = label_and_annotator % K
            # print("Real_label:", real_label)
            # print("Annotator:", annotator_index)
            emission_p = _pi[annotator_index, real_label]
            # print("i_la:", i_la)
            # print("counts:", counts_la[i_la])
            # emission_p[np.isnan(emission_p)] = 0.  # TODO (OM, 20121207): Hack added to escape NaN probabilities.
            emitted_labels = np.random.choice(J, size=counts_la[i_la], p=emission_p)
            ca_indexes = np.equal(inverse_la, i_la)
            # print(ca_indexes.shape)
            # print(ca_indexes)
            crowd_labels[:, 1][ca_indexes] = annotator_index
            crowd_labels[:, 2][ca_indexes] = emitted_labels
        return crowd_labels

    def sample(self, I, num_annotators, parameters=None):
        """

        Args:
            p:
            _pi:
            I: number of tasks
            num_annotators: number of annotators

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        if parameters is None:
            p = self.p
            _pi = np.exp(self.logpi)
        else:
            p, _pi = self._get_parameters_from_dict(parameters)

        J = len(p)  # number of labels
        # Sample the real labels
        real_labels = np.random.choice(J, size=I, p=p)
        crowd_labels = self.generate_crowd_labels(real_labels, num_annotators, parameters)
        return real_labels, crowd_labels

    def generate_datasets(self, I, annotators, parameters=None):
        """

        Args:
            I (int): number of tasks
            annotators List[int]: list of annotator numbers

        Yields:
            Tuple[numpy.ndarray, numpy.ndarray, int]: (real_labels, crowd_labels, K) tuple

        """
        for K in annotators:
            real_labels, crowd_labels = self.sample(I, K, parameters)
            yield real_labels, crowd_labels, K

    def generate_linked_datasets(self, parameters_ref, parameters_others, I, annotators):
        for real_labels, crowd_labels_ref, K in self.generate_datasets(I, annotators=annotators,
                                                                       parameters=parameters_ref):
            crowd_labels_others = {}
            for other_name, parameters_other in parameters_others.items():
                crowd_labels_others[other_name] = self.generate_crowd_labels(real_labels, num_annotators=K,
                                                                             parameters=parameters_other)
            yield K, real_labels, crowd_labels_ref, crowd_labels_others

    def compute_linked_consensuses(self, parameters_ref, parameters_others, models, I, annotators):
        p, _pi = self._get_parameters_from_dict(parameters_ref)
        J = len(p)
        K = _pi.shape[0]
        for K, real_labels, crowd_labels_ref, crowd_labels_others in self.generate_linked_datasets(parameters_ref,
                                                                                                   parameters_others,
                                                                                                   I, annotators):
            for model_name, model in models.items():
                consensus_ref, _ = model._fit_and_compute_consensus(crowd_labels_ref, I, J, K)
                consensus_others = {}
                for other_name, crowd_labels_other in crowd_labels_others.items():
                    consensus_others[other_name], _ = model._fit_and_compute_consensus(crowd_labels_other, I, J, K)

                yield K, model_name, real_labels, consensus_ref, consensus_others

    def evaluate_linked_consensuses(self, parameters_ref, parameters_others, models, measures, I, annotators):
        for K, model_name, real_labels, consensus_ref, consensus_others in self.compute_linked_consensuses(parameters_ref,
                                                                                              parameters_others,
                                                                                              models, I, annotators):
            for measure_name, measure in measures.items():
                yield K, measure_name, model_name, "reference", measure(real_labels, consensus_ref)
                for other_name, consensus_other in consensus_others.items():
                    yield K, measure_name, model_name, other_name, measure(real_labels, consensus_other)



    def success_rate(self, real_labels, crowd_labels):
        I = real_labels.shape[0]  # number of tasks
        prob_consensus, consensus = self.compute_consensus_with_fixed_parameters(self.p, np.exp(self.logpi),
                                                                                 crowd_labels, I)
        num_successes = np.sum(real_labels == consensus)
        return num_successes / I

    def success_rates(self, I, annotators):
        success_p = np.zeros(len(annotators))
        for real_labels, crowd_labels, K in self.generate_datasets(I, annotators):
            success_p[annotators.index(K)] = self.success_rate(real_labels, crowd_labels)
        return success_p

    def success_rate_with_fixed_parameters(self, p, _pi, I, K):
        real_labels, crowd_labels = self.fast_sample(p, _pi, I, K)
        # print(crowd_labels)
        prob_consensus, consensus = self.compute_consensus_with_fixed_parameters(p, _pi, crowd_labels, I)
        num_successes = np.sum(real_labels == consensus)
        return num_successes / I
