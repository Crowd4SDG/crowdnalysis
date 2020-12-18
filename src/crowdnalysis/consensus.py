import numpy as np
from .data import Data


class AbstractConsensus:
    """ Base class for a consensus algorithm."""
    name = None
    
    @staticmethod
    def compute_counts(m, I, J):
        n = np.zeros((I, J))
        for i, k, j in m:
            n[i, j] += 1
        return n
        # print(n)

    @staticmethod
    def get_question_matrix_and_ranges(d, question):
        m = d.get_question_matrix(question)
        I = d.n_tasks
        J = d.n_labels(question)
        K = d.n_annotators
        return m, I, J, K

    def fit_and_compute_consensuses(self, d, questions, **kwargs):
        consensuses = {}
        parameters = {}
        for q in questions:
            consensuses[q], parameters[q] = self.fit_and_compute_consensus(d, q)
        return consensuses, parameters

    def fit_and_compute_consensus(self, d: Data, question, **kwargs):
        """Computes consensus and fits model for question question from Data d.

        returns consensus, model parameters"""
        m, I, J, K = self.get_question_matrix_and_ranges(d, question)
        return self.m_fit_and_compute_consensus(m, I, J, K, **kwargs)

    def m_fit_and_compute_consensus(self, m, I, J, K, **kwargs):
        # TODO (OM, 20201210): A return class for model parameters instead of dictionary
        raise NotImplementedError

    def get_dimensions(self, parameters):
        """ Returns the number of labels and number of annotators of the model encoded in the parameters"""
        raise NotImplementedError

    def fit_many(self, d: Data, reference_consensuses):
        parameters = {}
        for q, consensus in reference_consensuses.items():
            parameters[q] = self.fit(d, q, consensus)
        return parameters

    def fit(self, d: Data, question, reference_consensus, prior=1.0):
        m, I, J, K = self.get_question_matrix_and_ranges(d, question)
        return self.m_fit(m, I, J, K, reference_consensus, prior)

    def m_fit(self, m, I, J, K, reference_consensus, **kwargs):
        """ Fits the model parameters provided that the consensus is already known.
        This is useful to determine the errors of a different set of annotators than the
        ones that were used to determine the consensus.

        returns parameters """
        raise NotImplementedError

    def compute_consensus(self, d: Data, question, parameters):
        m, I, J, K = self.get_question_matrix_and_ranges(d, question)
        return self.m_compute_consensus(m, I, J, K, parameters)

    def m_compute_consensus(self, m, I, J, K, parameters):
        """ Computes the consensus with a fixed pre-determined set of parameters.

        returns consensus """
        raise NotImplementedError


class GenerativeAbstractConsensus(AbstractConsensus):

    def generate_crowd_labels(self, real_labels, num_annotators, parameters=None):
        """

        Args:
            real_labels (numpy.ndarray): 1D array with dimension (I)
            num_annotators (int): number of annotators

        Returns:
            numpy.ndarray: 2D array with dimensions (I * K, 3)

        """
        raise NotImplementedError

    def sample(self, I, num_annotators, parameters=None):
        """

        Args:
            I: number of tasks
            num_annotators: number of annotators
            parameters: Dict with parameters of the model.
        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        raise NotImplementedError

    def generate_datasets(self, I, annotators, parameters=None):
        """

        Args:
            I (int): number of tasks
            annotators List[int]: list of annotator numbers
            parameters: Dict with parameters of the model.

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
        J, K = self.get_dimensions(parameters_ref)

        for K, real_labels, crowd_labels_ref, crowd_labels_others in self.generate_linked_datasets(parameters_ref,
                                                                                                   parameters_others,
                                                                                                   I, annotators):
            for model_name, model in models.items():
                consensus_ref, _ = model.m_fit_and_compute_consensus(crowd_labels_ref, I, J, K)
                consensus_others = {}
                for other_name, crowd_labels_other in crowd_labels_others.items():
                    consensus_others[other_name], _ = model.m_fit_and_compute_consensus(crowd_labels_other, I, J, K)

                yield K, model_name, real_labels, consensus_ref, consensus_others

    def evaluate_linked_consensuses(self, parameters_ref, parameters_others, models, measures, I, annotators):
        for K, model_name, real_labels, consensus_ref, consensus_others in self.compute_linked_consensuses(parameters_ref,
                                                                                              parameters_others,
                                                                                              models, I, annotators):
            for measure_name, measure in measures.items():
                yield K, measure_name, model_name, "reference", measure(real_labels, consensus_ref)
                for other_name, consensus_other in consensus_others.items():
                    yield K, measure_name, model_name, other_name, measure(real_labels, consensus_other)

