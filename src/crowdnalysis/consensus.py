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
            consensuses[q], parameters[q] = self.fit_and_compute_consensus(d, q, **kwargs)
        return consensuses, parameters

    def fit_and_compute_consensus(self, d: Data, question, **kwargs):
        """Computes consensus and fits model for question question from Data d.

        returns consensus, model parameters"""
        m, I, J, K = self.get_question_matrix_and_ranges(d, question)
        return self.m_fit_and_compute_consensus(m, I, J, K, **kwargs)

    def m_fit_and_compute_consensus(self, m, I, J, K, **kwargs):
        # TODO (OM, 20201210): A return class for model parameters instead of dictionary
        raise NotImplementedError

    def n_fit_and_compute_consensus(self, n, **kwargs):
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

    def n_fit(self, n, reference_consensus, **kwargs):
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

    def n_compute_consensus(self, n, parameters):
        """ Computes the consensus with a fixed pre-determined set of parameters.

        returns consensus """
        raise NotImplementedError

# TODO: Evaluate whether it makes sense to keep both the m and n methods


class GenerativeAbstractConsensus(AbstractConsensus):

    def sample_real_labels(self, I, parameters=None):
        """

        Args:
            I: number of tasks
            num_annotators: number of annotators

        Returns:
            Tuple[numpy.ndarray]:
        """
        return NotImplementedError

    def generate_crowd_labels(self, real_labels, num_annotations_per_task, parameters=None):
        """

        Args:
            real_labels (numpy.ndarray): 1D array with dimension (I)
            num_annotations_per_task (int):number of annotations per task

        Returns:
            numpy.ndarray: 2D array with dimensions (I * num_annotations_per_task, 3)

        """
        raise NotImplementedError

    def sample(self, I, num_annotations_per_task, parameters=None):
        """

        Args:
            I: number of tasks
            num_annotators: number of annotators

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        real_labels = self.sample_real_labels(I, parameters)
        crowd_labels = self.generate_crowd_labels(real_labels, num_annotations_per_task, parameters)
        return real_labels, crowd_labels

    def linked_samples(self, real_parameters, crowds_parameters, I, num_annotations_per_task):
        real_labels = self.sample_real_labels(I, parameters=real_parameters)
        crowds_labels= {}
        for crowd_name, parameters in crowds_parameters.items():
            crowds_labels[crowd_name] = self.generate_crowd_labels(real_labels, num_annotations_per_task,
                                                                  parameters=parameters)
        return real_labels, crowds_labels

    def compute_consensuses(self, crowds_labels, model, I, J ,K):
        crowds_consensus = {}
        for crowd_name, crowd_labels in crowds_labels.items():
            crowds_consensus[crowd_name], _ = model.m_fit_and_compute_consensus(crowd_labels, I, J, K)
        return crowds_consensus

    def evaluate_consensuses_on_linked_samples(self, real_parameters, crowds_parameters, models, measures, sample_sizes,
                                               annotations_per_task, repeats):
        J, K = self.get_dimensions(real_parameters)
        for I in sample_sizes:
            for num_annotations_per_task in annotations_per_task:
                for repetition in range(repeats):
                    real_labels, crowds_labels = self.linked_samples(real_parameters, crowds_parameters, I, num_annotations_per_task)
                    for model_name, model in models.items():
                        crowds_consensus = self.compute_consensuses(crowds_labels, model, I, J, K)
                        for measure_name, measure in measures.items():
                            crowds_evals = measure.evaluate_crowds(real_labels, crowds_consensus)
                            for crowd_name, eval_value in crowds_evals.items():
                                yield {"num_samples":I,
                                       "num_annotations_per_task":num_annotations_per_task,
                                       "consensus_algorithm": model_name,
                                       "repetition":repetition,
                                       "crowd_name": crowd_name,
                                       "measure": measure_name,
                                       "value": eval_value}

