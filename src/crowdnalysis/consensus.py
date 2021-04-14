import numpy as np

from .common import vprint
from .data import Data
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DiscreteConsensusProblem:
    m: Optional[np.ndarray] = None
    n_tasks: int = 100
    n_labels: int = 2
    n_annotators: int = 1
    classes: Optional[List[int]] = None

class AbstractConsensus:
    """ Base class for a consensus algorithm."""
    name = None
    
    @staticmethod
    def compute_counts(dcp: DiscreteConsensusProblem):
        n = np.zeros((dcp.n_tasks, dcp.n_labels))
        for i, k, j in dcp.m:
            n[i, j] += 1
        return n
        # print(n)

    @staticmethod
    def get_problem(d: Data, question):
        return DiscreteConsensusProblem(m = d.get_question_matrix(question),
                                        n_tasks = d.n_tasks,
                                        n_labels = d.n_labels(question),
                                        n_annotators = d.n_annotators,
                                        classes = d.classes(question))

    def fit_and_compute_consensuses(self, d: Data, questions, **kwargs):
        consensuses = {}
        parameters = {}
        for q in questions:
            consensuses[q], parameters[q] = self.fit_and_compute_consensus(d, q, **kwargs)
        return consensuses, parameters

    def fit_and_compute_consensus(self, d: Data, question, **kwargs):
        """Computes consensus and fits model for question question from Data d.

        returns consensus, model parameters"""
        dcp = self.get_problem(d, question)
        return self.m_fit_and_compute_consensus(dcp, **kwargs)

    def m_fit_and_compute_consensus(self, dcp:DiscreteConsensusProblem, **kwargs):
        # TODO (OM, 20201210): A return class for model parameters instead of dictionary
        raise NotImplementedError

    def get_dimensions(self, parameters):
        """ Returns the number of labels and number of annotators and number of classes of the model encoded in the parameters"""
        raise NotImplementedError

    def fit_many(self, d: Data, reference_consensuses):
        parameters = {}
        for q, consensus in reference_consensuses.items():
            parameters[q] = self.fit(d, q, consensus)
        return parameters

    def fit(self, d: Data, question, reference_consensus, prior=1.0):
        dcp = self.get_problem(d, question)
        return self.m_fit(dcp, reference_consensus, prior)

    def m_fit(self, dcp: DiscreteConsensusProblem, reference_consensus, **kwargs):
        """ Fits the model parameters provided that the consensus is already known.
        This is useful to determine the errors of a different set of annotators than the
        ones that were used to determine the consensus.

        returns parameters """
        raise NotImplementedError

    def compute_consensus(self, d: Data, question, parameters):
        dcp = self.get_problem(d, question)
        return self.m_compute_consensus(dcp, parameters)

    def m_compute_consensus(self, dcp: DiscreteConsensusProblem, n_classes, parameters):
        """ Computes the consensus with a fixed pre-determined set of parameters.

        returns consensus """
        raise NotImplementedError


class GenerativeAbstractConsensus(AbstractConsensus):

    def sample_tasks(self, n_tasks, parameters=None):
        """

        Args:
            n_tasks: number of tasks

        Returns:
            numpy.ndarray:
        """
        return NotImplementedError

    def sample_annotations(self, real_labels, num_annotations_per_task, parameters=None):
        """

        Args:
            real_labels (numpy.ndarray): 1D array with dimension (n_tasks)
            num_annotations_per_task (int):number of annotations per task

        Returns:
            numpy.ndarray: 2D array with dimensions (n_tasks * num_annotations_per_task, 3)

        """
        raise NotImplementedError

    def sample(self, n_tasks, num_annotations_per_task, parameters=None):
        """

        Args:
            n_tasks: number of tasks
            num_annotators: number of annotators

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        real_labels = self.sample_tasks(n_tasks, parameters)
        crowd_labels = self.sample_annotations(real_labels, num_annotations_per_task, parameters)
        return real_labels, crowd_labels

    def linked_samples(self, real_parameters, crowds_parameters, n_tasks, num_annotations_per_task):
        real_labels = self.sample_tasks(n_tasks, parameters=real_parameters)
        crowds_labels= {}
        for crowd_name, parameters in crowds_parameters.items():
            #print("parameters:", parameters)
            crowds_labels[crowd_name] = self.sample_annotations(real_labels, num_annotations_per_task,
                                                                parameters=parameters)
        return real_labels, crowds_labels

    def compute_consensuses(self, crowds_labels, model, n_tasks, n_labels, n_annotators, crowd_parameters=None, **kwargs):
        crowds_consensus = {}
        for crowd_name, crowd_labels in crowds_labels.items():
            if crowd_parameters is None:
                crowds_consensus[crowd_name], _ = model.m_fit_and_compute_consensus(crowd_labels, n_tasks, n_labels, n_annotators, **kwargs)
            else:
                crowds_consensus[crowd_name], _ = model.m_fit_and_compute_consensus(
                    crowd_labels, n_tasks, n_labels, n_annotators, init_params=crowd_parameters[crowd_name],**kwargs)

        return crowds_consensus

    def evaluate_consensuses_on_linked_samples(self, real_parameters, crowds_parameters, models, measures, sample_sizes,
                                               annotations_per_task, repeats, verbose=False, init_params=False):
        n_labels, n_annotators, n_classes = self.get_dimensions(real_parameters)
        for n_tasks in sample_sizes:
            vprint("Sample size:", n_tasks, verbose=verbose)
            for num_annotations_per_task in annotations_per_task:
                vprint(".# of annotations per task:", num_annotations_per_task, verbose=verbose)
                for repetition in range(repeats):
                    vprint("..Repetition:", repetition, verbose=verbose)
                    real_labels, crowds_labels = self.linked_samples(real_parameters, crowds_parameters, n_tasks, num_annotations_per_task)
                    for model_name, model in models.items():
                        vprint("...Model:", model_name, verbose=verbose)
                        crowds_consensus = self.compute_consensuses(crowds_labels, model, n_tasks, n_labels, n_annotators,
                                                                    crowds_parameters if init_params else None,
                                                                    verbose=verbose)
                        for measure_name, measure in measures.items():
                            # vprint("....Measure:", measure_name, verbose=verbose)
                            crowds_evals = measure.evaluate_crowds(real_labels, crowds_consensus)
                            for crowd_name, eval_value in crowds_evals.items():
                                yield {"num_samples": n_tasks,
                                       "num_annotations_per_task": num_annotations_per_task,
                                       "consensus_algorithm": model_name,
                                       "repetition": repetition,
                                       "crowd_name": crowd_name,
                                       "measure": measure_name,
                                       "value": eval_value}

