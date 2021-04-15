import numpy as np

from .common import vprint
from .data import Data
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ConsensusProblem:
    n_tasks: Optional[int] = field(default=None)
    # features of the tasks
    tasks: Optional[np.ndarray] = None
    n_workers: Optional[int] = field(default=None)
    # features of the workers
    workers: Optional[np.ndarray] = None
    n_annotations: int = field(init=False)
    # features of each of the annotations
    annotations: np.ndarray = np.array([])

    def __post_init__(self):
        self.n_annotations = self.annotations.shape[0]
        if self.n_tasks is None:
            if self.tasks is None:
                raise Exception("Undetermined number of tasks")
            else:
                self.n_tasks = self.tasks.shape[0]
        if self.n_workers is None:
            if self.workers is None:
                raise Exception("Undetermined number of workers")
            else:
                self.n_workers = self.workers.shape[0]

"""
The DiscreteConsensusProblem enforces that:
- There is a discrete set of real classes
- There is a single attribute of each annotation, 
  which is also discrete and which at least contains the real_classes.
  It can eventually contain additional answers such as "I do not know", or "in doubt", or "does not apply"
"""
@dataclass
class DiscreteConsensusProblem(ConsensusProblem):
    # number of different labels in the annotation
    n_labels: Optional[int] = None
    # Which labels in an annotation correspond to real hidden classes
    classes: Optional[List[int]] = None

    def __post_init__(self):
        ConsensusProblem.__post_init__(self)
        if self.n_labels is None:
            self.n_labels = np.unique(self.annotations[:, 0])
        if self.classes is None:
            self.classes = list(range(self.n_labels))

"""
This dataclass should be subclassed by each of the different models used to compute consensus
"""
@dataclass
class Parameters:
    pass




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

    def m_fit_and_compute_consensus(self, dcp: DiscreteConsensusProblem, **kwargs):
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

    def m_compute_consensus(self, dcp: DiscreteConsensusProblem, parameters):
        """ Computes the consensus with a fixed pre-determined set of parameters.

        returns consensus """
        raise NotImplementedError


"""
This dataclass should be subclassed by each of the different generative consensus models
"""
@dataclass
class DataGenerationParameters:
    pass

class GenerativeAbstractConsensus(AbstractConsensus):

    def sample_tasks(self, dgp: DataGenerationParameters, parameters: Optional[Parameters]=None):
        return NotImplementedError

    def sample_workers(self, dgp: DataGenerationParameters, parameters: Optional[Parameters]=None):
        return NotImplementedError

    def sample_annotations(self, tasks, workers, dgp: DataGenerationParameters, parameters: Optional[Parameters]=None):
        raise NotImplementedError

    def sample(self, dgp:DataGenerationParameters, parameters: Optional[Parameters] = None):
        tasks = self.sample_tasks(dgp, parameters)
        workers = self.sample_workers(dgp, parameters)
        crowd_labels = self.sample_annotations(tasks, workers, dgp, parameters)
        return tasks, workers, crowd_labels

    # TODO: Everything down this comment has to be worked on after freezing the main interfaces.
    # Creates a set of linked discrete consensus problems, with linked meaning that they share the very same set of tasks.
    # Each problem represents how it will be labeled by a different community
    def linked_samples(self, real_parameters, crowds_parameters, dgp: DataGenerationParameters):
        tasks = self.sample_tasks(dgp, parameters=real_parameters)
        crowds_labels= {}
        crowds_workers = {}
        for crowd_name, parameters in crowds_parameters.items():
            crowds_workers[crowd_name] = self.sample_workers(dgp, parameters=real_parameters)
            #print("parameters:", parameters)
            crowds_labels[crowd_name] = self.sample_annotations(tasks, crowds_workers[crowd_name], dgp,
                                                                parameters=parameters)
        return tasks, crowds_workers, crowds_labels

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

