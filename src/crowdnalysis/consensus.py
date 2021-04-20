import dataclasses

import numpy as np

from . import log
from .data import Data
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Optional, Tuple
import json
from numpyencoder import NumpyEncoder

@dataclass
class JSONDataClass:
    def to_json(self):
        return json.dumps(dataclasses.asdict(self), cls=NumpyEncoder)

    @classmethod
    def from_json(cls, s: str):
        d = json.loads(s)
        return cls(**d)

@dataclass
class ConsensusProblem(JSONDataClass):
    """
    Example:

        TODO (OM, 20210416): Add examples to init params
    """
    n_tasks: int = 0
    # features of the tasks
    f_T: Optional[np.ndarray] = None
    n_workers: int = 0
    # features of the workers
    f_W: Optional[np.ndarray] = None
    n_annotations: Optional[int] = None
    # task of an annotation
    t_A: np.ndarray = np.array([])
    # worker of an annotation
    w_A: np.ndarray = np.array([])
    # features of each of the annotations
    f_A: np.ndarray = np.array([])

    def __post_init__(self):
        """

        Raises:
            ValueError: If # of tasks or of workers is not determinable
        """
        if isinstance(self.f_T, list):
            self.f_T = np.array(self.f_T)
        if isinstance(self.f_W, list):
            self.f_W = np.array(self.f_W)
        if isinstance(self.t_A, list):
            self.t_A = np.array(self.t_A)
        if isinstance(self.w_A, list):
            self.w_A = np.array(self.w_A)
        if isinstance(self.f_A, list):
            self.f_A = np.array(self.f_A)
        if len(self.f_A.shape) == 1:
            self.f_A = self.f_A[:, np.newaxis]
        self.n_annotations = self.f_A.shape[0]



@dataclass
class DiscreteConsensusProblem(ConsensusProblem):
    """
    Notes:
        The DiscreteConsensusProblem enforces that:
        - There is a discrete set of real classes
        - The labels should be integers starting from 0.
        - There is a single attribute of each annotation,
          which is also discrete and which at least contains the real_classes.
          It can eventually contain additional answers such as "I do not know", or "in doubt", or "does not apply"

    Example:
        TODO (OM, 20210416): Add examples to init params
    """
    # number of different labels in the annotation
    n_labels: Optional[int] = None
    # Which labels in an annotation correspond to real hidden classes
    classes: Optional[List[int]] = None

    def __post_init__(self):
        ConsensusProblem.__post_init__(self)
        if (self.n_labels is None) and (self.f_A.shape[0] > 0):
            self.n_labels = np.max(self.f_A[:, 0]) + 1
        if self.classes is None:
            # By default every label is a real class
            self.classes = list(range(self.n_labels))

    @staticmethod
    def from_data(d: Data, question):
        return DiscreteConsensusProblem(n_tasks=d.n_tasks,
                                        n_workers=d.n_annotators,
                                        t_A=d.get_tasks(question),
                                        w_A=d.get_workers(question),
                                        f_A=d.get_annotations(question),
                                        n_labels=d.n_labels(question))

    def compute_n(self):
        # TODO: This should be optimized
        # print(m)
        #N = m.shape[0]

        # print("N=", N, "n_tasks=", self.n_tasks, "n_labels=", self.n_labels, "n_annotators=", self.n_annotators)

        # Compute the n matrix

        n = np.zeros((self.n_workers, self.n_tasks, self.n_labels))
        for i in range(self.n_annotations):
            n[self.w_A[i], self.t_A[i], self.f_A[i, 0]] += 1
        return n


@dataclass
class Parameters:
    """
    Notes:
        This dataclass should be subclassed by each of the different models used to compute consensus
    """
    pass

# This wrapper deals with the Data interface

class ConsensusWrapper:
    def __init__(self, consensus):
        self._consensus = consensus

    def fit_and_compute_consensuses(self, d: Data, questions, **kwargs):
        consensuses = {}
        parameters = {}
        for q in questions:
            consensuses[q], parameters[q] = self.fit_and_compute_consensus(d, q, **kwargs)
        return consensuses, parameters

    def fit_and_compute_consensus(self, d: Data, question, **kwargs):
        """Computes consensus and fits model for question question from Data d.

        returns consensus, model parameters"""
        dcp = DiscreteConsensusProblem.from_data(d, question)
        return self._consensus.fit_and_compute_consensus(dcp, **kwargs)

    def fit_many(self, d: Data, reference_consensuses):
        parameters = {}
        for q, consensus in reference_consensuses.items():
            parameters[q] = self.fit(d, q, consensus)
        return parameters

    def fit(self, d: Data, question, reference_consensus, prior=1.0):
        dcp = DiscreteConsensusProblem.from_data(d, question)
        return self._consensus.fit(dcp, reference_consensus, prior)

    def compute_consensus(self, d: Data, question, parameters):
        dcp = DiscreteConsensusProblem.from_data(d, question)
        return self._consensus.compute_consensus(dcp, parameters)


class AbstractConsensus:
    """ Base class for a consensus algorithm."""
    name = None

    def fit_and_compute_consensus(self, dcp: DiscreteConsensusProblem, **kwargs):
        # TODO (OM, 20201210): A return class for model parameters instead of dictionary
        raise NotImplementedError

    def get_dimensions(self, parameters):
        """ Returns the number of labels and number of annotators and number of classes of the model encoded in the parameters"""
        raise NotImplementedError

    def fit(self, dcp: DiscreteConsensusProblem, reference_consensus, **kwargs):
        """ Fits the model parameters provided that the consensus is already known.
        This is useful to determine the errors of a different set of annotators than the
        ones that were used to determine the consensus.

        returns parameters """
        raise NotImplementedError


    def compute_consensus(self, dcp: DiscreteConsensusProblem, parameters):
        """ Computes the consensus with a fixed pre-determined set of parameters.

        returns consensus """
        raise NotImplementedError


@dataclass
class DataGenerationParameters:
    """
    Notes:
        This dataclass should be subclassed by each of the different generative consensus models
    """
    pass


class GenerativeAbstractConsensus(AbstractConsensus):
    """Base class for a consensus algorithm that also samples tasks, workers and annotations."""

    def sample_tasks(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None) \
            -> Tuple[int, Optional[np.ndarray]]:
        raise NotImplementedError

    def sample_workers(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None)\
            -> Tuple[int, Optional[np.ndarray]]:
        raise NotImplementedError

    def sample_annotations(self, tasks, workers, dgp: DataGenerationParameters, parameters: Optional[Parameters]=None)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sample(self, dgp: DataGenerationParameters, parameters: Optional[Parameters] = None):
        n_tasks, tasks = self.sample_tasks(dgp, parameters)
        n_workers, workers = self.sample_workers(dgp, parameters)
        w_A, t_A, f_A = self.sample_annotations(tasks, workers, dgp, parameters)
        return DiscreteConsensusProblem(n_tasks=n_tasks,
                                        f_T=tasks,
                                        n_workers=n_workers,
                                        w_A=w_A,
                                        t_A=t_A,
                                        f_A=f_A)

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
                crowds_consensus[crowd_name], _ = model.fit_and_compute_consensus(crowd_labels, n_tasks, n_labels, n_annotators, **kwargs)
            else:
                crowds_consensus[crowd_name], _ = model.fit_and_compute_consensus(
                    crowd_labels, n_tasks, n_labels, n_annotators, init_params=crowd_parameters[crowd_name], **kwargs)

        return crowds_consensus

    def evaluate_consensuses_on_linked_samples(self, real_parameters, crowds_parameters, models, measures, sample_sizes,
                                               annotations_per_task, repeats, init_params=False):
        n_labels, n_annotators, n_classes = self.get_dimensions(real_parameters)
        for n_tasks in sample_sizes:
            log.info("Sample size:", n_tasks)
            for num_annotations_per_task in annotations_per_task:
                log.info(".# of annotations per task:", num_annotations_per_task)
                for repetition in range(repeats):
                    log.info("..Repetition:", repetition)
                    real_labels, crowds_labels = self.linked_samples(real_parameters, crowds_parameters, n_tasks, num_annotations_per_task)
                    for model_name, model in models.items():
                        log.info("...Model:", model_name)
                        crowds_consensus = self.compute_consensuses(crowds_labels, model, n_tasks, n_labels, n_annotators,
                                                                    crowds_parameters if init_params else None)
                        for measure_name, measure in measures.items():
                            log.info("....Measure:", measure_name)
                            crowds_evals = measure.evaluate_crowds(real_labels, crowds_consensus)
                            for crowd_name, eval_value in crowds_evals.items():
                                yield {"num_samples": n_tasks,
                                       "num_annotations_per_task": num_annotations_per_task,
                                       "consensus_algorithm": model_name,
                                       "repetition": repetition,
                                       "crowd_name": crowd_name,
                                       "measure": measure_name,
                                       "value": eval_value}

