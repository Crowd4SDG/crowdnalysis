from . import log
from .data import Data
from .problems import DiscreteConsensusProblem, JSONDataClass

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

DiscreteConsensus = np.ndarray


class AbstractSimpleConsensus:
    """ Base class for very simple consensus algorithms."""
    name = None

    @dataclass
    class Parameters(JSONDataClass):
        """
        Notes:
            This dataclass should be subclassed by each of the different models used to compute consensus
        """
        pass

    def fit_and_compute_consensus(self, dcp: DiscreteConsensusProblem, **kwargs) \
            -> Tuple[DiscreteConsensus, Parameters]:
        raise NotImplementedError

    def fit_and_compute_consensuses_from_data(self, d: Data, questions, **kwargs):
        consensuses = {}
        parameters = {}
        for q in questions:
            consensuses[q], parameters[q] = self.fit_and_compute_consensus_from_data(d, q, **kwargs)
        return consensuses, parameters

    def fit_and_compute_consensus_from_data(self, d: Data, question, **kwargs):
        """Computes consensus and fits model for question question from Data d.

        returns consensus, model parameters"""
        dcp = d.get_dcp(question)
        return self.fit_and_compute_consensus(dcp, **kwargs)


class AbstractConsensus(AbstractSimpleConsensus):
    """ Base class for a consensus algorithm."""
    name = None

    def fit_many_from_data(self, d: Data, reference_consensuses):
        parameters = {}
        for q, consensus in reference_consensuses.items():
            parameters[q] = self.fit_from_data(d, q, consensus)
        return parameters

    def fit_from_data(self, d: Data, question, reference_consensus):
        dcp = d.get_dcp(question)
        return self.fit(dcp, reference_consensus)

    def compute_consensus_from_data(self, d: Data, question, parameters):
        dcp = d.get_dcp(question)
        return self.compute_consensus(dcp, parameters)

    def fit(self, dcp: DiscreteConsensusProblem, reference_consensus: DiscreteConsensus, **kwargs) \
            -> AbstractSimpleConsensus.Parameters:
        """ Fits the model parameters provided that the consensus is already known.
        This is useful to determine the errors of a different set of annotators than the
        ones that were used to determine the consensus.

        returns parameters """
        raise NotImplementedError
    def compute_consensus(self, dcp: DiscreteConsensusProblem, parameters: AbstractSimpleConsensus.Parameters) \
            -> DiscreteConsensus:
        """ Computes the consensus with a fixed pre-determined set of parameters.

        returns consensus """
        raise NotImplementedError


class GenerativeAbstractConsensus(AbstractConsensus):
    """Base class for a consensus algorithm that also samples tasks, workers and annotations."""

    @dataclass
    class DataGenerationParameters(JSONDataClass):
        """
        Notes:
            This dataclass should be subclassed by each of the different generative consensus models
        """
        pass

    def get_dimensions(self, parameters: AbstractSimpleConsensus.Parameters):
        """ Returns the number of labels and number of annotators and number of classes of the model encoded in the
        parameters"""
        raise NotImplementedError

    def sample_tasks(self, dgp: DataGenerationParameters, parameters: Optional[AbstractConsensus.Parameters] = None) \
            -> Tuple[int, np.ndarray]:
        raise NotImplementedError

    def sample_workers(self, dgp: DataGenerationParameters, parameters: Optional[AbstractConsensus.Parameters] = None)\
            -> Tuple[int, Optional[np.ndarray]]:
        raise NotImplementedError

    def sample_annotations(self, tasks, workers, dgp: DataGenerationParameters,
                           parameters: Optional[AbstractConsensus.Parameters] = None)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        raise NotImplementedError

    def sample(self, dgp: DataGenerationParameters, parameters: Optional[AbstractConsensus.Parameters] = None):
        n_tasks, tasks = self.sample_tasks(dgp, parameters)
        return self._sample_others(n_tasks, tasks, dgp, parameters)

    def _sample_others(self, n_tasks, tasks, dgp: DataGenerationParameters,
                       parameters: Optional[AbstractConsensus.Parameters] = None):
        n_workers, workers = self.sample_workers(dgp, parameters)
        w_A, t_A, f_A, classes = self.sample_annotations(tasks, workers, dgp, parameters)
        # log.debug(type(w_A.dtype))
        return DiscreteConsensusProblem(n_tasks=n_tasks,
                                        f_T=tasks,
                                        n_workers=n_workers,
                                        w_A=w_A,
                                        t_A=t_A,
                                        f_A=f_A,
                                        classes=classes)

    # TODO: Everything down this comment has to be worked on after freezing the main interfaces.
    # Creates a set of linked discrete consensus problems, with linked meaning that they share the very
    # same set of tasks.
    # Each problem represents how it will be labeled by a different community
    def linked_samples(self, real_parameters, crowds_parameters, dgp: DataGenerationParameters):
        n_tasks, tasks = self.sample_tasks(dgp, parameters=real_parameters)
        crowds_dcps = {}
        for crowd_name, parameters in crowds_parameters.items():
            crowds_dcps[crowd_name] = self._sample_others(n_tasks, tasks, dgp, parameters)

        return tasks, crowds_dcps

    def compute_consensuses(self, crowds_dcps, model, crowd_parameters=None, **kwargs):
        crowds_consensus = {}
        for crowd_name, dcp in crowds_dcps.items():
            # print(dcp)
            # print(model)
            if crowd_parameters is None:
                crowds_consensus[crowd_name], _ = model.fit_and_compute_consensus(dcp, **kwargs)
            else:
                crowds_consensus[crowd_name], _ = \
                    model.fit_and_compute_consensus(dcp, init_params=crowd_parameters[crowd_name], **kwargs)

        return crowds_consensus

    def evaluate_consensuses_on_linked_samples(self, real_parameters, crowds_parameters, models, measures,
                                               dgps: List[DataGenerationParameters],
                                               repeats, init_params=False):
        for dgp in dgps:
            for repetition in range(repeats):
                log.info("..Repetition:", repetition)
                real_labels, crowds_dcps = self.linked_samples(real_parameters, crowds_parameters, dgp)
                for model_name, model in models.items():
                    log.info("...Model:", model_name)
                    crowds_consensus = self.compute_consensuses(crowds_dcps, model,
                                                                crowds_parameters if init_params else None)
                    for measure_name, measure in measures.items():
                        log.info("....Measure:", measure_name)
                        crowds_evals = measure.evaluate_crowds(real_labels, crowds_consensus)
                        for crowd_name, eval_value in crowds_evals.items():
                            d = dataclasses.asdict(dgp)
                            d.update({
                                "consensus_algorithm": model_name,
                                "repetition": repetition,
                                "crowd_name": crowd_name,
                                "measure": measure_name,
                                "value": eval_value})
                            log.info(d)
                            # print(d)
                            yield d
