import dataclasses
import pytest
from dataclasses import dataclass
from typing import Callable, List, Tuple, Type

import numpy as np

from . import close, distance, TOLERANCE
from .. import log
from ..consensus import GenerativeAbstractConsensus
from ..problems import ConsensusProblem


@dataclass
class SampleForTest:
    """Class for a sampling function's output"""
    problem: ConsensusProblem
    parameters: GenerativeAbstractConsensus.Parameters


class BaseTestGenerativeConsensusModel:
    """Abstract test class for generative consensus models.

    Conducts following tests:
    1. `model_cls` class attribute is valid;
    2. `sampling_funcs` class attribute is valid;
    3. Asserts each sample provided by sampling functions is valid;
    4. Runs tests for the following methods of the child consensus model:
        * `fit_and_compute_consensus`,
        * `fit`,
        * `compute_consensus`.

    Notes:
        A child test class only has to override `model_cls` and `sampling_funcs` class attributes.
    """

    # Subclass' type; e.g. StanMultinomialOptimizeConsensus
    model_cls: Type[GenerativeAbstractConsensus] = None
    # A list of zero-argument sampling functions that return `SampleTest` objects
    sampling_funcs: List[Callable] = []
    # Absolute tolerance for the `numpy.allclose` function that may also be overridden, if need be
    ABSOLUTE_TOLERANCE = TOLERANCE

    @classmethod
    @pytest.fixture(scope="class")
    def samples(cls) -> List[SampleForTest]:
        # print("In samples for the model {}".format(cls.model_cls.name))
        return [sf() for sf in cls.sampling_funcs]

    @classmethod
    @pytest.fixture(scope="class")
    def ref_consensus_params_problem(cls, samples) -> List[Tuple[np.ndarray,
                                                                 GenerativeAbstractConsensus.Parameters,
                                                                 ConsensusProblem]]:
        # print("In ref_consensus_params_problem for the model {}".format(cls.model_cls.name))
        model = cls.model_cls()
        list_ref = []
        for sample in samples:
            consensus_ref, parameters_ref = model.fit_and_compute_consensus(sample.problem)
            list_ref.append((consensus_ref, parameters_ref, sample.problem))
        return list_ref

    @classmethod
    def _test_fit_and_compute_consensus(cls, sample: SampleForTest):
        model = cls.model_cls()
        consensus, parameters_learned = model.fit_and_compute_consensus(sample.problem)
        dict_parameters_learned = dataclasses.asdict(parameters_learned)
        dict_sample_parameters = dataclasses.asdict(sample.parameters)
        for p in dict_sample_parameters.keys():
            log.debug("Distance between learned {p} and real {p}: {d:f}".format(
                p=p, d=distance(dict_parameters_learned[p], dict_sample_parameters[p])))
            assert close(dict_parameters_learned[p], dict_sample_parameters[p])

    @classmethod
    def _test_fit(cls, consensus_ref: np.ndarray, parameters_ref: GenerativeAbstractConsensus.Parameters,
                  problem: ConsensusProblem):
        model = cls.model_cls()
        parameters_learned = model.fit(problem, consensus_ref)
        dict_parameters_learned = dataclasses.asdict(parameters_learned)
        dict_parameters_ref = dataclasses.asdict(parameters_ref)
        for p in dict_parameters_ref.keys():
            log.debug("Distance between learned {p} and reference {p}: {d:f}".format(
                p=p, d=distance(dict_parameters_learned[p], dict_parameters_ref[p])))
            assert close(dict_parameters_learned[p], dict_parameters_ref[p])

    @classmethod
    def _test_compute_consensus(cls, consensus_ref: np.ndarray, parameters_ref: GenerativeAbstractConsensus.Parameters,
                                problem: ConsensusProblem):
        model = cls.model_cls()
        # print("In _test_compute_consensus:", parameters_ref)
        consensus_calculated = model.compute_consensus(problem, parameters_ref)
        if isinstance(consensus_calculated, Tuple):
            consensus_calculated = consensus_calculated[0]
            # TODO (OM, 20210511): Check if the `kwargs_opt` return value of
            #  `AbstractStanOptimizeConsensus.compute_consensus()`is necessary
        assert np.allclose(consensus_calculated, consensus_ref, atol=cls.ABSOLUTE_TOLERANCE)

    # ==============================  Actual tests  ==============================
    def test_model_cls(self):
        # print("In test_model_cls for the model {}".format(self.model_cls.name))
        assert self.model_cls
        assert issubclass(self.model_cls, GenerativeAbstractConsensus)

    def test_sampling_funcs(self):
        # print("In test_sampling_funcs for the model {}".format(self.model_cls.name))
        assert self.sampling_funcs  # Not empty
        for sf in self.sampling_funcs:
            assert isinstance(sf, Callable)

    def test_sampling(self, samples):
        # print("In test_sampling for the model {}".format(self.model_cls.name))
        assert isinstance(samples, list)
        assert samples  # Not empty
        for sample in samples:
            assert isinstance(sample, SampleForTest)

    def test_fit_and_compute_consensus(self, samples):
        # print("In test_fit_and_compute_consensus for the model {}".format(self.model_cls.name))
        for sample in samples:
            self._test_fit_and_compute_consensus(sample)

    def test_fit(self, ref_consensus_params_problem):
        # print("In test_fit for the model {}".format(self.model_cls.name))
        for consensus_ref, parameters_ref, problem in ref_consensus_params_problem:
            self._test_fit(consensus_ref, parameters_ref, problem)

    def test_compute_consensus(self, ref_consensus_params_problem):
        # print("In test_compute_consensus for the model {}".format(self.model_cls.name))
        for consensus_ref, parameters_ref, problem in ref_consensus_params_problem:
            self._test_compute_consensus(consensus_ref, parameters_ref, problem)