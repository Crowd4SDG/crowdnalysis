import dataclasses
from typing import List, Callable, Type

import pytest
from dataclasses import dataclass

import numpy as np

from .. import log
from ..problems import ConsensusProblem
from ..cmdstan import AbstractStanOptimizeConsensus, StanMultinomialOptimizeConsensus
from . import close, distance, TOLERANCE


@dataclass()
class SampleForTest:
    problem: ConsensusProblem
    parameters: AbstractStanOptimizeConsensus.Parameters


def easy_sample() -> SampleForTest:
    dgp = StanMultinomialOptimizeConsensus.DataGenerationParameters(n_tasks=1000, num_annotations_per_task=20)
    parameters = StanMultinomialOptimizeConsensus.Parameters(tau=np.array([0.3, 0.7]),
                                                             pi=np.array([[0.9, 0.1], [0.2, 0.8]]))
    smoc = StanMultinomialOptimizeConsensus()
    problem = smoc.sample(dgp, parameters)
    return SampleForTest(problem, parameters)


def sample_non_finite_gradient() -> SampleForTest:
    dgp = StanMultinomialOptimizeConsensus.DataGenerationParameters(n_tasks=100000, num_annotations_per_task=3)
    parameters = StanMultinomialOptimizeConsensus.Parameters(tau=np.array([0.05, 0.95]),
                                                             pi=np.array([[0.999, 0.001], [0.2, 0.8]]))
    smoc = StanMultinomialOptimizeConsensus()
    problem = smoc.sample(dgp, parameters)
    log.info("sample_non_finite_gradient")
    log.info(problem)
    log.info(parameters)
    return SampleForTest(problem, parameters)


@pytest.fixture
def sample_funcs() -> List[Callable]:
    return [easy_sample, sample_non_finite_gradient]


@pytest.fixture
def samples(sample_funcs) -> List[SampleForTest]:
    return [sf() for sf in sample_funcs]


def _test_fit_and_compute_consensus(model_cls: Type[AbstractStanOptimizeConsensus], sample: SampleForTest):
    model = model_cls()
    consensus, parameters_learned = model.fit_and_compute_consensus(sample.problem)
    dict_parameters_learned = dataclasses.asdict(parameters_learned)
    dict_sample_parameters = dataclasses.asdict(sample.parameters)
    for p in dict_sample_parameters.keys():
        log.debug("Distance between learned {p} and real {p}: {d:f}".format(
            p=p, d=distance(dict_parameters_learned[p], dict_sample_parameters[p])))
        assert close(dict_parameters_learned[p], dict_sample_parameters[p])


def _test_fit(model_cls: Type[AbstractStanOptimizeConsensus], sample: SampleForTest):
    model = model_cls()
    consensus_ref, parameters_ref = model.fit_and_compute_consensus(sample.problem)
    model = model_cls()
    parameters_learned = model.fit(sample.problem, consensus_ref)
    dict_parameters_learned = dataclasses.asdict(parameters_learned)
    dict_parameters_ref = dataclasses.asdict(parameters_ref)
    for p in dict_parameters_learned.keys():
        log.debug("Distance between learned {p} and reference {p}: {d:f}".format(
            p=p, d=distance(dict_parameters_learned[p], dict_parameters_ref[p])))
        assert close(dict_parameters_learned[p], dict_parameters_ref[p])


def _test_compute_consensus(model_cls: Type[AbstractStanOptimizeConsensus], sample: SampleForTest):
    model = model_cls()
    consensus_ref, parameters_ref = model.fit_and_compute_consensus(sample.problem)
    model = model_cls()
    consensus_calculated, _ = model.compute_consensus(sample.problem, data=parameters_ref.to_json())
    assert np.allclose(consensus_calculated, consensus_ref, atol=TOLERANCE)


def test_multinomial_optimize_fit_and_compute_consensus(samples):
    for sample in samples:
        _test_fit_and_compute_consensus(StanMultinomialOptimizeConsensus, sample=sample)


def test_multinomial_optimize_fit(samples):
    for sample in samples:
        _test_fit(StanMultinomialOptimizeConsensus, sample=sample)


def test_multinomial_optimize_compute_consensus(samples):
    for sample in samples:
        _test_compute_consensus(StanMultinomialOptimizeConsensus, sample=sample)
