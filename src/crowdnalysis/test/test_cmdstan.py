import dataclasses
import pytest
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Type

import numpy as np

from . import close, distance, TOLERANCE
from .. import log
from ..problems import ConsensusProblem
from ..cmdstan import AbstractStanOptimizeConsensus, StanMultinomialOptimizeConsensus


@dataclass
class SampleForTest:
    problem: ConsensusProblem
    parameters: AbstractStanOptimizeConsensus.Parameters


def easy_sample() -> SampleForTest:
    dgp = StanMultinomialOptimizeConsensus.DataGenerationParameters(n_tasks=1000, n_annotations_per_task=20)
    parameters = StanMultinomialOptimizeConsensus.Parameters(tau=np.array([0.3, 0.7]),
                                                             pi=np.array([[0.9, 0.1], [0.2, 0.8]]))
    smoc = StanMultinomialOptimizeConsensus()
    problem = smoc.sample(dgp, parameters)
    return SampleForTest(problem, parameters)


def sample_non_finite_gradient() -> SampleForTest:
    dgp = StanMultinomialOptimizeConsensus.DataGenerationParameters(n_tasks=100000, n_annotations_per_task=3)
    parameters = StanMultinomialOptimizeConsensus.Parameters(tau=np.array([0.05, 0.95]),
                                                             pi=np.array([[0.999, 0.001], [0.2, 0.8]]))
    smoc = StanMultinomialOptimizeConsensus()
    problem = smoc.sample(dgp, parameters)
    log.info("sample_non_finite_gradient")
    log.info(problem)
    log.info(parameters)
    return SampleForTest(problem, parameters)


@pytest.fixture(scope="module")
def models() -> Dict[str, Type[AbstractStanOptimizeConsensus]]:
    # Model classes can be integrated into `sample_funcs` too. We keep it separate for simplicity's sake. OM, 20210508
    dict_models = {StanMultinomialOptimizeConsensus.name: StanMultinomialOptimizeConsensus}
    return dict_models


@pytest.fixture(scope="module")
def sample_funcs() -> Dict[str, List[Callable]]:
    dict_funcs = {StanMultinomialOptimizeConsensus.name: [easy_sample, sample_non_finite_gradient]}
    return dict_funcs


@pytest.fixture(scope="module")
def samples(sample_funcs) -> Dict[str, List[SampleForTest]]:
    dict_samples = {}
    for model_name in sample_funcs.keys():
        dict_samples[model_name] = [sf() for sf in sample_funcs[model_name]]
    return dict_samples


@pytest.fixture(scope="module")
def ref_consensus_params_problem(models, samples) -> Dict[str, List[Tuple[np.ndarray,
                                                                          AbstractStanOptimizeConsensus.Parameters,
                                                                          ConsensusProblem]]]:
    dict_ref = {}
    for model_name, model_cls in models.items():
        model = model_cls()
        dict_ref[model_name] = []
        for sample in samples[model_name]:
            consensus_ref, parameters_ref = model.fit_and_compute_consensus(sample.problem)
            dict_ref[model_name].append((consensus_ref, parameters_ref, sample.problem))
    return dict_ref


def _test_fit_and_compute_consensus(model_cls: Type[AbstractStanOptimizeConsensus], sample: SampleForTest):
    model = model_cls()
    consensus, parameters_learned = model.fit_and_compute_consensus(sample.problem)
    dict_parameters_learned = dataclasses.asdict(parameters_learned)
    dict_sample_parameters = dataclasses.asdict(sample.parameters)
    for p in dict_sample_parameters.keys():
        log.debug("Distance between learned {p} and real {p}: {d:f}".format(
            p=p, d=distance(dict_parameters_learned[p], dict_sample_parameters[p])))
        assert close(dict_parameters_learned[p], dict_sample_parameters[p])


def _test_fit(model_cls: Type[AbstractStanOptimizeConsensus], consensus_ref: np.ndarray,
              parameters_ref: AbstractStanOptimizeConsensus.Parameters, problem: ConsensusProblem):
    model = model_cls()
    parameters_learned = model.fit(problem, consensus_ref)
    dict_parameters_learned = dataclasses.asdict(parameters_learned)
    dict_parameters_ref = dataclasses.asdict(parameters_ref)
    for p in dict_parameters_ref.keys():
        log.debug("Distance between learned {p} and reference {p}: {d:f}".format(
            p=p, d=distance(dict_parameters_learned[p], dict_parameters_ref[p])))
        assert close(dict_parameters_learned[p], dict_parameters_ref[p])


def _test_compute_consensus(model_cls: Type[AbstractStanOptimizeConsensus], consensus_ref: np.ndarray,
                            parameters_ref: AbstractStanOptimizeConsensus.Parameters, problem: ConsensusProblem):
    model = model_cls()
    consensus_calculated, _ = model.compute_consensus(problem, data=parameters_ref.to_json())
    assert np.allclose(consensus_calculated, consensus_ref, atol=TOLERANCE)


def test_multinomial_optimize_fit_and_compute_consensus(samples):
    for sample in samples[StanMultinomialOptimizeConsensus.name]:
        _test_fit_and_compute_consensus(StanMultinomialOptimizeConsensus, sample)


def test_multinomial_optimize_fit(ref_consensus_params_problem):
    for consensus_ref, parameters_ref, problem in ref_consensus_params_problem[StanMultinomialOptimizeConsensus.name]:
        _test_fit(StanMultinomialOptimizeConsensus, consensus_ref, parameters_ref, problem)


def test_multinomial_optimize_compute_consensus(ref_consensus_params_problem):
    for consensus_ref, parameters_ref, problem in ref_consensus_params_problem[StanMultinomialOptimizeConsensus.name]:
        _test_compute_consensus(StanMultinomialOptimizeConsensus, consensus_ref, parameters_ref, problem)
