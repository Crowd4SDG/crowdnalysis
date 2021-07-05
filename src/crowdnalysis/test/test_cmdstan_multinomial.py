import numpy as np

from .. import log
from .common import BaseTestGenerativeConsensusModel, SampleForTest
from ..cmdstan.multinomial import StanMultinomialOptimizeConsensus, StanMultinomialEtaOptimizeConsensus


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


def more_labels_than_classes_sample() -> SampleForTest:
    dgp = StanMultinomialOptimizeConsensus.DataGenerationParameters(n_tasks=1000, n_annotations_per_task=20)
    parameters = StanMultinomialOptimizeConsensus.Parameters(tau=np.array([0.3, 0.7]),
                                                                pi=np.array([[0.6, 0.1, 0.3], [0.3, 0.5, 0.2]]))
    smoc = StanMultinomialOptimizeConsensus()
    problem = smoc.sample(dgp, parameters)
    return SampleForTest(problem, parameters)


class TestStanMultinomialOptimizeConsensus(BaseTestGenerativeConsensusModel):
    model_cls = StanMultinomialOptimizeConsensus
    sampling_funcs = [easy_sample, sample_non_finite_gradient, more_labels_than_classes_sample]


class TestStanMultinomialEtaOptimizeConsensus(BaseTestGenerativeConsensusModel):
    model_cls = StanMultinomialEtaOptimizeConsensus
    sampling_funcs = [easy_sample, sample_non_finite_gradient, more_labels_than_classes_sample]

