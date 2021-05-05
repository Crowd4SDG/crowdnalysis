from .. import log
from ..cmdstan import StanMultinomialOptimizeConsensus
from . import close, distance
import numpy as np


def easy_sample():
    dgp = StanMultinomialOptimizeConsensus.DataGenerationParameters(n_tasks=1000, num_annotations_per_task=20)
    parameters = StanMultinomialOptimizeConsensus.Parameters(tau=np.array([0.3, 0.7]),
                                                             pi=np.array([[0.9, 0.1], [0.2, 0.8]]))
    smoc = StanMultinomialOptimizeConsensus()
    problem = smoc.sample(dgp, parameters)
    return problem, parameters

def sample_non_finite_gradient():
    dgp = StanMultinomialOptimizeConsensus.DataGenerationParameters(n_tasks=100000, num_annotations_per_task=3)
    parameters = StanMultinomialOptimizeConsensus.Parameters(tau=np.array([0.05, 0.95]),
                                                             pi=np.array([[0.999, 0.001], [0.2, 0.8]]))
    smoc = StanMultinomialOptimizeConsensus()
    problem = smoc.sample(dgp, parameters)
    return problem, parameters


def test_sampling():
    problem, parameters = easy_sample()
    # log.info(problem)
    # log.info(parameters)

def test_all_samples():
    sample_functions = [easy_sample, sample_non_finite_gradient]
    for s in sample_functions:
        _test_fit_and_compute_consensus(s)

def _test_fit_and_compute_consensus(sample_f):
    problem, parameters = sample_f()
    multinomial = StanMultinomialOptimizeConsensus()
    consensus, parameters_learned = multinomial.fit_and_compute_consensus(problem)
    tau_distance = distance(parameters_learned.tau, parameters.tau)
    log.debug("Distance between learned tau and real tau: %f", tau_distance)
    pi_distance = distance(parameters_learned.pi, parameters.pi)
    log.debug("Distance between learned pi and real pi: %f", pi_distance)
    assert close(parameters_learned.tau, parameters.tau)
    assert close(parameters_learned.pi, parameters.pi)


# TODO: Fill in the test methods below

def test_fit():
    pass


def test_compute_consensus():
    pass
