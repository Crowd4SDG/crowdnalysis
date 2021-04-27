from .. import log
from ..cmdstan import StanMultinomialOptimizeConsensus
from . import close, distance
import numpy as np


def sample():
    dgp = StanMultinomialOptimizeConsensus.DataGenerationParameters(n_tasks=1000, num_annotations_per_task=20)
    parameters = StanMultinomialOptimizeConsensus.Parameters(tau=np.array([0.3, 0.7]),
                                                             pi=np.array([[0.9, 0.1], [0.2, 0.8]]))
    smoc = StanMultinomialOptimizeConsensus()
    problem = smoc.sample(dgp, parameters)
    return problem, parameters


def test_sampling():
    problem, parameters = sample()
    # log.info(problem)
    # log.info(parameters)


def test_fit_and_compute_consensus():
    problem, parameters = sample()
    ds = StanMultinomialOptimizeConsensus()
    consensus, parameters_learned = ds.fit_and_compute_consensus(problem)
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
