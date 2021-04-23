from ..dawid_skene import DawidSkene
from ..simple import Probabilistic, MajorityVoting
from ..measures import Accuracy
import numpy as np


def sample():
    dgp = DawidSkene.DataGenerationParameters(n_tasks=1000, n_annotations_per_task=200)
    parameters = DawidSkene.Parameters(tau=np.array([0.3, 0.7]), pi=np.array([[[0.6, 0.3, 0.1], [0.3, 0.6, 0.1]]]))
    ds = DawidSkene()
    problem = ds.sample(dgp, parameters)

    return problem, parameters


def test_fit_and_compute_consensus():
    pr = Probabilistic()
    problem, params = sample()
    consensus, _ = pr.fit_and_compute_consensus(problem)
    assert Accuracy.evaluate(problem.f_T, consensus) == 1.0
    # print(Accuracy.evaluate(problem.f_T, consensus))
    # raise Exception()


def test_majority_voting():
    mv = MajorityVoting()
    problem, params = sample()
    consensus, _ = mv.fit_and_compute_consensus(problem)
    assert Accuracy.evaluate(problem.f_T, consensus) == 1.0
