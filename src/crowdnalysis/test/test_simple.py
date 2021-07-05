import numpy as np

from .common import BaseTestSimpleConsensusModel, SampleForTest
from ..dawid_skene import DawidSkene
from ..simple import Probabilistic, MajorityVoting


def sample() -> SampleForTest:
    dgp = DawidSkene.DataGenerationParameters(n_tasks=1000, n_annotations_per_task=200)
    parameters = DawidSkene.Parameters(tau=np.array([0.3, 0.7]), pi=np.array([[[0.6, 0.3, 0.1], [0.3, 0.6, 0.1]]]))
    ds = DawidSkene()
    problem = ds.sample(dgp, parameters)
    return SampleForTest(problem, None)


class TestProbabilisticConsensus(BaseTestSimpleConsensusModel):
    model_cls = Probabilistic
    sampling_funcs = [sample]


class TestMajorityVotingConsensus(BaseTestSimpleConsensusModel):
    model_cls = MajorityVoting
    sampling_funcs = [sample]