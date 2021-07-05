import numpy as np

from .common import BaseTestGenerativeConsensusModel, SampleForTest
from ..dawid_skene import DawidSkene


def sample() -> SampleForTest:
    dgp = DawidSkene.DataGenerationParameters(n_tasks=1000, n_annotations_per_task=20)
    parameters = DawidSkene.Parameters(tau=np.array([0.3, 0.7]), pi=np.array([
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
        [[0.5, 0.2, 0.3], [0.2, 0.5, 0.3]]
    ]))
    ds = DawidSkene()
    problem = ds.sample(dgp, parameters)
    return SampleForTest(problem, parameters)


class TestDawidSkeneConsensus(BaseTestGenerativeConsensusModel):
    model_cls = DawidSkene
    sampling_funcs = [sample]
