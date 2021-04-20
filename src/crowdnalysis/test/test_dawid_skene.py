from .. import log
from ..dawid_skene import DawidSkene
import numpy as np


def test_sampling():
    dgp = DawidSkene.DataGenerationParameters(n_tasks=1000,num_annotations_per_task=10)
    parameters = DawidSkene.Parameters(tau=np.array([0.3, 0.7]), pi=np.array([[[0.9, 0.1], [0.2, 0.8]]]))
    ds = DawidSkene()
    p = ds.sample(dgp, parameters)
    log.info(p)
    log.info(p.to_json())
