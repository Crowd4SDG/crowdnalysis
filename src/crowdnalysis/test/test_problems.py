from .. import log
from ..consensus import DiscreteConsensusProblem
import dataclasses
import numpy as np


def test_constructors():
    dcp = DiscreteConsensusProblem(
        n_tasks=2,
        n_workers=2,
        t_A=np.array([0, 1, 1]),
        w_A=np.array([0, 0, 1]),
        n_labels=2,
        f_A=np.array([0, 0, 1]))
    log.info(dcp)
    d = {
        "n_tasks": 2,
        "n_workers": 2,
        "t_A": [0, 1, 1],
        "w_A": [0, 0, 1],
        "n_labels": 2,
        "f_A": [0, 0, 1]}
    dcp = DiscreteConsensusProblem(**d)
    log.info(dcp)
    log.info(dataclasses.asdict(dcp))
    a = dcp.to_json()
    log.info(a)
    b = DiscreteConsensusProblem.from_json(a)
    log.info(b)
