import pytest
from typing import Any, Dict

import numpy as np

from .. import log
from ..consensus import DiscreteConsensusProblem


@pytest.fixture(scope="module")
def dcp_kwargs() -> Dict[str, Any]:
    return {
        "n_tasks": 2,
        "n_workers": 2,
        "t_A": [0, 1, 1],
        "w_A": [0, 0, 1],
        "n_labels": 2,
        "f_A": [0, 0, 1]}


@pytest.fixture(scope="module")
def dcp(dcp_kwargs) -> DiscreteConsensusProblem:
    dcp_ = DiscreteConsensusProblem(**dcp_kwargs)
    log.info("dcp from kwargs: {}".format(dcp_))
    log.info("dcp.to_json(): {}".format(dcp_.to_json()))
    return dcp_


def test_constructors(dcp_kwargs, dcp):
    assert dcp.n_tasks == dcp_kwargs["n_tasks"]
    assert np.array_equal(dcp.t_A, dcp_kwargs["t_A"])
    assert dcp.f_T is None
    kw = dcp_kwargs.copy()
    kw["n_tasks"] = dcp_kwargs["n_tasks"] * 2
    assert not dcp == DiscreteConsensusProblem(**kw)


def test_json_methods(dcp):
    assert dcp == DiscreteConsensusProblem.from_json(dcp.to_json())


# TODO (OM, 20210512): Fill in below. Related method is never used.
def test_product_from_dict():
    pass

