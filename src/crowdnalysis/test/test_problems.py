from dataclasses import dataclass

import numpy as np
import pytest
from typing import Any, Dict

from .. import log
from ..consensus import DiscreteConsensusProblem
from ..problems import JSONDataClass


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
def manual_n() -> np.ndarray:
    """Hand-computed `n` array from `dcp_kwargs`"""
    return np.array([[[1, 0],
                      [1, 0]],
                     [[0, 0],
                      [0, 1]]])


@pytest.fixture(scope="module")
def dcp_kwargs_2() -> Dict[str, Any]:
    return {
        "n_tasks": 2,
        "n_workers": 2,
        "t_A": [1, 1, 1],  # task 0 has no annotations
        "w_A": [0, 0, 1],
        "n_labels": 2,
        "f_A": [0, 0, 1]}


@pytest.fixture(scope="module")
def manual_n2_filtered() -> np.ndarray:
    """Hand-computed `n` array from `dcp_kwargs_2` w/ `ignore_zero_annots=True`"""
    return np.array([[[2, 0]],  # task 0 is filtered
                     [[0, 1]]])


@pytest.fixture(scope="module")
def manual_n2_unfiltered() -> np.ndarray:
    """Hand-computed `n` array from `dcp_kwargs_2` w/ `ignore_zero_annots=False`"""
    return np.array([[[0, 0],
                      [2, 0]],
                     [[0, 0],
                      [0, 1]]])


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


def test_compute_n(dcp, manual_n, dcp_kwargs_2, manual_n2_filtered, manual_n2_unfiltered):
    n, _ = dcp.compute_n()
    assert (dcp.n_workers, dcp.n_tasks, dcp.n_labels) == n.shape
    assert np.array_equal(n, manual_n)
    assert np.array_equal(dcp.compute_n(ignore_zero_annots=True)[0], dcp.compute_n(ignore_zero_annots=False)[0])

    dcp2 = DiscreteConsensusProblem(**dcp_kwargs_2)
    n2_unfiltered, _ = dcp2.compute_n(ignore_zero_annots=False)
    assert np.array_equal(n2_unfiltered, manual_n2_unfiltered)
    n2_filtered, filtered_tasks = dcp2.compute_n(ignore_zero_annots=True)
    assert np.array_equal(n2_filtered, manual_n2_filtered)
    assert np.array_equal(filtered_tasks, np.array([0]))  # task 0 was filtered


def test_jsondataclass_product_from_dict():
    @dataclass
    class C(JSONDataClass):
        a: int
        b: int
    options = {"a": [10, 100], "b": [3, 9]}
    assert C.product_from_dict(options) == [C(10, 3), C(10, 9), C(100, 3), C(100, 9)]


