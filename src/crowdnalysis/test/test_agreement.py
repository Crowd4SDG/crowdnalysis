from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from . import close
from .. import agreement
from ..data import Data


class BaseTestDataSimple:
    TASK_IDS = None
    TASK_RUN_IDS = None
    USER_IDS = None
    ANSWERS = None
    QUESTION = None


class TEST_SIMPLE(BaseTestDataSimple):
    TASK_IDS = [0, 1, 2, 3, 4]
    TASK_RUN_IDS = [0, 0, 0, 1, 1, 1, 2, 2, 3]  # task 2 has two, task 3 has one, task 4 has zero annotations
    USER_IDS = ["u0", "u1", "u2", "u0", "u1", "u2", "u0", "u1", "u0"]
    # USER_IDS = [0, 1, 2, 0, 1, 2, 0, 1, 0]
    ANSWERS = ["No", "No", "No", "No", "No", "Yes", "No", "Yes", "No"]
    QUESTION = "question"


class _TEST_KAPPA(BaseTestDataSimple):
    pass


def _make_data(test_data: BaseTestDataSimple) -> Data:
    records = np.array([(test_data.USER_IDS[i], test_data.TASK_RUN_IDS[i], test_data.ANSWERS[i])
                        for i in range(len(test_data.TASK_RUN_IDS))],
                       dtype=[(Data.COL_USER_ID, "U15"), (Data.COL_TASK_ID, "i4"), (test_data.QUESTION, "U15")])
    df = pd.DataFrame.from_records(records)
    d = Data.from_df(df, task_ids=test_data.TASK_IDS, questions=["question"], annotator_id_col_name=Data.COL_USER_ID)
    # print("\n", d.df)
    return d


@pytest.fixture(scope="module")
def data_():
    d = _make_data(TEST_SIMPLE)
    # print("\nData simple:\n", d.df)
    return d


@pytest.fixture(scope="module")
def data_kappa() -> Tuple[BaseTestDataSimple, str, float]:
    """Table for test data copied from Wikipedia

    see https://en.wikipedia.org/wiki/Fleiss%27_kappa#Worked_example

    Returns:
        Tuple: (annotation data, question, kappa value)
    """
    table = np.asarray("""\
        0   0   0   0   14
        0   2   6   4   2
        0   0   3   5   6
        0   3   9   2   0
        2   2   8   1   1
        7   7   0   0   0
        3   2   6   3   0
        2   5   3   2   2
        6   5   2   1   0
        0   2   2   3   7""".split(), float).reshape(10, 5)
    kappa = 0.210

    num_ann_per_task = np.max(np.sum(table, axis=1))
    _TEST_KAPPA.TASK_IDS = np.arange(table.shape[0])
    _TEST_KAPPA.TASK_RUN_IDS = np.array([np.repeat(i, num_ann_per_task) for i in _TEST_KAPPA.TASK_IDS]).flatten()
    _TEST_KAPPA.USER_IDS = np.array([np.arange(num_ann_per_task) for _ in _TEST_KAPPA.TASK_IDS], dtype="U15").flatten()
    _TEST_KAPPA.ANSWERS = np.array([])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            a = np.repeat(j, table[i, j])
            if a.size > 0:
                _TEST_KAPPA.ANSWERS = np.append(_TEST_KAPPA.ANSWERS, a)
    _TEST_KAPPA.QUESTION = "question"
    d = _make_data(_TEST_KAPPA)
    # print("\nData kappa:\n", d.df)
    return d, _TEST_KAPPA.QUESTION, kappa


def test_get_n_ranges(data_):
    n, n_tasks = agreement.get_n_and_ranges(data_, TEST_SIMPLE.QUESTION, ignore_le1_annots=True)
    assert n_tasks == len(TEST_SIMPLE.TASK_IDS)
    assert np.array_equal(n, np.array([[3, 0], [2, 1], [1, 1]]))
    n, _n_tasks = agreement.get_n_and_ranges(data_, TEST_SIMPLE.QUESTION, ignore_le1_annots=False)
    assert n_tasks == len(TEST_SIMPLE.TASK_IDS)
    assert np.array_equal(n, np.array([[3, 0], [2, 1], [1, 1], [1, 0], [0, 0]]))


def test_full_agreement(data_):
    assert agreement.full_agreement_percentage(data_, TEST_SIMPLE.QUESTION) == .2  # Only for task 0.


def test_fleiss_kappa(data_, data_kappa):
    with pytest.raises(ValueError):  # All tasks must have equal number of annotations for standard Fleiss' kappa
        agreement.fleiss_kappa(data_, TEST_SIMPLE.QUESTION)
    # Assert kappa
    d, q, kappa = data_kappa
    assert close(agreement.fleiss_kappa(d, q), kappa, eps=0.001)


def test_gen_fleiss_kappa(data_, data_kappa):
    # TODO (OM, 20210521): Manually calculate the kappa and ensure that it is ~ -0.385
    assert close(agreement.gen_fleiss_kappa(data_, TEST_SIMPLE.QUESTION), -0.385, eps=0.001)
    d, q, kappa = data_kappa
    assert close(agreement.gen_fleiss_kappa(d, q), kappa, eps=0.001)