import pytest

import numpy as np
import pandas as pd

from . import close
from .. import agreement
from ..data import Data


class TEST_SIMPLE:
    TASK_IDS = [0, 1, 2, 3, 4]
    TASK_RUN_IDS = [0, 0, 0, 1, 1, 1, 2, 2, 3]  # task 2 has two, task 3 has one, task 4 has zero annotations
    USER_IDS = ["u0", "u1", "u2", "u0", "u1", "u2", "u0", "u1", "u0"]
    ANSWERS = ["No", "No", "No", "No", "No", "Yes", "No", "Yes", "No"]
    QUESTION = "question"


@pytest.fixture(scope="module")
def data_():
    records = np.array([(TEST_SIMPLE.USER_IDS[i], TEST_SIMPLE.TASK_RUN_IDS[i], TEST_SIMPLE.ANSWERS[i])
                        for i in range(len(TEST_SIMPLE.TASK_RUN_IDS))],
                       dtype=[(Data.COL_USER_ID, "U15"), (Data.COL_TASK_ID, "i4"), (TEST_SIMPLE.QUESTION, "U15")])
    df = pd.DataFrame.from_records(records)
    d = Data.from_df(df, task_ids=TEST_SIMPLE.TASK_IDS, questions=["question"], annotator_id_col_name=Data.COL_USER_ID)
    # print("\n", d.df)
    return d


def test_get_n_ranges(data_):
    n, n_tasks = agreement.get_n_and_ranges(data_, TEST_SIMPLE.QUESTION, ignore_le1_annots=True)
    assert n_tasks == len(TEST_SIMPLE.TASK_IDS)
    assert np.array_equal(n, np.array([[3, 0], [2, 1], [1, 1]]))
    n, _n_tasks = agreement.get_n_and_ranges(data_, TEST_SIMPLE.QUESTION, ignore_le1_annots=False)
    assert n_tasks == len(TEST_SIMPLE.TASK_IDS)
    assert np.array_equal(n, np.array([[3, 0], [2, 1], [1, 1], [1, 0], [0, 0]]))


def test_full_agreement(data_):
    assert agreement.full_agreement_percentage(data_, TEST_SIMPLE.QUESTION) == .2  # Only for task 0.


def test_fleiss_kappa(data_):
    with pytest.raises(ValueError):  # All tasks must have equal number of annotations for standard Fleiss' kappa
        agreement.fleiss_kappa(data_, TEST_SIMPLE.QUESTION)


def test_gen_fleiss_kappa(data_):
    # TODO (OM, 20210521): Manually calculate the kappa and ensure that it is ~ -0.385
    assert close(agreement.gen_fleiss_kappa(data_, TEST_SIMPLE.QUESTION), -0.385, eps=0.001)

