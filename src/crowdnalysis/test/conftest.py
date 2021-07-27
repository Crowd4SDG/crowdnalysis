import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from ..data import Data


class TEST:
    TASK_IDS = [12, 10, 11, 13]  # Task 13 has no annotations
    TASK_KEYS = [f"t{i}" for i in TASK_IDS]
    TASK_RUN_IDS = [10, 11, 12, 11, 12, 11, 10, 10, 12, 10, 11, 12]
    USER_IDS = ["u2", "u0", "u1", "u1", "u2", "u2", "u0", "u1", "u0", "u3", "u3", "u3"]
    QUESTIONS = ["question_0", "question_1"]
    CATEGORIES = {QUESTIONS[0]: CategoricalDtype(categories=["Yes", "No", "Not sure"], ordered=False),
                  QUESTIONS[1]: CategoricalDtype(categories=["A", "B", "C", "Not answered"], ordered=False)}
    ANSWER_0 = ["No", "No", "Yes", "Yes", "Not sure", "Yes", "Not sure", "Yes", "No", "No", "Yes", "No"]
    ANSWER_1 = ["C", "Not answered", "B", "B", "C", "A", "B", "A", "B", "Not answered", "A", "A"]
    EXTRA_COL = "extra_col"
    EXTRA_COL_VAL = [f"img_{id_}" for id_ in TASK_RUN_IDS]


@pytest.fixture(scope="session")
def fixt_single_file_records() -> np.ndarray:
    return np.array([(TEST.USER_IDS[i], TEST.TASK_RUN_IDS[i], TEST.ANSWER_0[i], TEST.ANSWER_1[i], TEST.EXTRA_COL_VAL[i])
                     for i in range(len(TEST.TASK_RUN_IDS))],
                    dtype=[(Data.COL_USER_ID, "U15"), (Data.COL_TASK_ID, "i4"),
                           (TEST.QUESTIONS[0], "U15"), (TEST.QUESTIONS[1], "U15"), (TEST.EXTRA_COL, "U15")])


@pytest.fixture(scope="session")
def fixt_df(fixt_single_file_records):
    df = pd.DataFrame.from_records(fixt_single_file_records)
    return df


@pytest.fixture(scope="session")
def fixt_data_factory(fixt_df):
    class DataFactory:
        @classmethod
        def make(cls, data_src="test"):
            d = Data.from_df(fixt_df, data_src=data_src, questions=TEST.QUESTIONS, task_ids=TEST.TASK_IDS,
                             categories=TEST.CATEGORIES, annotator_id_col_name=Data.COL_USER_ID)
            # print("\n", d.df)
            return d
    return DataFactory


@pytest.fixture(scope="session")
def fixt_data(fixt_data_factory) -> Data:
    d = fixt_data_factory.make()
    # print("\n", d.df)
    return d

