import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from ..data import Data
from ..problems import DiscreteConsensusProblem


TASK_IDS = [12, 10, 11]
TASK_KEYS = [f"t{i}" for i in TASK_IDS]
TASK_RUN_IDS = [10, 11, 12, 11, 12, 11, 10, 10, 12, 10, 11, 12]
USER_IDS = ["u2", "u0", "u1", "u1", "u2", "u2", "u0", "u1", "u0", "u3", "u3", "u3"]
CATEGORIES = {"question_0": CategoricalDtype(categories=["Yes", "No", "Not sure"], ordered=False),
              "question_1": CategoricalDtype(categories=["A", "B", "C", "Not answered"], ordered=False)}
ANSWER_0 = ["No", "No", "Yes", "Yes", "Not sure", "Yes", "Not sure", "Yes", "No", "No", "Yes", "No"]
ANSWER_1 = ["C", "Not answered", "B", "B", "C", "A", "B", "A", "B", "Not answered", "A", "A"]
EXTRA_COL = ["".join(random.sample("XYZ", 3)) for i in range(len(TASK_RUN_IDS))]


@pytest.fixture(scope="module")
def dcp_kwargs() -> Dict[str, Dict[str, Any]]:
    kwargs_common = {
        "n_tasks": len(TASK_IDS),
        "n_workers": np.unique(USER_IDS).size,
        "t_A": [TASK_IDS.index(id_) for id_ in TASK_RUN_IDS],
        "w_A": [np.unique(USER_IDS).tolist().index(id_) for id_ in USER_IDS]}
    kwargs_return = {}
    for q, answers in [("question_0", ANSWER_0), ("question_1", ANSWER_1)]:
        kwargs_q = kwargs_common.copy()
        kwargs_q.update({"n_labels": CATEGORIES[q].categories.size,
                         "f_A": [CATEGORIES[q].categories.tolist().index(a) for a in answers]})
        kwargs_return[q] = kwargs_q
    # print("\n", kwargs_return)
    return kwargs_return


@pytest.fixture(scope="module")
def single_file_records():
    return np.array([(USER_IDS[i], TASK_RUN_IDS[i], ANSWER_0[i], ANSWER_1[i], EXTRA_COL[i])
                     for i in range(len(TASK_RUN_IDS))],
                    dtype=[("user_id", "U15"), ("task_id", "i4"),
                           ("question_0", "U15"), ("question_1", "U15"), ("extra_col", "U15")])


@pytest.fixture
def mock_pybossa_csv(monkeypatch, single_file_records):
    """Mocks `pandas.read_csv()` for three different Pybossa files"""
    def mock_read(*args, **kwargs):
        file_name = args[0]
        if file_name == "task.csv":
            records = np.array(TASK_KEYS, dtype=[("task_key", "U15")])
        elif file_name == "task_info.csv":
            records = np.array(TASK_IDS, dtype=[("task_id", "i4")])
        else:  # "task_run.csv"
            records = single_file_records
        df = pd.DataFrame.from_records(records)
        # print("df:\n", df)
        return df
    monkeypatch.setattr(pd, "read_csv", mock_read, raising=True)


@pytest.fixture
def mock_mturk_csv(monkeypatch, single_file_records):
    """Mocks `pandas.read_csv()` for the single MTurk file"""
    def mock_read(*args, **kwargs):
        records = single_file_records
        df = pd.DataFrame.from_records(records)
        return df
    monkeypatch.setattr(pd, "read_csv", mock_read, raising=True)


@pytest.fixture
def data(mock_mturk_csv) -> Data:
    d = Data.from_mturk(
        "mturk_task_run.csv",
        questions=["question_0", "question_1"],
        data_src="test",
        preprocess=lambda x: x,
        task_ids=TASK_IDS,
        categories=CATEGORIES,
        other_columns=["extra_col"])
    # print("\n", d.df)
    return d


def assert_data_object(d: Data):
    """Helper for assertion of a created `Data` object"""
    # First assert the shape
    assert d.df.shape == (len(TASK_RUN_IDS),
                          len(["task_id", "user_id", "question_0", "question_1", "extra_col", "task_index",
                               "annotator_index", "question_0_index", "question_1_index"]))
    # Assert data source
    assert d.data_src == "test"
    # Assert task indices follow the order of the given TASK_IDS
    assert np.array_equal(d.df["task_id"], [TASK_IDS[ix] for ix in d.df[Data.COL_TASK_INDEX]])
    # Assert annotator indices follow the order of the sorted unique USER_IDS
    assert np.array_equal(d.df["user_id"], [np.unique(USER_IDS)[ix] for ix in d.df[Data.COL_WORKER_INDEX]])
    # Assert answer indices for a question follow the order of its given CATEGORIES
    for q in ["question_0", "question_1"]:
        assert np.array_equal(d.df[q], [CATEGORIES[q].categories[ix] for ix in d.df[f"{q}_index"]])


def test_from_pybossa(mock_pybossa_csv):
    d = Data.from_pybossa(
            "task_run.csv",
            questions=["question_0", "question_1"],
            data_src="test",
            preprocess=lambda x: x,
            task_ids=TASK_IDS,
            categories=CATEGORIES,
            task_info_file="task_info.csv",
            task_file="task.csv",
            field_task_key="task_key",
            other_columns=["extra_col"])
    # print("\n", d.df)
    assert_data_object(d)


def test_from_mturk(data):
    # print("\n", data.df)
    assert_data_object(data)


def test_get_dcp(dcp_kwargs, data):
    for q in ["question_0", "question_1"]:
        dcp1 = data.get_dcp(q)
        # print("\n", dcp_kwargs[q])
        dcp2 = DiscreteConsensusProblem(**dcp_kwargs[q])
        assert dcp1 == dcp2


def test_get_field(data):
    task_indices = [0, 1]  # -> TASK_IDS: 12, 10
    for field, answers in [("question_0", ANSWER_0), ("question_1", ANSWER_1)]:
        indices = []
        for ix in task_indices:
            indices += [i for i, x in enumerate(TASK_RUN_IDS) if x == TASK_IDS[ix]]
        # print("indices:", indices)
        output = np.array(answers)[indices]
        # print("output:", output)
        # print("return:", data.get_field(task_indices=task_indices, field=field, unique=False))
        assert np.array_equal(output, data.get_field(task_indices=task_indices, field=field, unique=False))
        assert np.array_equal(np.unique(output),
                              np.unique(data.get_field(task_indices=task_indices, field=field, unique=True)))


def test_set_condition(data):
    q = "question_1"
    data.set_condition(q, "question_0 in ['Yes', 'Not sure']")
    assert np.array_equal(data.valid_rows(q), [ix for ix, val in enumerate(ANSWER_0) if val in ["Yes", "Not sure"]])
    # Clear the condition
    data.set_condition(q, None)
    assert np.array_equal(data.valid_rows(q), data.df.index)
    data.set_condition(q, "question_0 in ['Yes', 'Not sure']")


def test_set_classes(data):
    for q in ["question_0", "question_1"]:
        classes_ = CATEGORIES[q].categories.tolist()
        assert data.get_classes(q) == list(range(len(classes_)))
        classes_.pop(-1)
        data.set_classes(q, classes_)
        assert data.get_classes(q) == list(range(len(classes_)))
        # Reset the classes
        data.set_classes(q, None)
        assert data.get_classes(q) == list(range(len(CATEGORIES[q].categories.tolist())))


def test_make_and_condition():
    assert Data.make_and_condition([]) == ""
    assert Data.make_and_condition([('A', 5), ('B', 'Yes'), ('C C', ['Yes', True])]) \
           == "`A`==5 & `B`=='Yes' & `C C` in ['Yes', True]"
    with pytest.raises(ValueError):
        Data.make_and_condition([('A', 5), ('B', 'Yes'), ('C C', )])
