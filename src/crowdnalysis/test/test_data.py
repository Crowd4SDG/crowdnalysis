import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from ..data import Data
from ..problems import DiscreteConsensusProblem


class TEST:
    TASK_IDS = [12, 10, 11]
    TASK_KEYS = [f"t{i}" for i in TASK_IDS]
    TASK_RUN_IDS = [10, 11, 12, 11, 12, 11, 10, 10, 12, 10, 11, 12]
    USER_IDS = ["u2", "u0", "u1", "u1", "u2", "u2", "u0", "u1", "u0", "u3", "u3", "u3"]
    QUESTIONS = ["question_0", "question_1"]
    CATEGORIES = {QUESTIONS[0]: CategoricalDtype(categories=["Yes", "No", "Not sure"], ordered=False),
                  QUESTIONS[1]: CategoricalDtype(categories=["A", "B", "C", "Not answered"], ordered=False)}
    ANSWER_0 = ["No", "No", "Yes", "Yes", "Not sure", "Yes", "Not sure", "Yes", "No", "No", "Yes", "No"]
    ANSWER_1 = ["C", "Not answered", "B", "B", "C", "A", "B", "A", "B", "Not answered", "A", "A"]
    EXTRA_COL = "extra_col"
    EXTRA_COL_VAL = ["".join(random.sample("XYZ", 3)) for i in range(len(TASK_RUN_IDS))]


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
def fixt_data(fixt_df) -> Data:
    d = Data.from_df(fixt_df, data_src="test", questions=TEST.QUESTIONS, task_ids=TEST.TASK_IDS,
                     categories=TEST.CATEGORIES, annotator_id_col_name=Data.COL_USER_ID)
    # print("\n", d.df)
    return d


@pytest.fixture(scope="module")
def dcp_kwargs() -> Dict[str, Dict[str, Any]]:
    kwargs_common = {
        "n_tasks": len(TEST.TASK_IDS),
        "n_workers": np.unique(TEST.USER_IDS).size,
        "t_A": [TEST.TASK_IDS.index(id_) for id_ in TEST.TASK_RUN_IDS],
        "w_A": [np.where(np.unique(TEST.USER_IDS) == id_)[0][0] for id_ in TEST.USER_IDS]}
    kwargs_return = {}
    for q, answers in [(TEST.QUESTIONS[0], TEST.ANSWER_0), (TEST.QUESTIONS[1], TEST.ANSWER_1)]:
        kwargs_q = kwargs_common.copy()
        kwargs_q.update({"n_labels": TEST.CATEGORIES[q].categories.size,
                         "f_A": [TEST.CATEGORIES[q].categories.get_loc(a) for a in answers]})
        kwargs_return[q] = kwargs_q
    # print("\n", kwargs_return)
    return kwargs_return


@pytest.fixture
def mock_pybossa_csv(monkeypatch, fixt_single_file_records):
    """Mocks `pandas.read_csv()` for three different Pybossa files"""
    def mock_read(*args, **kwargs):
        file_name = args[0]
        if file_name == "task.csv":
            records = np.array(TEST.TASK_KEYS, dtype=[("task_key", "U15")])
        elif file_name == "task_info.csv":
            records = np.array(TEST.TASK_IDS, dtype=[(Data.COL_TASK_ID, "i4")])
        else:  # "task_run.csv"
            records = fixt_single_file_records
        df = pd.DataFrame.from_records(records)
        # print("df:\n", df)
        return df
    monkeypatch.setattr(pd, "read_csv", mock_read, raising=True)


@pytest.fixture
def mock_single_file_csv(monkeypatch, fixt_df):
    def mock_read(*args, **kwargs):
        return fixt_df
    monkeypatch.setattr(pd, "read_csv", mock_read, raising=True)


def assert_data_object(d: Data):
    """Helper for assertion of a created `Data` object"""
    # First assert the shape
    asked_questions = set(d.df.columns).intersection(TEST.QUESTIONS)
    assert d.df.shape == (len(TEST.TASK_RUN_IDS),
                          len([Data.COL_TASK_ID, Data.COL_USER_ID, *asked_questions, TEST.EXTRA_COL,
                               Data.COL_TASK_INDEX, Data.COL_WORKER_INDEX, *[Data.COL_QUESTION_INDEX(q)
                                                                             for q in asked_questions]]))
    # Assert task indices follow the order of the given TEST.TASK_IDS
    assert np.array_equal(d.df[Data.COL_TASK_ID], [TEST.TASK_IDS[ix] for ix in d.df[Data.COL_TASK_INDEX]])
    # Assert annotator indices follow the order of the sorted unique TEST.USER_IDS
    assert np.array_equal(d.df[Data.COL_USER_ID], [np.unique(TEST.USER_IDS)[ix] for ix in d.df[Data.COL_WORKER_INDEX]])
    # Assert answer indices for a question follow the order of its given TEST.CATEGORIES
    for q in asked_questions:
        assert isinstance(d.df[q].dtype, CategoricalDtype)
        if d.data_src == "test":  # != "test_aidr", because in `Data.from_aidr()` we do not send categories
            assert np.array_equal(d.df[q], [TEST.CATEGORIES[q].categories[ix]
                                            for ix in d.df[Data.COL_QUESTION_INDEX(q)]])


def test_from_df(fixt_data):
    # print("\n", data.df)
    assert_data_object(fixt_data)


def test_from_pybossa(mock_pybossa_csv):
    d = Data.from_pybossa(
            "task_run.csv",
            questions=TEST.QUESTIONS,
            data_src="test",
            preprocess=lambda x: x,
            task_ids=TEST.TASK_IDS,
            categories=TEST.CATEGORIES,
            task_info_file="task_info.csv",
            task_file="task.csv",
            field_task_key="task_key",
            other_columns=[TEST.EXTRA_COL])
    # print("\n", d.df)
    assert_data_object(d)


def test_from_mturk(mock_single_file_csv):
    d = Data.from_mturk(
        "mturk_task.csv",  # dummy value
        questions=TEST.QUESTIONS,
        data_src="test",
        preprocess=lambda x: x,
        task_ids=TEST.TASK_IDS,
        categories=TEST.CATEGORIES,
        other_columns=[TEST.EXTRA_COL])
    # print("\n", d.df)
    assert_data_object(d)


def test_from_aidr(mock_single_file_csv):
    d = Data.from_aidr(
        "aidr.csv",  # dummy value
        questions=[TEST.QUESTIONS[1]],
        data_src="test_aidr",
        preprocess=lambda x: x,
        task_ids=TEST.TASK_IDS,
        other_columns=[TEST.EXTRA_COL])
    # print("\n", d.df)
    assert_data_object(d)


def test_get_categories(fixt_data):
    assert fixt_data.get_categories() == TEST.CATEGORIES


def test_get_dcp(dcp_kwargs, fixt_data):
    for q in TEST.QUESTIONS:
        dcp1 = fixt_data.get_dcp(q)
        # print("\n", dcp_kwargs[q])
        dcp2 = DiscreteConsensusProblem(**dcp_kwargs[q])
        assert dcp1 == dcp2


def test_get_field(fixt_data):
    task_indices = [0, 1]  # -> TEST.TASK_IDS: 12, 10
    for field, answers in [(TEST.QUESTIONS[0], TEST.ANSWER_0), (TEST.QUESTIONS[1], TEST.ANSWER_1)]:
        indices = []
        for ix in task_indices:
            indices += [i for i, x in enumerate(TEST.TASK_RUN_IDS) if x == TEST.TASK_IDS[ix]]
        # print("indices:", indices)
        output = np.array(answers)[indices]
        # print("output:", output)
        # print("return:", data.get_field(task_indices=task_indices, field=field, unique=False))
        assert np.array_equal(output, fixt_data.get_field(task_indices=task_indices, field=field, unique=False))
        assert np.array_equal(np.unique(output),
                              np.unique(fixt_data.get_field(task_indices=task_indices, field=field, unique=True)))


def test_set_classes(fixt_data):
    for q in TEST.QUESTIONS:
        classes_ = TEST.CATEGORIES[q].categories.tolist()
        # Assert a ValueError is raised if the classes is not a sublist of the categories starting from index 0
        with pytest.raises(ValueError):
            fixt_data.set_classes(q, [classes_[0], classes_[-1]])
        assert fixt_data.get_classes(q) == list(range(len(classes_)))
        classes_.pop(-1)
        fixt_data.set_classes(q, classes_)
        assert fixt_data.get_classes(q) == list(range(len(classes_)))
        # Reset the classes
        fixt_data.set_classes(q, None)
        assert fixt_data.get_classes(q) == list(range(len(TEST.CATEGORIES[q].categories.tolist())))


def test_valid_rows(fixt_data):
    for q in TEST.QUESTIONS:
        fixt_data.set_condition(q, None)
        assert np.array_equal(fixt_data.valid_rows(q), fixt_data.df.index)


def test_set_condition(fixt_data):
    q = TEST.QUESTIONS[1]
    fixt_data.set_condition(q, f"{TEST.QUESTIONS[0]} in ['Yes', 'Not sure']")
    assert np.array_equal(fixt_data.valid_rows(q),
                          [ix for ix, val in enumerate(TEST.ANSWER_0) if val in ["Yes", "Not sure"]])
    # Clear the condition
    fixt_data.set_condition(q, None)
    assert np.array_equal(fixt_data.valid_rows(q), fixt_data.df.index)


def test_make_and_condition():
    assert Data.make_and_condition([]) == ""
    assert Data.make_and_condition([('A', 5), ('B', 'Yes'), ('C C', ['Yes', True])]) \
           == "`A`==5 & `B`=='Yes' & `C C` in ['Yes', True]"
    with pytest.raises(ValueError):
        Data.make_and_condition([('A', 5), ('B', 'Yes'), ('C C', )])


def test_get_others(fixt_data):
    """`Data.[get_tasks, get_workers, get_annotations]`"""
    q = TEST.QUESTIONS[1]
    for method_, col_name in [(fixt_data.get_tasks, Data.COL_TASK_INDEX),
                              (fixt_data.get_workers, Data.COL_WORKER_INDEX),
                              (fixt_data.get_annotations, Data.COL_QUESTION_INDEX(q))]:
        assert np.array_equal(method_(q), fixt_data.df[col_name])
        fixt_data.set_condition(q, f"{TEST.QUESTIONS[0]} in ['Yes', 'Not sure']")
        assert np.array_equal(method_(q),
                          [fixt_data.df[col_name][ix]
                           for ix, val in enumerate(TEST.ANSWER_0) if val in ["Yes", "Not sure"]])
        # Clear the condition
        fixt_data.set_condition(q, None)
