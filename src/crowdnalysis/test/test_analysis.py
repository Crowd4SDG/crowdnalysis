import dataclasses

import numpy as np
import pandas as pd
import pytest

from . import close
from .conftest import TEST
from .. import analysis, factory
from ..measures import Accuracy


@pytest.fixture(scope="module")
def d_expert(fixt_data_factory):  # see conftest.py
    d = fixt_data_factory.make(data_src="test_expert")
    return d


@pytest.fixture(scope="module")
def d_others(fixt_data_factory):
    d_others_ = {}
    for i in range(2):
        data_src = f"test_crowd_{i}"
        d_others_[data_src] = fixt_data_factory.make(data_src=data_src)
    # print([d.data_src for d in d_others_.values()])
    return d_others_


@pytest.fixture(scope="module")
def expert_consensuses_n_parameters(d_expert):
    model_ = factory.Factory.make("DawidSkene")
    expert_consensuses, expert_parameters = model_.fit_and_compute_consensuses_from_data(d_expert, TEST.QUESTIONS)
    return expert_consensuses, expert_parameters


@pytest.fixture(scope="module")
def parameters_others(expert_consensuses_n_parameters, d_others):
    model_ = factory.Factory.make("DawidSkene")
    expert_consensuses, _ = expert_consensuses_n_parameters
    parameters_others_ = analysis.compute_crossed(model_, d_others, expert_consensuses)
    return parameters_others_


def test_compute_crossed(expert_consensuses_n_parameters, parameters_others):
    _, expert_parameters = expert_consensuses_n_parameters
    for question, dgp in expert_parameters.items():
        expert_dgp = dataclasses.asdict(dgp)
        for param_name, expert_param_val in expert_dgp.items():
            for other in parameters_others:
                other_dgp = dataclasses.asdict(parameters_others[other][question])
                assert np.array_equal(expert_param_val, other_dgp[param_name])


def test_prospective_analysis(d_expert, expert_consensuses_n_parameters, parameters_others):
    _, expert_parameters = expert_consensuses_n_parameters
    measures = {Accuracy.name: Accuracy}
    repeats = 2
    question = TEST.QUESTIONS[0]
    model_ = factory.Factory.make("DawidSkene")
    ds = factory.Factory.make("DawidSkene")
    mv = factory.Factory.make("MajorityVoting")
    models = {"DS": ds, "MV": mv}
    options = {"n_tasks": [10, 100],
               "n_annotations_per_task": [3, 9]}
    scenarios = ds.DataGenerationParameters.product_from_dict(options)
    df = analysis.prospective_analysis(question, d_expert.data_src, expert_parameters, parameters_others,
                                       generative_model=model_, models=models, measures=measures, dgps=scenarios,
                                       repeats=repeats)
    # print(df)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == repeats * len(models) * len(options["n_tasks"]) * \
           len(options["n_annotations_per_task"]) * (len(parameters_others.keys()) + 1)  # +1 for experts


def test_compare_data_to_consensus(d_expert, d_others, expert_consensuses_n_parameters):
    expert_consensuses, _ = expert_consensuses_n_parameters
    for d_other in d_others.values():
        for question in TEST.QUESTIONS:
            df = analysis.compare_data_to_consensus(d_expert, d_other, expert_consensuses, question,
                                                    add_total_cols=True)
            # assert df is DataFrame
            assert isinstance(df, pd.DataFrame)
            # #Rows == #Labels
            assert df.shape[0] == d_other.n_labels(question)
            # #Columns == #Classes
            assert (df.shape[1] - 1 == len(d_other.get_classes(question)))
            # Total column == Sum(other columns)
            assert np.allclose(df.iloc[:, -1].to_numpy(), df.iloc[:, :-1].sum(axis=1).to_numpy())


def test_gen_confusion_matrix(d_expert, expert_consensuses_n_parameters):
    expert_consensuses, _ = expert_consensuses_n_parameters
    for question in TEST.QUESTIONS:
        df = analysis.gen_confusion_matrix(expert_consensuses[question], expert_consensuses[question],
                                           d_expert, question)
        # assert df is DataFrame
        assert isinstance(df, pd.DataFrame)
        # assert all categories are in the index and in the columns
        categories = d_expert.get_categories()[question].categories.tolist()
        assert df.index.to_list() == categories
        assert df.columns.to_list() == categories
        # assert dataframe total sum = # of tasks
        assert close(np.nansum(df.to_numpy()), len(np.unique(d_expert.get_tasks(question))))
        # TODO (OM, 20210716): assert the values of rows?
