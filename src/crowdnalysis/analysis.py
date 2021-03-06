"""Module for analysing crowd-sourced data"""

from typing import Dict

import numpy as np
import pandas as pd

from . import consensus, data, measures


def compute_crossed(model, d_others, ref_consensuses):
    """Compute parameters when the true labels are known

    Args:
        model (consensus.AbstractConsensus):
        d_others (Dict[str, data.Data]): Dictionary of (data source name, Data) key-value pairs
        ref_consensuses (Dict[str, np.ndarray]): Dictionary of (question, consensus) key-value pairs

    Returns:
        Dict[str, Dict[str, Dict[str, np.ndarray]]]: {data_source: {question: {_p: [...], _pi: [...]}}}
    """
    parameters_others = {}
    for d_name in d_others:
        parameters_others[d_name] = model.fit_many(d_others[d_name], ref_consensuses)
    return parameters_others


def compare_data_to_consensus(d_base, d_compare, base_consensuses, question, add_total_cols=False):
    """Compares the cross comparison of a crowd-sourced data with the consensus

    Args:
        d_base (data.Data): `Data` used to generate the `base_consensuses
        d_compare (data.Data): `Data` to be compared
        base_consensuses (Dict[str, np.ndarray]): Dictionary of (question, consensus) key-value pairs
        question (str): The question to be compared. e.g. "severity"
        add_total_cols (bool): If True, adds checksum columns.

    Returns:
        pd.DataFrame
    """
    consensus_col_labels = dict(zip(list(range(base_consensuses[question].shape[1])),
                                    d_base.get_categories()[question].categories.tolist()))
    df_consensus = pd.DataFrame(base_consensuses[question]).rename(columns=consensus_col_labels)
    df_consensus[data.Data.COL_TASK_INDEX] = list(range(df_consensus.shape[0]))
    df_out = df_consensus.merge(d_compare.df[[question, data.Data.COL_TASK_INDEX]], how="right",
                                left_on=data.Data.COL_TASK_INDEX, right_on=data.Data.COL_TASK_INDEX)
    df_out.drop([data.Data.COL_TASK_INDEX], axis=1, inplace=True)
    df_out = df_out.groupby([question]).sum()
    if add_total_cols:
        # df_out["Total Consensus"] = df_out.sum(axis=1)
        df_count_compare = pd.DataFrame(d_compare.df[question], columns=[question])
        df_out["Total"] = df_count_compare.groupby([question]).size()
    return df_out


def prospective_analysis(question, expert_data_src, expert_parameters, parameters_others, generative_model, models,
                         measures, numbers_of_tasks, annotations_per_task, repeats=2, verbose=False):
    """Makes a predictive analysis for each community based on the expert parameters.

    `repeats` times analysis is made for each model, for each no of tasks, for each no of annotations per task.

    Args:
        question (str): The question to be compared. e.g. "severity"
        expert_data_src (str): Data source of the experts' data
        expert_parameters (Dict[str, np.ndarray]): {_p: [...], _pi: [...]}
        parameters_others (Dict[str, Dict[str, np.ndarray]]): {data_source: {_p: [...], _pi: [...]}}:
        generative_model (consensus.GenerativeAbstractConsensus): #TODO (OM, 20210304): When GenerativeAbstractConsensus methods are converted to class/static methods this argument can be omitted.
        models (Dict[str, consensus.AbstractConsensus]]: Dictionary of (model_abbreviation, model_instance) pairs
        measures (Dict[str, measures.AbstractMeasure]): Dictionary of (measure name, measure class) pairs
        numbers_of_tasks (List[int]): List of different numbers of tasks
        annotations_per_task (List[int]): List of different numbers of annotations per task
        repeats (int): Number of times the analysis should be repeated
        verbose (bool): If True, prints stages of the analysis

    Returns:
        pd.DataFrame.
    """
    crowds_parameters = {name: parameters_others[name][question] for name in parameters_others}
    crowds_parameters[expert_data_src] = expert_parameters[question]
    return pd.DataFrame.from_records(
        generative_model.evaluate_consensuses_on_linked_samples(
            expert_parameters[question], crowds_parameters, models, measures,
            numbers_of_tasks, annotations_per_task, repeats=repeats, verbose=verbose))


