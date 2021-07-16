"""Module for analysing crowd-sourced data"""

from typing import List

import numpy as np
import pandas as pd

from .import data
from .consensus import AbstractSimpleConsensus, GenerativeAbstractConsensus


def compute_crossed(model, d_others, ref_consensuses):
    """Compute parameters when the true labels are known

    Args:
        model (consensus.AbstractConsensus):
        d_others (Dict[str, data.Data]): Dictionary of (data source name, Data) key-value pairs
        ref_consensuses (Dict[str, np.ndarray]): Dictionary of (question, consensus) key-value pairs

    Returns:
        Dict[str, Dict[str, AbstractSimpleConsensus.Parameters]]: {data_source: {question: {_p: [...], _pi: [...]}}}
    """
    parameters_others = {}
    for d_name in d_others:
        parameters_others[d_name] = model.fit_many_from_data(d_others[d_name], ref_consensuses)
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
    # print("consensus_col_labels:", consensus_col_labels)
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
                         measures, dgps: List[GenerativeAbstractConsensus.DataGenerationParameters], repeats=2):
    """Makes a predictive analysis for each community based on the expert parameters.

    `repeats` times analysis is made for each model, for each no of tasks, for each no of annotations per task.

    Args:
        question (str): The question to be compared. e.g. "severity"
        expert_data_src (str): Data source of the experts' data
        expert_parameters (Dict[str, np.ndarray]): {_p: [...], _pi: [...]}
        parameters_others (Dict[str, Dict[str, np.ndarray]]): {data_source: {_p: [...], _pi: [...]}}:
        generative_model (consensus.GenerativeAbstractConsensus):
        # TODO (OM, 20210304): When GenerativeAbstractConsensus methods are converted to class/static methods
        # this argument can be omitted.
        models (Dict[str, consensus.AbstractConsensus]]: Dictionary of (model_abbreviation, model_instance) pairs
        measures (Dict[str, measures.AbstractMeasure]): Dictionary of (measure name, measure class) pairs
        dgps: List of data generation parameters
        repeats (int): Number of times the analysis should be repeated

    Returns:
        pd.DataFrame.
    """
    crowds_parameters = {name: parameters_others[name][question] for name in parameters_others}
    crowds_parameters[expert_data_src] = expert_parameters[question]
    # print(expert_parameters[question])
    return pd.DataFrame.from_records(
        generative_model.evaluate_consensuses_on_linked_samples(
            expert_parameters[question], crowds_parameters, models, measures, dgps, repeats=repeats))


def gen_confusion_matrix(consensus_ref: np.ndarray, consensus_compare: np.ndarray,
                         d_ref: data.Data, question: str) -> pd.DataFrame:
    """Compute confusion matrix for a consensus based on another ref consensus.

    Cells values are the number of tasks. The values are merely the sum of probabilities for the corresponding
    (class, label) pair.
    """
    best_ref = np.argmax(consensus_ref, axis=1)
    categories = d_ref.get_categories()[question].categories.tolist()
    label_names = dict(zip(range(len(categories)), categories))
    # print("label_names:", label_names)
    df_out = pd.DataFrame(consensus_compare).rename(columns=label_names)
    df_out["Ground Truth"] = best_ref
    df_out = df_out.groupby("Ground Truth").sum()
    df_out = df_out.rename(index=label_names)
    df_out = df_out.reindex(categories)  # In case there are labels which were not in `best_ref`
    return df_out
