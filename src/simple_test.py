import os
import sys
import traceback

import pandas as pd

from crowdnalysis.factory import Factory
from crowdnalysis.data import Data


DATA_PATH = "../../albania-analysis/data"
TEST_FILE = "albaniaeq-expert-twitter"


class DF_COL:
    USER_ID = "user_id"
    TASK_ID = "task_id"
    RELEVANT = "relevant"
    SEVERITY = "severity"
    COMPACT_SEVERITY = "compact_severity"


class CitSciDataEarthquake:
    """Citizen Science Data base class"""

    # TODO: (OM, 20201201): An upper abstract class CitSciData that allows parametric definition of Qs & As

    class AnswerSeverity:
        NO_DAMAGE = "no-damage"
        MODERATE_DAMAGE = "moderate-damage"
        IRRELEVANT = "irrelevant"
        MINIMAL_DAMAGE = "minimal-damage"
        SEVERE_DAMAGE = "severe-damage"

    def __init__(self, data_dir, task_file, field_task_key, field_task_id, field_user_id,
                 field_severity, field_relevant, task_run_file=None, data_src=""):
        self.data_src = data_src
        self.data_dir = data_dir
        self.task_file = task_file
        self.task_run_file = task_run_file if task_run_file else self.task_file
        self.field_task_key = field_task_key
        self.field_task_id = field_task_id
        self.field_user_id = field_user_id
        self.field_severity = field_severity
        self.field_relevant = field_relevant

    def get_tasks(self, **kwargs):
        """Extract task keys & ids

        Returns:
            pd.DataFrame: Two column labels <task_key>, <task_id> are of the given data file.
        """
        raise NotImplementedError

    def _unify_annotators(self, df, lbl="Anonymous"):
        df_copy = df.copy(deep=True)
        df_copy[self.field_user_id] = lbl  # Treat all annotators as one
        return df_copy

    def _fill_missing_severity(self, df, lbl=None):
        df_copy = df.copy(deep=True)
        df_copy.loc[df[self.field_severity].isnull(), self.field_severity] = lbl if lbl else self.AnswerSeverity.IRRELEVANT
        return df_copy

    def preprocess(self, df):
        """Preprocessing of the given data

        Args:
            df: pd.DataFrame

        Returns:
            pd.DataFrame
        """
        return df

    def get_data(self, questions, preprocess=None, task_ids=None, categories=None):
        """
        Calls the appropriate crowdnalysis.data.Data.from~ method

        Returns:
            Data

        """
        raise NotImplementedError


class PyBossaData(CitSciDataEarthquake):

    def __init__(self, task_file_ext="_task.csv", task_info_file_ext="_task_info_only.csv",
                 task_run_file_ext="_task_run.csv", **kwargs):
        super().__init__(**kwargs)  # Invoke parent class init
        self.task_file_ext = task_file_ext  # Extension for the file with task details, alas without task ids!
        self.task_info_file_ext = task_info_file_ext  # Extension for the file with task ids only, coherent with above
        self.task_run_file_ext = task_run_file_ext  # Extension for the file with task answers
        self.task_run_file = self.task_file + self.task_run_file_ext

    def get_tasks(self):
        t_csv = os.path.join(self.data_dir, self.task_file + self.task_file_ext)
        t_df = pd.read_csv(t_csv)
        t_df = t_df[self.field_task_key]
        ti_csv = os.path.join(self.data_dir, self.task_file + self.task_info_file_ext)
        ti_df = pd.read_csv(ti_csv)
        ti_df = ti_df[self.field_task_id]
        t_df = pd.merge(t_df, ti_df, left_index=True, right_index=True)
        # print("get_tasks {} {} ({}) -> \n{}".format(self.__class__.__name__, self.task_file, len(t_df.index), t_df))
        return t_df

    def preprocess(self, df, unify_annotators=True):
        df = self._fill_missing_severity(df)
        if unify_annotators:
            df = self._unify_annotators(df)
        df = df.rename(columns={self.field_relevant: DF_COL.RELEVANT, self.field_severity: DF_COL.SEVERITY})
        mapping_compact_severity = {self.AnswerSeverity.NO_DAMAGE: self.AnswerSeverity.IRRELEVANT,
                                    self.AnswerSeverity.MODERATE_DAMAGE: self.AnswerSeverity.MODERATE_DAMAGE,
                                    self.AnswerSeverity.IRRELEVANT: self.AnswerSeverity.IRRELEVANT,
                                    self.AnswerSeverity.MINIMAL_DAMAGE: self.AnswerSeverity.MODERATE_DAMAGE,
                                    self.AnswerSeverity.SEVERE_DAMAGE: self.AnswerSeverity.SEVERE_DAMAGE
                                    }
        df[DF_COL.COMPACT_SEVERITY] = df[DF_COL.SEVERITY].map(mapping_compact_severity)  # New column
        # print("Dataframe after Preprocess ({}):\n{}".format(self.__class__.__name__, df))
        return df

    def get_data(self, questions, preprocess=None, task_ids=None, categories=None):
        # print("Reading task run data from {}".format(os.path.join(self.data_dir, self.task_run_file)))
        d = Data.from_pybossa(os.path.join(self.data_dir, self.task_run_file),
                                                questions=questions,
                                                preprocess=preprocess if preprocess else self.preprocess,
                                                task_ids=task_ids,
                                                categories=categories)
        return d


def _run_consensus(d: Data, q: str, algorithm_name):
    alg = Factory.get_consensus_algorithm(algorithm_name)
    consensus, params = alg.compute_consensus(d, q)
    return consensus, params


def main():
    QUESTIONS = [DF_COL.RELEVANT, DF_COL.SEVERITY, DF_COL.COMPACT_SEVERITY]
    pybossa_expert = PyBossaData(
        data_src="PyBossa",
        data_dir=DATA_PATH,
        task_file=TEST_FILE,
        field_task_key="info_media_0",
        field_task_id="task_id",
        field_user_id="user_id",
        field_severity="info_answer_0_tags",
        field_relevant="info_answer_0_relevant")
    d_ref = pybossa_expert.get_data(questions=QUESTIONS)
    for algorithm_name in Factory.algorithms:
        print("Testing consensus algorithm {}".format(algorithm_name))
        consensus, params = _run_consensus(d_ref, QUESTIONS[1], algorithm_name)
        print(consensus[0])
        # try:
        #     consensus, params = _run_consensus(d_ref, QUESTIONS[1], algorithm_name)
        #     print(consensus[0])
        #     print("OK.")
        # except Exception as e:
        #     print("!!! Error with the {} algorithm:\n{}".format(algorithm_name, str(e)))
        #     traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
    main()
