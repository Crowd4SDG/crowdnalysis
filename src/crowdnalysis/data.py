from typing import Tuple, Any, List, Union

import pandas as pd
import numpy as np
from .problems import JSONDataClass, DiscreteConsensusProblem
import dataclasses
from dataclasses import dataclass
#
# class AbstractCondition:
#     def filter(self):
#
# @dataclass
# class BasicCondition(JSONDataClass, AbstractCondition):
#     question: Any = None
#     value: Any = None
#
# class AndCondition()
#     init(el1,el2):

class Data:
    """
    This is the main class for storing answers
    - by a set of annotators, 
    - to a set of tasks, 
    - each task with the very same questions.
    """

    COL_TASK_INDEX = "task_index"  # column name in Data.df
    COL_WORKER_INDEX = "annotator_index"

    def __init__(self):
        """
        The Data object must not be initialized by constructor. 
        Instead, use the from methods
        """
        # Make instance attributes' definitions explicit
        self.df = None
        self.data_src = None
        self.questions = None
        self.task_ids = None
        self.task_index_from_id = None
        self.n_tasks = None
        self.annotator_ids = None
        self.annotator_index_from_id = None
        self.n_annotators = None
        self.conditions = None  # type: List[Tuple[str, Union[Any, List[Any]]]]
        self.question_valid_rows = {}

    @classmethod
    def from_df(cls, df, data_src=None, task_id_col_name="task_id", annotator_id_col_name="annotator_id",
                questions=None, task_ids=None, categories=None):
        """ 
        Create a Data object from dataframe df.
        The dataframe contains one row for each annotation, 
        each row has a task_id, an annotator id and the answers to a set of 
        questions, each answer in one column
        """
        d = Data()

        d.data_src = data_src

        if questions is None:
            questions = list(df.columns)
            questions.remove(task_id_col_name)
            questions.remove(annotator_id_col_name)

        d.set_questions(questions)

        df = df.copy()

        np_task_ids = df[task_id_col_name].to_numpy()
        if task_ids is None:
            task_ids = np.unique(np_task_ids)
        #t = np.unique(np_task_ids)
        #print(t==task_ids)
        d.set_task_ids(task_ids)
        inverse = np.vectorize(d.task_index_from_id.get)(np_task_ids)
        df[cls.COL_TASK_INDEX] = inverse

        #print(df[annotator_id_col_name].to_numpy())
        annotator_ids, inverse = np.unique(df[annotator_id_col_name].to_numpy(), return_inverse=True)
        d.set_annotator_ids(annotator_ids)
        df["annotator_index"] = inverse

        d.set_df(df, categories)

        return d

    @staticmethod
    def _make_query(conditions: List[Tuple[str, Union[Any, List[Any]]]]):
        """Creates a query for use in DataFrame.query()

        Examples:
            >>> _make_query([('info_0', 5), ('info_2', ['Yes', 'Not answered'])])
            "`info_0`==5 & `info_2` in ['Yes', 'Not answered']"
        """
        def esc_str(v):
            return v if not isinstance(v, str) else "'{}'".format(v)
        return " & ".join("`{}`{}{}".format(q, "==" if not isinstance(a, list) else " in ", str(esc_str(a)))
                          for q, a in conditions)

    def set_condition(self, question: str, conditions: List[Tuple[str, Union[Any, List[Any]]]]):
        """Identifies valid rows of data.df for the `question` according to the `conditions`.

        The question is asked if all conditions are satisfied.
        Hence, `conditions` are treated as a conjunction of clauses.

        Args:
            question: Column name in the `Data.df` dataframe
            conditions: List of (column name for dependency question, answer).
                The answer can be a single literal or a list of literals.

        Examples:
            set_condition('info_3', [('info_0', 'Yes'), ('info_2', ['Yes', 'Not answered'])])

        """
        query_ = Data._make_query(conditions)
        # print("question:", question, "query:", query_)
        self.question_valid_rows[question] = self.df.query(query_).index.tolist()

    def valid_rows(self, question: str) -> List:
        """Return the indices of the valid rows for the `question`"""
        if question in self.question_valid_rows:
            return self.question_valid_rows[question]
        else:
            return self.df.index.tolist()

    @classmethod
    def _preprocess(cls, df, questions, preprocess=lambda x: x, other_columns=[]):
        df = df.copy()
        columns_to_retain = ["task_id", "user_id"] + questions + other_columns
        df = preprocess(df)
        df = df[columns_to_retain]
        return df

    @classmethod
    def _from_single_file(cls, file_name, questions, data_src=None, preprocess=lambda x: x, task_ids=None, categories=None,
                          other_columns=[], delimiter=","):
        """
        Create a Data object from a file.
        """
        df = pd.read_csv(file_name, delimiter=delimiter)
        # print("_from_single_file -> df columns before preprocessing: ", df.columns)
        df = cls._preprocess(df, questions, preprocess, other_columns)
        # print("_from_single_file -> df columns AFTER preprocessing: ", df.columns)

        return cls.from_df(df, data_src=data_src, annotator_id_col_name="user_id", questions=questions, task_ids=task_ids, categories=categories)

    @classmethod
    def from_pybossa(cls, file_name, questions, data_src=None, preprocess=lambda x: x, task_ids=None, categories=None,
                     task_info_file=None, task_file=None, field_task_key="info_media_0", other_columns=[], delimiter=","):
        """ 
        Create a Data object from a Pybossa file.
        """

        def get_tasks(t_csv, ti_csv, field_task_id, field_task_key):
            t_df = pd.read_csv(t_csv, delimiter=delimiter)
            t_df = t_df[field_task_key]
            ti_df = pd.read_csv(ti_csv, delimiter=delimiter)
            ti_df = ti_df[field_task_id]
            t_df = pd.merge(t_df, ti_df, left_index=True, right_index=True)
            # print("get_tasks {} {} ({}) -> \n{}".format("PyBossa", task_file, len(t_df.index), t_df))
            return t_df

        df = pd.read_csv(file_name, delimiter=delimiter)
        if task_info_file is not None:
            t_df = get_tasks(task_file, task_info_file, "task_id", field_task_key)
            df = pd.merge(df, t_df, how="left", left_on="task_id", right_on="task_id")
        # print("from_pybossa -> df columns before preprocessing: ", df.columns)
        df = cls._preprocess(df, questions, preprocess, other_columns=other_columns)
        #  print("from_pybossa -> df columns AFTER preprocessing: ", df.columns)
        # print(df)
        return cls.from_df(df, data_src=data_src, annotator_id_col_name="user_id", questions=questions, task_ids=task_ids, categories=categories)

    @classmethod
    def from_mturk(cls, file_name, questions, data_src=None, preprocess=lambda x: x, task_ids=None, categories=None, other_columns=[], delimiter=","):
        """Create a Data object from an Amazon MTurk file."""
        return cls._from_single_file(file_name, questions, data_src, preprocess, task_ids, categories, other_columns=other_columns, delimiter=delimiter)

    @classmethod
    def from_aidr(cls, file_name, questions, data_src=None, preprocess=lambda x: x, task_ids=None, other_columns=[], delimiter=","):
        """Create a Data object from an AIDR file."""
        # Note: Does NOT send 'categories' arg
        return cls._from_single_file(file_name, questions, data_src, preprocess, task_ids, other_columns=other_columns, delimiter=delimiter)

    def set_questions(self, questions):
        self.questions = questions

    def set_task_ids(self, task_ids):
        self.task_ids = task_ids
        self.task_index_from_id = {id: i for i, id in enumerate(task_ids)}
        self.n_tasks = len(task_ids)

    def set_annotator_ids(self, annotator_ids):
        self.annotator_ids = annotator_ids
        self.annotator_index_from_id = {id: i for i, id in enumerate(annotator_ids)}
        self.n_annotators = len(annotator_ids)

    def set_df(self, df, categories=None):
        self.df = df
        for q in self.questions:
            if categories is None:
                cat_vals = self.df[q].unique()
                t = pd.api.types.CategoricalDtype(cat_vals)
                self.df[q] = self.df[q].astype(t)
            else:
                self.df[q] = self.df[q].astype(categories[q])
        question_indexes = [x+"_index" for x in self.questions]
        self.df[question_indexes] = self.df[self.questions].apply(lambda x: x.cat.codes)

    def n_labels(self, question):
        return len(self.df[question].cat.categories)

    def get_tasks(self, question):
        ix = self.valid_rows(question)
        return self.df.iloc[ix][self.COL_TASK_INDEX].to_numpy()

    def get_workers(self, question):
        ix = self.valid_rows(question)
        return self.df.iloc[ix][self.COL_WORKER_INDEX].to_numpy()

    def get_annotations(self, question):
        ix = self.valid_rows(question)
        return self.df.iloc[ix][question + "_index"].to_numpy()

    #def get_question_matrix(self, question):
    #    df = self.df[[self.COL_TASK_INDEX, self.COL_WORKER_INDEX, question+"_index"]]
        #print(df[question].cat.codes)
        #df[question] = df[question].cat.codes
    #    return df.to_numpy()

    def get_categories(self):
        categories = {}
        for q in self.questions:
            #print(self.df[q].cat)
            #print(type(self.df[q].cat))
            #print(self.df[q].dtype)
            categories[q] = self.df[q].dtype
        return categories

    def get_field(self, task_indices, field, unique=False):
        """Return the `field values` within the `self.df` DataFrame that corresponds to the given `task_indices`

        Args:
            task_indices (np.ndarray): array of integer values on the `task_index` column.
            field (str): `data.df` DataFrame column name
            unique (bool): If True, only unique values will be returned

        Returns:
            np.ndarray: `field` values corresponding to and in the order of the given `task_indices`.
        """
        df_vals = self.df[self.df.task_index.isin(task_indices)][[self.COL_TASK_INDEX, field]]  # .isin() loses the order
        df_ind = pd.DataFrame(task_indices, columns=[self.COL_TASK_INDEX])
        df_vals = df_ind.merge(df_vals, how="left", left_on=self.COL_TASK_INDEX, right_on=self.COL_TASK_INDEX)  # Preserve the order
        df_vals = df_vals[field]
        if unique:
            df_vals = df_vals.drop_duplicates()
        return df_vals.to_numpy()

    def get_dcp(self, question):
        return DiscreteConsensusProblem(n_tasks=self.n_tasks,
                                        n_workers=self.n_annotators,
                                        t_A=self.get_tasks(question),
                                        w_A=self.get_workers(question),
                                        f_A=self.get_annotations(question),
                                        n_labels=self.n_labels(question))
