import pandas as pd
import numpy as np


class Data:
    """
    This is the main class for storing answers
    - by a set of annotators, 
    - to a set of tasks, 
    - each task with the very same questions.
    """
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
        df["task_index"] = inverse

        #print(df[annotator_id_col_name].to_numpy())
        annotator_ids, inverse = np.unique(df[annotator_id_col_name].to_numpy(), return_inverse=True)
        d.set_annotator_ids(annotator_ids)
        df["annotator_index"] = inverse

        d.set_df(df, categories)

        return d

    @classmethod
    def _preprocess(cls, df, questions, preprocess=lambda x: x, other_columns=[]):
        df = df.copy()
        columns_to_retain = ["task_id", "user_id"] + questions + other_columns
        df = preprocess(df)
        df = df[columns_to_retain]
        return df

    @classmethod
    def _from_single_file(cls, file_name, questions, data_src=None, preprocess=lambda x: x, task_ids=None, categories=None,
                          other_columns=[]):
        """
        Create a Data object from a file.
        """
        df = pd.read_csv(file_name)
        # print("_from_single_file -> df columns before preprocessing: ", df.columns)
        df = cls._preprocess(df, questions, preprocess, other_columns)
        # print("_from_single_file -> df columns AFTER preprocessing: ", df.columns)

        return cls.from_df(df, data_src=data_src, annotator_id_col_name="user_id", questions=questions, task_ids=task_ids, categories=categories)

    @classmethod
    def from_pybossa(cls, file_name, questions, data_src=None, preprocess=lambda x: x, task_ids=None, categories=None,
                     task_info_file=None, task_file=None, field_task_key="info_media_0", other_columns=[]):
        """ 
        Create a Data object from a Pyossa file.
        """

        def get_tasks(t_csv, ti_csv, field_task_id, field_task_key):
            t_df = pd.read_csv(t_csv)
            t_df = t_df[field_task_key]
            ti_df = pd.read_csv(ti_csv)
            ti_df = ti_df[field_task_id]
            t_df = pd.merge(t_df, ti_df, left_index=True, right_index=True)
            # print("get_tasks {} {} ({}) -> \n{}".format("PyBossa", task_file, len(t_df.index), t_df))
            return t_df

        df = pd.read_csv(file_name)
        if task_info_file is not None:
            t_df = get_tasks(task_file, task_info_file, "task_id", field_task_key)
            df = pd.merge(df, t_df, how="left", left_on="task_id", right_on="task_id")
        # print("from_pybossa -> df columns before preprocessing: ", df.columns)
        df = cls._preprocess(df, questions, preprocess, other_columns=other_columns)
        #  print("from_pybossa -> df columns AFTER preprocessing: ", df.columns)
        # print(df)
        return cls.from_df(df, data_src=data_src, annotator_id_col_name="user_id", questions=questions, task_ids=task_ids, categories=categories)

    @classmethod
    def from_mturk(cls, file_name, questions, data_src=None, preprocess=lambda x: x, task_ids=None, categories=None, other_columns=[]):
        """Create a Data object from an Amazon MTurk file."""
        return cls._from_single_file(file_name, questions, data_src, preprocess, task_ids, categories, other_columns)

    @classmethod
    def from_aidr(cls, file_name, questions, data_src=None, preprocess=lambda x: x, task_ids=None, other_columns=[]):
        """Create a Data object from an AIDR file."""
        # Note: Does NOT send 'categories' arg
        return cls._from_single_file(file_name, questions, data_src, preprocess, task_ids, other_columns=other_columns)

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

    def get_question_matrix(self, question):
        df = self.df[["task_index", "annotator_index", question+"_index"]]
        #print(df[question].cat.codes)
        #df[question] = df[question].cat.codes
        return df.to_numpy()

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
        df_vals = self.df[self.df.task_index.isin(task_indices)][["task_index", field]]  # .isin() loses the order
        df_ind = pd.DataFrame(task_indices, columns=["task_index"])
        df_vals = df_ind.merge(df_vals, how="left", left_on="task_index", right_on="task_index")  # Preserve the order
        df_vals = df_vals[field]
        if unique:
            df_vals = df_vals.drop_duplicates()
        return df_vals.to_numpy()
