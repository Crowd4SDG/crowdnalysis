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
        self.questions = None
        self.task_ids = None
        self.task_index_from_id = None
        self.n_tasks = None
        self.annotator_ids = None
        self.annotator_index_from_id = None
        self.n_annotators = None

    @classmethod
    def from_df(cls, df, task_id_col_name="task_id", annotator_id_col_name="annotator_id",
                questions=None, task_ids=None, categories=None):
        """ 
        Create a Data object from dataframe df.
        The dataframe contains one row for each annotation, 
        each row has a task_id, an annotator id and the answers to a set of 
        questions, each answer in one column
        """
        d = Data()

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
    def from_pybossa(cls, file_name, questions, preprocess=lambda x:x, task_ids=None, categories=None, other_columns=[]):
        """ 
        Create a Data object from a pybossa file.
        """
        df = pd.read_csv(file_name)
        columns_to_retain = ["task_id", "user_id"] + questions + other_columns
        df = preprocess(df)
        df = df[columns_to_retain]
        #print(df)

        return cls.from_df(df, annotator_id_col_name="user_id", task_ids=task_ids, categories=categories)

    @classmethod
    def from_mturk(cls, file_name, questions, preprocess=lambda x: x, task_ids=None, categories=None):
        """Create a Data object from an amazon mturk file."""
        return cls.from_pybossa(file_name, questions, preprocess, task_ids, categories)

    @classmethod
    def from_aidr(cls, file_name, questions, preprocess=lambda x: x, task_ids=None):
        """Create a Data object from an AIDR file."""
        return cls.from_pybossa(file_name, questions, preprocess, task_ids)  # Note: Does NOT send 'categories' arg

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
