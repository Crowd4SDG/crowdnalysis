import json
import numpy as np
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Dict
from numpyencoder import NumpyEncoder
from itertools import product

@dataclass
class JSONDataClass:
    def to_json(self):
        return json.dumps(dataclasses.asdict(self), cls=NumpyEncoder)

    @classmethod
    def from_json(cls, s: str):
        d = json.loads(s)
        return cls(**d)

    @classmethod
    def product_from_dict(cls, options: Dict[str, List]):
        keys, values = zip(*options.items())
        ds = [dict(zip(keys, bundle)) for bundle in product(*values)]
        return [cls(**d) for d in ds]

@dataclass
class ConsensusProblem(JSONDataClass):
    """
    Example:

        TODO (OM, 20210416): Add examples to init params
    """
    n_tasks: int = 0
    # features of the tasks
    f_T: Optional[np.ndarray] = None
    n_workers: int = 0
    # features of the workers
    f_W: Optional[np.ndarray] = None
    n_annotations: Optional[int] = None
    # task of an annotation
    t_A: np.ndarray = np.array([])
    # worker of an annotation
    w_A: np.ndarray = np.array([])
    # features of each of the annotations
    f_A: np.ndarray = np.array([])

    def __post_init__(self):
        """

        Raises:
            ValueError: If # of tasks or of workers is not determinable
        """
        if isinstance(self.f_T, list):
            self.f_T = np.array(self.f_T)
        if isinstance(self.f_W, list):
            self.f_W = np.array(self.f_W)
        if isinstance(self.t_A, list):
            self.t_A = np.array(self.t_A)
        if isinstance(self.w_A, list):
            self.w_A = np.array(self.w_A)
        if isinstance(self.f_A, list):
            self.f_A = np.array(self.f_A)
        if len(self.f_A.shape) == 1:
            self.f_A = self.f_A[:, np.newaxis]
        self.n_annotations = self.f_A.shape[0]



@dataclass
class DiscreteConsensusProblem(ConsensusProblem):
    """
    Notes:
        The DiscreteConsensusProblem enforces that:
        - There is a discrete set of real classes
        - The labels should be integers starting from 0.
        - There is a single attribute of each annotation,
          which is also discrete and which at least contains the real_classes.
          It can eventually contain additional answers such as "I do not know", or "in doubt", or "does not apply"

    Example:
        TODO (OM, 20210416): Add examples to init params
    """
    # number of different labels in the annotation
    n_labels: Optional[int] = None
    # Which labels in an annotation correspond to real hidden classes
    classes: Optional[List[int]] = None

    def __post_init__(self):
        ConsensusProblem.__post_init__(self)
        if (self.n_labels is None) and (self.f_A.shape[0] > 0):
            self.n_labels = int(np.max(self.f_A[:, 0]) + 1)
        if self.classes is None:
            # By default every label is a real class
            self.classes = list(range(self.n_labels))

    def compute_n(self):
        # TODO: This should be optimized
        # Compute the n matrix

        n = np.zeros((self.n_workers, self.n_tasks, self.n_labels))
        for i in range(self.n_annotations):
            n[self.w_A[i], self.t_A[i], self.f_A[i, 0]] += 1
        return n
