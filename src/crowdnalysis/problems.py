import dataclasses
import json
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpyencoder import NumpyEncoder

from . import log


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


@dataclass(eq=False)
class ConsensusProblem(JSONDataClass):
    """
    Notes: see https://www.mdpi.com/2227-7390/9/8/875, p.4
        W*      A*        T*
        ^        ^        ^
    f_W |    f_A |    f_T |
        |        |        |
        |   w_A  |  t_A   |
        W <----- A -----> T

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

    def __eq__(self, other):
        """Compare two `ConsensusProblem`s"""
        # TODO (OM, 20210512): This might need further elaboration depending on the fields of future subclasses
        if not issubclass(self.__class__, other.__class__):
            return False
        return self.to_json() == other.to_json()
        # for field_, val1 in dataclasses.asdict(self).items():
        #     val2 = getattr(other, field_)
        #     if isinstance(val1, np.ndarray):
        #         eq_ = np.array_equal(val1, val2)
        #     else:
        #         eq_ = val1 == val2
        #     if not eq_:
        #         return False
        # return True


@dataclass(eq=False)
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

    def __eq__(self, other):
        return super().__eq__(other)

    def compute_n(self, ignore_zero_annots: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the `n` three-dimensional array in Dawid-Skene (1979).

        Dimensions: (n_workers, n_tasks, n_labels)

        Args:
            ignore_zero_annots: If True, filters out tasks with zero # of annotations.

        Returns:
            Tuple of (`n`, indices of filtered tasks)
        """
        # TODO: This should be optimized
        n = np.zeros((self.n_workers, self.n_tasks, self.n_labels))
        for i in range(self.n_annotations):
            n[self.w_A[i], self.t_A[i], self.f_A[i, 0]] += 1
        zero_annots = np.array([])
        if ignore_zero_annots:
            n_sum_w_A = n.sum(axis=0)
            zero_annots = np.all(n_sum_w_A == 0, axis=1)
            if np.any(zero_annots):
                n = n[:, ~zero_annots, :]
                log.info("{} tasks with zero annotations out of {} tasks in 'n' are eliminated.".format(
                    np.sum(zero_annots), n_sum_w_A.shape[0]))
        return n, np.where(zero_annots)[0]
