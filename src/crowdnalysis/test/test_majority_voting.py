import pytest

from .. import log
from ..consensus import DiscreteConsensusProblem
from ..majority_voting import MajorityVoting

def test_majority_voting():
    x = MajorityVoting()
    #dcp = DiscreteConsensusProblem(n_tasks = 10)
