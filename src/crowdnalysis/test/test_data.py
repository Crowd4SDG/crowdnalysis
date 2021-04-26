import pytest
from ..data import Data


# TODO (OM, 20210426): Extend tests to all Data class functionalities

def test_make_and_condition():
    assert Data.make_and_condition([]) == ""
    assert Data.make_and_condition([('A', 5), ('B', 'Yes'), ('C C', ['Yes', True])]) \
           == "`A`==5 & `B`=='Yes' & `C C` in ['Yes', True]"
    with pytest.raises(ValueError):
        Data.make_and_condition([('A', 5), ('B', 'Yes'), ('C C', )])
