import pytest

from ..consensus import AbstractSimpleConsensus
from ..factory import Factory


class DummyConsensus(AbstractSimpleConsensus):
    name = "Dummy"


class NotAnAbstractSimpleConsensus:
    name = "Dummier"


def test_all_factory_methods():
    # Register
    Factory.register_consensus_algorithm(DummyConsensus)
    with pytest.raises(ValueError):
        Factory.register_consensus_algorithm(NotAnAbstractSimpleConsensus)
    # Get
    assert Factory.get_consensus_algorithm(DummyConsensus.name) == DummyConsensus
    with pytest.raises(KeyError):
        Factory.get_consensus_algorithm(NotAnAbstractSimpleConsensus.name)
    # List
    assert DummyConsensus.name in Factory.list_registered_algorithms()
    # Make
    assert isinstance(Factory.make(DummyConsensus.name), DummyConsensus)
    with pytest.raises(KeyError):
        Factory.make(NotAnAbstractSimpleConsensus.name)
    # Unregister
    Factory.unregister_consensus_algorithm(DummyConsensus.name)
    with pytest.raises(KeyError):
        assert Factory.unregister_consensus_algorithm(DummyConsensus.name)
