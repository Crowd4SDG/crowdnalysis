from typing import Type

from . import cmdstan, dawid_skene, simple
from .consensus import AbstractSimpleConsensus


class Factory:
    """Factory class for consensus algorithms"""
    algorithms = {}

    @classmethod
    def make(cls, name: str, **kwargs) -> AbstractSimpleConsensus:
        """Return an instance of the algorithm registered with the specified `name` and created by the `kwargs`.

        Raises:
            KeyError: If the algorithm is not registered.

        """
        try:
            return cls.algorithms[name](**kwargs)
        except KeyError:
            raise KeyError("{} algorithm is not registered. "
                           "Available options are {}.".format(name, list(cls.algorithms.keys())))

    @classmethod
    def get_consensus_algorithm(cls, name: str) -> Type[AbstractSimpleConsensus]:
        """Return the corresponding consensus algorithm class (not its instance)

        Raises:
            KeyError: If the algorithm is not registered.

        """
        try:
            return cls.algorithms[name]
        except KeyError:
            raise KeyError("{} algorithm is not registered. "
                           "Available options are {}.".format(name, list(cls.algorithms.keys())))

    @classmethod
    def register_consensus_algorithm(cls, algorithm: AbstractSimpleConsensus) -> None:
        """Register a new consensus algorithm.

        Raises:
            ValueError: If `algorithm` is not a `AbstractConsensus`

        """
        if not issubclass(algorithm, AbstractSimpleConsensus):
            raise ValueError(f"{str(algorithm)} is not a 'AbstractConsensus' subclass.")
        cls.algorithms[algorithm.name] = algorithm
        return None

    @classmethod
    def unregister_consensus_algorithm(cls, name: str) -> None:
        """Unregisters an existing consensus algorithm.

        Raises:
            KeyError: If the algorithm is not registered.

        """
        try:
            del cls.algorithms[name]
        except KeyError:
            raise KeyError(f"{name} algorithm is not registered.")
        return None


Factory.register_consensus_algorithm(simple.MajorityVoting)
Factory.register_consensus_algorithm(simple.Probabilistic)
Factory.register_consensus_algorithm(dawid_skene.DawidSkene)
Factory.register_consensus_algorithm(cmdstan.StanMultinomialOptimizeConsensus)
Factory.register_consensus_algorithm(cmdstan.StanMultinomialEtaOptimizeConsensus)
Factory.register_consensus_algorithm(cmdstan.StanDSOptimizeConsensus)
Factory.register_consensus_algorithm(cmdstan.StanDSEtaOptimizeConsensus)
Factory.register_consensus_algorithm(cmdstan.StanDSEtaHOptimizeConsensus)
