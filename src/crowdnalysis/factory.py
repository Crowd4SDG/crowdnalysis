from typing import Type

from . import consensus
from .consensus import DiscreteConsensusProblem
from . import dawid_skene
from . import majority_voting
from . import probabilistic
from . import cmdstan
from .data import Data


class Factory:
    """Factory class for consensus algorithms"""
    algorithms = {}

    @classmethod
    def make(cls, name, **kwargs):
        """Return an instance of the algorithm registered with the name specified

        Args:
            name (str):

        Returns:
            Type[consensus.AbstractConsensus]: The class not its instance.

        Raises:
            KeyError: If the algorithm is not registered.
        """

        try:
            return cls.algorithms[name](**kwargs)
        except KeyError:
            raise KeyError("{} algorithm is not registered. "
                           "Available options are {}.".format(name, list(cls.algorithms.keys())))

    @classmethod
    def get_consensus_algorithm(cls, name):
        """Return the corresponding consensus algorithm class

        Args:
            name (str):

        Returns:
            Type[consensus.AbstractConsensus]: The class not its instance.

        Raises:
            KeyError: If the algorithm is not registered.
        """

        try:
            return cls.algorithms[name]
        except KeyError:
            raise KeyError("{} algorithm is not registered. "
                           "Available options are {}.".format(name, list(cls.algorithms.keys())))

    @classmethod
    def register_consensus_algorithm(cls, algorithm):
        """Register a new consensus algorithm

        Args:
            algorithm (Type[AbstractConsensus]):

        Returns:
            None

        """
        cls.algorithms[algorithm.name] = algorithm
        return None

# This wrapper deals with the Data interface





Factory.register_consensus_algorithm(majority_voting.MajorityVoting)
Factory.register_consensus_algorithm(probabilistic.Probabilistic)
Factory.register_consensus_algorithm(dawid_skene.DawidSkene)
Factory.register_consensus_algorithm(cmdstan.StanMultinomialOptimizeConsensus)
Factory.register_consensus_algorithm(cmdstan.StanMultinomialEtaOptimizeConsensus)
Factory.register_consensus_algorithm(cmdstan.StanDSOptimizeConsensus)
Factory.register_consensus_algorithm(cmdstan.StanDSEtaOptimizeConsensus)
Factory.register_consensus_algorithm(cmdstan.StanDSEtaHOptimizeConsensus)