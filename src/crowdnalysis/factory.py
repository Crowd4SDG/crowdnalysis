from . import dawid_skene
from . import majority_voting
from . import probabilistic
from . import stan


class Factory:
    """Factory class for consensus algorithms"""
    algorithms = {}

    @classmethod
    def get_consensus_algorithm(cls, name):
        """Return the corresponding consensus algorithm class

        Args:
            name (str):

        Returns:
            Type(AbstractConsensus):

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
            algorithm (Type(AbstractConsensus)):

        Returns:

        """
        cls.algorithms[algorithm.name] = algorithm


Factory.register_consensus_algorithm(majority_voting.MajorityVoting)
Factory.register_consensus_algorithm(probabilistic.Probabilistic)
Factory.register_consensus_algorithm(dawid_skene.DawidSkene)
Factory.register_consensus_algorithm(stan.StanDSOptimizeConsensus)
# Factory.register_consensus_algorithm(stan.StanHDSOptimizeConsensus)
# Factory.register_consensus_algorithm(stan.StanDSSampleConsensus)