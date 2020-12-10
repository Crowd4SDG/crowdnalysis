from . import dawid_skene
from . import majority_voting
from . import probabilistic
from . import stan


class Factory:
    """Factory class for consensus algorithms"""
    algorithms = {}

    @classmethod
    def get_consensus_algorithm(cls, name, **kwargs):
        """Return an instance of the corresponding consensus algorithm class

        Args:
            name (str):

        Returns:
            AbstractConsensus:

        """
        return cls.algorithms[name](**kwargs)

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