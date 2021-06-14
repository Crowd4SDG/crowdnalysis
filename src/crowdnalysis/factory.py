from typing import List, Type


#from . import dawid_skene, simple
from .consensus import AbstractSimpleConsensus
#from .cmdstan import multinomial as sm
#from .cmdstan import dawid_skene as sds
#from crowdnalysis.cmdstan.multinomial import StanMultinomialOptimizeConsensus

class Factory:
    """Factory class for consensus algorithms"""
    _algorithms = {}

    @classmethod
    def _msg_exception(cls, name: str, options: bool = True) -> str:
        """Return the message to be used in `KeyError`s"""
        msg = "{} algorithm is not registered.".format(name)
        if options:
            msg = "{}. Available options are {}.".format(msg, str(cls.list_registered_algorithms()))
        return msg

    @classmethod
    def make(cls, name: str, **kwargs) -> AbstractSimpleConsensus:
        """Return an instance of the algorithm registered with the specified `name` and created by the `kwargs`.

        Raises:
            KeyError: If the algorithm is not registered.

        """
        try:
            return cls._algorithms[name](**kwargs)
        except KeyError:
            raise KeyError(cls._msg_exception(name, options=True))

    @classmethod
    def get_consensus_algorithm(cls, name: str) -> Type[AbstractSimpleConsensus]:
        """Return the corresponding consensus algorithm class (not its instance)

        Raises:
            KeyError: If the algorithm is not registered.

        """
        try:
            return cls._algorithms[name]
        except KeyError:
            raise KeyError(cls._msg_exception(name, options=True))

    @classmethod
    def register_consensus_algorithm(cls, algorithm: AbstractSimpleConsensus) -> None:
        """Register a new consensus algorithm.

        Raises:
            ValueError: If `algorithm` is not a `AbstractConsensus`

        """
        if not issubclass(algorithm, AbstractSimpleConsensus):
            raise ValueError(f"{str(algorithm)} is not a 'AbstractConsensus' subclass.")
        cls._algorithms[algorithm.name] = algorithm
        return None

    @classmethod
    def unregister_consensus_algorithm(cls, name: str) -> None:
        """Unregisters an existing consensus algorithm.

        Raises:
            KeyError: If the algorithm is not registered.

        """
        try:
            del cls._algorithms[name]
        except KeyError:
            raise KeyError(cls._msg_exception(name, options=False))
        return None

    @classmethod
    def list_registered_algorithms(cls) -> List[str]:
        """Return the list of registered algorithm names"""
        return list(cls._algorithms.keys())





