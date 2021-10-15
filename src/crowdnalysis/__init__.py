import logging
from logging import NullHandler

log = logging.getLogger(__name__)
log.addHandler(NullHandler())

# Set package version
from crowdnalysis import _version
__version__ = _version.__version__
del _version

from crowdnalysis import (
    agreement,
    analysis,
    cmdstan,
    consensus,
    data,
    dawid_skene,
    factory,
    measures,
    problems,
    simple,
    visualization
)

from crowdnalysis.cmdstan import (
    multinomial,
    dawid_skene
)
