import logging
from logging import NullHandler

log = logging.getLogger(__name__)
log.addHandler(NullHandler())

import crowdnalysis.simple
import crowdnalysis.dawid_skene
import crowdnalysis.cmdstan.multinomial
import crowdnalysis.cmdstan.dawid_skene
