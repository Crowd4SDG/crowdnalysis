import logging
from logging import NullHandler

log = logging.getLogger(__name__)
log.addHandler(NullHandler())
