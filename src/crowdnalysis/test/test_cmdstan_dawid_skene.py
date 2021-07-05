from .common import BaseTestGenerativeConsensusModel
from ..cmdstan.dawid_skene import StanDSOptimizeConsensus, StanDSEtaHOptimizeConsensus

from .test_dawid_skene import sample

class TestStanDSOptimizeConsensus(BaseTestGenerativeConsensusModel):
    model_cls = StanDSOptimizeConsensus
    sampling_funcs = [sample]

#class TestStanDSEtaOptimizeConsensus(BaseTestGenerativeConsensusModel):
#    model_cls = StanDSEtaOptimizeConsensus
#    sampling_funcs = [sample]

class TestStanDSEtaHOptimizeConsensus(BaseTestGenerativeConsensusModel):
    model_cls = StanDSEtaHOptimizeConsensus
    sampling_funcs = [sample]

