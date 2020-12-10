# Load data from Pybossa
import data 
d = data.Data.from_pybossa(...)

import consensus
from consensus import Factory

def run_consensus(d, q, algorithm_name):
  alg = Factory.getConsensusAlgorithm(algorithm_name)
  consensus, params = alg.compute_consensus(d, q)
  return consensus, params
  
for algorithm_name in Factory.algorithms:
  consensus, params = run_consensus(d, q, algorithm_name)
  print(consensus[0])
