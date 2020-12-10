import consensus
import pystan

class StanOptimizeConsensus(consensus.AbstractConsensus):
    name = "StanOptimize"
    def __init__(self, stan_model_filename):
        self.stan_model = pystan.StanModel(file = model_filename)
    
    def map_data_to_model(m, I, J, K):
        raise NotImplementedError
        
    def compute_consensus(self, d: Data, question):
        """Computes consensus for question question from Data d.
        returns consensus, model parameters""" 
       
        m, I, J, K = self._get_question_matrix_and_ranges(d, question)
        stan_data, init_data, args = map_to_model(m, I, J, K)
        "Here you should implement the mapping from to the input data for this Stan Model"
        "This mapping should work for any question"
        results = sm.optimizing(data = stan_data, init = init_data, **args)
        
        "Here you should obtain the consensus (the q's) from Stan and also return the additional parameters. 
        return results["q"], results 

    def success_rate(self, real_labels, crowd_labels):
        Apply this model to the crowd_labels and compare against the real_labels
        """"""
        raise NotImplementedError

class StanDSOptimizeConsensus(StanOptimizeConsensus):
    name = "StanDSOptimize"
    def __init__(self):
         StanOptimizeConsensus("DS.stan")
    
    def map_data_to_model(m,I,J,K):
        # TODO: Modify the line below to work properly with m, I, J, and K whatsoever
        stan_data = {'J': newK, 'K': newJ, 'N': newN,'I': newI,'ii': newii,'jj': newjj,'y': newyy, 'alpha':np.array([1,1])}
        # TODO: Modify the line below to define init_beta correctly for any m, I, J and K.
        init_beta = 0
        init_data = {"beta": init_beta}
        args = {"iter":2000}
        return stan_data, init_data, args
