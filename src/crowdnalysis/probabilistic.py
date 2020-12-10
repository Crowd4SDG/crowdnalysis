import consensus

class Probabilistic(consensus.AbstractConsensus):
    name = "Probabilisitic"
    
    def __init__(self):
        pass

    @classmethod
    def _probabilistic_consensus(cls, m, I, J, softening=0.1):
        n = cls.compute_counts(m, I, J)
        n += softening
        consensus = n / np.sum(n, axis=1)[:, np.newaxis]
        # print("Probabilistic._probabilistic_consensus ({}) -> \n".format(consensus.shape), consensus)
        return consensus, None

    def compute_consensus(self, d: Data, question, softening=0.1):
        m, I, J, K = self._get_question_matrix_and_ranges(d, question)
        return self._probabilistic_consensus(m, I, J, softening)
