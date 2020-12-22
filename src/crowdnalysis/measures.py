import numpy as np

class AbstractMeasure:
    @classmethod
    def evaluate(cls, real_labels, consensus):
        return NotImplementedError

    @classmethod
    def evaluate_crowds(cls, real_labels, crowds_consensus):
        return {crowd_name: cls.evaluate(real_labels, consensus) for crowd_name, consensus in crowds_consensus.items()}

class Accuracy(AbstractMeasure):
    name = "accuracy"
    @classmethod
    def evaluate(cls, real_labels, consensus):
        I, J = consensus.shape
        real_indexes = real_labels + np.arange(0, I * J, J)
        strict_consensus = np.argmax(consensus, axis=1)
        return np.sum(strict_consensus == real_labels) / I
