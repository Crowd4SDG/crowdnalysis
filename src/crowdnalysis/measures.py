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
        #real_indexes = real_labels + np.arange(0, I * J, J)
        strict_consensus = np.argmax(consensus, axis=1)
        acc = np.sum(strict_consensus == real_labels) / I

        return acc

# Accuracy measure taking into account potential label switching

class LSAccuracy(AbstractMeasure):
    name = "LS-accuracy"

    @classmethod
    def evaluate(cls, real_labels, consensus):
        def gen_exchanges():
            exchanges = []
            labels = np.unique(real_labels)
            for x in labels:
                for y in range(x):
                    exchanges.append((x, y))
            return exchanges

        def test_exchanges():
            for x, y in exchanges:
                copysc = np.where(strict_consensus == x, -1, strict_consensus)
                copysc = np.where(strict_consensus == y, x, copysc)
                copysc = np.where(copysc == -1, y, copysc)
                alt_acc = np.sum(copysc == real_labels) / I
                if alt_acc > acc:
                    # print(x, y, "swap OK!")
                    return True, alt_acc, copysc[:]
               # else:
                    # print(x, y, "no swap")
            return False, acc, strict_consensus

        I, J = consensus.shape
        # real_indexes = real_labels + np.arange(0, I * J, J)
        strict_consensus = np.argmax(consensus, axis=1)
        acc = np.sum(strict_consensus == real_labels) / I

        # Now we analyze potential class exchanges one by one

        exchanges = gen_exchanges()
        found, acc, strict_consensus = test_exchanges()
        while found:
            found, acc, strict_consensus = test_exchanges()

        #print("acc=", acc)
        return acc
