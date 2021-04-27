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
        n_tasks, _ = consensus.shape
        # real_indexes = real_labels + np.arange(0, I * J, J)
        strict_consensus = np.argmax(consensus, axis=1)
        acc = np.sum(strict_consensus == real_labels) / n_tasks

        return acc


# Accuracy measure taking into account potential label switching
class LSAccuracy(AbstractMeasure):
    name = "LS-accuracy"

    @classmethod
    def evaluate(cls, real_labels, consensus):
        def gen_exchanges():
            swaps = []
            labels = np.unique(real_labels)
            for x in labels:
                for y in range(x):
                    swaps.append((x, y))
            return swaps

        def test_exchanges():
            for x, y in exchanges:
                copy_sc = np.where(strict_consensus == x, -1, strict_consensus)
                copy_sc = np.where(strict_consensus == y, x, copy_sc)
                copy_sc = np.where(copy_sc == -1, y, copy_sc)
                alt_acc = np.sum(copy_sc == real_labels) / n_tasks
                if alt_acc > acc:
                    # print(x, y, "swap OK!")
                    return True, alt_acc, copy_sc[:]
                # else:
                    # print(x, y, "no swap")
            return False, acc, strict_consensus

        n_tasks, _ = consensus.shape
        # real_indexes = real_labels + np.arange(0, I * J, J)
        strict_consensus = np.argmax(consensus, axis=1)
        acc = np.sum(strict_consensus == real_labels) / n_tasks

        # Now we analyze potential class exchanges one by one

        exchanges = gen_exchanges()
        found, acc, strict_consensus = test_exchanges()
        while found:
            found, acc, strict_consensus = test_exchanges()

        # print("acc=", acc)
        return acc
